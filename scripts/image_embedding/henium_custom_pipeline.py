#!/usr/bin/env python
"""
Custom H&Enium-style pipeline for Xenium + registered H&E.

Stages:
1) prepare: QC/filter and build a combined counts matrix across samples
2) embed-rna: RNA embeddings via CellPLM HF checkpoint (or PCA fallback)
3) embed-he: H&E patch embeddings via UNI2
4) align: contrastive alignment into joint latent space (BLEEPinput or CLIP)
5) umap: Harmony (optional) + UMAP on fused aligned embeddings

This implementation follows the methodology in:
https://doi.org/10.1101/2025.07.22.665986
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.decomposition import TruncatedSVD


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PreparedPaths:
    meta_parquet: Path
    counts_npz: Path
    genes_txt: Path
    prep_summary_json: Path
    rna_npy: Path
    he_npy: Path
    ai_npy: Path
    ag_npy: Path
    fused_npy: Path
    align_summary_json: Path
    umap_csv: Path
    umap_h5ad: Path


def build_paths(outdir: Path) -> PreparedPaths:
    return PreparedPaths(
        meta_parquet=outdir / "prepared_meta.parquet",
        counts_npz=outdir / "prepared_counts_gene_expr.npz",
        genes_txt=outdir / "prepared_genes.txt",
        prep_summary_json=outdir / "prepare_summary.json",
        rna_npy=outdir / "rna_embeddings.npy",
        he_npy=outdir / "he_embeddings.npy",
        ai_npy=outdir / "aligned_image.npy",
        ag_npy=outdir / "aligned_rna.npy",
        fused_npy=outdir / "aligned_fused.npy",
        align_summary_json=outdir / "alignment_summary.json",
        umap_csv=outdir / "joint_umap_coordinates.csv",
        umap_h5ad=outdir / "joint_umap.h5ad",
    )


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _make_unique(items: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for x in items:
        if x not in seen:
            seen[x] = 0
            out.append(x)
        else:
            seen[x] += 1
            out.append(f"{x}_{seen[x]}")
    return out


def _read_um_per_px_from_tiff(path: Path) -> float:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        xres = page.tags.get("XResolution")
        if xres is None:
            raise ValueError(f"Missing XResolution tag in {path}")
        xnum, xden = xres.value
        px_per_cm = float(xnum) / float(xden)
        um_per_px = 10000.0 / px_per_cm
    return um_per_px


def read_he_shape_and_resolution(path: Path) -> Tuple[int, int, float]:
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape
        if len(shape) != 3:
            raise ValueError(f"Expected YXS image shape for {path}, got {shape}")
        h, w, _ = shape
    um_per_px = _read_um_per_px_from_tiff(path)
    return h, w, um_per_px


def read_xenium_cell_feature_matrix(h5_path: Path) -> Tuple[sp.csr_matrix, List[str], List[str], List[str]]:
    """
    Returns:
      X_gene: cells x genes CSR matrix (Gene Expression features only)
      barcodes: cell ids in row order
      genes: gene names in column order
      feature_types: feature type labels for all original features
    """
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][:].tolist())  # features x cells
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]

        barcodes = [b.decode("utf-8") for b in m["barcodes"][:]]
        feat_names = [b.decode("utf-8") for b in m["features"]["name"][:]]
        feat_types = [b.decode("utf-8") for b in m["features"]["feature_type"][:]]

    feat_names = _make_unique(feat_names)
    x = sp.csc_matrix((data, indices, indptr), shape=shape, dtype=np.float32).T.tocsr()  # cells x features

    gene_idx = np.where(np.array(feat_types) == "Gene Expression")[0]
    x_gene = x[:, gene_idx].tocsr()
    genes = [feat_names[i] for i in gene_idx]
    return x_gene, barcodes, genes, feat_types


def sparse_row_scale(x: sp.csr_matrix, scale: np.ndarray) -> sp.csr_matrix:
    d = sp.diags(scale.astype(np.float32))
    return d @ x


def normalize_log1p_cpm(x: sp.csr_matrix, target_sum: float = 1e6) -> sp.csr_matrix:
    lib = np.asarray(x.sum(axis=1)).ravel()
    lib = np.maximum(lib, 1.0)
    scale = target_sum / lib
    x2 = sparse_row_scale(x, scale)
    x2.data = np.log1p(x2.data)
    return x2


def load_cells_parquet(path: Path) -> pd.DataFrame:
    cols = [
        "cell_id",
        "x_centroid",
        "y_centroid",
        "transcript_counts",
        "control_probe_counts",
        "control_codeword_counts",
        "genomic_control_counts",
        "total_counts",
        "cell_area",
        "nucleus_count",
        "nucleus_area",
    ]
    df = pd.read_parquet(path, columns=cols)
    return df


def apply_qc(
    df: pd.DataFrame,
    transcript_min_count: int,
    max_control_fraction: float,
    cell_area_qmin: float,
    cell_area_qmax: float,
) -> pd.DataFrame:
    df = df.copy()
    ctrl = df["control_probe_counts"].fillna(0) + df["control_codeword_counts"].fillna(0) + df["genomic_control_counts"].fillna(0)
    total = df["total_counts"].fillna(0)
    ctrl_frac = np.divide(ctrl, total, out=np.zeros(len(df), dtype=np.float32), where=(total > 0))
    df["control_fraction"] = ctrl_frac

    area_lo = float(df["cell_area"].quantile(cell_area_qmin))
    area_hi = float(df["cell_area"].quantile(cell_area_qmax))

    mask = (
        (df["transcript_counts"] >= transcript_min_count)
        & (df["control_fraction"] <= max_control_fraction)
        & (df["cell_area"] >= area_lo)
        & (df["cell_area"] <= area_hi)
    )
    return df.loc[mask].copy()


def apply_edge_filter(
    df: pd.DataFrame,
    img_h: int,
    img_w: int,
    um_per_px: float,
    patch_size: int,
    upsample_factor: float,
) -> pd.DataFrame:
    df = df.copy()
    df["x_px"] = df["x_centroid"] / um_per_px
    df["y_px"] = df["y_centroid"] / um_per_px

    # Equivalent crop in original image if using upsample-first strategy.
    effective_crop = patch_size / max(upsample_factor, 1e-8)
    half = effective_crop / 2.0

    mask = (
        (df["x_px"] >= half)
        & (df["x_px"] < (img_w - half))
        & (df["y_px"] >= half)
        & (df["y_px"] < (img_h - half))
    )
    return df.loc[mask].copy()


def prepare_data(cfg: dict, outdir: Path, max_cells_per_sample: int | None = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    p = build_paths(outdir)

    qc_cfg = cfg["qc"]
    patch_cfg = cfg["patch"]
    apply_edge = bool(patch_cfg.get("apply_edge_filter", True))

    all_meta = []
    all_x = []
    genes_ref = None
    summary = {"samples": []}

    for s in cfg["samples"]:
        sid = s["sample_id"]
        xenium_dir = Path(s["xenium_dir"])
        he_path = Path(s["he_image"])

        log(f"[prepare] sample={sid}: loading cell metadata")
        cells = load_cells_parquet(xenium_dir / "cells.parquet")
        n_raw = len(cells)

        h, w, um_per_px = read_he_shape_and_resolution(he_path)
        if "he_um_per_px_override" in s and s["he_um_per_px_override"] is not None:
            um_per_px = float(s["he_um_per_px_override"])

        log(f"[prepare] sample={sid}: applying QC (transcripts >= {qc_cfg['transcript_min_count']})")
        cells = apply_qc(
            cells,
            transcript_min_count=int(qc_cfg["transcript_min_count"]),
            max_control_fraction=float(qc_cfg["max_control_fraction"]),
            cell_area_qmin=float(qc_cfg["cell_area_quantile_min"]),
            cell_area_qmax=float(qc_cfg["cell_area_quantile_max"]),
        )
        n_qc = len(cells)

        if apply_edge:
            cells = apply_edge_filter(
                cells,
                img_h=h,
                img_w=w,
                um_per_px=um_per_px,
                patch_size=int(patch_cfg["patch_size_px"]),
                upsample_factor=float(patch_cfg["upsample_factor"]),
            )
        n_edge = len(cells)

        if max_cells_per_sample is not None and n_edge > max_cells_per_sample:
            cells = cells.sample(n=max_cells_per_sample, random_state=17).copy()
            n_edge = len(cells)

        log(f"[prepare] sample={sid}: loading cell feature matrix")
        x_gene, barcodes, genes, _ = read_xenium_cell_feature_matrix(xenium_dir / "cell_feature_matrix.h5")
        barcode_index = pd.Index(barcodes)

        ridx = barcode_index.get_indexer(cells["cell_id"].astype(str))
        valid = ridx >= 0
        if (~valid).any():
            cells = cells.loc[valid].copy()
            ridx = ridx[valid]

        x_sub = x_gene[ridx].tocsr()

        if genes_ref is None:
            genes_ref = genes
        else:
            if genes != genes_ref:
                raise ValueError(
                    f"Gene order mismatch for sample={sid}. "
                    "This script currently expects identical gene order across samples."
                )

        cells["sample_id"] = sid
        cells["he_image"] = str(he_path)
        cells["um_per_px"] = um_per_px

        all_meta.append(cells.reset_index(drop=True))
        all_x.append(x_sub)

        summary["samples"].append(
            {
                "sample_id": sid,
                "raw_cells": int(n_raw),
                "post_qc_cells": int(n_qc),
                "post_edge_cells": int(n_edge),
                "he_shape": [int(h), int(w)],
                "he_um_per_px": float(um_per_px),
            }
        )

    meta = pd.concat(all_meta, axis=0, ignore_index=True)
    x = sp.vstack(all_x).tocsr()

    summary["total_cells"] = int(meta.shape[0])
    summary["total_genes"] = int(x.shape[1])

    meta.to_parquet(p.meta_parquet, index=False)
    sp.save_npz(p.counts_npz, x)
    with open(p.genes_txt, "w", encoding="utf-8") as f:
        for g in genes_ref:
            f.write(f"{g}\n")
    with open(p.prep_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"[prepare] wrote: {p.meta_parquet}")
    log(f"[prepare] wrote: {p.counts_npz}")
    log(f"[prepare] wrote: {p.prep_summary_json}")


def _load_prepared(outdir: Path) -> Tuple[pd.DataFrame, sp.csr_matrix, List[str], PreparedPaths]:
    p = build_paths(outdir)
    meta = pd.read_parquet(p.meta_parquet)
    x = sp.load_npz(p.counts_npz).tocsr()
    with open(p.genes_txt, "r", encoding="utf-8") as f:
        genes = [ln.strip() for ln in f if ln.strip()]
    if x.shape[0] != meta.shape[0]:
        raise ValueError(f"Row mismatch: counts={x.shape[0]} meta={meta.shape[0]}")
    if x.shape[1] != len(genes):
        raise ValueError(f"Column mismatch: counts={x.shape[1]} genes={len(genes)}")
    return meta, x, genes, p


def _scipy_csr_to_torch_sparse(csr: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = csr.tocoo()
    idx = np.vstack((coo.row, coo.col)).astype(np.int64)
    indices = torch.from_numpy(idx).to(device)
    values = torch.from_numpy(coo.data.astype(np.float32)).to(device)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


def _load_cellplm_modules(cellplm_code_dir: str | None = None):
    try:
        from CellPLM.model import OmicsFormer
        return OmicsFormer
    except Exception:
        if cellplm_code_dir:
            sys.path.insert(0, str(cellplm_code_dir))
            from CellPLM.model import OmicsFormer  # type: ignore
            return OmicsFormer
        raise


def _resolve_hf_files(repo_id: str, model_file: str, config_file: str, token: str | None) -> Tuple[str, str]:
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(repo_id=repo_id, filename=config_file, token=token)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_file, token=token)
    return cfg_path, model_path


def embed_rna_cellplm_hf(
    x: sp.csr_matrix,
    genes: List[str],
    rna_cfg: dict,
    device: torch.device,
    batch_size: int = 4096,
    cellplm_code_dir: str | None = None,
) -> np.ndarray:
    repo_id = rna_cfg["hf_repo_id"]
    model_file = rna_cfg.get("hf_model_file", "model.pth")
    config_file = rna_cfg.get("hf_config_file", "config.pkl")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    cfg_path, model_path = _resolve_hf_files(repo_id, model_file, config_file, token)

    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Unexpected CellPLM config format at {cfg_path}")

    model_genes = cfg.get("gene_list")
    if model_genes is None:
        raise ValueError("CellPLM config missing 'gene_list'")

    model_gene_set = set(model_genes)
    overlap_genes = [g for g in genes if g in model_gene_set]
    overlap = len(overlap_genes)
    log(f"[embed-rna] CellPLM gene overlap: {overlap}/{len(genes)} ({overlap/len(genes):.4f})")

    # Guardrail: avoid silently running with near-zero overlap.
    min_overlap = int(rna_cfg.get("min_gene_overlap", 200))
    if overlap < min_overlap:
        raise ValueError(
            "CellPLM gene overlap too small. "
            f"Got {overlap}, expected >= {min_overlap}. "
            "This usually means species/gene-ID mismatch (e.g., mouse CellPLM with human Xenium panel)."
        )

    gene_to_col = {g: i for i, g in enumerate(genes)}
    col_idx = np.array([gene_to_col[g] for g in overlap_genes], dtype=np.int64)
    x_sub = x[:, col_idx].tocsr()

    OmicsFormer = _load_cellplm_modules(cellplm_code_dir)
    model_cfg = dict(cfg)
    model_cfg["head_type"] = "embedder"
    model = OmicsFormer(**model_cfg)

    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Unsupported CellPLM checkpoint structure")

    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    log(f"[embed-rna] CellPLM load_state_dict missing={len(missing)} unexpected={len(unexpected)}")

    model.eval().to(device)

    embs = []
    n = x_sub.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = x_sub[start:end]
            x_sparse = _scipy_csr_to_torch_sparse(xb, device=device)
            inp = {
                "x_seq": x_sparse,
                "coord": torch.full((end - start, 2), -1.0, dtype=torch.float32, device=device),
                "batch": torch.zeros(end - start, dtype=torch.long, device=device),
            }
            out, _ = model(inp, input_gene_list=overlap_genes)
            z = out["pred"].detach().cpu().numpy().astype(np.float32)
            embs.append(z)

    return np.vstack(embs)

def embed_rna_cellplm_official(
    x: sp.csr_matrix,
    genes: List[str],
    rna_cfg: dict,
    device: torch.device,
) -> np.ndarray:
    try:
        import anndata as ad
    except Exception as e:
        raise RuntimeError(f"anndata is required for CellPLM official pipeline: {e}")

    try:
        from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
    except Exception as e:
        raise RuntimeError(f"CellPLM package not available: {e}")

    pretrain_prefix = rna_cfg.get("pretrain_prefix")
    pretrain_dir = rna_cfg.get("pretrain_directory")
    if not pretrain_prefix or not pretrain_dir:
        raise ValueError("CellPLM official method requires pretrain_prefix and pretrain_directory")

    config_path = Path(pretrain_dir) / f"{pretrain_prefix}.config.json"
    ckpt_path = Path(pretrain_dir) / f"{pretrain_prefix}.best.ckpt"
    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(
            "CellPLM checkpoint files not found. "
            f"Expected {config_path} and {ckpt_path}"
        )

    adata = ad.AnnData(X=x)
    adata.var = pd.DataFrame(index=genes)

    pipeline = CellEmbeddingPipeline(
        pretrain_prefix=pretrain_prefix,
        pretrain_directory=str(pretrain_dir),
    )

    emb = pipeline.predict(
        adata,
        inference_config={"batch_size": int(rna_cfg.get("batch_size", 50000))},
        ensembl_auto_conversion=bool(rna_cfg.get("ensembl_auto_conversion", True)),
        device=str(device),
    )
    return emb.detach().cpu().numpy().astype(np.float32)


def embed_rna_pca(
    x: sp.csr_matrix,
    n_components: int,
    global_median_scale: bool,
    random_state: int = 17,
) -> np.ndarray:
    x_use = x.copy().tocsr()

    if global_median_scale:
        lib = np.asarray(x_use.sum(axis=1)).ravel()
        positive = lib[lib > 0]
        if len(positive) == 0:
            raise ValueError("All cells have zero library size")
        global_median = float(np.median(positive))
        scale = np.divide(global_median, lib, out=np.ones_like(lib, dtype=np.float32), where=(lib > 0))
        x_use = sparse_row_scale(x_use, scale)

    x_use = normalize_log1p_cpm(x_use, target_sum=1e6)
    k = max(2, min(n_components, x_use.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    z = svd.fit_transform(x_use).astype(np.float32)
    return z


def embed_rna(cfg: dict, outdir: Path, device: torch.device, cellplm_code_dir: str | None = None) -> None:
    meta, x, genes, p = _load_prepared(outdir)
    _ = meta

    rna_cfg = cfg["rna_embedding"]
    method = rna_cfg.get("method", "cellplm_hf")

    if method == "cellplm_hf":
        try:
            z = embed_rna_cellplm_hf(
                x=x,
                genes=genes,
                rna_cfg=rna_cfg,
                device=device,
                batch_size=int(rna_cfg.get("batch_size", 4096)),
                cellplm_code_dir=cellplm_code_dir,
            )
            used_method = "cellplm_hf"
        except Exception as e:
            if bool(rna_cfg.get("allow_pca_fallback", True)):
                log(f"[embed-rna] CellPLM failed: {e}")
                log("[embed-rna] Falling back to PCA embeddings")
                z = embed_rna_pca(
                    x=x,
                    n_components=int(rna_cfg.get("pca_components", 256)),
                    global_median_scale=bool(rna_cfg.get("global_median_scale", False)),
                    random_state=17,
                )
                used_method = "pca_fallback"
            else:
                raise
    elif method == "cellplm_official":
        try:
            z = embed_rna_cellplm_official(
                x=x,
                genes=genes,
                rna_cfg=rna_cfg,
                device=device,
            )
            used_method = "cellplm_official"
        except Exception as e:
            if bool(rna_cfg.get("allow_pca_fallback", True)):
                log(f"[embed-rna] CellPLM official failed: {e}")
                log("[embed-rna] Falling back to PCA embeddings")
                z = embed_rna_pca(
                    x=x,
                    n_components=int(rna_cfg.get("pca_components", 256)),
                    global_median_scale=bool(rna_cfg.get("global_median_scale", False)),
                    random_state=17,
                )
                used_method = "pca_fallback"
            else:
                raise
    elif method == "pca":
        z = embed_rna_pca(
            x=x,
            n_components=int(rna_cfg.get("pca_components", 256)),
            global_median_scale=bool(rna_cfg.get("global_median_scale", False)),
            random_state=17,
        )
        used_method = "pca"
    else:
        raise ValueError(f"Unsupported RNA method: {method}")

    np.save(p.rna_npy, z.astype(np.float32))
    with open(outdir / "rna_embedding_summary.json", "w", encoding="utf-8") as f:
        json.dump({"method": used_method, "shape": list(z.shape)}, f, indent=2)
    log(f"[embed-rna] wrote: {p.rna_npy} shape={z.shape}")


def _load_he_model(model_id: str, token: str | None, device: torch.device) -> tuple[nn.Module, tuple[float, float, float], tuple[float, float, float]]:
    import timm

    if token:
        os.environ["HF_TOKEN"] = token

    # UNI2-h requires explicit architecture kwargs from the model card.
    if model_id == "MahmoodLab/UNI2-h":
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model(f"hf-hub:{model_id}", pretrained=True, **timm_kwargs)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif model_id == "bioptimus/H-optimus-1":
        model = timm.create_model(
            f"hf-hub:{model_id}",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
            num_classes=0,
        )
        mean = (0.707223, 0.578729, 0.703617)
        std = (0.211883, 0.230117, 0.177517)
    else:
        # For standard timm HF-Hub models, num_classes=0 returns penultimate embeddings.
        model = timm.create_model(f"hf-hub:{model_id}", pretrained=True, num_classes=0)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    model.eval().to(device)
    return model, mean, std


def _preprocess_patch_rgb_uint8(
    patch: np.ndarray,
    mean_rgb: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std_rgb: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    x = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(mean_rgb, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(std_rgb, dtype=torch.float32).view(3, 1, 1)
    x = (x - mean) / std
    return x


def _extract_patch(
    image: np.ndarray,
    x_px: float,
    y_px: float,
    patch_size: int,
    upsample_factor: float,
) -> np.ndarray:
    # Equivalent to upsample-then-crop strategy by shrinking FOV before resizing.
    effective = int(round(patch_size / max(upsample_factor, 1e-8)))
    effective = max(8, effective)
    half = effective // 2

    cx = int(round(x_px))
    cy = int(round(y_px))
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + effective
    y1 = y0 + effective

    crop = image[y0:y1, x0:x1, :]
    if crop.shape[0] != effective or crop.shape[1] != effective:
        # Should be rare because edge filtering already handled this.
        pad = np.zeros((effective, effective, 3), dtype=image.dtype)
        pad[: crop.shape[0], : crop.shape[1], :] = crop
        crop = pad

    if effective != patch_size:
        crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LANCZOS4)

    return crop


def embed_he(cfg: dict, outdir: Path, device: torch.device) -> None:
    meta, _, _, p = _load_prepared(outdir)
    meta = meta.copy()

    img_cfg = cfg["image_embedding"]
    patch_cfg = cfg["patch"]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model_id = img_cfg.get("model_id", "MahmoodLab/UNI2-h")
    batch_size = int(img_cfg.get("batch_size", 128))

    log(f"[embed-he] loading H&E model: {model_id}")
    model, mean_rgb, std_rgb = _load_he_model(model_id=model_id, token=token, device=device)

    patch_size = int(patch_cfg.get("patch_size_px", 224))
    upsample_factor = float(patch_cfg.get("upsample_factor", 1.0))

    # Edge-filtered runs already materialize pixel-space centroids during prepare.
    # For no-filter runs, derive them here from Xenium coordinates.
    if "x_px" not in meta.columns or "y_px" not in meta.columns:
        if "x_centroid" not in meta.columns or "y_centroid" not in meta.columns or "um_per_px" not in meta.columns:
            raise ValueError("prepared_meta missing x/y centroid columns needed for H&E patch extraction")
        meta["x_px"] = meta["x_centroid"].astype(np.float32) / meta["um_per_px"].astype(np.float32)
        meta["y_px"] = meta["y_centroid"].astype(np.float32) / meta["um_per_px"].astype(np.float32)

    emb_all = np.zeros((meta.shape[0], 1536), dtype=np.float32)  # resized after first batch if needed
    first = True

    total_done = 0
    with torch.no_grad():
        for sid, g in meta.groupby("sample_id", sort=False):
            he_path = Path(g["he_image"].iloc[0])
            log(f"[embed-he] loading image for sample={sid}: {he_path}")
            img = tifffile.imread(he_path)

            sample_indices = g.index.to_numpy()
            for start in range(0, len(sample_indices), batch_size):
                end = min(start + batch_size, len(sample_indices))
                ib = sample_indices[start:end]
                patches = []
                for i in ib:
                    row = meta.iloc[i]
                    patch = _extract_patch(
                        image=img,
                        x_px=float(row["x_px"]),
                        y_px=float(row["y_px"]),
                        patch_size=patch_size,
                        upsample_factor=upsample_factor,
                    )
                    patches.append(_preprocess_patch_rgb_uint8(patch, mean_rgb=mean_rgb, std_rgb=std_rgb))

                xb = torch.stack(patches, dim=0).to(device)
                zb = model(xb)
                if isinstance(zb, (list, tuple)):
                    zb = zb[0]
                if zb.ndim > 2:
                    zb = zb.mean(dim=1)
                zb = zb.detach().cpu().numpy().astype(np.float32)

                if first:
                    emb_all = np.zeros((meta.shape[0], zb.shape[1]), dtype=np.float32)
                    first = False
                emb_all[ib] = zb

                total_done += len(ib)
                if (total_done // batch_size) % 50 == 0:
                    log(f"[embed-he] processed {total_done}/{meta.shape[0]} cells")

            # Free image array before loading next sample.
            del img

    np.save(p.he_npy, emb_all)
    with open(outdir / "he_embedding_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": model_id,
                "shape": list(emb_all.shape),
                "mean": list(mean_rgb),
                "std": list(std_rgb),
            },
            f,
            indent=2,
        )
    log(f"[embed-he] wrote: {p.he_npy} shape={emb_all.shape}")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeniumAlignModel(nn.Module):
    def __init__(self, d_i: int, d_g: int, d_a: int, dropout: float = 0.3):
        super().__init__()
        self.pi = ProjectionHead(d_i, d_a, dropout=dropout)
        self.pg = ProjectionHead(d_g, d_a, dropout=dropout)

    def forward(self, zi: torch.Tensor, zg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ai = self.pi(zi)
        ag = self.pg(zg)
        ai = F.normalize(ai, p=2, dim=1)
        ag = F.normalize(ag, p=2, dim=1)
        return ai, ag


class SoftContrastiveLoss(nn.Module):
    def __init__(
        self,
        lambda_image: float,
        target_type: str,
        alpha: float,
        target_temperature: float,
        learnable_logit_temperature: bool,
        init_tau: float = 0.07,
    ):
        super().__init__()
        self.lambda_image = float(lambda_image)
        self.target_type = target_type
        self.alpha = float(alpha)
        self.target_temperature = float(target_temperature)
        self.learnable_logit_temperature = bool(learnable_logit_temperature)
        if self.learnable_logit_temperature:
            self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau), dtype=torch.float32))
        else:
            self.register_buffer("fixed_tau", torch.tensor(float(init_tau), dtype=torch.float32))

    def _tau(self) -> torch.Tensor:
        if self.learnable_logit_temperature:
            return self.log_tau.exp().clamp(1e-3, 2.0)
        return self.fixed_tau

    @staticmethod
    def _cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return x @ y.T

    def _target(self, zi: torch.Tensor, zg: torch.Tensor, device: torch.device) -> torch.Tensor:
        b = zi.shape[0]
        if self.target_type == "clip":
            return torch.eye(b, device=device)
        if self.target_type == "bleepinput":
            sgg = self._cosine_sim(zg, zg)
            sii = self._cosine_sim(zi, zi)
            t = self.alpha * sgg + (1.0 - self.alpha) * sii
            return F.softmax(t / self.target_temperature, dim=1)
        raise ValueError(f"Unsupported target_type: {self.target_type}")

    def _soft_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tau = self._tau()
        logp = F.log_softmax(logits / tau, dim=1)
        return -(targets * logp).sum(dim=1).mean()

    def forward(self, zi: torch.Tensor, zg: torch.Tensor, ai: torch.Tensor, ag: torch.Tensor) -> torch.Tensor:
        s = self._cosine_sim(ag, ai)
        t = self._target(zi=zi, zg=zg, device=zi.device)
        l_gene = self._soft_ce(s, t)
        l_image = self._soft_ce(s.T, t.T)
        return self.lambda_image * l_image + (1.0 - self.lambda_image) * l_gene


def _iterate_minibatches(n: int, batch_size: int, shuffle: bool = True) -> List[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def align_embeddings(cfg: dict, outdir: Path, device: torch.device) -> None:
    meta, _, _, p = _load_prepared(outdir)
    _ = meta

    zi_np = np.load(p.he_npy)
    zg_np = np.load(p.rna_npy)

    if zi_np.shape[0] != zg_np.shape[0]:
        raise ValueError("Embedding row mismatch between image and RNA")

    zi = torch.from_numpy(zi_np.astype(np.float32))
    zg = torch.from_numpy(zg_np.astype(np.float32))

    n = zi.shape[0]
    idx = np.arange(n)
    np.random.seed(17)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    tr_idx = idx[:split]
    va_idx = idx[split:]

    acfg = cfg["alignment"]
    model = HeniumAlignModel(
        d_i=int(zi.shape[1]),
        d_g=int(zg.shape[1]),
        d_a=int(acfg.get("latent_dim", 128)),
        dropout=float(acfg.get("dropout", 0.3)),
    ).to(device)

    criterion = SoftContrastiveLoss(
        lambda_image=float(acfg.get("lambda_gene_to_image", 0.5)),
        target_type=str(acfg.get("target_type", "bleepinput")).lower(),
        alpha=float(acfg.get("alpha", 0.5)),
        target_temperature=float(acfg.get("target_temperature", 0.1)),
        learnable_logit_temperature=bool(acfg.get("use_learnable_logit_temperature", True)),
        init_tau=float(acfg.get("init_logit_temperature", 0.07)),
    ).to(device)

    params = list(model.parameters()) + list(criterion.parameters())
    optim = torch.optim.AdamW(
        params,
        lr=float(acfg.get("learning_rate", 1e-3)),
        weight_decay=float(acfg.get("weight_decay", 1e-4)),
    )

    batch_size = int(acfg.get("batch_size", 64))
    max_epochs = int(acfg.get("max_epochs", 20))
    patience = int(acfg.get("early_stop_patience", 5))

    best_val = float("inf")
    best_state = None
    stale = 0
    history = []

    log("[align] training alignment model")
    for epoch in range(1, max_epochs + 1):
        model.train()
        criterion.train()
        tr_losses = []

        for bi in _iterate_minibatches(len(tr_idx), batch_size, shuffle=True):
            ridx = tr_idx[bi]
            zi_b = zi[ridx].to(device)
            zg_b = zg[ridx].to(device)

            ai_b, ag_b = model(zi_b, zg_b)
            loss = criterion(zi_b, zg_b, ai_b, ag_b)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        criterion.eval()
        va_losses = []
        with torch.no_grad():
            for bi in _iterate_minibatches(len(va_idx), batch_size, shuffle=False):
                ridx = va_idx[bi]
                zi_b = zi[ridx].to(device)
                zg_b = zg[ridx].to(device)
                ai_b, ag_b = model(zi_b, zg_b)
                loss = criterion(zi_b, zg_b, ai_b, ag_b)
                va_losses.append(float(loss.item()))

        tr_mean = float(np.mean(tr_losses))
        va_mean = float(np.mean(va_losses))
        history.append({"epoch": epoch, "train_loss": tr_mean, "val_loss": va_mean})
        log(f"[align] epoch={epoch:02d} train={tr_mean:.4f} val={va_mean:.4f}")

        if va_mean < best_val - 1e-6:
            best_val = va_mean
            stale = 0
            best_state = {
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
            }
        else:
            stale += 1
            if stale >= patience:
                log(f"[align] early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("No model state saved during alignment training")

    model.load_state_dict(best_state["model"])  # type: ignore[arg-type]
    criterion.load_state_dict(best_state["criterion"])  # type: ignore[arg-type]

    # Full inference in chunks.
    model.eval()
    ai_all = np.zeros((n, int(acfg.get("latent_dim", 128))), dtype=np.float32)
    ag_all = np.zeros((n, int(acfg.get("latent_dim", 128))), dtype=np.float32)

    with torch.no_grad():
        for bi in _iterate_minibatches(n, max(2048, batch_size * 8), shuffle=False):
            zi_b = zi[bi].to(device)
            zg_b = zg[bi].to(device)
            ai_b, ag_b = model(zi_b, zg_b)
            ai_all[bi] = ai_b.detach().cpu().numpy().astype(np.float32)
            ag_all[bi] = ag_b.detach().cpu().numpy().astype(np.float32)

    fused = (ai_all + ag_all) / 2.0
    fused = fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8)

    np.save(p.ai_npy, ai_all)
    np.save(p.ag_npy, ag_all)
    np.save(p.fused_npy, fused)

    align_summary = {
        "best_val_loss": best_val,
        "history": history,
        "input_shape_image": list(zi_np.shape),
        "input_shape_rna": list(zg_np.shape),
        "aligned_shape": list(fused.shape),
        "target_type": acfg.get("target_type", "bleepinput"),
    }
    with open(p.align_summary_json, "w", encoding="utf-8") as f:
        json.dump(align_summary, f, indent=2)

    log(f"[align] wrote: {p.ai_npy}, {p.ag_npy}, {p.fused_npy}")


def run_umap(cfg: dict, outdir: Path) -> None:
    import anndata as ad
    import scanpy as sc
    import matplotlib.pyplot as plt

    meta, _, _, p = _load_prepared(outdir)
    fused = np.load(p.fused_npy).astype(np.float32)

    adata = ad.AnnData(X=fused)
    adata.obs = meta[["cell_id", "sample_id", "transcript_counts", "control_fraction", "cell_area"]].copy()

    icfg = cfg["integration"]
    pca_mode = str(icfg.get("pca_mode", "scanpy")).lower()
    if pca_mode not in {"scanpy", "none"}:
        raise ValueError("integration.pca_mode must be one of: scanpy, none")

    run_harmony_cfg = str(icfg.get("run_harmony", "auto")).lower()
    run_harmony = False
    if run_harmony_cfg == "true":
        run_harmony = True
    elif run_harmony_cfg == "false":
        run_harmony = False
    elif run_harmony_cfg == "auto":
        # Conservative default for multi-sample integration.
        run_harmony = adata.obs["sample_id"].nunique() > 1
    else:
        raise ValueError("integration.run_harmony must be one of: true, false, auto")

    if pca_mode == "none":
        if run_harmony:
            raise ValueError(
                "integration.pca_mode='none' requires integration.run_harmony='false'. "
                "Set run_harmony: false for direct embedding UMAP."
            )
        direct_rep_key = str(icfg.get("direct_rep_key", "X_fused"))
        adata.obsm[direct_rep_key] = fused
        rep = direct_rep_key
    else:
        n_pcs_req = int(icfg.get("pca_components", 50))
        # sklearn/scanpy arpack requires n_components < min(n_samples, n_features)
        max_npcs = min(adata.n_obs - 1, fused.shape[1] - 1)
        if max_npcs < 2:
            max_npcs = min(adata.n_obs - 1, fused.shape[1])
        if max_npcs < 2:
            raise ValueError(f"Insufficient dimensions for PCA: n_obs={adata.n_obs}, n_features={fused.shape[1]}")
        n_pcs = max(2, min(n_pcs_req, max_npcs))

        sc.pp.pca(adata, n_comps=n_pcs)

        if run_harmony:
            try:
                import harmonypy as hm
            except Exception:
                log("[umap] harmonypy is not installed. Falling back to PCA without Harmony.")
                run_harmony = False

        if run_harmony:
            key = str(icfg.get("harmony_key", "sample_id"))
            ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, key)
            zcorr = np.asarray(ho.Z_corr)
            if zcorr.shape[0] == adata.n_obs:
                x_joint = zcorr
            elif zcorr.shape[1] == adata.n_obs:
                x_joint = zcorr.T
            else:
                raise ValueError(
                    f"Unexpected Harmony shape {zcorr.shape} for n_obs={adata.n_obs}"
                )
            adata.obsm["X_joint"] = x_joint
            rep = "X_joint"
        else:
            rep = "X_pca"

    umap_metric = str(icfg.get("umap_metric", "euclidean"))
    sc.pp.neighbors(
        adata,
        use_rep=rep,
        n_neighbors=int(icfg.get("umap_n_neighbors", 30)),
        metric=umap_metric,
    )
    sc.tl.umap(
        adata,
        min_dist=float(icfg.get("umap_min_dist", 0.3)),
        random_state=int(icfg.get("random_state", 17)),
    )

    coords = pd.DataFrame(adata.obsm["X_umap"], columns=["umap1", "umap2"])
    out = pd.concat([adata.obs.reset_index(drop=True), coords], axis=1)
    out.to_csv(p.umap_csv, index=False)
    adata.write_h5ad(p.umap_h5ad)

    # Simple diagnostic plots.
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    for sid in sorted(out["sample_id"].unique()):
        m = out["sample_id"] == sid
        ax[0].scatter(out.loc[m, "umap1"], out.loc[m, "umap2"], s=1, alpha=0.6, label=sid)
    ax[0].set_title("Joint UMAP by sample")
    ax[0].legend(markerscale=6, fontsize=7)

    scmap = ax[1].scatter(out["umap1"], out["umap2"], s=1, c=out["transcript_counts"], cmap="viridis", alpha=0.7)
    ax[1].set_title("Joint UMAP by transcript_counts")
    plt.colorbar(scmap, ax=ax[1], fraction=0.046, pad=0.04)

    for a in ax:
        a.set_xlabel("UMAP1")
        a.set_ylabel("UMAP2")

    fig.tight_layout()
    fig_path = outdir / "joint_umap_qc_panels.png"
    fig.savefig(fig_path)
    plt.close(fig)

    with open(outdir / "umap_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_cells": int(adata.n_obs),
                "n_features": int(adata.n_vars),
                "used_representation": rep,
                "pca_mode": pca_mode,
                "run_harmony": bool(run_harmony),
                "n_neighbors": int(icfg.get("umap_n_neighbors", 30)),
                "min_dist": float(icfg.get("umap_min_dist", 0.3)),
                "metric": umap_metric,
            },
            f,
            indent=2,
        )

    log(f"[umap] wrote: {p.umap_csv}")
    log(f"[umap] wrote: {p.umap_h5ad}")
    log(f"[umap] wrote: {fig_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Custom H&Enium-style Xenium+H&E integration pipeline")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument(
        "--step",
        required=True,
        choices=["prepare", "embed-rna", "embed-he", "align", "umap", "all"],
        help="Pipeline step",
    )
    ap.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-cells-per-sample", type=int, default=None)
    ap.add_argument("--cellplm-code-dir", default=None, help="Optional local path containing CellPLM source package")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(Path(args.config))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    log(f"Running step={args.step} on device={device}")

    if args.step in ["prepare", "all"]:
        prepare_data(cfg=cfg, outdir=outdir, max_cells_per_sample=args.max_cells_per_sample)

    if args.step in ["embed-rna", "all"]:
        embed_rna(cfg=cfg, outdir=outdir, device=device, cellplm_code_dir=args.cellplm_code_dir)

    if args.step in ["embed-he", "all"]:
        embed_he(cfg=cfg, outdir=outdir, device=device)

    if args.step in ["align", "all"]:
        align_embeddings(cfg=cfg, outdir=outdir, device=device)

    if args.step in ["umap", "all"]:
        run_umap(cfg=cfg, outdir=outdir)

    log("Done")


if __name__ == "__main__":
    main()
