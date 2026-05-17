#!/usr/bin/env python3
"""Image-graph UMAP from Harmony-corrected image embeddings."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from harmonypy import run_harmony
from pynndescent import NNDescent
from sklearn.decomposition import PCA
from umap.umap_ import fuzzy_simplicial_set


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build post-Harmony image graph UMAP from image embeddings")
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--embedding-file", default="he_embeddings.npy")
    ap.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="PCA components before Harmony. Use 0 to skip PCA and use the embedding directly.",
    )
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--min-dist", type=float, default=0.3)
    ap.add_argument("--harmony-key", default="sample_id")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return (x / denom).astype(np.float32, copy=False)


def canonicalize_csr(mat: sp.spmatrix) -> sp.csr_matrix:
    out = mat.tocsr(copy=True)
    out.sum_duplicates()
    out.sort_indices()
    out.eliminate_zeros()
    return out


def make_obs_names(meta: pd.DataFrame) -> np.ndarray:
    keys = meta["sample_id"].astype(str).str.strip() + "__" + meta["cell_id"].astype(str).str.strip()
    if not pd.Index(keys).has_duplicates:
        return keys.to_numpy(dtype=object)
    seen: dict[str, int] = {}
    out: list[str] = []
    for key in keys:
        count = seen.get(str(key), 0)
        out.append(str(key) if count == 0 else f"{key}__{count}")
        seen[str(key)] = count + 1
    return np.asarray(out, dtype=object)


def orient_harmony(z: np.ndarray, n_obs: int) -> np.ndarray:
    z = np.asarray(z)
    if z.shape[0] == n_obs:
        return z.astype(np.float32, copy=False)
    if z.shape[1] == n_obs:
        return z.T.astype(np.float32, copy=False)
    raise ValueError(f"Could not orient Harmony output {z.shape} for n_obs={n_obs}")


def build_graph(x: np.ndarray, n_neighbors: int, metric: str, seed: int) -> sp.csr_matrix:
    index = NNDescent(
        x,
        n_neighbors=int(n_neighbors),
        metric=str(metric),
        random_state=int(seed),
        low_memory=True,
        compressed=True,
        n_jobs=1,
    )
    knn_indices, knn_dists = index.neighbor_graph
    graph, _, _ = fuzzy_simplicial_set(
        x,
        n_neighbors=int(n_neighbors),
        random_state=np.random.RandomState(int(seed)),
        metric=str(metric),
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )
    del index, knn_indices, knn_dists
    gc.collect()
    graph = canonicalize_csr((graph + graph.T) * 0.5)
    if graph.nnz == 0:
        raise RuntimeError("Harmony image graph is empty")
    return graph


def write_qc(outdir: Path, coords: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    for sid in sorted(coords["sample_id"].astype(str).unique()):
        m = coords["sample_id"].astype(str) == sid
        ax[0].scatter(coords.loc[m, "umap1"], coords.loc[m, "umap2"], s=1, alpha=0.6, label=sid, linewidths=0)
    ax[0].set_title("Image graph post-Harmony UMAP by sample")
    ax[0].legend(markerscale=6, fontsize=7)
    scmap = ax[1].scatter(coords["umap1"], coords["umap2"], s=1, c=coords["transcript_counts"], cmap="viridis", alpha=0.7, linewidths=0)
    ax[1].set_title("Image graph post-Harmony UMAP by transcript_counts")
    plt.colorbar(scmap, ax=ax[1], fraction=0.046, pad=0.04)
    for a in ax:
        a.set_xlabel("UMAP1")
        a.set_ylabel("UMAP2")
    fig.tight_layout()
    p = outdir / "joint_umap_qc_panels.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    meta_path = source_dir / "prepared_meta.parquet"
    image_path = source_dir / args.embedding_file
    for p in [meta_path, image_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    meta = pd.read_parquet(meta_path)
    required = {"cell_id", "sample_id", "transcript_counts", "cell_area"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    image_mm = np.load(image_path, mmap_mode="r")
    if image_mm.shape[0] != meta.shape[0]:
        raise ValueError(f"row mismatch: meta={meta.shape[0]} aligned_image={image_mm.shape[0]}")
    embed_shape = tuple(int(v) for v in image_mm.shape)

    log(f"[i-graph-harmony] loading and L2-normalizing {args.embedding_file}")
    x = l2_normalize_rows(np.asarray(image_mm, dtype=np.float32))
    del image_mm
    gc.collect()
    if not np.isfinite(x).all():
        raise ValueError("aligned_image contains non-finite values after normalization")

    requested_pcs = int(args.pca_components)
    if requested_pcs > 0:
        n_pcs = min(requested_pcs, x.shape[1], max(1, x.shape[0] - 1))
        log(f"[i-graph-harmony] PCA n_components={n_pcs}")
        xp = PCA(n_components=n_pcs, svd_solver="randomized", random_state=int(args.seed)).fit_transform(x)
        xp = l2_normalize_rows(xp)
        pca_mode = f"pca{n_pcs}_before_harmony"
        l2_after_pca = True
        del x
    else:
        n_pcs = int(x.shape[1])
        log(f"[i-graph-harmony] PCA skipped; using direct {n_pcs}D embedding")
        xp = x
        pca_mode = "none_direct_embedding_before_harmony"
        l2_after_pca = False
    gc.collect()

    hkey = str(args.harmony_key)
    if hkey not in meta.columns:
        raise ValueError(f"Harmony key missing from metadata: {hkey}")
    log(f"[i-graph-harmony] running Harmony on {pca_mode} using {hkey}")
    hm_meta = meta[[hkey]].copy()
    hm_meta[hkey] = hm_meta[hkey].astype(str)
    ho = run_harmony(xp, hm_meta, hkey, verbose=True, random_state=int(args.seed))
    xh = orient_harmony(np.asarray(ho.Z_corr), n_obs=meta.shape[0])
    xh = l2_normalize_rows(xh)
    del xp, ho
    gc.collect()

    log("[i-graph-harmony] building post-Harmony image graph")
    graph = build_graph(xh, n_neighbors=int(args.n_neighbors), metric=str(args.metric), seed=int(args.seed))

    log("[i-graph-harmony] writing compatibility embedding")
    compat = outdir / "aligned_fused.npy"
    np.save(compat, xh.astype(np.float32, copy=False))
    alias = outdir / "i_graph_post_harmony.npy"
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    os.symlink(compat.name, alias)

    obs_names = make_obs_names(meta)
    adata = ad.AnnData(X=np.zeros((meta.shape[0], 1), dtype=np.float32))
    obs_cols = [c for c in ["cell_id", "sample_id", "transcript_counts", "control_fraction", "cell_area"] if c in meta.columns]
    adata.obs = meta[obs_cols].copy()
    adata.obs_names = obs_names
    adata.obsm["X_image_harmony"] = xh
    adata.obsp["connectivities"] = graph
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "params": {
            "n_neighbors": int(args.n_neighbors),
            "metric": str(args.metric),
            "method": "umap",
            "graph_mode": "image_post_harmony",
            "pca_mode": pca_mode,
            "run_harmony": True,
            "harmony_key": hkey,
            "source_dir": str(source_dir),
        },
    }

    log("[i-graph-harmony] running UMAP from post-Harmony image graph")
    sc.tl.umap(adata, min_dist=float(args.min_dist), random_state=int(args.seed))
    coords = pd.DataFrame(adata.obsm["X_umap"], columns=["umap1", "umap2"])
    coords_out = pd.concat([adata.obs.reset_index(drop=True), coords], axis=1)
    coords_out.to_csv(outdir / "joint_umap_coordinates.csv", index=False)
    adata.write_h5ad(outdir / "joint_umap.h5ad")
    qc_png = write_qc(outdir, coords_out)

    summary = {
        "method": "image_embedding_harmony_cosine_knn_umap",
        "source_dir": str(source_dir),
        "embedding_file": str(image_path),
        "n_cells": int(meta.shape[0]),
        "embedding_shape": [int(embed_shape[0]), int(embed_shape[1])],
        "graph_method": {
            "graph_mode": "image_post_harmony",
            "metric": str(args.metric),
            "n_neighbors": int(args.n_neighbors),
            "symmetrized": True,
            "selected_graph_nnz": int(graph.nnz),
            "l2_normalize_before_pca": True,
            "pca_components": int(n_pcs),
            "pca_requested_components": requested_pcs,
            "pca_mode": pca_mode,
            "l2_normalize_after_pca": l2_after_pca,
            "l2_normalize_after_harmony": True,
            "run_harmony": True,
            "harmony_key": hkey,
            "umap_from_modality_graph": True,
            "umap_min_dist": float(args.min_dist),
            "seed": int(args.seed),
        },
        "outputs": {
            "compat_embedding_npy": str(alias),
            "aligned_fused_npy": str(compat),
            "joint_umap_h5ad": str(outdir / "joint_umap.h5ad"),
            "joint_umap_coordinates_csv": str(outdir / "joint_umap_coordinates.csv"),
            "joint_umap_qc_panels_png": str(qc_png),
        },
    }
    (outdir / "i_graph_post_harmony.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "umap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[i-graph-harmony] wrote {outdir / 'joint_umap.h5ad'}")
    log(f"[i-graph-harmony] wrote {outdir / 'joint_umap_coordinates.csv'}")


if __name__ == "__main__":
    main()
