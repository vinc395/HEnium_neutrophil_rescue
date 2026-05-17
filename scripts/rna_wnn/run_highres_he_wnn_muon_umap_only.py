#!/usr/bin/env python3
"""High-res HEnium Muon WNN graph branch.

Builds separate gene/image kNN graphs from aligned embeddings, runs Muon WNN,
clusters/UMAPs from the WNN graph, and writes full diagnostic tables/plots.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from harmonypy import run_harmony
from pynndescent import NNDescent
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import fuzzy_simplicial_set


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Muon WNN on high-res HEnium RNA/image embeddings")
    ap.add_argument("--source-dir", required=True, help="cellplm_uni2_source directory")
    ap.add_argument("--outdir", required=True, help="Output directory for wnn_muon_auto_harmony_umap_only")
    ap.add_argument("--gene-embedding", default="", help="Optional gene embedding .npy path; defaults to source-dir/aligned_rna.npy")
    ap.add_argument("--image-embedding", default="", help="Optional image embedding .npy path; defaults to source-dir/aligned_image.npy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dims", type=int, default=128, help="Embedding dimensions to use after variance ranking")
    ap.add_argument("--gene-dims", type=int, default=0, help="Optional gene-specific dimensions; defaults to --dims")
    ap.add_argument("--image-dims", type=int, default=0, help="Optional image-specific dimensions; defaults to --dims")
    ap.add_argument("--final-k", type=int, default=30)
    ap.add_argument("--graph-ks", default="15,30", help="Comma-separated graph k values to build/save")
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--n-multineighbors", type=int, default=60)
    ap.add_argument("--n-bandwidth-neighbors", type=int, default=15)
    ap.add_argument("--low-memory", choices=["true", "false", "auto"], default="true")
    ap.add_argument("--auto-harmony", choices=["true", "false"], default="true")
    ap.add_argument("--harmony-key", default="sample_id")
    ap.add_argument("--sample-effect-threshold", type=float, default=0.10)
    ap.add_argument("--diagnostic-subsample", type=int, default=50000)
    ap.add_argument("--umap-min-dist", type=float, default=0.3)
    ap.add_argument("--leiden-resolutions", default="0.2,0.5,0.8,1.0")
    ap.add_argument("--target-clusters", type=int, default=15)
    ap.add_argument("--target-res-min", type=float, default=0.10)
    ap.add_argument("--target-res-max", type=float, default=3.00)
    ap.add_argument("--target-res-step", type=float, default=0.02)
    ap.add_argument("--skip-target-scan", action="store_true")
    ap.add_argument("--skip-clustering", choices=["true", "false"], default="false", help="Write WNN graph/UMAP outputs without Leiden clustering")
    ap.add_argument("--baseline-fused-embedding-clusters", default="", help="Optional existing fused-embedding cluster CSV")
    ap.add_argument("--skip-muon-use-fixed-graph", action="store_true", help="Skip Muon WNN and use a fixed 50/50 RNA+image graph fusion with equal modality weights")
    return ap.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_ints(text: str) -> list[int]:
    out = []
    for x in text.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def parse_floats(text: str) -> list[float]:
    out = []
    for x in text.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def canonicalize_csr(mat: sp.spmatrix) -> sp.csr_matrix:
    out = mat.tocsr(copy=True)
    out.sum_duplicates()
    out.sort_indices()
    out.eliminate_zeros()
    return out


def graph_sanity_metrics(graph: sp.csr_matrix) -> dict:
    graph = canonicalize_csr(graph)
    row_nnz = np.diff(graph.indptr)
    data = graph.data
    return {
        "nnz": int(graph.nnz),
        "row_nnz_min": int(row_nnz.min()) if row_nnz.size else 0,
        "row_nnz_median": float(np.median(row_nnz)) if row_nnz.size else 0.0,
        "row_nnz_p99": float(np.quantile(row_nnz, 0.99)) if row_nnz.size else 0.0,
        "row_nnz_max": int(row_nnz.max()) if row_nnz.size else 0,
        "edge_weight_min": float(data.min()) if data.size else 0.0,
        "edge_weight_median": float(np.median(data)) if data.size else 0.0,
        "edge_weight_p99": float(np.quantile(data, 0.99)) if data.size else 0.0,
        "edge_weight_max": float(data.max()) if data.size else 0.0,
    }


def sanitize_wnn_connectivities(
    graph: sp.csr_matrix,
    dist: sp.csr_matrix | None,
    final_k: int,
    outdir: Path,
) -> sp.csr_matrix:
    graph = canonicalize_csr(graph)
    before = graph_sanity_metrics(graph)
    max_reasonable_degree = max(int(final_k) * 20, 500)
    invalid = before["row_nnz_max"] > max_reasonable_degree or before["edge_weight_max"] > 1.000001
    summary = {
        "status": "ok",
        "max_reasonable_degree": int(max_reasonable_degree),
        "before": before,
        "repair": None,
        "after": before,
    }
    if invalid:
        if dist is None:
            raise RuntimeError(
                "Muon WNN connectivities failed sanity checks and no distance graph is available for repair: "
                f"{before}"
            )
        log(f"[wnn] WNN connectivity sanity repair triggered: {before}")
        mask = canonicalize_csr(dist.maximum(dist.T))
        mask.data = np.ones(mask.nnz, dtype=np.float32)
        repaired = canonicalize_csr(graph.multiply(mask))
        repaired.data = np.clip(repaired.data, 0.0, 1.0).astype(np.float32, copy=False)
        repaired = canonicalize_csr(repaired.maximum(repaired.T))
        after = graph_sanity_metrics(repaired)
        if after["row_nnz_max"] > max_reasonable_degree or after["edge_weight_max"] > 1.000001:
            raise RuntimeError(f"Muon WNN connectivity repair failed sanity checks: before={before} after={after}")
        sp.save_npz(outdir / "wnn_graph_connectivities_raw_muon.npz", graph)
        graph = repaired
        summary = {
            "status": "repaired",
            "max_reasonable_degree": int(max_reasonable_degree),
            "before": before,
            "repair": "intersected raw Muon connectivities with symmetrized WNN distance graph and clipped edge weights to [0,1]",
            "after": after,
        }
        log(f"[wnn] WNN connectivity repair complete: {after}")
    (outdir / "wnn_graph_sanity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return graph


def make_obs_names(meta: pd.DataFrame) -> np.ndarray:
    keys = (meta["sample_id"].astype(str).str.strip() + "__" + meta["cell_id"].astype(str).str.strip()).to_numpy(dtype=object)
    if not pd.Index(keys).has_duplicates:
        return keys
    seen: dict[str, int] = {}
    unique: list[str] = []
    for key in keys:
        n = seen.get(key, 0)
        unique.append(key if n == 0 else f"{key}__{n}")
        seen[key] = n + 1
    return np.asarray(unique, dtype=object)


def row_l2_normalize(x: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    qc = {
        "nonfinite_values": int((~np.isfinite(x)).sum()),
        "zero_norm_rows": 0,
        "zero_variance_rows": 0,
    }
    if qc["nonfinite_values"]:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    row_var = np.var(x, axis=1)
    qc["zero_variance_rows"] = int((row_var <= 1e-12).sum())
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    zero = denom[:, 0] <= 1e-12
    qc["zero_norm_rows"] = int(zero.sum())
    denom = np.maximum(denom, 1e-8)
    return (x / denom).astype(np.float32, copy=False), qc


def select_dims_by_variance(x: np.ndarray, dims: int) -> tuple[np.ndarray, list[int]]:
    dims = max(2, min(int(dims), x.shape[1]))
    order = np.argsort(np.var(x, axis=0))[::-1][:dims]
    return x[:, order].astype(np.float32, copy=False), [int(i) for i in order]


def read_inputs(
    source_dir: Path,
    gene_embedding: str,
    image_embedding: str,
    dims: int,
    gene_dims: int = 0,
    image_dims: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    meta_path = source_dir / "prepared_meta.parquet"
    g_path = Path(gene_embedding).expanduser() if gene_embedding else source_dir / "aligned_rna.npy"
    i_path = Path(image_embedding).expanduser() if image_embedding else source_dir / "aligned_image.npy"
    for path in [meta_path, g_path, i_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    required = {"sample_id", "cell_id"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"prepared_meta.parquet missing required columns: {sorted(missing)}")
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["cell_id"] = meta["cell_id"].astype(str)

    g0 = np.load(g_path).astype(np.float32)
    i0 = np.load(i_path).astype(np.float32)
    if g0.shape[0] != i0.shape[0]:
        raise ValueError(f"gene/image row mismatch: {g0.shape} vs {i0.shape}")
    if g0.shape[0] != meta.shape[0]:
        raise ValueError(f"metadata/embedding row mismatch: meta={meta.shape[0]}, embeddings={g0.shape[0]}")
    g0_shape = [int(g0.shape[0]), int(g0.shape[1])]
    i0_shape = [int(i0.shape[0]), int(i0.shape[1])]

    g_dim_target = int(gene_dims) if int(gene_dims) > 0 else int(dims)
    i_dim_target = int(image_dims) if int(image_dims) > 0 else int(dims)
    g, g_dims = select_dims_by_variance(g0, g_dim_target)
    i, i_dims = select_dims_by_variance(i0, i_dim_target)
    del g0, i0
    gc.collect()

    g, g_qc = row_l2_normalize(g)
    i, i_qc = row_l2_normalize(i)
    obs_names = make_obs_names(meta)
    input_qc = {
        "n_cells": int(meta.shape[0]),
        "gene_embedding_path": str(g_path),
        "image_embedding_path": str(i_path),
        "gene_embedding_shape_before_dim_select": g0_shape,
        "image_embedding_shape_before_dim_select": i0_shape,
        "gene_embedding_shape_after_dim_select": [int(g.shape[0]), int(g.shape[1])],
        "image_embedding_shape_after_dim_select": [int(i.shape[0]), int(i.shape[1])],
        "gene_dim_target": int(g_dim_target),
        "image_dim_target": int(i_dim_target),
        "duplicated_sample_cell_keys": int(pd.Index(meta["sample_id"] + "__" + meta["cell_id"]).duplicated().sum()),
        "gene_embedding_qc": g_qc,
        "image_embedding_qc": i_qc,
        "gene_selected_dim_indices": g_dims,
        "image_selected_dim_indices": i_dims,
    }
    return meta, obs_names, g, i, input_qc


def sample_effect_score(x: np.ndarray, sample: np.ndarray, seed: int, n_sub: int, k: int = 30) -> dict:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = np.arange(n) if n <= n_sub else rng.choice(n, size=n_sub, replace=False)
    xs = x[idx]
    ss = sample[idx]
    nn = NearestNeighbors(n_neighbors=min(k + 1, xs.shape[0]), metric="cosine")
    nn.fit(xs)
    neigh = nn.kneighbors(xs, return_distance=False)[:, 1:]
    same = (ss[neigh] == ss[:, None]).mean(axis=1)
    obs = float(np.mean(same))
    freq = pd.Series(ss).value_counts(normalize=True)
    expected = float(np.sum(np.square(freq.to_numpy())))
    excess = obs - expected
    return {
        "n_subsample": int(len(idx)),
        "k": int(k),
        "observed_same_sample_neighbor_fraction": obs,
        "expected_same_sample_fraction": expected,
        "excess_same_sample_fraction": excess,
        "ratio_observed_to_expected": float(obs / expected) if expected > 0 else None,
    }


def run_harmony_if_needed(x: np.ndarray, meta: pd.DataFrame, key: str, apply: bool) -> np.ndarray:
    if not apply:
        return x
    ho = run_harmony(x, meta[[key]].assign(**{key: meta[key].astype(str)}), key, verbose=False)
    z = np.asarray(ho.Z_corr)
    if z.shape[0] == x.shape[0]:
        out = z.astype(np.float32)
    elif z.shape[1] == x.shape[0]:
        out = z.T.astype(np.float32)
    else:
        raise ValueError(f"Unexpected Harmony output shape {z.shape} for input {x.shape}")
    out, _ = row_l2_normalize(out)
    return out


def plot_sample_umap(x: np.ndarray, meta: pd.DataFrame, out_png: Path, title: str, seed: int, n_sub: int) -> None:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = np.arange(n) if n <= n_sub else rng.choice(n, size=n_sub, replace=False)
    adata = ad.AnnData(X=x[idx].astype(np.float32, copy=False))
    adata.obs["sample_id"] = meta.iloc[idx]["sample_id"].astype(str).to_numpy()
    # Explicitly use the embedding matrix as-is. Without use_rep="X", Scanpy may
    # silently run PCA when X has >50 dimensions, which is both conceptually wrong
    # for this diagnostic and has triggered native crashes on the full dataset.
    sc.pp.neighbors(adata, n_neighbors=30, metric="cosine", random_state=seed, use_rep="X")
    sc.tl.umap(adata, min_dist=0.3, random_state=seed)
    coords = pd.DataFrame(adata.obsm["X_umap"], columns=["umap1", "umap2"])
    coords["sample_id"] = adata.obs["sample_id"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=160)
    for sid in sorted(coords["sample_id"].unique()):
        d = coords[coords["sample_id"] == sid]
        ax.scatter(d["umap1"], d["umap2"], s=2, alpha=0.65, linewidths=0, label=sid)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(markerscale=5, fontsize=7, title="sample_id", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    del adata, coords, fig, ax
    gc.collect()


def build_graph_from_embedding(x: np.ndarray, k: int, metric: str, seed: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    index = NNDescent(
        x.astype(np.float32, copy=False),
        n_neighbors=int(k),
        metric=metric,
        random_state=int(seed),
        low_memory=True,
        compressed=True,
        n_jobs=1,
    )
    knn_indices, knn_dists = index.neighbor_graph
    graph, _, _ = fuzzy_simplicial_set(
        x,
        n_neighbors=int(k),
        random_state=np.random.RandomState(int(seed)),
        metric=metric,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )
    n = x.shape[0]
    rows = np.repeat(np.arange(n, dtype=np.int64), knn_indices.shape[1])
    cols = knn_indices.reshape(-1).astype(np.int64)
    vals = knn_dists.reshape(-1).astype(np.float32)
    dist = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    dist = canonicalize_csr(dist)
    graph = canonicalize_csr(graph)
    dist = ensure_distance_rows(dist, graph, k)
    del index, knn_indices, knn_dists
    gc.collect()
    return graph, dist


def ensure_distance_rows(dist: sp.csr_matrix, graph: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Ensure Muon sees at least one distance-neighbor row per cell.

    NNDescent can emit degenerate all-self neighbor rows for duplicated or near-
    duplicated low-dimensional RNA PCA points. The fuzzy connectivity graph can
    still contain valid neighbors, but Muon checks the distance graph and fails
    if a modality row has no distance neighbors. For those rare rows, add
    conservative fallback distances to the strongest connectivity neighbors.
    """
    dist = canonicalize_csr(dist)
    graph = canonicalize_csr(graph)
    row_nnz = np.diff(dist.indptr)
    zero_rows = np.flatnonzero(row_nnz == 0)
    if zero_rows.size == 0:
        return dist

    add_rows: list[int] = []
    add_cols: list[int] = []
    add_vals: list[float] = []
    max_fallback = max(1, min(int(k), 30))
    for r in zero_rows:
        start, end = graph.indptr[r], graph.indptr[r + 1]
        cols = graph.indices[start:end]
        vals = graph.data[start:end]
        keep = cols != r
        cols = cols[keep]
        vals = vals[keep]
        if cols.size == 0:
            continue
        order = np.argsort(vals)[::-1][:max_fallback]
        for c, w in zip(cols[order], vals[order], strict=False):
            add_rows.append(int(r))
            add_cols.append(int(c))
            add_vals.append(float(max(1e-6, 1.0 - float(w))))
    if add_rows:
        fallback = sp.csr_matrix((np.asarray(add_vals, dtype=np.float32), (add_rows, add_cols)), shape=dist.shape)
        dist = canonicalize_csr(dist + fallback)
    return dist


def make_modality_adata(x: np.ndarray, obs_names: np.ndarray, graph: sp.csr_matrix, dist: sp.csr_matrix, k: int, metric: str) -> ad.AnnData:
    adata = ad.AnnData(X=x.astype(np.float32, copy=False))
    adata.obs_names = obs_names
    adata.var_names = [f"dim_{j+1}" for j in range(x.shape[1])]
    adata.obsp["connectivities"] = graph
    adata.obsp["distances"] = dist
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {
            "n_neighbors": int(k),
            "metric": metric,
            "method": "umap",
            "use_rep": "X",
            "n_pcs": 0,
        },
    }
    return adata


def add_graph_to_adata(meta: pd.DataFrame, obs_names: np.ndarray, graph: sp.csr_matrix, dist: sp.csr_matrix | None, seed: int, min_dist: float) -> ad.AnnData:
    obs_cols = [c for c in ["sample_id", "cell_id", "transcript_counts", "control_fraction", "cell_area"] if c in meta.columns]
    adata = ad.AnnData(X=np.zeros((meta.shape[0], 1), dtype=np.float32), obs=meta[obs_cols].copy())
    adata.obs_names = obs_names
    adata.obsp["connectivities"] = graph
    if dist is not None:
        adata.obsp["distances"] = dist
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances" if dist is not None else None,
        "params": {"method": "umap", "metric": "cosine"},
    }
    sc.tl.umap(adata, min_dist=float(min_dist), random_state=int(seed))
    return adata


def leiden_on_graph(adata: ad.AnnData, graph: sp.csr_matrix, resolution: float, key: str, seed: int) -> np.ndarray:
    sc.tl.leiden(
        adata,
        resolution=float(resolution),
        random_state=int(seed),
        key_added=key,
        adjacency=graph,
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    return adata.obs[key].astype(str).to_numpy()


def target_resolution_scan(adata: ad.AnnData, graph: sp.csr_matrix, target: int, r_min: float, r_max: float, r_step: float, seed: int) -> tuple[float, int, pd.DataFrame, np.ndarray]:
    rows = []
    resolutions = np.round(np.arange(r_min, r_max + r_step / 2, r_step), 2)
    for r in resolutions:
        key = f"_scan_{r:.2f}"
        labs = leiden_on_graph(adata, graph, float(r), key, seed)
        nclu = int(pd.Series(labs).nunique())
        rows.append({"resolution": float(r), "n_clusters": nclu, "distance_to_target": abs(nclu - target)})
        del adata.obs[key]
    scan = pd.DataFrame(rows).sort_values(["distance_to_target", "resolution"]).reset_index(drop=True)
    best_r = float(scan.iloc[0]["resolution"])
    best_key = f"leiden_r{best_r:.2f}"
    labs = leiden_on_graph(adata, graph, best_r, best_key, seed)
    return best_r, int(pd.Series(labs).nunique()), scan, labs


def write_cluster_outputs(outdir: Path, meta: pd.DataFrame, adata: ad.AnnData, cluster_labels: np.ndarray, scan: pd.DataFrame, best_r: float, best_n: int, target: int) -> None:
    coords = pd.DataFrame(adata.obsm["X_umap"], columns=["umap1", "umap2"])
    coords.insert(0, "cell_id", meta["cell_id"].astype(str).to_numpy())
    coords.insert(0, "sample_id", meta["sample_id"].astype(str).to_numpy())
    coords.to_csv(outdir / "joint_umap_coordinates.csv", index=False)

    cluster_df = coords.copy()
    cluster_df["cluster"] = cluster_labels.astype(str)
    target_tag = f"target{int(target)}"
    cluster_df.to_csv(outdir / "joint_umap_clusters_10ish.csv", index=False)
    cluster_df.to_csv(outdir / f"joint_umap_clusters_{target_tag}.csv", index=False)
    embedding_path = outdir / f"embedding_clusters_{target_tag}.csv"
    sizes_path = outdir / f"embedding_cluster_sizes_{target_tag}.csv"
    cluster_df[["sample_id", "cell_id", "cluster"]].to_csv(embedding_path, index=False)
    sizes = cluster_df["cluster"].astype(str).value_counts().rename_axis("cluster").reset_index(name="n_cells")
    sizes = sizes.sort_values("cluster", key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else 999999))
    sizes.to_csv(sizes_path, index=False)
    scan.to_csv(outdir / "joint_umap_leiden_resolution_scan.csv", index=False)
    scan.to_csv(outdir / f"joint_umap_leiden_{target_tag}_resolution_scan.csv", index=False)

    exp = outdir / "per_sample_cluster_exports"
    exp.mkdir(parents=True, exist_ok=True)
    for sid, g in cluster_df.groupby("sample_id", sort=True):
        g[["cell_id", "cluster"]].rename(columns={"cluster": "group"}).sort_values("cell_id").to_csv(exp / f"{sid}_cellid_group.csv", index=False)
    cluster_df[["sample_id", "cell_id", "cluster"]].rename(columns={"cluster": "group"}).to_csv(outdir / "all_samples_cellid_group.csv", index=False)

    counts = pd.crosstab(cluster_df["sample_id"], cluster_df["cluster"])
    counts.to_csv(outdir / "wnn_cluster_by_sample_counts.csv")
    counts.to_csv(outdir / f"cluster_{target_tag}_by_sample_counts.csv")
    pct = counts.div(counts.sum(axis=1), axis=0) * 100.0
    pct.to_csv(outdir / "wnn_cluster_by_sample_percent.csv")
    pct.to_csv(outdir / f"cluster_{target_tag}_by_sample_percent.csv")

    summary = {
        "target_clusters": int(target),
        "best_resolution": float(best_r),
        "n_clusters": int(best_n),
        "n_cells": int(cluster_df.shape[0]),
        "outputs": {
            f"joint_umap_clusters_{target_tag}_csv": str(outdir / f"joint_umap_clusters_{target_tag}.csv"),
            f"embedding_clusters_{target_tag}_csv": str(embedding_path),
            f"embedding_cluster_sizes_{target_tag}_csv": str(sizes_path),
        },
    }
    (outdir / f"cluster_{target_tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_wnn_panels(outdir: Path, coords: pd.DataFrame, weights: pd.DataFrame, clusters: np.ndarray) -> None:
    df = coords.copy()
    df["cluster"] = clusters.astype(str)
    weight_cols = ["sample_id", "cell_id", "gene_weight", "image_weight"]
    if "transcript_counts" in weights.columns:
        weight_cols.append("transcript_counts")
    df = df.merge(weights[weight_cols], on=["sample_id", "cell_id"], how="left")
    has_transcripts = "transcript_counts" in df.columns
    ncols = 3 if has_transcripts else 2
    fig, axes = plt.subplots(2, ncols, figsize=(18 if has_transcripts else 13, 10), dpi=150)
    axes = np.asarray(axes).reshape(2, ncols)
    ax = axes[0, 0]
    for sid in sorted(df["sample_id"].unique()):
        d = df[df["sample_id"] == sid]
        ax.scatter(d["umap1"], d["umap2"], s=1, alpha=0.65, linewidths=0, label=sid)
    ax.set_title("WNN UMAP by sample")
    ax.legend(markerscale=5, fontsize=6, title="sample_id", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax = axes[0, 1]
    cats = sorted(df["cluster"].unique(), key=lambda x: int(x) if str(x).isdigit() else 999999)
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(cats), 2)))
    for i, c in enumerate(cats):
        d = df[df["cluster"] == c]
        ax.scatter(d["umap1"], d["umap2"], s=1, alpha=0.75, linewidths=0, color=pal[i % len(pal)], label=c)
    ax.set_title("WNN UMAP by cluster")
    ax.legend(markerscale=5, fontsize=6, title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")

    for ax, col, title in [(axes[1, 0], "gene_weight", "Gene modality weight"), (axes[1, 1], "image_weight", "Image modality weight")]:
        sca = ax.scatter(df["umap1"], df["umap2"], s=1, c=df[col], cmap="viridis", alpha=0.8, linewidths=0, vmin=0, vmax=1)
        ax.set_title(title)
        plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    if has_transcripts:
        ax = axes[0, 2]
        sca = ax.scatter(df["umap1"], df["umap2"], s=1, c=df["transcript_counts"], cmap="magma", alpha=0.8, linewidths=0)
        ax.set_title("Transcript counts")
        plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
        axes[1, 2].axis("off")
    for ax in axes.flat:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    fig.tight_layout()
    fig.savefig(outdir / "wnn_umap_diagnostic_panels.png")
    plt.close(fig)


def plot_cluster_umap(outdir: Path, coords: pd.DataFrame, clusters: np.ndarray, title: str, target: int = 0) -> None:
    df = coords.copy()
    df["cluster"] = clusters.astype(str)
    cats = sorted(df["cluster"].unique(), key=lambda x: int(x) if str(x).isdigit() else 999999)
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(cats), 2)))
    centers = df.groupby("cluster")[["umap1", "umap2"]].median().reset_index()
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    for i, c in enumerate(cats):
        d = df[df["cluster"] == c]
        ax.scatter(d["umap1"], d["umap2"], s=1, alpha=0.8, linewidths=0, color=pal[i % len(pal)], label=c)
    for _, r in centers.iterrows():
        ax.text(float(r["umap1"]), float(r["umap2"]), str(r["cluster"]), ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(title="cluster", markerscale=8, fontsize=7, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "joint_umap_leiden_10ish.png")
    if int(target) > 0:
        fig.savefig(outdir / f"joint_umap_leiden_target{int(target)}.png")
    plt.close(fig)


def baseline_comparisons(outdir: Path, meta: pd.DataFrame, obs_names: np.ndarray, graphs: dict[str, sp.csr_matrix], wnn_clusters: np.ndarray, resolutions: list[float], seed: int, fused_cluster_csv: str, target: int) -> None:
    rows = []
    ari_col = f"ari_vs_wnn_target{int(target)}"
    nmi_col = f"nmi_vs_wnn_target{int(target)}"
    baseline_labels: dict[str, np.ndarray] = {}
    for name, graph in graphs.items():
        adata = ad.AnnData(X=np.zeros((meta.shape[0], 1), dtype=np.float32), obs=meta[["sample_id", "cell_id"]].copy())
        adata.obs_names = obs_names
        for r in resolutions:
            key = f"{name}_r{r:.2f}"
            labels = leiden_on_graph(adata, graph, r, key, seed)
            ari = adjusted_rand_score(wnn_clusters, labels)
            nmi = normalized_mutual_info_score(wnn_clusters, labels)
            rows.append({"baseline": name, "resolution": r, "n_clusters": int(pd.Series(labels).nunique()), ari_col: ari, nmi_col: nmi})
            if abs(r - 0.5) < 1e-9:
                baseline_labels[name] = labels
    if fused_cluster_csv:
        p = Path(fused_cluster_csv)
        if p.exists():
            d = pd.read_csv(p, dtype={"sample_id": str, "cell_id": str})
            if "group" in d.columns and "cluster" not in d.columns:
                d = d.rename(columns={"group": "cluster"})
            d["key"] = d["sample_id"].astype(str) + "__" + d["cell_id"].astype(str)
            ref = pd.DataFrame({"key": meta["sample_id"].astype(str) + "__" + meta["cell_id"].astype(str)})
            merged = ref.merge(d[["key", "cluster"]], on="key", how="left")
            if merged["cluster"].notna().all():
                labels = merged["cluster"].astype(str).to_numpy()
                rows.append({"baseline": "fused_embedding_first_existing", "resolution": np.nan, "n_clusters": int(pd.Series(labels).nunique()), ari_col: adjusted_rand_score(wnn_clusters, labels), nmi_col: normalized_mutual_info_score(wnn_clusters, labels)})
    pd.DataFrame(rows).to_csv(outdir / "wnn_baseline_cluster_agreement.csv", index=False)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    np.random.seed(args.seed)
    sc.settings.seed = args.seed
    source_dir = Path(args.source_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    log("[wnn] reading and validating inputs")
    meta, obs_names, gene, image, input_qc = read_inputs(
        source_dir,
        args.gene_embedding,
        args.image_embedding,
        args.dims,
        args.gene_dims,
        args.image_dims,
    )
    input_qc["source_dir"] = str(source_dir)
    input_qc["sample_counts"] = meta["sample_id"].value_counts().sort_index().to_dict()
    (outdir / "input_qc_summary.json").write_text(json.dumps(input_qc, indent=2), encoding="utf-8")

    log("[wnn] assessing sample effects")
    sample = meta["sample_id"].astype(str).to_numpy()
    gene_effect_pre = sample_effect_score(gene, sample, args.seed, args.diagnostic_subsample)
    image_effect_pre = sample_effect_score(image, sample, args.seed + 1, args.diagnostic_subsample)
    plot_sample_umap(gene, meta, diag_dir / "gene_embedding_umap_by_sample_pre_harmony.png", "Gene embedding by sample before Harmony", args.seed, args.diagnostic_subsample)
    plot_sample_umap(image, meta, diag_dir / "image_embedding_umap_by_sample_pre_harmony.png", "Image embedding by sample before Harmony", args.seed + 1, args.diagnostic_subsample)

    auto_harmony = args.auto_harmony == "true"
    gene_apply_harmony = bool(auto_harmony and gene_effect_pre["excess_same_sample_fraction"] > args.sample_effect_threshold)
    image_apply_harmony = bool(auto_harmony and image_effect_pre["excess_same_sample_fraction"] > args.sample_effect_threshold)

    log(f"[wnn] auto Harmony: gene={gene_apply_harmony} image={image_apply_harmony}")
    gene_work = run_harmony_if_needed(gene, meta, args.harmony_key, gene_apply_harmony)
    image_work = run_harmony_if_needed(image, meta, args.harmony_key, image_apply_harmony)
    if gene_apply_harmony:
        del gene
    if image_apply_harmony:
        del image
    if gene_apply_harmony or image_apply_harmony:
        gc.collect()
    if gene_apply_harmony:
        plot_sample_umap(gene_work, meta, diag_dir / "gene_embedding_umap_by_sample_post_harmony.png", "Gene embedding by sample after Harmony", args.seed, args.diagnostic_subsample)
    if image_apply_harmony:
        plot_sample_umap(image_work, meta, diag_dir / "image_embedding_umap_by_sample_post_harmony.png", "Image embedding by sample after Harmony", args.seed + 1, args.diagnostic_subsample)
    gene_effect_post = sample_effect_score(gene_work, sample, args.seed, args.diagnostic_subsample)
    image_effect_post = sample_effect_score(image_work, sample, args.seed + 1, args.diagnostic_subsample)
    sample_effects = pd.DataFrame([
        {"modality": "gene", "stage": "pre_harmony", **gene_effect_pre},
        {"modality": "image", "stage": "pre_harmony", **image_effect_pre},
        {"modality": "gene", "stage": "post_harmony", **gene_effect_post},
        {"modality": "image", "stage": "post_harmony", **image_effect_post},
    ])
    sample_effects.to_csv(outdir / "sample_effect_diagnostics.csv", index=False)

    graph_ks = sorted(set(parse_ints(args.graph_ks) + [int(args.final_k)]))
    final_k = int(args.final_k)
    final_gene_graph = final_gene_dist = final_image_graph = final_image_dist = None
    graph_summaries = []
    for k in graph_ks:
        log(f"[wnn] building gene graph k={k}")
        g_graph, g_dist = build_graph_from_embedding(gene_work, k, args.metric, args.seed)
        log(f"[wnn] building image graph k={k}")
        i_graph, i_dist = build_graph_from_embedding(image_work, k, args.metric, args.seed + 1)
        suffix = "" if k == final_k else f"_k{k}"
        sp.save_npz(outdir / f"g_graph_connectivities{suffix}.npz", g_graph)
        sp.save_npz(outdir / f"g_graph_distances{suffix}.npz", g_dist)
        sp.save_npz(outdir / f"i_graph_connectivities{suffix}.npz", i_graph)
        sp.save_npz(outdir / f"i_graph_distances{suffix}.npz", i_dist)
        graph_summaries.append({"k": int(k), "gene_graph_nnz": int(g_graph.nnz), "image_graph_nnz": int(i_graph.nnz)})
        if k == final_k:
            final_gene_graph, final_gene_dist = g_graph, g_dist
            final_image_graph, final_image_dist = i_graph, i_dist
        else:
            del g_graph, g_dist, i_graph, i_dist
            gc.collect()

    if final_gene_graph is None or final_image_graph is None:
        raise RuntimeError("Final modality graphs were not created")
    pd.DataFrame(graph_summaries).to_csv(outdir / "modality_graph_summary.csv", index=False)

    if args.skip_muon_use_fixed_graph:
        log("[wnn] skipping Muon; using fixed 50/50 RNA+image graph fusion")
        wnn_graph = canonicalize_csr((final_gene_graph + final_image_graph) * 0.5)
        wnn_dist = canonicalize_csr((final_gene_dist + final_image_dist) * 0.5)
        sp.save_npz(outdir / "wnn_graph_connectivities.npz", wnn_graph)
        sp.save_npz(outdir / "wnn_graph_distances.npz", wnn_dist)
        (outdir / "wnn_graph_sanity.json").write_text(json.dumps({
            "status": "fixed_graph_fusion",
            "note": "Muon WNN skipped; graph is 0.5 * gene_graph + 0.5 * image_graph.",
            "connectivities": graph_sanity_metrics(wnn_graph),
        }, indent=2), encoding="utf-8")
        weights = pd.DataFrame({
            "sample_id": meta["sample_id"].astype(str).to_numpy(),
            "cell_id": meta["cell_id"].astype(str).to_numpy(),
            "gene_weight": np.full(meta.shape[0], 0.5, dtype=np.float32),
            "image_weight": np.full(meta.shape[0], 0.5, dtype=np.float32),
        })
        if "transcript_counts" in meta.columns:
            weights["transcript_counts"] = pd.to_numeric(meta["transcript_counts"], errors="coerce").to_numpy()
        weights.to_csv(outdir / "wnn_modality_weights.csv", index=False)
        muon_mode = "fixed_graph_50_50_fallback"
    else:
        log("[wnn] constructing MuData and running Muon WNN")
        adata_g = make_modality_adata(gene_work, obs_names, final_gene_graph, final_gene_dist, final_k, args.metric)
        adata_i = make_modality_adata(image_work, obs_names, final_image_graph, final_image_dist, final_k, args.metric)
        mdata = md.MuData({"gene": adata_g, "image": adata_i})
        low_memory = None if args.low_memory == "auto" else args.low_memory == "true"
        try:
            mu.pp.neighbors(
                mdata,
                n_neighbors=final_k,
                n_multineighbors=int(args.n_multineighbors),
                n_bandwidth_neighbors=int(args.n_bandwidth_neighbors),
                neighbor_keys={"gene": "neighbors", "image": "neighbors"},
                metric=args.metric,
                low_memory=low_memory,
                key_added="wnn",
                weight_key="mod_weight",
                add_weights_to_modalities=False,
                random_state=int(args.seed),
            )
        except Exception as exc:
            fail = {
                "status": "failed",
                "stage": "muon_wnn",
                "error": repr(exc),
                "policy": "strict Muon WNN; fixed 50/50 graph fusion is not used by this tutorial",
                "next_step": "diagnose the Muon WNN graph construction error before rerunning",
            }
            (outdir / "run_summary.json").write_text(json.dumps(fail, indent=2), encoding="utf-8")
            raise

        if "wnn_connectivities" not in mdata.obsp:
            fail = {"status": "failed", "stage": "muon_wnn", "error": f"wnn_connectivities missing; available={list(mdata.obsp.keys())}", "policy": "strict Muon WNN; fixed 50/50 graph fusion is not used by this tutorial"}
            (outdir / "run_summary.json").write_text(json.dumps(fail, indent=2), encoding="utf-8")
            raise RuntimeError(fail["error"])
        weight_cols = set(mdata.obs.columns)
        gene_weight_col = "gene:mod_weight"
        image_weight_col = "image:mod_weight"
        if gene_weight_col not in weight_cols or image_weight_col not in weight_cols:
            fail = {"status": "failed", "stage": "muon_wnn", "error": f"modality weights missing; available columns={sorted(weight_cols)}", "policy": "strict unless --skip-muon-use-fixed-graph is set"}
            (outdir / "run_summary.json").write_text(json.dumps(fail, indent=2), encoding="utf-8")
            raise RuntimeError(fail["error"])

        wnn_dist = canonicalize_csr(mdata.obsp["wnn_distances"]) if "wnn_distances" in mdata.obsp else None
        wnn_graph = sanitize_wnn_connectivities(
            canonicalize_csr(mdata.obsp["wnn_connectivities"]),
            wnn_dist,
            final_k,
            outdir,
        )
        sp.save_npz(outdir / "wnn_graph_connectivities.npz", wnn_graph)
        if wnn_dist is not None:
            sp.save_npz(outdir / "wnn_graph_distances.npz", wnn_dist)

        weights = pd.DataFrame({
            "sample_id": meta["sample_id"].astype(str).to_numpy(),
            "cell_id": meta["cell_id"].astype(str).to_numpy(),
            "gene_weight": mdata.obs[gene_weight_col].to_numpy(dtype=np.float32),
            "image_weight": mdata.obs[image_weight_col].to_numpy(dtype=np.float32),
        })
        if "transcript_counts" in meta.columns:
            weights["transcript_counts"] = pd.to_numeric(meta["transcript_counts"], errors="coerce").to_numpy()
        weights.to_csv(outdir / "wnn_modality_weights.csv", index=False)
        muon_mode = "muon_wnn"

    log("[wnn] running UMAP from WNN graph")
    adata_wnn = add_graph_to_adata(meta, obs_names, wnn_graph, wnn_dist, args.seed, args.umap_min_dist)
    coords_tmp = pd.DataFrame(adata_wnn.obsm["X_umap"], columns=["umap1", "umap2"])
    coords_tmp.insert(0, "cell_id", meta["cell_id"].astype(str).to_numpy())
    coords_tmp.insert(0, "sample_id", meta["sample_id"].astype(str).to_numpy())

    coords_tmp.to_csv(outdir / "joint_umap_coordinates.csv", index=False)
    adata_wnn.write_h5ad(outdir / "joint_umap.h5ad")


    clustering_summary = None
    if args.skip_clustering == "true":
        log("[wnn] skip-clustering=true; wrote graph, modality weights, and UMAP coordinates only")
    else:
        log("[wnn] running fixed Leiden resolution grid")
        resolution_rows = []
        for r in parse_floats(args.leiden_resolutions):
            key = f"leiden_r{r:.2f}"
            labs = leiden_on_graph(adata_wnn, wnn_graph, r, key, args.seed)
            resolution_rows.append({"resolution": float(r), "n_clusters": int(pd.Series(labs).nunique())})
        pd.DataFrame(resolution_rows).to_csv(outdir / "wnn_leiden_resolution_grid.csv", index=False)

        if args.skip_target_scan:
            best_r = 0.5
            best_key = "leiden_r0.50"
            if best_key not in adata_wnn.obs:
                clusters = leiden_on_graph(adata_wnn, wnn_graph, best_r, best_key, args.seed)
            else:
                clusters = adata_wnn.obs[best_key].astype(str).to_numpy()
            best_n = int(pd.Series(clusters).nunique())
            scan = pd.DataFrame([{"resolution": best_r, "n_clusters": best_n, "distance_to_target": abs(best_n - args.target_clusters)}])
        else:
            log("[wnn] running target-cluster Leiden scan")
            best_r, best_n, scan, clusters = target_resolution_scan(
                adata_wnn,
                wnn_graph,
                args.target_clusters,
                args.target_res_min,
                args.target_res_max,
                args.target_res_step,
                args.seed,
            )
        adata_wnn.obs["leiden_10ish"] = clusters.astype(str)
        adata_wnn.write_h5ad(outdir / "joint_umap_clustered.h5ad")
        write_cluster_outputs(outdir, meta, adata_wnn, clusters, scan, best_r, best_n, args.target_clusters)
        coords = pd.read_csv(outdir / "joint_umap_coordinates.csv")
        plot_wnn_panels(outdir, coords, weights, clusters)
        plot_cluster_umap(outdir, coords, clusters, f"WNN UMAP Leiden clustering (resolution={best_r:.2f}, n={best_n})", args.target_clusters)

        fixed_graph = canonicalize_csr((final_gene_graph + final_image_graph) * 0.5)
        baseline_comparisons(
            outdir,
            meta,
            obs_names,
            {"gene_graph": final_gene_graph, "image_graph": final_image_graph, "fixed_graph_50_50": fixed_graph},
            clusters,
            parse_floats(args.leiden_resolutions),
            args.seed,
            args.baseline_fused_embedding_clusters,
            args.target_clusters,
        )

        weights_by_cluster = weights.assign(cluster=clusters.astype(str)).groupby("cluster")[["gene_weight", "image_weight"]].agg(["mean", "median", "std", "count"])
        weights_by_cluster.to_csv(outdir / "wnn_modality_weights_by_cluster.csv")
        clustering_summary = {"best_resolution": float(best_r), "n_clusters": int(best_n), "target_clusters": int(args.target_clusters)}

    weights_by_sample = weights.groupby("sample_id")[["gene_weight", "image_weight"]].agg(["mean", "median", "std", "count"])
    weights_by_sample.to_csv(outdir / "wnn_modality_weights_by_sample.csv")
    if "transcript_counts" in weights.columns:
        weights_for_bins = weights.copy()
        weights_for_bins["transcript_bin"] = pd.cut(
            weights_for_bins["transcript_counts"],
            bins=[-1, 4, 9, 29, 99, np.inf],
            labels=["0-4", "5-9", "10-29", "30-99", ">=100"],
        )
        bin_summary = weights_for_bins.groupby("transcript_bin", observed=True)[["transcript_counts", "gene_weight", "image_weight"]].agg(["count", "mean", "median", "std"])
        bin_summary.to_csv(outdir / "wnn_modality_weights_by_transcript_bin.csv")
        sample_bin_summary = weights_for_bins.groupby(["sample_id", "transcript_bin"], observed=True)[["gene_weight", "image_weight"]].agg(["mean", "median", "std", "count"])
        sample_bin_summary.to_csv(outdir / "wnn_modality_weights_by_sample_transcript_bin.csv")

    summary = {
        "status": "completed",
        "method": "muon_wnn_auto_harmony_umap_only" if muon_mode == "muon_wnn" else "fixed_50_50_rna_image_graph_fusion",
        "source_dir": str(source_dir),
        "outdir": str(outdir),
        "n_cells": int(meta.shape[0]),
        "gene_dims": int(gene_work.shape[1]),
        "image_dims": int(image_work.shape[1]),
        "gene_embedding": input_qc.get("gene_embedding_path"),
        "image_embedding": input_qc.get("image_embedding_path"),
        "final_k": int(final_k),
        "graph_ks": graph_ks,
        "metric": args.metric,
        "integration_mode": muon_mode,
        "muon": {
            "n_multineighbors": int(args.n_multineighbors),
            "n_bandwidth_neighbors": int(args.n_bandwidth_neighbors),
            "low_memory": args.low_memory,
        },
        "auto_harmony": {
            "enabled": args.auto_harmony == "true",
            "threshold_excess_same_sample_fraction": float(args.sample_effect_threshold),
            "gene_applied": bool(gene_apply_harmony),
            "image_applied": bool(image_apply_harmony),
        },
        "clustering_mode": "skipped" if args.skip_clustering == "true" else "leiden",
        "target_clustering": clustering_summary,
        "runtime_seconds": float(time.time() - t0),
        "outputs": {
            "joint_umap_h5ad": str(outdir / "joint_umap.h5ad"),
            "joint_umap_coordinates_csv": str(outdir / "joint_umap_coordinates.csv"),
            "wnn_modality_weights_csv": str(outdir / "wnn_modality_weights.csv"),
            "wnn_modality_weights_by_transcript_bin_csv": str(outdir / "wnn_modality_weights_by_transcript_bin.csv"),
            "wnn_graph_connectivities_npz": str(outdir / "wnn_graph_connectivities.npz"),
        },
    }
    if args.skip_clustering != "true":
        summary["outputs"].update({
            "joint_umap_clustered_h5ad": str(outdir / "joint_umap_clustered.h5ad"),
            f"joint_umap_clusters_target{int(args.target_clusters)}_csv": str(outdir / f"joint_umap_clusters_target{int(args.target_clusters)}.csv"),
            f"embedding_clusters_target{int(args.target_clusters)}_csv": str(outdir / f"embedding_clusters_target{int(args.target_clusters)}.csv"),
        })
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[wnn] completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
