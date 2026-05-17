#!/usr/bin/env python3
"""Compute all-gene Xenium-normalized log1p scaled PCA embeddings.

This intentionally does not perform HVG selection. It is designed for targeted
Xenium panels where preserving marker genes is preferred over gene filtering.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import IncrementalPCA


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Xenium median-normalized log1p scale PCA using all genes")
    ap.add_argument("--source-dir", required=True, help="Directory with prepared_meta/counts/genes")
    ap.add_argument("--output-npy", default="rna_xenium_norm_log1p_scale_pca128.npy")
    ap.add_argument("--summary-json", default="rna_xenium_norm_log1p_scale_pca128_summary.json")
    ap.add_argument("--n-components", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def batches(n: int, batch_size: int, min_batch: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + batch_size)
        if n - end > 0 and n - end < min_batch:
            end = n
        out.append((start, end))
        start = end
    return out


def dense_norm_log1p(counts: sp.csr_matrix, lib: np.ndarray, global_median: float, start: int, end: int) -> np.ndarray:
    x = counts[start:end].toarray().astype(np.float32, copy=False)
    scale = np.divide(global_median, lib[start:end], out=np.ones(end - start, dtype=np.float32), where=(lib[start:end] > 0))
    x *= scale[:, None]
    np.log1p(x, out=x)
    return x


def main() -> None:
    args = parse_args()
    t0 = time.time()
    np.random.seed(int(args.seed))
    source = Path(args.source_dir).resolve()
    meta_path = source / "prepared_meta.parquet"
    counts_path = source / "prepared_counts_gene_expr.npz"
    genes_path = source / "prepared_genes.txt"
    for p in [meta_path, counts_path, genes_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    meta = pd.read_parquet(meta_path, columns=["sample_id", "cell_id", "transcript_counts"])
    counts = sp.load_npz(counts_path).tocsr().astype(np.float32)
    genes = [g.strip() for g in genes_path.read_text(encoding="utf-8").splitlines() if g.strip()]
    if counts.shape[0] != meta.shape[0]:
        raise ValueError(f"Row mismatch: meta={meta.shape[0]} counts={counts.shape[0]}")
    if counts.shape[1] != len(genes):
        raise ValueError(f"Gene mismatch: counts={counts.shape[1]} genes={len(genes)}")

    n_obs, n_vars = counts.shape
    n_components = max(2, min(int(args.n_components), n_vars - 1, n_obs - 1))
    batch_size = max(int(args.batch_size), n_components)
    batch_slices = batches(n_obs, batch_size, n_components)

    lib = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    positive = lib[lib > 0]
    if positive.size == 0:
        raise ValueError("All cells have zero library size")
    global_median = float(np.median(positive))

    log(f"[rna-allgenes] source={source}")
    log(f"[rna-allgenes] n_cells={n_obs} n_genes={n_vars} n_components={n_components} batches={len(batch_slices)}")
    log(f"[rna-allgenes] xenium_global_median_libsize={global_median:.6g}")

    sums = np.zeros(n_vars, dtype=np.float64)
    sumsqs = np.zeros(n_vars, dtype=np.float64)
    for bi, (start, end) in enumerate(batch_slices, start=1):
        x = dense_norm_log1p(counts, lib, global_median, start, end)
        sums += x.sum(axis=0, dtype=np.float64)
        sumsqs += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
        if bi == 1 or bi % 25 == 0 or bi == len(batch_slices):
            log(f"[rna-allgenes] moments batch {bi}/{len(batch_slices)} rows={end-start}")
        del x
    mean = sums / float(n_obs)
    var = np.maximum((sumsqs / float(n_obs)) - mean * mean, 1e-12)
    std = np.sqrt(var)
    zero_std = int((std <= 1e-6).sum())
    std = np.maximum(std, 1e-6)

    stats = pd.DataFrame({"gene": genes, "mean_log1p_norm": mean, "std_log1p_norm": std})
    stats.to_csv(source / "rna_xenium_norm_log1p_gene_scaling_stats.csv", index=False)

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for bi, (start, end) in enumerate(batch_slices, start=1):
        x = dense_norm_log1p(counts, lib, global_median, start, end)
        x = (x - mean.astype(np.float32)) / std.astype(np.float32)
        if args.clip and float(args.clip) > 0:
            np.clip(x, -float(args.clip), float(args.clip), out=x)
        ipca.partial_fit(x)
        if bi == 1 or bi % 10 == 0 or bi == len(batch_slices):
            log(f"[rna-allgenes] fit PCA batch {bi}/{len(batch_slices)}")
        del x

    out_path = source / args.output_npy
    emb = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_obs, n_components))
    for bi, (start, end) in enumerate(batch_slices, start=1):
        x = dense_norm_log1p(counts, lib, global_median, start, end)
        x = (x - mean.astype(np.float32)) / std.astype(np.float32)
        if args.clip and float(args.clip) > 0:
            np.clip(x, -float(args.clip), float(args.clip), out=x)
        emb[start:end] = ipca.transform(x).astype(np.float32, copy=False)
        if bi == 1 or bi % 10 == 0 or bi == len(batch_slices):
            log(f"[rna-allgenes] transform batch {bi}/{len(batch_slices)}")
        del x
    emb.flush()

    summary = {
        "method": "xenium_global_median_norm_log1p_allgenes_scale_incremental_pca",
        "source_dir": str(source),
        "n_cells": int(n_obs),
        "n_genes": int(n_vars),
        "n_components": int(n_components),
        "batch_size": int(batch_size),
        "clip": float(args.clip),
        "xenium_global_median_libsize": global_median,
        "zero_std_genes": zero_std,
        "output_npy": str(out_path),
        "runtime_seconds": float(time.time() - t0),
        "explained_variance_ratio_sum": float(np.sum(ipca.explained_variance_ratio_)),
    }
    (source / args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[rna-allgenes] wrote={out_path}")
    log(f"[rna-allgenes] completed runtime_seconds={summary['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
