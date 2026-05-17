#!/usr/bin/env python3
"""Plot per-cell Xenium-normalized gene expression on an existing UMAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--clusters", required=True, help="CSV with sample_id, cell_id, umap1, umap2, cluster")
    ap.add_argument("--branch-name", required=True)
    ap.add_argument("--gene", default="CXCL8")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-summary", required=True)
    return ap.parse_args()


def read_genes(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    args = parse_args()
    source = Path(args.source_dir).resolve()
    clusters_path = Path(args.clusters).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_png = Path(args.out_png).resolve()
    out_summary = Path(args.out_summary).resolve()
    for p in [out_csv.parent, out_png.parent, out_summary.parent]:
        p.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(source / "prepared_meta.parquet", columns=["sample_id", "cell_id"]).reset_index(drop=True)
    counts = sp.load_npz(source / "prepared_counts_gene_expr.npz").tocsr().astype(np.float32)
    genes = read_genes(source / "prepared_genes.txt")
    clusters = pd.read_csv(clusters_path, dtype={"sample_id": str, "cell_id": str})
    required = {"sample_id", "cell_id", "umap1", "umap2", "cluster"}
    if not required.issubset(clusters.columns):
        raise ValueError(f"Cluster CSV missing columns: {sorted(required - set(clusters.columns))}")
    if counts.shape[0] != meta.shape[0]:
        raise ValueError(f"Row mismatch: counts={counts.shape[0]} meta={meta.shape[0]}")

    ref = meta.assign(_row=np.arange(meta.shape[0]))
    merged = ref.merge(
        clusters[["sample_id", "cell_id", "umap1", "umap2", "cluster"]],
        on=["sample_id", "cell_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.shape[0] != meta.shape[0]:
        raise ValueError(f"Cluster row alignment mismatch: meta={meta.shape[0]} merged={merged.shape[0]}")
    if not np.array_equal(merged["_row"].to_numpy(), np.arange(meta.shape[0])):
        raise ValueError("Cluster CSV row order does not match prepared_meta after sample_id/cell_id merge")

    gene_lookup = {g.upper(): i for i, g in enumerate(genes)}
    gene = str(args.gene).strip()
    if gene.upper() not in gene_lookup:
        raise ValueError(f"Gene not found: {gene}")

    lib = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    positive = lib[lib > 0]
    if positive.size == 0:
        raise ValueError("All cells have zero library size")
    global_median = float(np.median(positive))
    scale = np.divide(global_median, lib, out=np.ones_like(lib, dtype=np.float32), where=lib > 0)

    raw = np.asarray(counts[:, gene_lookup[gene.upper()]].toarray()).ravel().astype(np.float32)
    expr = np.log1p(raw * scale).astype(np.float32)
    out = merged[["sample_id", "cell_id", "cluster", "umap1", "umap2"]].copy()
    out[f"{gene}_raw_count"] = raw
    out[f"{gene}_log1p_global_median_norm"] = expr
    out.to_csv(out_csv, index=False)

    order = np.argsort(expr)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=180)
    ax.scatter(
        out["umap1"].to_numpy()[order],
        out["umap2"].to_numpy()[order],
        c=expr[order],
        s=2.0,
        cmap="magma",
        linewidths=0,
        alpha=0.85,
    )
    sca = ax.collections[-1]
    ax.set_title(f"{args.branch_name}: {gene} expression")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{gene} log1p global-median normalized")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    summary = {
        "branch": args.branch_name,
        "source_dir": str(source),
        "clusters": str(clusters_path),
        "gene": gene,
        "n_cells": int(meta.shape[0]),
        "n_clusters": int(out["cluster"].nunique()),
        "normalization": "raw counts scaled by global median library size per cell, then log1p",
        "global_median_library_size": global_median,
        "n_cells_raw_count_gt0": int((raw > 0).sum()),
        "max_log1p_global_median_norm": float(np.max(expr)) if expr.size else 0.0,
        "outputs": {"csv": str(out_csv), "png": str(out_png)},
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
