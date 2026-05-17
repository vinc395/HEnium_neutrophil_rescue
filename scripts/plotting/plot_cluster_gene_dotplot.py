#!/usr/bin/env python3
"""Cluster-level Xenium-normalized gene dotplots."""

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
    ap.add_argument("--source-dir", required=True, help="Prepared source with metadata, counts, and genes")
    ap.add_argument("--clusters", required=True, help="Cluster CSV with sample_id, cell_id, cluster")
    ap.add_argument("--branch-name", required=True, help="Branch label written to tables and plot titles")
    ap.add_argument("--genes", default="CXCL8,CXCR2", help="Comma-separated gene symbols")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-summary", required=True)
    return ap.parse_args()


def read_genes(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sort_cluster_labels(labels: pd.Series) -> list[str]:
    def key_fn(v: str) -> tuple[int, int | str]:
        s = str(v)
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted(labels.astype(str).unique(), key=key_fn)


def main() -> None:
    args = parse_args()
    source = Path(args.source_dir).resolve()
    clusters_path = Path(args.clusters).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_png = Path(args.out_png).resolve()
    out_summary = Path(args.out_summary).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(source / "prepared_meta.parquet", columns=["sample_id", "cell_id"]).reset_index(drop=True)
    counts = sp.load_npz(source / "prepared_counts_gene_expr.npz").tocsr().astype(np.float32)
    genes = read_genes(source / "prepared_genes.txt")
    clusters = pd.read_csv(clusters_path, dtype={"sample_id": str, "cell_id": str})
    if counts.shape[0] != meta.shape[0]:
        raise ValueError(f"Row mismatch: counts={counts.shape[0]} meta={meta.shape[0]}")
    if counts.shape[1] != len(genes):
        raise ValueError(f"Gene mismatch: counts={counts.shape[1]} genes={len(genes)}")
    if not {"sample_id", "cell_id", "cluster"}.issubset(clusters.columns):
        raise ValueError("Cluster CSV must contain sample_id, cell_id, cluster")

    ref = meta.assign(_row=np.arange(meta.shape[0]))
    merged = ref.merge(clusters[["sample_id", "cell_id", "cluster"]], on=["sample_id", "cell_id"], how="inner", validate="one_to_one")
    if merged.shape[0] != meta.shape[0]:
        raise ValueError(f"Cluster row alignment mismatch: meta={meta.shape[0]} merged={merged.shape[0]}")
    if not np.array_equal(merged["_row"].to_numpy(), np.arange(meta.shape[0])):
        raise ValueError("Cluster CSV row order does not match prepared_meta after sample_id/cell_id merge")

    wanted = [g.strip() for g in args.genes.split(",") if g.strip()]
    gene_lookup = {g.upper(): i for i, g in enumerate(genes)}
    missing = [g for g in wanted if g.upper() not in gene_lookup]
    present = [g for g in wanted if g.upper() in gene_lookup]
    if not present:
        raise ValueError(f"None of the requested genes were found. Missing: {missing}")

    lib = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    positive = lib[lib > 0]
    if positive.size == 0:
        raise ValueError("All cells have zero library size")
    global_median = float(np.median(positive))
    scale = np.divide(global_median, lib, out=np.ones_like(lib, dtype=np.float32), where=lib > 0)

    expr = {}
    raw = {}
    for gene in present:
        col = gene_lookup[gene.upper()]
        raw_vec = np.asarray(counts[:, col].toarray()).ravel().astype(np.float32)
        raw[gene] = raw_vec
        expr[gene] = np.log1p(raw_vec * scale).astype(np.float32)

    rows = []
    merged["cluster"] = merged["cluster"].astype(str)
    for cluster in sort_cluster_labels(merged["cluster"]):
        idx = merged.index[merged["cluster"] == cluster].to_numpy()
        for gene in present:
            vals = expr[gene][idx]
            raw_vals = raw[gene][idx]
            rows.append(
                {
                    "branch": args.branch_name,
                    "cluster": cluster,
                    "gene": gene,
                    "n_cells": int(idx.size),
                    "mean_log1p_global_median_norm": float(np.mean(vals)) if idx.size else 0.0,
                    "median_log1p_global_median_norm": float(np.median(vals)) if idx.size else 0.0,
                    "pct_expressing_raw_count_gt0": float(np.mean(raw_vals > 0) * 100.0) if idx.size else 0.0,
                    "sum_raw_counts": float(np.sum(raw_vals)) if idx.size else 0.0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    clusters_order = sort_cluster_labels(out["cluster"])
    genes_order = present
    x_map = {c: i for i, c in enumerate(clusters_order)}
    y_map = {g: i for i, g in enumerate(genes_order)}
    plot_df = out.copy()
    plot_df["x"] = plot_df["cluster"].map(x_map)
    plot_df["y"] = plot_df["gene"].map(y_map)

    fig_w = max(8, 0.28 * len(clusters_order) + 2.5)
    fig_h = max(2.6, 0.75 * len(genes_order) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    sizes = 12 + plot_df["pct_expressing_raw_count_gt0"].to_numpy() * 2.8
    sca = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=sizes,
        c=plot_df["mean_log1p_global_median_norm"],
        cmap="viridis",
        edgecolors="0.25",
        linewidths=0.25,
    )
    ax.set_xticks(range(len(clusters_order)))
    ax.set_xticklabels(clusters_order, rotation=90, fontsize=6)
    ax.set_yticks(range(len(genes_order)))
    ax.set_yticklabels(genes_order, fontsize=9)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Gene")
    ax.set_title(f"{args.branch_name}: Xenium global-median normalized log1p expression")
    cbar = fig.colorbar(sca, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean log1p normalized expression")
    for pct in [5, 25, 50, 75]:
        ax.scatter([], [], s=12 + pct * 2.8, c="white", edgecolors="0.25", linewidths=0.25, label=f"{pct}%")
    ax.legend(title="% expressing", loc="upper left", bbox_to_anchor=(1.04, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    summary = {
        "branch": args.branch_name,
        "source_dir": str(source),
        "clusters": str(clusters_path),
        "genes_requested": wanted,
        "genes_present": present,
        "genes_missing": missing,
        "n_cells": int(meta.shape[0]),
        "n_clusters": int(merged["cluster"].nunique()),
        "normalization": "raw counts scaled by global median library size per cell, then log1p",
        "global_median_library_size": global_median,
        "outputs": {"csv": str(out_csv), "png": str(out_png)},
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
