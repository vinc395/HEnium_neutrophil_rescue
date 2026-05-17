#!/usr/bin/env python3
"""Fast Leiden scan on an existing UMAP AnnData graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import scanpy as sc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--target-clusters", type=int, default=30)
    ap.add_argument("--resolutions", default="0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0,2.5,3.0")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def parse_resolutions(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    h5ad = Path(args.h5ad).resolve()
    outdir = h5ad.parent
    target = int(args.target_clusters)
    adata = sc.read_h5ad(h5ad)
    if "neighbors" not in adata.uns:
        raise ValueError("Input h5ad has no neighbors graph")
    if "X_umap" not in adata.obsm:
        raise ValueError("Input h5ad has no X_umap")

    rows = []
    for r in parse_resolutions(args.resolutions):
        key = f"_scan_{r:g}"
        print(f"[cluster-fast] resolution={r:g}", flush=True)
        sc.tl.leiden(
            adata,
            resolution=float(r),
            random_state=int(args.seed),
            key_added=key,
            flavor="igraph",
            directed=False,
            n_iterations=2,
        )
        n = int(adata.obs[key].nunique())
        rows.append({"resolution": float(r), "n_clusters": n, "distance_to_target": abs(n - target)})
        print(f"[cluster-fast] resolution={r:g} n_clusters={n}", flush=True)

    scan = pd.DataFrame(rows).sort_values(["distance_to_target", "resolution"])
    best = scan.iloc[0]
    best_r = float(best["resolution"])
    cluster_key = f"leiden_target{target}_r{best_r:g}"
    print(f"[cluster-fast] best_resolution={best_r:g}", flush=True)
    sc.tl.leiden(
        adata,
        resolution=best_r,
        random_state=int(args.seed),
        key_added=cluster_key,
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    adata.obs["cluster"] = adata.obs[cluster_key].astype(str)
    adata.write_h5ad(outdir / "joint_umap_clustered.h5ad")

    coords = pd.DataFrame(adata.obsm["X_umap"], columns=["umap1", "umap2"])
    cols = [c for c in ["sample_id", "cell_id", "transcript_counts"] if c in adata.obs.columns]
    out = pd.concat([adata.obs[cols].reset_index(drop=True), coords], axis=1)
    out["cluster"] = adata.obs["cluster"].to_numpy()
    out.to_csv(outdir / f"joint_umap_clusters_target{target}.csv", index=False)
    out.to_csv(outdir / "joint_umap_clusters_10ish.csv", index=False)

    sizes = out["cluster"].value_counts().rename_axis("cluster").reset_index(name="n_cells")
    sizes = sizes.sort_values("cluster", key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else 999999))
    sizes.to_csv(outdir / f"embedding_cluster_sizes_target{target}.csv", index=False)
    out[["sample_id", "cell_id", "cluster"]].to_csv(outdir / f"embedding_clusters_target{target}.csv", index=False)
    scan.to_csv(outdir / f"joint_umap_leiden_target{target}_resolution_scan.csv", index=False)

    cats = sorted(out["cluster"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(cats), 2)))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    for i, c in enumerate(cats):
        m = out["cluster"].astype(str) == c
        ax.scatter(out.loc[m, "umap1"], out.loc[m, "umap2"], s=1, alpha=0.75, color=palette[i % len(palette)], linewidths=0)
    centers = out.groupby("cluster")[["umap1", "umap2"]].median().reset_index()
    for _, row in centers.iterrows():
        txt = ax.text(float(row["umap1"]), float(row["umap2"]), str(row["cluster"]), fontsize=8, weight="bold", ha="center", va="center")
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"Image H-Optimus UMAP Leiden (r={best_r:g}, n={int(best['n_clusters'])})")
    fig.tight_layout()
    fig.savefig(outdir / f"joint_umap_leiden_target{target}.png")
    plt.close(fig)

    summary = {
        "h5ad": str(h5ad),
        "target_clusters": target,
        "best_resolution": best_r,
        "best_n_clusters": int(best["n_clusters"]),
        "n_cells": int(out.shape[0]),
        "outputs": {
            "clusters": str(outdir / f"joint_umap_clusters_target{target}.csv"),
            "cluster_sizes": str(outdir / f"embedding_cluster_sizes_target{target}.csv"),
            "plot": str(outdir / f"joint_umap_leiden_target{target}.png"),
        },
    }
    (outdir / f"cluster_target{target}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
