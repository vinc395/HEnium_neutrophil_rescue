#!/usr/bin/env python3
"""Export cluster assignments as per-sample cell_id,group CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clusters", required=True, help="CSV with sample_id, cell_id, cluster")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run-name", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    clusters_path = Path(args.clusters).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(clusters_path, dtype={"sample_id": str, "cell_id": str})
    required = {"sample_id", "cell_id", "cluster"}
    if not required.issubset(df.columns):
        raise ValueError(f"Cluster CSV missing columns: {sorted(required - set(df.columns))}")

    export_df = df[["sample_id", "cell_id", "cluster"]].rename(columns={"cluster": "group"}).copy()
    export_df["group"] = export_df["group"].astype(str)
    export_df.to_csv(outdir / "all_samples_cellid_group.csv", index=False)

    per_sample = []
    for sid, g in export_df.groupby("sample_id", sort=True):
        out = outdir / f"{sid}_cellid_group.csv"
        g[["cell_id", "group"]].sort_values("cell_id").to_csv(out, index=False)
        per_sample.append({"sample_id": sid, "n_cells": int(g.shape[0]), "path": str(out)})

    counts = export_df.groupby(["sample_id", "group"]).size().unstack(fill_value=0)
    counts = counts.reindex(sorted(counts.columns, key=lambda x: int(x) if str(x).isdigit() else str(x)), axis=1)
    counts.reset_index().to_csv(outdir / "cluster_target40_by_sample_counts.csv", index=False)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100.0
    pct.reset_index().to_csv(outdir / "cluster_target40_by_sample_percent.csv", index=False)
    sizes = export_df.groupby("group").size().reset_index(name="n_cells")
    sizes["group_sort"] = sizes["group"].map(lambda x: int(x) if str(x).isdigit() else 10**9)
    sizes.sort_values(["group_sort", "group"]).drop(columns=["group_sort"]).to_csv(outdir / "cluster_target40_sizes.csv", index=False)

    manifest = {
        "run_name": args.run_name,
        "source_cluster_csv": str(clusters_path),
        "n_cells": int(export_df.shape[0]),
        "n_samples": int(export_df["sample_id"].nunique()),
        "n_groups": int(export_df["group"].nunique()),
        "group_column": "group",
        "per_sample_exports": per_sample,
        "outputs": {
            "all_samples_cellid_group_csv": str(outdir / "all_samples_cellid_group.csv"),
            "cluster_target40_by_sample_counts_csv": str(outdir / "cluster_target40_by_sample_counts.csv"),
            "cluster_target40_by_sample_percent_csv": str(outdir / "cluster_target40_by_sample_percent.csv"),
            "cluster_target40_sizes_csv": str(outdir / "cluster_target40_sizes.csv"),
        },
    }
    (outdir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
