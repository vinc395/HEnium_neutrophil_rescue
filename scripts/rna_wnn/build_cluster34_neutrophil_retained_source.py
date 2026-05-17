#!/usr/bin/env python3
"""Build a retained source for Notebook 3 WNN analysis.

Confirmed neutrophils are taken from a morphology cluster CSV and retained
regardless of transcript count. All other cells are filtered by a minimum
transcript threshold.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--cluster-csv", required=True)
    ap.add_argument("--confirmed-cluster", default="34")
    ap.add_argument("--min-transcripts-non-neutrophil", type=int, default=10)
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.source_dir).resolve()
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta_path = src / "prepared_meta.parquet"
    counts_path = src / "prepared_counts_gene_expr.npz"
    genes_path = src / "prepared_genes.txt"
    image_path = src / "aligned_image.npy"
    cluster_csv = Path(args.cluster_csv).resolve()
    for p in [meta_path, counts_path, genes_path, image_path, cluster_csv]:
        if not p.exists():
            raise FileNotFoundError(p)

    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    clusters = pd.read_csv(cluster_csv, dtype={"sample_id": str, "cell_id": str, "cluster": str})
    required = {"sample_id", "cell_id", "cluster"}
    if not required.issubset(clusters.columns):
        raise ValueError(f"cluster CSV missing columns: {sorted(required - set(clusters.columns))}")
    if not {"sample_id", "cell_id", "transcript_counts"}.issubset(meta.columns):
        raise ValueError("prepared_meta.parquet must contain sample_id, cell_id, transcript_counts")

    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["cell_id"] = meta["cell_id"].astype(str)
    meta_key = meta["sample_id"] + "__" + meta["cell_id"]
    cluster_key = clusters["sample_id"].astype(str) + "__" + clusters["cell_id"].astype(str)
    if meta_key.duplicated().any():
        raise ValueError("source metadata contains duplicated sample_id__cell_id keys")
    if cluster_key.duplicated().any():
        raise ValueError("cluster CSV contains duplicated sample_id__cell_id keys")

    cluster_map = pd.Series(clusters["cluster"].astype(str).to_numpy(), index=cluster_key)
    cluster_labels = meta_key.map(cluster_map)
    if cluster_labels.isna().any():
        missing = int(cluster_labels.isna().sum())
        raise ValueError(f"{missing} metadata rows are missing from the cluster CSV")

    confirmed = cluster_labels.astype(str).to_numpy() == str(args.confirmed_cluster)
    tx = pd.to_numeric(meta["transcript_counts"], errors="coerce").fillna(0).to_numpy()
    non_neut_pass = tx >= int(args.min_transcripts_non_neutrophil)
    retain = confirmed | non_neut_pass

    source_row_index = np.arange(len(meta), dtype=np.int64)
    meta_out = meta.loc[retain].copy()
    meta_out["source_row_index"] = source_row_index[retain]
    meta_out = meta_out.reset_index(drop=True)
    meta_out["image_cluster_source"] = "rna_aligned_bleep_harmony_target40"
    meta_out["image_cluster_target40"] = cluster_labels.loc[retain].astype(str).to_numpy()
    meta_out["confirmed_neutrophil"] = confirmed[retain]
    meta_out["neutrophil_source_method"] = np.where(
        meta_out["confirmed_neutrophil"],
        f"rna_aligned_bleep_harmony_target40_cluster{args.confirmed_cluster}",
        "not_confirmed_neutrophil",
    )
    meta_out["passes_non_neutrophil_min_transcripts"] = non_neut_pass[retain]
    meta_out["retention_reason"] = np.where(
        meta_out["confirmed_neutrophil"],
        "confirmed_neutrophil_retained_regardless_transcripts",
        f"non_neutrophil_transcript_counts_ge_{args.min_transcripts_non_neutrophil}",
    )

    dropped = meta.loc[~retain, ["sample_id", "cell_id", "transcript_counts"]].copy()
    dropped["source_row_index"] = source_row_index[~retain]
    dropped["image_cluster_target40"] = cluster_labels.loc[~retain].astype(str).to_numpy()
    dropped["confirmed_neutrophil"] = confirmed[~retain]
    dropped["drop_reason"] = f"non_neutrophil_transcript_counts_lt_{args.min_transcripts_non_neutrophil}"

    counts = sp.load_npz(counts_path).tocsr()
    if counts.shape[0] != len(meta):
        raise ValueError(f"count rows {counts.shape[0]} != metadata rows {len(meta)}")
    sp.save_npz(out / "prepared_counts_gene_expr.npz", counts[retain].tocsr())
    meta_out.to_parquet(out / "prepared_meta.parquet", index=False)
    shutil.copy2(genes_path, out / "prepared_genes.txt")

    image = np.load(image_path, mmap_mode="r")
    if image.shape[0] != len(meta):
        raise ValueError(f"image rows {image.shape[0]} != metadata rows {len(meta)}")
    np.save(out / "aligned_image.npy", np.asarray(image[retain], dtype=np.float32))

    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    retained_table = meta_out[
        [
            "sample_id",
            "cell_id",
            "transcript_counts",
            "source_row_index",
            "image_cluster_target40",
            "confirmed_neutrophil",
            "retention_reason",
        ]
    ].copy()
    retained_table.to_csv(tables / "retained_cells_cluster34_neutrophil_tc10.csv", index=False)
    dropped.to_csv(tables / "dropped_non_neutrophil_lt10_cells.csv", index=False)
    pd.crosstab(meta_out["sample_id"], meta_out["retention_reason"]).to_csv(tables / "retained_cells_by_sample_reason.csv")

    summary = {
        "source_dir": str(src),
        "cluster_csv": str(cluster_csv),
        "confirmed_cluster": str(args.confirmed_cluster),
        "min_transcripts_non_neutrophil": int(args.min_transcripts_non_neutrophil),
        "n_source_cells": int(len(meta)),
        "n_retained_cells": int(len(meta_out)),
        "n_dropped_cells": int(len(dropped)),
        "n_confirmed_neutrophils_source": int(confirmed.sum()),
        "n_confirmed_neutrophils_retained": int(meta_out["confirmed_neutrophil"].sum()),
        "n_confirmed_neutrophils_dropped": int((confirmed & ~retain).sum()),
        "all_confirmed_neutrophils_retained": bool((confirmed & ~retain).sum() == 0),
        "n_non_neutrophils_retained": int((~meta_out["confirmed_neutrophil"]).sum()),
        "n_non_neutrophils_dropped_lt_threshold": int((~confirmed & ~retain).sum()),
        "outputs": {
            "prepared_meta": str(out / "prepared_meta.parquet"),
            "prepared_counts": str(out / "prepared_counts_gene_expr.npz"),
            "prepared_genes": str(out / "prepared_genes.txt"),
            "aligned_image": str(out / "aligned_image.npy"),
            "retained_cells": str(tables / "retained_cells_cluster34_neutrophil_tc10.csv"),
            "dropped_cells": str(tables / "dropped_non_neutrophil_lt10_cells.csv"),
        },
    }
    (out / "retained_source_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
