#!/usr/bin/env python3
"""Prepare Cellpose Xenium outputs for H-Optimus H&E embedding.

This source builder uses Cellpose Xenium `cell_boundaries.csv.gz` as the
authoritative cell geometry source and keeps low-transcript cells unless they
are outside the H&E patchable image area.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import tifffile


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sample",
        action="append",
        required=True,
        metavar="SAMPLE_ID|XENIUM_DIR|REGISTERED_HE",
        help="Sample triple. Can be supplied multiple times.",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--boundaries-dir", required=True)
    ap.add_argument("--patch-size", type=int, default=224)
    ap.add_argument("--upsample-factor", type=float, default=4.0)
    ap.add_argument("--um-per-px", type=float, default=0.2124998718994902)
    ap.add_argument("--max-cells-per-sample", type=int, default=0, help="0 means keep all post-edge cells.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def read_he_shape(path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape
    if len(shape) != 3:
        raise ValueError(f"Expected YXS image shape for {path}, got {shape}")
    return int(shape[0]), int(shape[1])


def read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t")[0] for line in f if line.strip()]


def read_genes(path: Path) -> list[str]:
    genes: list[str] = []
    seen: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            gene = parts[1] if len(parts) > 1 and parts[1] else parts[0]
            if gene in seen:
                seen[gene] += 1
                gene = f"{gene}_{seen[gene]}"
            else:
                seen[gene] = 0
            genes.append(gene)
    return genes


def load_counts(xenium_dir: Path) -> tuple[sp.csr_matrix, list[str], list[str]]:
    mtx = xenium_dir / "cell_feature_matrix" / "matrix.mtx.gz"
    barcodes = xenium_dir / "cell_feature_matrix" / "barcodes.tsv.gz"
    features = xenium_dir / "cell_feature_matrix" / "features.tsv.gz"
    for p in [mtx, barcodes, features]:
        if not p.exists():
            raise FileNotFoundError(p)
    x_feature_by_cell = scipy.io.mmread(mtx).tocsc().astype(np.float32)
    cell_ids = read_lines_gz(barcodes)
    genes = read_genes(features)
    if x_feature_by_cell.shape != (len(genes), len(cell_ids)):
        raise ValueError(
            f"Matrix shape {x_feature_by_cell.shape} does not match genes/cells {len(genes)} {len(cell_ids)}"
        )
    return x_feature_by_cell.T.tocsr(), cell_ids, genes


def export_boundaries_and_centroids(xenium_dir: Path, sample_id: str, boundaries_dir: Path) -> pd.DataFrame:
    src = xenium_dir / "cell_boundaries.csv.gz"
    if not src.exists():
        raise FileNotFoundError(src)

    out_csv = boundaries_dir / f"{sample_id}_boundaries.csv"
    out_cent = boundaries_dir / f"{sample_id}_boundary_centroids.csv"
    pieces = []
    vertex_offsets: dict[str, int] = {}

    with pd.read_csv(src, chunksize=1_000_000) as reader:
        first = True
        for chunk in reader:
            req = {"cell_id", "vertex_x", "vertex_y"}
            if not req.issubset(chunk.columns):
                raise ValueError(f"{src} missing required columns {sorted(req - set(chunk.columns))}")
            chunk = chunk[["cell_id", "vertex_x", "vertex_y"]].copy()
            chunk["cell_id"] = chunk["cell_id"].astype(str)
            chunk.rename(columns={"vertex_x": "x", "vertex_y": "y"}, inplace=True)
            chunk.insert(0, "sample_id", sample_id)
            chunk["polygon_id"] = 1
            # Vertex IDs only need to be monotonic within a cell for plotting.
            chunk["vertex_id"] = chunk.groupby("cell_id").cumcount()
            if vertex_offsets:
                offsets = chunk["cell_id"].map(vertex_offsets).fillna(0).astype(int)
                chunk["vertex_id"] = chunk["vertex_id"].astype(int) + offsets
            ends = chunk.groupby("cell_id")["vertex_id"].max() + 1
            vertex_offsets.update({str(k): int(v) for k, v in ends.items()})
            chunk = chunk[["sample_id", "cell_id", "polygon_id", "vertex_id", "x", "y"]]
            chunk.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
            first = False
            g = chunk.groupby("cell_id", sort=False).agg(x_sum=("x", "sum"), y_sum=("y", "sum"), n=("x", "size"))
            pieces.append(g)

    if not pieces:
        raise ValueError(f"No boundaries found in {src}")
    sums = pd.concat(pieces).groupby(level=0).sum()
    cent = pd.DataFrame(
        {
            "cell_id": sums.index.astype(str),
            "x_centroid": sums["x_sum"].to_numpy(dtype=float) / sums["n"].to_numpy(dtype=float),
            "y_centroid": sums["y_sum"].to_numpy(dtype=float) / sums["n"].to_numpy(dtype=float),
            "n_boundary_vertices": sums["n"].to_numpy(dtype=int),
        }
    )
    cent.to_csv(out_cent, index=False)
    return cent


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    boundaries_dir = Path(args.boundaries_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    boundaries_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    all_meta = []
    all_counts = []
    genes_ref: list[str] | None = None
    summary = {
        "method": "cellpose_xenium_cell_boundaries_counts",
        "id_source": "cellpose_outs cell_boundaries.csv.gz cell_id",
        "boundary_source": "cell_boundaries.csv.gz only",
        "counts_source": "cell_feature_matrix/matrix.mtx.gz",
        "samples": [],
        "patch_size": int(args.patch_size),
        "upsample_factor": float(args.upsample_factor),
        "um_per_px": float(args.um_per_px),
        "max_cells_per_sample": int(args.max_cells_per_sample),
    }

    effective = int(round(int(args.patch_size) / max(float(args.upsample_factor), 1e-8)))
    effective = max(8, effective)
    half = effective / 2.0

    for spec in args.sample:
        parts = spec.split("|")
        if len(parts) != 3:
            raise ValueError("--sample must be SAMPLE_ID|XENIUM_DIR|REGISTERED_HE")
        sample_id, xenium_dir_s, he_s = parts
        xenium_dir = Path(xenium_dir_s).resolve()
        he_path = Path(he_s).resolve()
        if not xenium_dir.exists():
            raise FileNotFoundError(xenium_dir)
        if not he_path.exists():
            raise FileNotFoundError(he_path)

        cells = pd.read_csv(xenium_dir / "cells.csv.gz")
        cells["cell_id"] = cells["cell_id"].astype(str)
        n_cells_input = int(cells.shape[0])
        n_lt20_input = int((cells["transcript_counts"] < 20).sum()) if "transcript_counts" in cells.columns else None

        cent = export_boundaries_and_centroids(xenium_dir, sample_id, boundaries_dir)
        meta = cells.drop(columns=["x_centroid", "y_centroid"], errors="ignore").merge(cent, on="cell_id", how="inner")
        n_with_boundary = int(meta.shape[0])

        h, w = read_he_shape(he_path)
        meta["sample_id"] = sample_id
        meta["he_image"] = str(he_path)
        meta["um_per_px"] = float(args.um_per_px)
        meta["x_px"] = meta["x_centroid"].astype(float) / float(args.um_per_px)
        meta["y_px"] = meta["y_centroid"].astype(float) / float(args.um_per_px)
        keep_edge = (
            (meta["x_px"] >= half)
            & (meta["x_px"] < (w - half))
            & (meta["y_px"] >= half)
            & (meta["y_px"] < (h - half))
        )
        meta = meta.loc[keep_edge].copy()
        n_post_edge = int(meta.shape[0])
        n_lt20_post_edge = int((meta["transcript_counts"] < 20).sum()) if "transcript_counts" in meta.columns else None

        if int(args.max_cells_per_sample) > 0 and meta.shape[0] > int(args.max_cells_per_sample):
            low = meta[meta["transcript_counts"] < 20].copy() if "transcript_counts" in meta.columns else meta.iloc[0:0].copy()
            remaining_n = max(0, int(args.max_cells_per_sample) - low.shape[0])
            rest = meta.drop(low.index)
            if rest.shape[0] > remaining_n:
                rest = rest.sample(n=remaining_n, random_state=int(args.seed))
            meta = pd.concat([low, rest], axis=0).sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)

        counts, barcodes, genes = load_counts(xenium_dir)
        if genes_ref is None:
            genes_ref = genes
        elif genes != genes_ref:
            raise ValueError(f"Gene order mismatch for {sample_id}")
        barcode_index = pd.Index(barcodes)
        ridx = barcode_index.get_indexer(meta["cell_id"].astype(str))
        valid = ridx >= 0
        if not valid.all():
            meta = meta.loc[valid].copy()
            ridx = ridx[valid]
        x_sub = counts[ridx].tocsr()

        meta = meta.reset_index(drop=True)
        all_meta.append(meta)
        all_counts.append(x_sub)
        summary["samples"].append(
            {
                "sample_id": sample_id,
                "xenium_dir": str(xenium_dir),
                "he_image": str(he_path),
                "n_cells_input": n_cells_input,
                "n_cells_with_boundary": n_with_boundary,
                "n_cells_post_edge": n_post_edge,
                "n_cells_output": int(meta.shape[0]),
                "min_transcripts_input": int(cells["transcript_counts"].min()) if "transcript_counts" in cells.columns else None,
                "n_lt20_input": n_lt20_input,
                "n_lt20_post_edge": n_lt20_post_edge,
                "n_lt20_output": int((meta["transcript_counts"] < 20).sum()) if "transcript_counts" in meta.columns else None,
                "he_shape": [h, w],
            }
        )

    if genes_ref is None:
        raise ValueError("No samples prepared")
    meta_all = pd.concat(all_meta, axis=0, ignore_index=True)
    x_all = sp.vstack(all_counts).tocsr()
    if meta_all.shape[0] != x_all.shape[0]:
        raise ValueError("metadata/count row mismatch")

    meta_all.to_parquet(outdir / "prepared_meta.parquet", index=False)
    sp.save_npz(outdir / "prepared_counts_gene_expr.npz", x_all)
    (outdir / "prepared_genes.txt").write_text("\n".join(genes_ref) + "\n", encoding="utf-8")
    summary["n_cells_output"] = int(meta_all.shape[0])
    summary["n_genes"] = int(len(genes_ref))
    summary["n_lt20_output"] = int((meta_all["transcript_counts"] < 20).sum()) if "transcript_counts" in meta_all.columns else None
    (outdir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
