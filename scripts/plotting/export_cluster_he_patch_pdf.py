#!/usr/bin/env python3
"""Export grouped H&E patch examples as a multi-page PDF."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export grouped H&E patch grids to a multi-page PDF")
    ap.add_argument("--meta", required=True, help="prepared_meta.parquet path")
    ap.add_argument("--clusters", required=True, help="CSV with sample_id, cell_id, and a grouping column")
    ap.add_argument("--label-column", default="cluster", help="Grouping column in --clusters")
    ap.add_argument("--out-pdf", required=True, help="Output PDF path")
    ap.add_argument("--out-csv", required=True, help="Output sampled metadata CSV path")
    ap.add_argument("--patch-size", type=int, default=224, help="Patch size in pixels")
    ap.add_argument(
        "--upsample-factor",
        type=float,
        default=1.0,
        help="Upsample factor used by UNI2 extraction (effective crop=patch_size/upsample_factor, then resize)",
    )
    ap.add_argument("--grid-size", type=int, default=5, help="Grid width/height (5 => 25 patches)")
    ap.add_argument(
        "--n-patches-per-cluster",
        type=int,
        default=None,
        help="Exact number of patches per cluster (overrides grid-size^2 when provided).",
    )
    ap.add_argument(
        "--n-cols",
        type=int,
        default=None,
        help="Number of columns for rectangular grids when --n-patches-per-cluster is set (default: grid-size).",
    )
    ap.add_argument("--seed", type=int, default=17, help="Random seed")
    ap.add_argument(
        "--boundaries-dir",
        default="",
        help="Optional directory containing <sample>_boundaries.csv with x/y polygon vertices.",
    )
    ap.add_argument(
        "--draw-boundaries",
        action="store_true",
        help="Draw the selected cell boundary as a dotted line when --boundaries-dir is supplied.",
    )
    return ap.parse_args()


def to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    x = np.squeeze(x)

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3:
        if x.shape[-1] in (3, 4):
            x = x[..., :3]
        elif x.shape[0] in (3, 4):
            x = np.moveaxis(x[:3, ...], 0, -1)
        else:
            # Fallback to first 3 planes on last axis if possible.
            x = x[..., :3]
            if x.shape[-1] != 3:
                raise ValueError(f"Cannot infer RGB channels from image shape {arr.shape}")
    else:
        raise ValueError(f"Unsupported image ndim={x.ndim} shape={arr.shape}")

    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating):
            vmax = float(np.nanmax(x)) if x.size else 0.0
            if vmax <= 1.0:
                x = np.clip(x * 255.0, 0, 255)
            else:
                x = np.clip(x, 0, 255)
        else:
            x = np.clip(x, 0, 255)
        x = x.astype(np.uint8)
    return x


def extract_patch(img: np.ndarray, x_px: float, y_px: float, patch_size: int, upsample_factor: float) -> np.ndarray:
    h, w, _ = img.shape

    # Match UNI2 pipeline behavior: smaller FOV for upsample>1, then resize to patch_size.
    effective = int(round(patch_size / max(float(upsample_factor), 1e-8)))
    effective = max(8, effective)
    half = effective // 2

    cx = int(round(x_px))
    cy = int(round(y_px))
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + effective
    y1 = y0 + effective

    patch = np.zeros((effective, effective, 3), dtype=np.uint8)

    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)

    if sx1 <= sx0 or sy1 <= sy0:
        return patch

    dx0 = sx0 - x0
    dy0 = sy0 - y0
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)
    patch[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]

    if effective != patch_size:
        patch = np.asarray(
            Image.fromarray(patch, mode="RGB").resize((patch_size, patch_size), resample=Image.Resampling.LANCZOS)
        )
    return patch


def patch_window(x_px: float, y_px: float, patch_size: int, upsample_factor: float) -> tuple[int, int, int, float]:
    effective = int(round(patch_size / max(float(upsample_factor), 1e-8)))
    effective = max(8, effective)
    half = effective // 2
    cx = int(round(x_px))
    cy = int(round(y_px))
    x0 = cx - half
    y0 = cy - half
    scale = patch_size / effective
    return x0, y0, effective, scale


def load_boundary_centroids(boundaries_dir: Path, sample_id: str) -> pd.DataFrame:
    cache = boundaries_dir / f"{sample_id}_boundary_centroids.csv"
    if cache.exists():
        out = pd.read_csv(cache)
        if {"boundary_cell_id", "boundary_x_centroid", "boundary_y_centroid"}.issubset(out.columns):
            return out
        rename = {}
        if "cell_id" in out.columns and "boundary_cell_id" not in out.columns:
            rename["cell_id"] = "boundary_cell_id"
        if "x_centroid" in out.columns and "boundary_x_centroid" not in out.columns:
            rename["x_centroid"] = "boundary_x_centroid"
        if "y_centroid" in out.columns and "boundary_y_centroid" not in out.columns:
            rename["y_centroid"] = "boundary_y_centroid"
        out = out.rename(columns=rename)
        required = {"boundary_cell_id", "boundary_x_centroid", "boundary_y_centroid"}
        if not required.issubset(out.columns):
            raise ValueError(f"{cache} missing required centroid columns after normalization: {sorted(required - set(out.columns))}")
        return out

    path = boundaries_dir / f"{sample_id}_boundaries.csv"
    if not path.exists():
        return pd.DataFrame(columns=["boundary_cell_id", "boundary_x_centroid", "boundary_y_centroid"])

    pieces = []
    for chunk in pd.read_csv(path, usecols=["cell_id", "x", "y"], chunksize=500000):
        g = chunk.groupby("cell_id", sort=False).agg(
            x_sum=("x", "sum"),
            y_sum=("y", "sum"),
            n=("x", "size"),
        )
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(columns=["boundary_cell_id", "boundary_x_centroid", "boundary_y_centroid"])
    sums = pd.concat(pieces).groupby(level=0).sum()
    out = pd.DataFrame(
        {
            "boundary_cell_id": sums.index.astype(str),
            "boundary_x_centroid": sums["x_sum"].to_numpy(dtype=float) / sums["n"].to_numpy(dtype=float),
            "boundary_y_centroid": sums["y_sum"].to_numpy(dtype=float) / sums["n"].to_numpy(dtype=float),
        }
    )
    out.to_csv(cache, index=False)
    return out


def assign_nearest_boundary_cells(boundaries_dir: Path, sample_id: str, selected: pd.DataFrame) -> dict[str, str]:
    centroids = load_boundary_centroids(boundaries_dir, sample_id)
    if centroids.empty:
        return {}
    tree = cKDTree(centroids[["boundary_x_centroid", "boundary_y_centroid"]].to_numpy(dtype=float))
    query = selected[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    _dist, idx = tree.query(query, k=1)
    return dict(zip(selected["cell_id"].astype(str), centroids.iloc[idx]["boundary_cell_id"].astype(str)))


def load_boundaries(boundaries_dir: Path, sample_id: str, boundary_ids: set[str]) -> pd.DataFrame:
    path = boundaries_dir / f"{sample_id}_boundaries.csv"
    cols = ["sample_id", "cell_id", "polygon_id", "vertex_id", "x", "y"]
    if not path.exists() or not boundary_ids:
        return pd.DataFrame(columns=cols)

    chunks = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in cols, chunksize=500000):
        chunk["cell_id"] = chunk["cell_id"].astype(str)
        keep = chunk["cell_id"].isin(boundary_ids)
        if keep.any():
            chunks.append(chunk.loc[keep, cols].copy())
    if not chunks:
        return pd.DataFrame(columns=cols)
    out = pd.concat(chunks, ignore_index=True)
    out["polygon_id"] = out["polygon_id"].fillna(1).astype(int)
    out["vertex_id"] = out["vertex_id"].fillna(0).astype(int)
    return out.sort_values(["cell_id", "polygon_id", "vertex_id"])


def _sort_cluster_labels(vals: list[object]) -> list[object]:
    def key_fn(v: object):
        s = str(v)
        if s.isdigit():
            return (0, int(s))
        return (1, s)

    return sorted(vals, key=key_fn)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    meta = pd.read_parquet(args.meta)
    clu = pd.read_csv(args.clusters)

    req_meta = {"cell_id", "sample_id", "x_centroid", "y_centroid", "x_px", "y_px", "he_image", "um_per_px"}
    group_col = str(args.label_column)
    req_clu = {"cell_id", "sample_id", group_col}
    if not req_meta.issubset(meta.columns):
        miss = sorted(req_meta - set(meta.columns))
        raise ValueError(f"Missing metadata columns: {miss}")
    if not req_clu.issubset(clu.columns):
        miss = sorted(req_clu - set(clu.columns))
        raise ValueError(f"Missing cluster columns: {miss}")

    m = meta[["cell_id", "sample_id", "x_centroid", "y_centroid", "x_px", "y_px", "he_image", "um_per_px"]].copy()
    d = clu[["cell_id", "sample_id", group_col]].copy()
    d = d.rename(columns={group_col: "cluster"})
    d = d.merge(m, on=["cell_id", "sample_id"], how="left", validate="one_to_one")
    if d["x_px"].isna().any() or d["he_image"].isna().any():
        missing = int(d["x_px"].isna().sum() + d["he_image"].isna().sum())
        raise ValueError(f"Could not map all clustered cells to metadata fields (missing entries={missing})")

    grid_n = int(args.grid_size)
    per_cluster = int(args.n_patches_per_cluster) if args.n_patches_per_cluster is not None else grid_n * grid_n
    if per_cluster <= 0:
        raise ValueError("--n-patches-per-cluster must be > 0")
    n_cols = int(args.n_cols) if args.n_cols is not None else grid_n
    if n_cols <= 0:
        raise ValueError("--n-cols must be > 0")
    n_rows = int(math.ceil(per_cluster / n_cols))
    clusters = _sort_cluster_labels(list(d["cluster"].dropna().unique()))

    picks: list[pd.DataFrame] = []
    for cl in clusters:
        sub = d[d["cluster"] == cl].copy()
        n = min(per_cluster, sub.shape[0])
        choose = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        pick = sub.loc[choose].copy().reset_index(drop=True)
        pick["grid_index"] = np.arange(n, dtype=int)
        picks.append(pick)

    sel = pd.concat(picks, axis=0, ignore_index=True)
    sel["boundary_drawn"] = False

    boundaries: dict[tuple[str, str], pd.DataFrame] = {}
    boundaries_dir = Path(args.boundaries_dir) if args.boundaries_dir else None
    if args.draw_boundaries and boundaries_dir is not None:
        for sid, g in sel.groupby("sample_id", sort=False):
            nearest = assign_nearest_boundary_cells(boundaries_dir, str(sid), g)
            b = load_boundaries(boundaries_dir, str(sid), set(nearest.values()))
            rev: dict[str, list[str]] = {}
            for selected_id, boundary_id in nearest.items():
                rev.setdefault(boundary_id, []).append(selected_id)
            for boundary_id, gb in b.groupby("cell_id", sort=False):
                for selected_id in rev.get(str(boundary_id), []):
                    boundaries[(str(sid), str(selected_id))] = gb

    # Extract all selected patches by loading each sample image once.
    patch_map: dict[tuple[str, int], np.ndarray] = {}
    for sid, g in sel.groupby("sample_id", sort=False):
        he_path = Path(g["he_image"].iloc[0])
        img = to_rgb_uint8(tifffile.imread(he_path))
        for _, row in g.iterrows():
            key = (str(row["cluster"]), int(row["grid_index"]))
            patch_map[key] = extract_patch(
                img=img,
                x_px=float(row["x_px"]),
                y_px=float(row["y_px"]),
                patch_size=int(args.patch_size),
                upsample_factor=float(args.upsample_factor),
            )
        del img

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sel.to_csv(out_csv, index=False)

    with PdfPages(out_pdf) as pdf:
        for cl in clusters:
            sub = sel[sel["cluster"] == cl].copy()
            n = sub.shape[0]

            fig_w = max(11.0, 0.8 * n_cols)
            fig_h = max(8.5, 0.8 * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=170)
            axes = np.asarray(axes).reshape(n_rows, n_cols)

            for i in range(per_cluster):
                ax = axes.flat[i]
                if i < n:
                    row = sub.iloc[i]
                    key = (str(cl), i)
                    patch = patch_map[key]
                    ax.imshow(patch)
                    b = boundaries.get((str(row["sample_id"]), str(row["cell_id"])))
                    if b is not None and not b.empty:
                        x0, y0, _effective, scale = patch_window(
                            float(row["x_px"]),
                            float(row["y_px"]),
                            int(args.patch_size),
                            float(args.upsample_factor),
                        )
                        drawn = False
                        for _pid, poly in b.groupby("polygon_id", sort=False):
                            coord_scale = 1.0 / float(row["um_per_px"])
                            xs = (poly["x"].to_numpy(dtype=float) * coord_scale - x0) * scale
                            ys = (poly["y"].to_numpy(dtype=float) * coord_scale - y0) * scale
                            in_view = (xs >= 0) & (xs <= args.patch_size) & (ys >= 0) & (ys <= args.patch_size)
                            if in_view.any():
                                ax.plot(xs, ys, color="yellow", linewidth=0.7, linestyle=(0, (1.5, 1.5)))
                                drawn = True
                        if drawn:
                            sel_idx = sub.index[i]
                            sel.loc[sel_idx, "boundary_drawn"] = True
                    ax.set_title(
                        f"{row['sample_id']} | {row['cell_id']}",
                        fontsize=6,
                        pad=2.0,
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            fig.suptitle(
                f"{group_col} {cl}: H&E patch examples ({n} shown, {n_rows}x{n_cols}, upsample={args.upsample_factor:g})",
                fontsize=14,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

    sel.to_csv(out_csv, index=False)

    print(f"groups={len(clusters)}")
    print(f"patches_per_group={per_cluster}")
    print(f"wrote_pdf={out_pdf}")
    print(f"wrote_csv={out_csv}")


if __name__ == "__main__":
    main()
