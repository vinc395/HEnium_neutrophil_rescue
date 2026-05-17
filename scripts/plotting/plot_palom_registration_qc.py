#!/usr/bin/env python3
"""Create registration overlay and random tile QC figures.

The script is deliberately conservative: it reads low-resolution arrays where
possible and writes static PNGs for tutorial notebooks. Boundary plotting uses
CSV exports produced by scripts/utils/extract_shape_boundaries.R when polygon
coordinates can be converted by the local R environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/tutorial_paths.yaml")
    ap.add_argument("--sample", action="append", help="Sample ID to plot. Defaults to all samples.")
    ap.add_argument("--registered-dir", default="", help="Directory containing <sample>_he_registered_palom.ome.tif")
    ap.add_argument("--outdir", default="", help="Output figure directory")
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--tile-count", type=int, default=4)
    ap.add_argument("--tile-zoom-factor", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def read_level(path: Path, max_dim: int = 3000) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = getattr(series, "levels", [series])
        chosen = levels[-1]
        for level in levels:
            h, w = spatial_shape(level.shape)
            if max(h, w) <= max_dim:
                chosen = level
                break
        arr = chosen.asarray()
    return normalize_image(arr)


def spatial_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3:
        if shape[0] <= 8 and shape[-1] > 8:
            return int(shape[1]), int(shape[2])
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported image shape: {shape}")


def read_base_crop(path: Path, x: int, y: int, size: int) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        arr = tif.series[0].levels[0].asarray()
    if arr.ndim == 3 and arr.shape[0] <= 8:
        arr = np.moveaxis(arr, 0, -1)
    h, w = spatial_shape(arr.shape)
    x = max(0, min(int(x), max(w - size, 0)))
    y = max(0, min(int(y), max(h - size, 0)))
    crop = arr[y : y + size, x : x + size]
    return normalize_image(crop)


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[-1] > 8:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        lo, hi = np.percentile(arr, [1, 99])
        out = np.clip((arr - lo) / max(hi - lo, 1), 0, 1)
        return np.dstack([out, out, out])
    arr = arr[..., :3].astype(np.float32)
    lo, hi = np.percentile(arr.reshape(-1, arr.shape[-1]), [1, 99], axis=0)
    return np.clip((arr - lo) / np.maximum(hi - lo, 1), 0, 1)


def overlay_rgb(he: np.ndarray, xen: np.ndarray) -> np.ndarray:
    h = min(he.shape[0], xen.shape[0])
    w = min(he.shape[1], xen.shape[1])
    he = he[:h, :w]
    xen_gray = xen[:h, :w].mean(axis=2)
    out = he.copy()
    out[..., 1] = np.maximum(out[..., 1] * 0.65, xen_gray)
    out[..., 2] = np.maximum(out[..., 2] * 0.65, xen_gray)
    return np.clip(out, 0, 1)


def sample_tiles(shape: tuple[int, int], tile_size: int, tile_count: int, seed: int) -> list[tuple[int, int]]:
    if tile_count <= 0:
        return []
    rng = np.random.default_rng(seed)
    h, w = shape
    if h <= tile_size or w <= tile_size:
        return [(0, 0)]
    xs = rng.integers(0, w - tile_size, size=tile_count)
    ys = rng.integers(0, h - tile_size, size=tile_count)
    return list(zip(xs.tolist(), ys.tolist()))


def base_shape(path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tif:
        return spatial_shape(tif.series[0].levels[0].shape)


def resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(im.resize((size, size), resample=Image.Resampling.BICUBIC)).astype(np.float32) / 255.0


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sample_table = pd.read_csv(cfg.get("registration_sample_table", cfg.get("sample_table")), sep="\t")
    requested = set(args.sample or sample_table["sample_id"].astype(str))
    sample_table = sample_table[sample_table["sample_id"].astype(str).isin(requested)]
    palom_dir = Path(args.registered_dir or Path(cfg["results"]["palom"]) / "registered_he")
    outdir = Path(args.outdir or Path(cfg["results"]["palom"]) / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for row in sample_table.to_dict("records"):
        sample = row["sample_id"]
        xenium = Path(row["xenium_morphology_ome_tif"])
        he = palom_dir / f"{sample}_he_registered_palom.ome.tif"
        if not he.exists():
            he = Path(cfg["results"]["palom"]) / "outputs" / sample / "registered_slides" / f"{sample}_he_registered_palom.ome.tif"
        if not he.exists():
            print(f"[skip] missing registered H&E for {sample}: {he}")
            continue

        he_low = read_level(he)
        xen_low = read_level(xenium)
        full_overlay = overlay_rgb(he_low, xen_low)
        full_path = outdir / f"{sample}_palom_full_overlay.png"
        plt.imsave(full_path, full_overlay)

        he_base_h, he_base_w = base_shape(he)
        low_h, low_w = he_low.shape[:2]
        scale_x = he_base_w / low_w
        scale_y = he_base_h / low_h
        close_size = max(32, int(round(args.tile_size / max(args.tile_zoom_factor, 1.0))))

        tile_paths = []
        for idx, (x, y) in enumerate(sample_tiles(he_low.shape[:2], args.tile_size, args.tile_count, args.seed), start=1):
            cx = int(round((x + args.tile_size / 2) * scale_x))
            cy = int(round((y + args.tile_size / 2) * scale_y))
            x0 = cx - close_size // 2
            y0 = cy - close_size // 2
            he_crop = read_base_crop(he, x0, y0, close_size)
            xen_crop = read_base_crop(xenium, x0, y0, close_size)
            crop = resize_rgb(overlay_rgb(he_crop, xen_crop), args.tile_size)
            tile_path = outdir / f"{sample}_palom_overlay_tile{idx}.png"
            plt.imsave(tile_path, crop)
            tile_paths.append(str(tile_path))

        manifest.append(
            {
                "sample_id": sample,
                "full_overlay": str(full_path),
                "tile_overlays": tile_paths,
                "tile_zoom_factor": float(args.tile_zoom_factor),
                "tile_crop_size_base_px": int(close_size),
            }
        )

    manifest_path = outdir / "palom_registration_qc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
