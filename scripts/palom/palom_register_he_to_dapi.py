import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import palom
import tifffile
import zarr
from palom import align
from palom import color
from palom import register_dev


def _img_resize(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _downsample_for_pyramid(
    image: np.ndarray, photometric: str, pyramid_scale: int
) -> np.ndarray:
    scale = 1.0 / pyramid_scale
    if image.ndim == 3 and image.shape[0] < image.shape[-1]:
        if photometric in ("minisblack", "rgb"):
            image = np.moveaxis(image, 0, -1)
            image = _img_resize(image, scale)
            image = np.moveaxis(image, -1, 0)
            return image
    if photometric == "minisblack" and image.ndim == 3 and image.shape[0] < image.shape[-1]:
        image = np.moveaxis(image, 0, -1)
        image = _img_resize(image, scale)
        image = np.moveaxis(image, -1, 0)
        return image
    return _img_resize(image, scale)


def _read_pixel_size_microns(tif: tifffile.TiffFile) -> float:
    if tif.ome_metadata:
        try:
            from ome_types import from_xml

            ome = from_xml(tif.ome_metadata)
            px_size = ome.images[0].pixels.physical_size_x
            unit = ome.images[0].pixels.physical_size_x_unit
            if px_size is not None and unit is not None:
                try:
                    import pint

                    ureg = pint.UnitRegistry()
                    return px_size * ureg(unit.value).to(ureg.micron).magnitude
                except Exception:
                    return float(px_size)
        except Exception:
            pass
    try:
        xres = tif.pages[0].tags["XResolution"].value
        return 10000.0 / (xres[0] / xres[1])
    except Exception:
        return 1.0


def _read_channel_names(tif: tifffile.TiffFile) -> list[str] | None:
    if not tif.ome_metadata:
        return None
    try:
        from ome_types import from_xml

        ome = from_xml(tif.ome_metadata)
        channels = ome.images[0].pixels.channels
        names = [c.name for c in channels if c.name]
        return names or None
    except Exception:
        return None


def _iter_tiles_contig(zarr_arr, tile_size: int, axes: str):
    if zarr_arr.ndim != 3:
        raise ValueError("Only 3D arrays are supported for RGB export.")
    axes = axes or "SYX"
    channel_first = axes[0] in ("S", "C")
    if channel_first:
        samples, height, width = zarr_arr.shape
    else:
        height, width, samples = zarr_arr.shape
    for y in range(0, height, tile_size):
        y_end = min(y + tile_size, height)
        for x in range(0, width, tile_size):
            x_end = min(x + tile_size, width)
            if channel_first:
                tile = zarr_arr[:, y:y_end, x:x_end]
                tile = np.moveaxis(tile, 0, -1)
            else:
                tile = zarr_arr[y:y_end, x:x_end, :]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                padded = np.zeros((tile_size, tile_size, samples), dtype=tile.dtype)
                padded[: tile.shape[0], : tile.shape[1], :] = tile
                tile = padded
            yield tile


def _open_level_zarr(level):
    store = level.aszarr()
    z = zarr.open(store, mode="r")
    if isinstance(z, zarr.hierarchy.Group):
        if "0" in z:
            return z["0"]
        return z[list(z.array_keys())[0]]
    return z


def _write_pyramidal_contig(
    tif: tifffile.TiffFile,
    output_path: Path,
    tile_size: int,
    compression: str,
    compression_level: int,
    pyramid_scale: int,
    max_workers: int,
):
    series = tif.series[0]
    axes = series.axes
    levels = series.levels
    px_size = _read_pixel_size_microns(tif)
    channel_names = _read_channel_names(tif)
    if channel_names is None and series.shape[0] in (3, 4):
        channel_names = list("RGB")[: series.shape[0]]
    meta = {
        "axes": "YXS",
        "Pixels": {
            "PhysicalSizeX": px_size,
            "PhysicalSizeXUnit": "\u00b5m",
            "PhysicalSizeY": px_size,
            "PhysicalSizeYUnit": "\u00b5m",
        },
    }
    if channel_names:
        meta["Channel"] = {"Name": channel_names}

    bigtiff = True
    compressionargs = {"level": compression_level} if compression == "zlib" else None
    write_kwargs = dict(
        tile=(tile_size, tile_size),
        compression=compression,
        compressionargs=compressionargs,
        photometric="rgb",
        planarconfig="contig",
        maxworkers=max_workers,
        resolutionunit="CENTIMETER",
    )
    with tifffile.TiffWriter(str(output_path), bigtiff=bigtiff) as out_tif:
        base = _open_level_zarr(levels[0])
        base_axes = axes
        if base_axes is None:
            base_axes = "SYX"
        dtype = base.dtype
        height = base.shape[1] if base_axes[0] in ("S", "C") else base.shape[0]
        width = base.shape[2] if base_axes[0] in ("S", "C") else base.shape[1]
        samples = base.shape[0] if base_axes[0] in ("S", "C") else base.shape[2]
        out_tif.write(
            data=_iter_tiles_contig(base, tile_size, base_axes),
            shape=(height, width, samples),
            dtype=dtype,
            subifds=int(len(levels) - 1),
            metadata=meta,
            resolution=(1e4 / px_size, 1e4 / px_size),
            **write_kwargs,
        )
        for idx, level in enumerate(levels[1:], start=1):
            lvl = _open_level_zarr(level)
            lvl_axes = base_axes
            height = lvl.shape[1] if lvl_axes[0] in ("S", "C") else lvl.shape[0]
            width = lvl.shape[2] if lvl_axes[0] in ("S", "C") else lvl.shape[1]
            samples = lvl.shape[0] if lvl_axes[0] in ("S", "C") else lvl.shape[2]
            mag = pyramid_scale ** idx
            out_tif.write(
                data=_iter_tiles_contig(lvl, tile_size, lvl_axes),
                shape=(height, width, samples),
                dtype=dtype,
                subfiletype=1,
                metadata=None,
                resolution=(1e4 / (px_size * mag), 1e4 / (px_size * mag)),
                **write_kwargs,
            )


def _compute_subresolutions(
    image: np.ndarray, photometric: str, pyramid_scale: int, tile_size: int
) -> int:
    if image.ndim == 2:
        height, width = image.shape
    elif image.shape[0] < image.shape[-1] and photometric in ("minisblack", "rgb"):
        height, width = image.shape[1], image.shape[2]
    else:
        height, width = image.shape[0], image.shape[1]
    subres = 0
    while max(height, width) > tile_size:
        height = math.ceil(height / pyramid_scale)
        width = math.ceil(width / pyramid_scale)
        subres += 1
    return max(subres, 1)


def _read_series_thumbnail(path: Path, max_size: int = 2048) -> tuple[np.ndarray, str]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes or ""
        if len(series.levels) > 1:
            thumb = series.levels[-1].asarray()
            return thumb, axes
        full = series.asarray()
    if full.ndim == 3:
        if full.shape[0] < full.shape[-1]:
            full = np.moveaxis(full, 0, -1)
        h, w = full.shape[:2]
    else:
        h, w = full.shape
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        full = cv2.resize(full, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return full, axes


def _normalize_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    return (img * 255).astype(np.uint8)


def _overlay_he_dapi(he_path: Path, dapi_path: Path, dapi_channel: int, out_png: Path) -> None:
    he_img, he_axes = _read_series_thumbnail(he_path)
    if he_img.ndim == 3 and he_img.shape[0] < he_img.shape[-1]:
        he_img = np.moveaxis(he_img, 0, -1)
    if he_img.ndim == 2:
        he_img = np.stack([he_img] * 3, axis=-1)
    he_rgb = _normalize_uint8(he_img[..., :3])

    dapi_img, dapi_axes = _read_series_thumbnail(dapi_path)
    if dapi_img.ndim == 3:
        if dapi_axes and dapi_axes[0] in ("C", "S"):
            dapi = dapi_img[dapi_channel]
        elif dapi_img.shape[0] < dapi_img.shape[-1]:
            dapi = dapi_img[dapi_channel]
        else:
            dapi = dapi_img[..., 0]
    else:
        dapi = dapi_img
    dapi = _normalize_uint8(dapi)

    if dapi.shape[:2] != he_rgb.shape[:2]:
        dapi = cv2.resize(dapi, (he_rgb.shape[1], he_rgb.shape[0]), interpolation=cv2.INTER_AREA)

    dapi_color = np.zeros_like(he_rgb)
    dapi_color[..., 2] = dapi
    overlay = cv2.addWeighted(he_rgb, 0.7, dapi_color, 0.3, 0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), overlay)


def write_xe_pyramid(
    input_path: Path,
    output_path: Path,
    tile_size: int = 1024,
    compression: str = "jpeg2000",
    compression_level: int = 85,
    subresolutions: int | None = None,
    pyramid_scale: int = 2,
    max_workers: int = 4,
    force: bool = False,
) -> bool:
    with tifffile.TiffFile(input_path) as tif:
        series = tif.series[0]
        already_pyramidal = getattr(series, "is_pyramidal", False) or len(series.levels) > 1
        if already_pyramidal:
            if not force:
                return False
            _write_pyramidal_contig(
                tif,
                output_path,
                tile_size=tile_size,
                compression=compression,
                compression_level=compression_level,
                pyramid_scale=pyramid_scale,
                max_workers=max_workers,
            )
            return True
        image = tif.asarray()
        photometric = "rgb" if tif.pages[0].samplesperpixel == 3 else "minisblack"
        px_size = _read_pixel_size_microns(tif)
        channel_names = _read_channel_names(tif)

    if channel_names is None and image.ndim == 3 and image.shape[0] in (3, 4):
        channel_names = list("RGB")[: image.shape[0]]

    meta = {
        "axes": "YXS" if photometric == "rgb" else "CYX",
        "Pixels": {
            "PhysicalSizeX": px_size,
            "PhysicalSizeXUnit": "\u00b5m",
            "PhysicalSizeY": px_size,
            "PhysicalSizeYUnit": "\u00b5m",
        }
    }
    if channel_names:
        meta["Channel"] = {"Name": channel_names}

    if subresolutions is None:
        subresolutions = _compute_subresolutions(
            image, photometric, pyramid_scale, tile_size
        )

    bigtiff = image.size * image.itemsize > 2_000_000_000
    compressionargs = {"level": compression_level} if compression == "zlib" else None

    options = dict(
        tile=(tile_size, tile_size),
        compression=compression,
        compressionargs=compressionargs,
        photometric=photometric,
        metadata=meta,
        maxworkers=max_workers,
    )
    if photometric == "rgb":
        options["planarconfig"] = "contig"

    with tifffile.TiffWriter(str(output_path), bigtiff=bigtiff) as tif:
        tif.write(image, subifds=int(subresolutions), **options)
        for _ in range(int(subresolutions)):
            image = _downsample_for_pyramid(image, photometric, pyramid_scale)
            tif.write(image, subfiletype=1, **options)
    return True


def main():
    parser = argparse.ArgumentParser(description="Register H&E to DAPI using PALOM.")
    parser.add_argument("--he-path", required=True, help="H&E RGB OME-TIFF (moving).")
    parser.add_argument("--dapi-path", required=True, help="mIF OME-TIFF (reference).")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--out-name", default="he_registered_palom.ome.tif")
    parser.add_argument("--he-mpp", type=float, default=0.25, help="H&E pixel size (micron/pixel).")
    parser.add_argument("--dapi-mpp", type=float, default=None, help="DAPI pixel size (micron/pixel).")
    parser.add_argument("--dapi-channel", type=int, default=0, help="DAPI channel index (0-based).")
    parser.add_argument("--n-keypoints", type=int, default=4000, help="Keypoints for affine init.")
    parser.add_argument("--thumbnail-max-size", type=int, default=2000)
    parser.add_argument(
        "--skip-xe-pyramid",
        action="store_true",
        help="Skip Xenium Explorer pyramid conversion step.",
    )
    parser.add_argument("--xe-out", default=None, help="Optional output path for XE pyramid.")
    parser.add_argument("--xe-subresolutions", type=int, default=None)
    parser.add_argument(
        "--xe-compression",
        choices=["jpeg2000", "zlib"],
        default="zlib",
    )
    parser.add_argument(
        "--xe-compression-level",
        type=int,
        default=6,
        help="Compression level (zlib: 0-9).",
    )
    parser.add_argument("--xe-max-workers", type=int, default=4)
    parser.add_argument("--xe-force", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--overlay-out", default=None, help="Optional path for overlay PNG.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    r_ref = palom.reader.OmePyramidReader(args.dapi_path, pixel_size=args.dapi_mpp)
    r_mov = palom.reader.OmePyramidReader(args.he_path, pixel_size=args.he_mpp)

    level_ref = 0
    level_mov = 0
    thumb_level_ref = r_ref.get_thumbnail_level_of_size(args.thumbnail_max_size)
    thumb_level_mov = r_mov.get_thumbnail_level_of_size(args.thumbnail_max_size)

    # DAPI reference channel
    ref_img = r_ref.read_level_channels(level_ref, args.dapi_channel)
    ref_thumb = r_ref.read_level_channels(thumb_level_ref, args.dapi_channel)

    # Hematoxylin from H&E
    hax = color.PyramidHaxProcessor(r_mov.pyramid, thumbnail_level=thumb_level_mov)
    mov_img = hax.get_processed_color(level_mov, mode="hematoxylin", out_dtype=np.float32)
    mov_thumb = hax.get_processed_color(thumb_level_mov, mode="hematoxylin", out_dtype=np.float32)

    aligner = align.Aligner(
        ref_img=ref_img,
        moving_img=mov_img,
        ref_thumbnail=ref_thumb,
        moving_thumbnail=mov_thumb,
        ref_thumbnail_down_factor=r_ref.level_downsamples[thumb_level_ref]
        / r_ref.level_downsamples[level_ref],
        moving_thumbnail_down_factor=r_mov.level_downsamples[thumb_level_mov]
        / r_mov.level_downsamples[level_mov],
    )

    mx = register_dev.search_then_register(
        np.asarray(aligner.ref_thumbnail),
        np.asarray(aligner.moving_thumbnail),
        n_keypoints=args.n_keypoints,
        auto_mask=True,
        max_size=args.thumbnail_max_size,
    )
    aligner.coarse_affine_matrix = np.vstack([mx, [0, 0, 1]])

    aligner.compute_shifts()
    aligner.constrain_shifts()
    block_mx = aligner.block_affine_matrices_da

    # Apply transform to full RGB H&E
    mosaic = align.block_affine_transformed_moving_img(
        ref_img=aligner.ref_img, moving_img=r_mov.pyramid[level_mov], mxs=block_mx
    )

    tifffile_kwarg = {"photometric": "rgb", "planarconfig": "separate"}
    palom.pyramid.write_pyramid(
        mosaics=[mosaic],
        output_path=out_path,
        pixel_size=r_ref.pixel_size * r_ref.level_downsamples[level_ref],
        channel_names=[list("RGB")],
        compression="zlib",
        downscale_factor=2,
        save_RAM=True,
        tile_size=1024,
        kwargs_tifffile=tifffile_kwarg,
    )

    print(f"Saved registered H&E to {out_path}")
    if not args.skip_overlay:
        overlay_path = (
            Path(args.overlay_out)
            if args.overlay_out
            else out_dir / f"{out_path.stem}_overlay.png"
        )
        _overlay_he_dapi(out_path, Path(args.dapi_path), args.dapi_channel, overlay_path)
        print(f"Saved overlay PNG to {overlay_path}")
    if not args.skip_xe_pyramid:
        xe_out = Path(args.xe_out) if args.xe_out else out_path
        tmp_out = xe_out
        if xe_out == out_path:
            tmp_out = out_path.with_suffix(".xe_tmp.ome.tif")
        converted = write_xe_pyramid(
            out_path,
            tmp_out,
            tile_size=1024,
            compression=args.xe_compression,
            compression_level=args.xe_compression_level,
            subresolutions=args.xe_subresolutions,
            pyramid_scale=2,
            max_workers=args.xe_max_workers,
            force=args.xe_force,
        )
        if converted:
            if tmp_out != xe_out:
                tmp_out.replace(xe_out)
            print(f"Saved XE pyramid to {xe_out}")
        else:
            if tmp_out.exists():
                tmp_out.unlink()
            print("XE pyramid conversion skipped (already pyramidal).")


if __name__ == "__main__":
    main()
