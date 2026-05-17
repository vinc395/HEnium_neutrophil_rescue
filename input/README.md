# Packaged Tutorial Inputs

This directory contains only the inputs currently read by the three tutorial notebooks.

## Notebook 1 registration inputs

`registration/` contains one subdirectory per sample. Each sample subdirectory contains the raw H&E OME-TIFF and the Xenium morphology-focus `0000` OME-TIFF listed in `../config/samples.tsv`. It also includes the matching `morphology_focus_0001.ome.tif` to `morphology_focus_0003.ome.tif` companion channel files using the generic filenames referenced by the `0000` OME-TIFF metadata.

## Notebook 2 and 3 Cellpose/Xenium inputs

`cellpose/cellpose_outs_AP0921a/` and `cellpose/cellpose_outs_AP2320a/` are minimal Xenium-format Cellpose exports. Each sample keeps only:

- `cells.csv.gz`
- `cell_boundaries.csv.gz`
- `cell_feature_matrix/matrix.mtx.gz`
- `cell_feature_matrix/features.tsv.gz`
- `cell_feature_matrix/barcodes.tsv.gz`

The notebooks do not read transcripts, zarr archives, morphology-focus image channels, overview images, parquet duplicates, nucleus boundaries, or Cell Ranger-style H5 files, so those files were intentionally not copied.

Notebook 2 expects the Palom-registered H&E files produced by Notebook 1 under `../results/01_palom_registration/outputs/<sample>/registered_slides/`.

## Notebook 3 label-transfer references

`reference/tiles_annot_cnt30_xen_only_res05_final_version.rds` is the coarse label-transfer reference RDS.

`reference/xenium_cellpose_subtypes_final_reference/` contains the subtype reference tables used for hierarchical fine-label transfer.

See `input_manifest.json` for the exact packaged file list and sizes.
