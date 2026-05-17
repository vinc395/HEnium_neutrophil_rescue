# Data Availability and Local Input Setup

This repository is intended to be committed as a code-only tutorial. The raw data and generated outputs are intentionally not included in GitHub.

## Required Local Inputs

Place or symlink the required files under `input/` following `input/README.md` and `input/input_manifest.json`.

The tutorial expects:

- H&E and Xenium morphology OME-TIFFs for Palom registration.
- Xenium-format Cellpose outputs for each sample, minimally including `cells.csv.gz`, `cell_boundaries.csv.gz`, and `cell_feature_matrix/`.
- A Seurat RDS reference object for coarse label transfer.
- Fine subtype reference tables for hierarchical subtype transfer.

## Why Inputs Are Not Committed

The local input bundle is several gigabytes and contains study-specific microscopy/RNA data. Generated outputs are also large and include registered OME-TIFFs, H-Optimus embeddings, WNN graphs, H&E patch PDFs, and label-transfer tables. These should be regenerated locally from the notebooks rather than stored in normal Git history.

## Reproducibility Contract

The committed configs describe the expected file layout and sample IDs. A user can reproduce the tutorial by placing equivalent inputs in the documented layout, installing the Python/R environments, and running the notebooks in order.
