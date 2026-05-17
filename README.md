# HEnium Morphology-Assisted Neutrophil Rescue Tutorial

This repository is a notebook-driven tutorial for rescuing neutrophils in Xenium data using registered H&E morphology. The motivating problem is that neutrophils can be poorly resolved by transcript-only calls from the Xenium 5k panel, especially when transcript counts are low. This workflow uses H&E morphology and RNA-aligned image embeddings to identify a neutrophil-like morphology cluster, then carries those morphology-confirmed neutrophils into RNA/image WNN label reconciliation.

The included executed notebooks are intended as the readable tutorial version. They show the expected outputs, plots, and review artifacts from the local two-sample run.

## Why Morphology Rescue?

Notebook 2 identifies image cluster `34` from the RNA-aligned BLEEP + Harmony morphology clustering as the neutrophil morphology cluster. In this local review, the morphology rescue was estimated to be about 80% accurate, but that should be treated as a dataset-specific manual-review estimate rather than a universal benchmark. Users should re-review the H&E patch PDFs for their own data before accepting any morphology cluster as neutrophil.

Notebook 3 intentionally applies an asymmetric QC rule:

- morphology-confirmed cluster-34 neutrophils are retained regardless of transcript count;
- all non-neutrophils are filtered to `transcript_counts >= 10`;
- retained cells are reconciled with RNA + image Muon WNN and hierarchical coarse/fine label transfer.

This avoids discarding low-transcript neutrophils before morphology has a chance to rescue them.

## Repository Contents

- `notebooks/01_palom_he_to_xenium_registration.executed.ipynb`: Palom H&E-to-Xenium morphology registration.
- `notebooks/02_hoptimus_image_embedding_neutrophil_review.executed.ipynb`: H-Optimus patch embeddings, BLEEP RNA-aligned image clustering, CXCL8/CXCR2 checks, and cluster-34 H&E morphology review.
- `notebooks/03_xenium_rna_image_wnn_label_reconciliation.executed.ipynb`: cluster-34 neutrophil rescue, non-neutrophil transcript filtering, Muon WNN, and hierarchical label transfer.
- `notebooks/01_*.ipynb`, `02_*.ipynb`, `03_*.ipynb`: output-free source notebooks for reruns.
- `scripts/`: reusable command-line helpers called by the notebooks.
- `config/`: sample tables and analysis settings.
- `envs/`: Python and R package requirements.
- `input/README.md` and `input/input_manifest.json`: local input layout and file recipe.
- `DATA_AVAILABILITY.md`: data packaging policy.

Large inputs and generated outputs are intentionally ignored by Git.

## Active Workflow

1. **Palom registration**
   - Registers each H&E OME-TIFF to the matching Xenium morphology-focus OME-TIFF.
   - Writes registered H&E images and full-slide overlays under `results/01_palom_registration/`.

2. **Morphology clustering and neutrophil review**
   - Builds the full Cellpose all-cell source from Xenium-format `cells.csv.gz`, `cell_boundaries.csv.gz`, and `cell_feature_matrix/`.
   - Extracts cell-centered registered H&E patches using Cellpose cell-boundary centroids.
   - Embeds patches with `bioptimus/H-optimus-1`.
   - Computes Xenium-style RNA PCA10 and aligns raw H-Optimus image embeddings to RNA with BLEEP-style soft-target contrastive alignment.
   - Runs the RNA-aligned BLEEP image target-40 clustering branch with Harmony by `sample_id`.
   - Exports cluster-level H&E patch PDFs and marker plots; cluster `34` is the morphology-confirmed neutrophil cluster used in Notebook 3.

3. **RNA/image WNN and label reconciliation**
   - Keeps all cluster-34 neutrophils.
   - Filters non-neutrophils to `transcript_counts >= 10`.
   - Runs Muon WNN from Xenium-normalized RNA PCA10 plus BLEEP-aligned image embeddings.
   - Transfers coarse labels first, then fine labels only within the predicted coarse parent.
   - Overrides confirmed cluster-34 cells to `neutrophil` for coarse and fine labels.

## Installation

Create the main tutorial Python environment:

```bash
cd HEnium_tutorial_2sample
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r envs/requirements-hennium-python.txt
python -m pip install ipykernel
python -m ipykernel install --user --name henium-tutorial --display-name "HEnium tutorial (.venv)"
```

Notebook 1 uses Palom. Palom is best installed in a separate Python 3.10 environment:

```bash
python -m venv .venv-palom
source .venv-palom/bin/activate
python -m pip install --upgrade pip
python -m pip install -r envs/requirements-palom.txt
```

The official Palom documentation is the source of truth for platform-specific installation details: <https://github.com/labsyspharm/palom/tree/main>

Notebook 3 also needs an R environment capable of reading the Seurat reference object; see `envs/r-seurat-session.md`.

## Data Availability and Local Inputs

The raw OME-TIFFs, Cellpose/Xenium-format outputs, Seurat RDS reference, H-Optimus embeddings, registered H&E images, WNN graphs, and review PDFs are too large and/or study-specific for normal GitHub storage. They are not committed.

To rerun the tutorial, place equivalent local inputs following:

- `input/README.md`
- `input/input_manifest.json`
- `config/samples.tsv`
- `config/cellpose_samples.tsv`
- `config/tutorial_paths.yaml`

The configs use paths relative to the repository root. Run notebooks in order so Notebook 1 creates the registered H&E files used by Notebooks 2 and 3.
