# R / Seurat Environment

Notebook 3 needs an R environment that can read the Xenium Seurat RDS object.

Minimum expected R packages:

- `Seurat`
- `SeuratObject`
- `Matrix`
- `dplyr`
- `readr`
- `jsonlite`
- `arrow`
- `sf`

The base R environment on this machine may not include Seurat. If `library(Seurat)` fails, create or activate an R environment with these packages before running the R extraction steps.
