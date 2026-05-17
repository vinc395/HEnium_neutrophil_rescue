#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default = NA_character_) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[[i + 1]]
}

rds_path <- arg_value("--rds")
outdir <- arg_value("--outdir")
label_col <- arg_value("--label-col", "cell_labels")
max_cells_per_label <- as.integer(arg_value("--max-cells-per-label", "25000"))
seed <- as.integer(arg_value("--seed", "42"))

if (is.na(rds_path) || is.na(outdir)) {
  stop("Usage: export_xenium_rds_cell_label_reference.R --rds path --outdir dir [--label-col cell_labels] [--max-cells-per-label 25000] [--seed 42]")
}

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)
if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("Matrix R package is required")
}
suppressPackageStartupMessages(library(Matrix))

message("[rds-export] reading RDS")
obj <- readRDS(rds_path)
if (!("sc_obj" %in% names(obj))) {
  stop("RDS does not contain sc_obj")
}
sc_obj <- obj[["sc_obj"]]
meta <- methods::slot(sc_obj, "meta.data")
if (!(label_col %in% colnames(meta))) {
  stop(sprintf("label column '%s' not found in sc_obj metadata", label_col))
}

labels <- as.character(meta[[label_col]])
valid <- !is.na(labels) & labels != ""
meta_out <- data.frame(
  cell = rownames(meta),
  sample = if ("sample" %in% colnames(meta)) as.character(meta[["sample"]]) else NA_character_,
  fov_cell = if ("fov_cell" %in% colnames(meta)) as.character(meta[["fov_cell"]]) else NA_character_,
  cell_labels = labels,
  stringsAsFactors = FALSE
)
write.csv(meta_out, file.path(outdir, "reference_cell_metadata.csv"), row.names = FALSE)

message("[rds-export] accessing RNA counts layer")
rna <- methods::slot(sc_obj, "assays")[["RNA"]]
layers <- methods::slot(rna, "layers")
if (!("counts" %in% names(layers))) {
  stop("RNA assay has no counts layer")
}
counts <- layers[["counts"]]
features <- methods::slot(rna, "features")
# Avoid SeuratObject method dispatch; this environment may not have that
# package installed, but the serialized LogMap still carries dimnames.
genes <- attr(features, "dimnames")[[1]]
if (is.null(genes) || length(genes) == 0) {
  genes <- attr(counts, "Dimnames")[[1]]
}
if (is.null(genes) || length(genes) == 0) {
  stop("Could not determine feature/gene names")
}
writeLines(genes, file.path(outdir, "reference_genes.txt"))

label_levels <- sort(unique(labels[valid]))
label_sum <- matrix(0, nrow = length(label_levels), ncol = length(genes))
rownames(label_sum) <- label_levels
colnames(label_sum) <- genes
label_n_total <- integer(length(label_levels))
label_n_used <- integer(length(label_levels))

sampled_idx <- integer(0)
for (i in seq_along(label_levels)) {
  lab <- label_levels[[i]]
  idx <- which(valid & labels == lab)
  label_n_total[[i]] <- length(idx)
  if (length(idx) > max_cells_per_label) {
    idx_use <- sort(sample(idx, max_cells_per_label))
  } else {
    idx_use <- idx
  }
  label_n_used[[i]] <- length(idx_use)
  sampled_idx <- c(sampled_idx, idx_use)
  message(sprintf("[rds-export] label=%s total=%d used=%d", lab, length(idx), length(idx_use)))
  sub <- counts[, idx_use, drop = FALSE]
  label_sum[i, ] <- as.numeric(Matrix::rowSums(sub))
}

sampled_idx <- sort(unique(sampled_idx))
sampled_meta <- meta_out[sampled_idx, , drop = FALSE]
write.csv(sampled_meta, file.path(outdir, "reference_sampled_cell_metadata.csv"), row.names = FALSE)

saveRDS(
  list(
    label_counts = label_sum,
    labels = label_levels,
    genes = genes,
    label_n_total = setNames(label_n_total, label_levels),
    label_n_used = setNames(label_n_used, label_levels),
    rds_path = rds_path,
    label_col = label_col,
    max_cells_per_label = max_cells_per_label,
    seed = seed
  ),
  file.path(outdir, "reference_label_count_sums.rds")
)

write.csv(
  data.frame(
    cell_labels = label_levels,
    n_total = label_n_total,
    n_used = label_n_used,
    stringsAsFactors = FALSE
  ),
  file.path(outdir, "reference_label_counts.csv"),
  row.names = FALSE
)

message("[rds-export] wrote reference outputs to ", outdir)
