#!/usr/bin/env python3
"""Per-cell coarse and hierarchical fine label transfer for Notebook 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wnn-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--coarse-reference-csv", default="")
    ap.add_argument("--coarse-reference-rds", required=True, help="Configured sc_obj RDS; used for provenance")
    ap.add_argument("--subtype-reference-dir", required=True)
    ap.add_argument("--out-prefix", default="label_transfer")
    ap.add_argument("--target", type=int, default=30, help="Accepted for backward compatibility; not used for transfer")
    ap.add_argument("--batch-size", type=int, default=2000)
    return ap.parse_args()


def read_genes(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def log1p_cpm(mat: np.ndarray) -> np.ndarray:
    lib = np.maximum(mat.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(mat / lib * 10000.0)


def load_reference(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(path, index_col=0)
    return df, df.index.astype(str).tolist(), df.columns.astype(str).tolist()


def normalize_ref(counts: np.ndarray) -> np.ndarray:
    return log1p_cpm(counts.astype(np.float64, copy=False))


def predict_profiles_chunked(
    query_counts: sp.csr_matrix,
    ref_profiles: np.ndarray,
    ref_labels: list[str],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = query_counts.shape[0]
    labels = np.asarray(ref_labels, dtype=object)
    pred = np.empty(n, dtype=object)
    score = np.full(n, np.nan, dtype=np.float64)
    second = np.empty(n, dtype=object)
    second_score = np.full(n, np.nan, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        q = log1p_cpm(query_counts[start:end].toarray().astype(np.float64, copy=False))
        sim = cosine_similarity(q, ref_profiles)
        sim_min = sim.min(axis=1, keepdims=True)
        shifted = sim - sim_min
        denom = shifted.sum(axis=1, keepdims=True)
        frac = np.divide(shifted, denom, out=np.zeros_like(shifted), where=denom > 0)
        order = np.argsort(frac, axis=1)[:, ::-1]
        top = order[:, 0]
        sec = order[:, 1] if frac.shape[1] > 1 else order[:, 0]
        rows = np.arange(end - start)
        pred[start:end] = labels[top]
        score[start:end] = frac[rows, top]
        second[start:end] = labels[sec]
        second_score[start:end] = frac[rows, sec]
    return pred, score, second, second_score


def numbered_umap(df: pd.DataFrame, label_col: str, out_stem: Path, title: str, x_col: str = "umap1", y_col: str = "umap2", axis_label: str = "UMAP") -> None:
    counts = df[label_col].astype(str).value_counts()
    labels = counts.index.tolist()
    legend = pd.DataFrame({"number": np.arange(1, len(labels) + 1), label_col: labels, "n_cells": counts.to_numpy(dtype=int)})
    legend.to_csv(out_stem.with_name(out_stem.name + "_legend.csv"), index=False)
    number_map = dict(zip(labels, legend["number"]))
    df = df.copy()
    df["_label"] = df[label_col].astype(str)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    for i, lab in enumerate(labels):
        sub = df[df["_label"] == lab]
        ax.scatter(sub[x_col], sub[y_col], s=1, alpha=0.75, linewidths=0, color=cmap(i % 20), label=f"{number_map[lab]}: {lab}")
    centers = df.groupby("_label")[[x_col, y_col]].median()
    for lab, row in centers.iterrows():
        ax.text(row[x_col], row[y_col], str(number_map[lab]), ha="center", va="center", fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.0))
    ax.set_title(title)
    ax.set_xlabel(f"{axis_label}1")
    ax.set_ylabel(f"{axis_label}2")
    ax.legend(markerscale=8, fontsize=5.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    summary = {"label_column": label_col, "n_labels": int(len(labels)), "x_column": x_col, "y_column": y_col, "legend_csv": str(out_stem.with_name(out_stem.name + "_legend.csv"))}
    out_stem.with_name(out_stem.name + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def by_cluster_majority(cell: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if "cluster" not in cell.columns:
        return pd.DataFrame(columns=["cluster", label_col, "n", "cluster_n", "fraction"])
    counts = cell.groupby(["cluster", label_col], dropna=False).size().reset_index(name="n")
    cluster_n = cell["cluster"].value_counts().rename_axis("cluster").reset_index(name="cluster_n")
    counts = counts.merge(cluster_n, on="cluster", how="left")
    counts["fraction"] = counts["n"] / counts["cluster_n"]
    counts = counts.sort_values(["cluster", "n"], ascending=[True, False])
    return counts.drop_duplicates("cluster").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    wnn = Path(args.wnn_dir).resolve()
    src = Path(args.source_dir).resolve()
    subtype_dir = Path(args.subtype_reference_dir).resolve()
    coords_path = wnn / "joint_umap_coordinates.csv"
    meta_path = src / "prepared_meta.parquet"
    counts_path = src / "prepared_counts_gene_expr.npz"
    genes_path = src / "prepared_genes.txt"
    for p in [coords_path, meta_path, counts_path, genes_path, subtype_dir / "reference_subtypes_final_count_sums.csv", subtype_dir / "reference_subtypes_final_metadata.csv"]:
        if not p.exists():
            raise FileNotFoundError(p)
    if not args.coarse_reference_csv:
        raise ValueError("--coarse-reference-csv is required for coarse label transfer")

    coords = pd.read_csv(coords_path, dtype={"sample_id": str, "cell_id": str})
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["cell_id"] = meta["cell_id"].astype(str)
    key_meta = meta["sample_id"] + "__" + meta["cell_id"]
    key_coords = coords["sample_id"].astype(str) + "__" + coords["cell_id"].astype(str)
    if not np.array_equal(key_meta.to_numpy(), key_coords.to_numpy()):
        raise ValueError("WNN UMAP coordinate row order does not match retained source metadata")

    genes = read_genes(genes_path)
    q_gene_to_idx = {g: i for i, g in enumerate(genes)}
    counts = sp.load_npz(counts_path).tocsr()
    if counts.shape[0] != len(meta):
        raise ValueError(f"count rows={counts.shape[0]} metadata rows={len(meta)}")

    coarse_ref, coarse_labels, coarse_genes = load_reference(Path(args.coarse_reference_csv))
    common_coarse = [g for g in coarse_genes if g in q_gene_to_idx]
    if len(common_coarse) < 50:
        raise ValueError(f"Too few common coarse genes: {len(common_coarse)}")
    q_idx = np.array([q_gene_to_idx[g] for g in common_coarse], dtype=int)
    r_idx = np.array([coarse_genes.index(g) for g in common_coarse], dtype=int)
    coarse_profiles = normalize_ref(coarse_ref.to_numpy(dtype=np.float64)[:, r_idx])
    coarse_pred, coarse_score, coarse_second, coarse_second_score = predict_profiles_chunked(
        counts[:, q_idx].tocsr(),
        coarse_profiles,
        coarse_labels,
        max(1, int(args.batch_size)),
    )

    cell = coords[["sample_id", "cell_id", "umap1", "umap2"]].copy()
    if "cluster" in coords.columns:
        cell["cluster"] = coords["cluster"].astype(str)
    cell["key"] = key_coords.to_numpy()
    cell["confirmed_neutrophil"] = meta["confirmed_neutrophil"].astype(bool).to_numpy()
    cell["coarse_predicted_id"] = coarse_pred
    cell["coarse_prediction_score"] = coarse_score
    cell["second_coarse_predicted_id"] = coarse_second
    cell["second_coarse_prediction_score"] = coarse_second_score
    cell["coarse_prediction_margin"] = coarse_score - coarse_second_score

    subtype_ref = pd.read_csv(subtype_dir / "reference_subtypes_final_count_sums.csv", index_col=0)
    subtype_meta = pd.read_csv(subtype_dir / "reference_subtypes_final_metadata.csv")
    subtype_labels = subtype_ref.index.astype(str).tolist()
    subtype_genes = subtype_ref.columns.astype(str).tolist()
    common_sub = [g for g in subtype_genes if g in q_gene_to_idx]
    if len(common_sub) < 50:
        raise ValueError(f"Too few common subtype genes: {len(common_sub)}")
    sq_idx = np.array([q_gene_to_idx[g] for g in common_sub], dtype=int)
    sr_idx = np.array([subtype_genes.index(g) for g in common_sub], dtype=int)
    subtype_counts = subtype_ref.to_numpy(dtype=np.float64)[:, sr_idx]
    subtype_parent = subtype_meta.set_index("subtype_label")["majority_coarse_label"].astype(str).to_dict()

    n = len(cell)
    sub_pred = np.empty(n, dtype=object)
    sub_score = np.full(n, np.nan, dtype=np.float64)
    sub_second = np.empty(n, dtype=object)
    sub_second_score = np.full(n, np.nan, dtype=np.float64)
    sub_assignment = np.empty(n, dtype=object)
    sub_matches = np.ones(n, dtype=bool)
    no_ref: dict[str, int] = {}
    subtype_labels_arr = np.asarray(subtype_labels)
    for parent in pd.unique(cell["coarse_predicted_id"].astype(str)):
        mask = cell["coarse_predicted_id"].astype(str).to_numpy() == parent
        idx_cells = np.flatnonzero(mask)
        subtype_keep = [i for i, lab in enumerate(subtype_labels) if subtype_parent.get(lab) == parent]
        if not subtype_keep:
            sub_pred[idx_cells] = parent
            sub_second[idx_cells] = parent
            sub_assignment[idx_cells] = "no_subtype_reference"
            no_ref[parent] = int(mask.sum())
            continue
        pred, score, second, second_score = predict_profiles_chunked(
            counts[idx_cells][:, sq_idx].tocsr(),
            normalize_ref(subtype_counts[subtype_keep]),
            subtype_labels_arr[subtype_keep].astype(str).tolist(),
            max(1, int(args.batch_size)),
        )
        sub_pred[idx_cells] = pred
        sub_score[idx_cells] = score
        sub_second[idx_cells] = second
        sub_second_score[idx_cells] = second_score
        sub_assignment[idx_cells] = "within_coarse_subtype_reference"
        sub_matches[idx_cells] = np.array([subtype_parent.get(str(x)) == parent for x in pred], dtype=bool)

    cell["subtypes_final_predicted_id"] = sub_pred
    cell["subtypes_final_prediction_score"] = sub_score
    cell["second_subtypes_final_predicted_id"] = sub_second
    cell["second_subtypes_final_prediction_score"] = sub_second_score
    cell["subtypes_final_prediction_margin"] = sub_score - sub_second_score
    cell["subtypes_final_assignment_type"] = sub_assignment
    cell["subtypes_final_matches_coarse_parent"] = sub_matches

    cell["coarse_predicted_id_neutrophil_override"] = np.where(cell["confirmed_neutrophil"], "neutrophil", cell["coarse_predicted_id"].astype(str))
    cell["subtypes_final_predicted_id_neutrophil_override"] = np.where(cell["confirmed_neutrophil"], "neutrophil", cell["subtypes_final_predicted_id"].astype(str))
    cell["neutrophil_override_action"] = np.where(cell["confirmed_neutrophil"], "cluster34_morphology_neutrophil_override", "original_label_transfer")

    cell.to_csv(wnn / "label_transfer_neutrophil_override_per_cell.csv", index=False)
    cell.to_csv(wnn / "label_transfer_subtypes_final_per_cell_predicted_id.csv", index=False)
    cell["coarse_predicted_id"].value_counts().rename_axis("coarse_predicted_id").reset_index(name="n_cells").to_csv(wnn / "label_transfer_coarse_predicted_id_counts.csv", index=False)
    cell[["coarse_predicted_id", "subtypes_final_predicted_id", "subtypes_final_assignment_type"]].value_counts().rename("n_cells").reset_index().to_csv(wnn / "label_transfer_subtypes_final_per_cell_predicted_id_counts.csv", index=False)
    cell[["coarse_predicted_id_neutrophil_override"]].value_counts().rename("n_cells").reset_index().to_csv(wnn / "label_transfer_coarse_neutrophil_override_counts.csv", index=False)
    cell[["subtypes_final_predicted_id_neutrophil_override"]].value_counts().rename("n_cells").reset_index().to_csv(wnn / "label_transfer_neutrophil_override_counts.csv", index=False)
    cell[["sample_id", "cell_id", "coarse_predicted_id_neutrophil_override"]].rename(columns={"coarse_predicted_id_neutrophil_override": "group"}).to_csv(wnn / "all_samples_coarse_neutrophil_override_cellid_group.csv", index=False)
    cell[["sample_id", "cell_id", "subtypes_final_predicted_id_neutrophil_override"]].rename(columns={"subtypes_final_predicted_id_neutrophil_override": "group"}).to_csv(wnn / "all_samples_subtypes_final_neutrophil_override_cellid_group.csv", index=False)

    by_cluster_majority(cell, "coarse_predicted_id").to_csv(wnn / "label_transfer_coarse_cluster_predictions.csv", index=False)
    by_cluster_majority(cell, "subtypes_final_predicted_id").to_csv(wnn / "label_transfer_subtypes_final_cluster_predictions.csv", index=False)
    if "cluster" in cell.columns:
        frac = (
            cell.groupby(["cluster", "coarse_predicted_id", "subtypes_final_predicted_id"], dropna=False)
            .size()
            .reset_index(name="n_cells")
        )
        cluster_n = cell["cluster"].value_counts().rename_axis("cluster").reset_index(name="cluster_n")
        frac = frac.merge(cluster_n, on="cluster", how="left")
        frac["fraction"] = frac["n_cells"] / frac["cluster_n"]
    else:
        frac = pd.DataFrame(columns=["cluster", "coarse_predicted_id", "subtypes_final_predicted_id", "n_cells", "cluster_n", "fraction"])
    frac.to_csv(wnn / "label_transfer_subtypes_final_cluster_label_fractions.csv", index=False)

    if no_ref:
        pd.DataFrame({"coarse_predicted_id": list(no_ref.keys()), "n_cells": list(no_ref.values())}).to_csv(wnn / "label_transfer_subtypes_final_no_reference_coarse_counts.csv", index=False)
    else:
        pd.DataFrame(columns=["coarse_predicted_id", "n_cells"]).to_csv(wnn / "label_transfer_subtypes_final_no_reference_coarse_counts.csv", index=False)

    numbered_umap(
        cell,
        "coarse_predicted_id_neutrophil_override",
        wnn / "joint_umap_coarse_predicted_id_neutrophil_override_numbered",
        "WNN UMAP per-cell coarse labels with cluster34 neutrophil override",
    )
    numbered_umap(
        cell,
        "subtypes_final_predicted_id_neutrophil_override",
        wnn / "joint_umap_subtypes_final_predicted_id_neutrophil_override_numbered",
        "WNN UMAP per-cell fine subtype labels with cluster34 neutrophil override",
    )
    scored = cell["subtypes_final_assignment_type"] == "within_coarse_subtype_reference"
    no_ref_mask = cell["subtypes_final_assignment_type"] == "no_subtype_reference"
    validation = {
        "prediction_rows": int(len(cell)),
        "expected_rows": int(len(meta)),
        "label_transfer_mode": "per_cell",
        "coarse_reference_rds": str(Path(args.coarse_reference_rds).resolve()),
        "coarse_reference_csv": str(Path(args.coarse_reference_csv).resolve()),
        "subtype_reference_dir": str(subtype_dir),
        "coarse_reference_labels": int(len(coarse_labels)),
        "subtype_reference_labels": int(len(subtype_labels)),
        "reference_majority_coarse_groups": sorted(set(subtype_parent.values())),
        "n_no_reference_cells": int(no_ref_mask.sum()),
        "n_scored_cells": int(scored.sum()),
        "all_scored_within_coarse_parent": bool(cell.loc[scored, "subtypes_final_matches_coarse_parent"].astype(bool).all()),
        "all_no_reference_kept_as_coarse": bool((cell.loc[no_ref_mask, "coarse_predicted_id"].astype(str) == cell.loc[no_ref_mask, "subtypes_final_predicted_id"].astype(str)).all()),
        "confirmed_neutrophil_count": int(cell["confirmed_neutrophil"].sum()),
        "coarse_neutrophil_override_count": int((cell["coarse_predicted_id_neutrophil_override"] == "neutrophil").sum()),
        "fine_neutrophil_override_count": int((cell["subtypes_final_predicted_id_neutrophil_override"] == "neutrophil").sum()),
        "coarse_predicted_label_count": int(cell["coarse_predicted_id"].astype(str).nunique()),
        "subtypes_final_predicted_label_count": int(cell["subtypes_final_predicted_id"].astype(str).nunique()),
        "subtypes_final_neutrophil_override_label_count": int(cell["subtypes_final_predicted_id_neutrophil_override"].astype(str).nunique()),
    }
    validation["all_checks_passed"] = bool(
        validation["prediction_rows"] == validation["expected_rows"]
        and validation["all_scored_within_coarse_parent"]
        and validation["all_no_reference_kept_as_coarse"]
        and validation["confirmed_neutrophil_count"] == validation["coarse_neutrophil_override_count"] == validation["fine_neutrophil_override_count"]
    )
    (wnn / "label_transfer_subtypes_final_validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if not validation["all_checks_passed"]:
        raise RuntimeError("Label-transfer validation failed")


if __name__ == "__main__":
    main()
