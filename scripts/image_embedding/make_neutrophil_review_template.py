#!/usr/bin/env python3
"""Create a manual review table for image clusters."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster-sizes", required=True, help="embedding_cluster_sizes_target30.csv or equivalent")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sizes = pd.read_csv(args.cluster_sizes)
    cluster_col = "cluster" if "cluster" in sizes.columns else sizes.columns[0]
    count_col = "n_cells" if "n_cells" in sizes.columns else sizes.columns[-1]
    out = pd.DataFrame(
        {
            "image_cluster": sizes[cluster_col].astype(str),
            "n_cells": sizes[count_col],
            "lobulated_neutrophil_morphology": "",
            "confirmed_neutrophil": "",
            "reviewer": "",
            "review_notes": "",
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
