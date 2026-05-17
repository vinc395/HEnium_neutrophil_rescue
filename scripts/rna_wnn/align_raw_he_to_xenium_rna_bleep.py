#!/usr/bin/env python3
"""BLEEP-style alignment of raw H&E embeddings to Xenium-normalized RNA PCA.

This script is intentionally small and explicit for the tutorial branch. It
does not call the broader HEnium pipeline RNA embedder; the RNA-side input is a
user-provided Xenium-normalized embedding, usually PCA from raw Xenium counts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[1]
IMAGE_SCRIPT_DIR = PROJECT / "scripts" / "image_embedding"
sys.path.insert(0, str(IMAGE_SCRIPT_DIR))

from henium_custom_pipeline import HeniumAlignModel, SoftContrastiveLoss, _iterate_minibatches  # noqa: E402


class BidirectionalInfoNCELoss(nn.Module):
    """Standard symmetric InfoNCE with same-row image/RNA positives."""

    def __init__(self, learnable_logit_temperature: bool, init_tau: float = 0.07):
        super().__init__()
        self.learnable_logit_temperature = bool(learnable_logit_temperature)
        if self.learnable_logit_temperature:
            self.log_tau = nn.Parameter(torch.tensor(float(np.log(init_tau)), dtype=torch.float32))
        else:
            self.register_buffer("fixed_tau", torch.tensor(float(init_tau), dtype=torch.float32))

    def _tau(self) -> torch.Tensor:
        if self.learnable_logit_temperature:
            return self.log_tau.exp().clamp(1e-3, 2.0)
        return self.fixed_tau

    def forward(self, zi: torch.Tensor, zg: torch.Tensor, ai: torch.Tensor, ag: torch.Tensor) -> torch.Tensor:
        del zi, zg
        logits = F.normalize(ai, p=2, dim=1) @ F.normalize(ag, p=2, dim=1).T
        logits = logits / self._tau()
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i2g = F.cross_entropy(logits, labels)
        loss_g2i = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_i2g + loss_g2i)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Align raw H-Optimus image embeddings to Xenium-normalized RNA PCA")
    ap.add_argument("--source-dir", required=True, help="QC-passing source directory with prepared_meta.parquet")
    ap.add_argument("--image-embedding", default="he_embeddings.npy", help="Image embedding .npy path or source-dir filename")
    ap.add_argument("--rna-embedding", required=True, help="RNA embedding .npy path or source-dir filename")
    ap.add_argument("--out-image", default="aligned_image.npy")
    ap.add_argument("--out-rna", default="aligned_rna.npy")
    ap.add_argument("--out-fused", default="aligned_fused.npy")
    ap.add_argument("--summary-json", default="alignment_summary.json")
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-image", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--target-temperature", type=float, default=0.1)
    ap.add_argument("--init-tau", type=float, default=0.07)
    ap.add_argument("--learnable-logit-temperature", action="store_true", default=True)
    ap.add_argument(
        "--loss-mode",
        choices=["bleepinput", "infonce"],
        default="bleepinput",
        help="Contrastive target mode. bleepinput uses soft BLEEP-style targets; infonce uses hard same-cell positives.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def resolve_source_path(source: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source / path
    return path.resolve()


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    source = Path(args.source_dir).expanduser().resolve()
    meta_path = source / "prepared_meta.parquet"
    image_path = resolve_source_path(source, args.image_embedding)
    rna_path = resolve_source_path(source, args.rna_embedding)
    out_image = resolve_source_path(source, args.out_image)
    out_rna = resolve_source_path(source, args.out_rna)
    out_fused = resolve_source_path(source, args.out_fused)
    summary_path = resolve_source_path(source, args.summary_json)
    for p in [meta_path, image_path, rna_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        device = torch.device(args.device)
        torch.cuda.manual_seed_all(int(args.seed))
    else:
        device = torch.device("cpu")
    log(f"[bleep-align] device={device}")

    meta = pd.read_parquet(meta_path, columns=["sample_id", "cell_id"]).reset_index(drop=True)
    zi_np = np.load(image_path).astype(np.float32, copy=False)
    zg_np = np.load(rna_path).astype(np.float32, copy=False)
    if zi_np.ndim != 2 or zg_np.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got image={zi_np.shape} rna={zg_np.shape}")
    if zi_np.shape[0] != zg_np.shape[0] or zi_np.shape[0] != meta.shape[0]:
        raise ValueError(f"Row mismatch: meta={meta.shape[0]} image={zi_np.shape} rna={zg_np.shape}")
    if not np.isfinite(zi_np).all() or not np.isfinite(zg_np).all():
        raise ValueError("Embedding arrays contain non-finite values")

    zi = torch.from_numpy(zi_np)
    zg = torch.from_numpy(zg_np)
    n = int(zi.shape[0])
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = max(1, int(0.8 * n))
    tr_idx = idx[:split]
    va_idx = idx[split:] if split < n else idx[: min(n, int(args.batch_size))]

    model = HeniumAlignModel(
        d_i=int(zi.shape[1]),
        d_g=int(zg.shape[1]),
        d_a=int(args.latent_dim),
        dropout=float(args.dropout),
    ).to(device)
    if args.loss_mode == "infonce":
        criterion = BidirectionalInfoNCELoss(
            learnable_logit_temperature=bool(args.learnable_logit_temperature),
            init_tau=float(args.init_tau),
        ).to(device)
    else:
        criterion = SoftContrastiveLoss(
            lambda_image=float(args.lambda_image),
            target_type="bleepinput",
            alpha=float(args.alpha),
            target_temperature=float(args.target_temperature),
            learnable_logit_temperature=bool(args.learnable_logit_temperature),
            init_tau=float(args.init_tau),
        ).to(device)
    optim = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    best_val = float("inf")
    best_state = None
    stale = 0
    history: list[dict] = []
    batch_size = int(args.batch_size)

    log(f"[bleep-align] image={image_path} shape={zi_np.shape}")
    log(f"[bleep-align] rna={rna_path} shape={zg_np.shape}")
    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        criterion.train()
        tr_losses = []
        for bi in _iterate_minibatches(len(tr_idx), batch_size, shuffle=True):
            ridx = tr_idx[bi]
            zi_b = zi[ridx].to(device)
            zg_b = zg[ridx].to(device)
            ai_b, ag_b = model(zi_b, zg_b)
            loss = criterion(zi_b, zg_b, ai_b, ag_b)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        criterion.eval()
        va_losses = []
        with torch.no_grad():
            for bi in _iterate_minibatches(len(va_idx), batch_size, shuffle=False):
                ridx = va_idx[bi]
                zi_b = zi[ridx].to(device)
                zg_b = zg[ridx].to(device)
                ai_b, ag_b = model(zi_b, zg_b)
                va_losses.append(float(criterion(zi_b, zg_b, ai_b, ag_b).item()))

        tr_mean = float(np.mean(tr_losses))
        va_mean = float(np.mean(va_losses))
        history.append({"epoch": int(epoch), "train_loss": tr_mean, "val_loss": va_mean})
        log(f"[bleep-align] epoch={epoch:02d} train={tr_mean:.4f} val={va_mean:.4f}")
        if va_mean < best_val - 1e-6:
            best_val = va_mean
            stale = 0
            best_state = {"model": model.state_dict(), "criterion": criterion.state_dict()}
        else:
            stale += 1
            if stale >= int(args.early_stop_patience):
                log(f"[bleep-align] early_stop epoch={epoch}")
                break

    if best_state is None:
        raise RuntimeError("No best alignment state was captured")
    model.load_state_dict(best_state["model"])  # type: ignore[arg-type]
    criterion.load_state_dict(best_state["criterion"])  # type: ignore[arg-type]
    model.eval()

    ai_all = np.zeros((n, int(args.latent_dim)), dtype=np.float32)
    ag_all = np.zeros((n, int(args.latent_dim)), dtype=np.float32)
    infer_batch = max(2048, batch_size * 8)
    with torch.no_grad():
        for bi in _iterate_minibatches(n, infer_batch, shuffle=False):
            zi_b = zi[bi].to(device)
            zg_b = zg[bi].to(device)
            ai_b, ag_b = model(zi_b, zg_b)
            ai_all[bi] = ai_b.detach().cpu().numpy().astype(np.float32)
            ag_all[bi] = ag_b.detach().cpu().numpy().astype(np.float32)

    fused = (ai_all + ag_all) / 2.0
    fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8)
    np.save(out_image, ai_all)
    np.save(out_rna, ag_all)
    np.save(out_fused, fused.astype(np.float32, copy=False))

    summary = {
        "method": (
            "standard_bidirectional_infonce_alignment_raw_hoptimus_to_xenium_norm_rna_pca"
            if args.loss_mode == "infonce"
            else "bleep_style_alignment_raw_hoptimus_to_xenium_norm_rna_pca"
        ),
        "loss_mode": str(args.loss_mode),
        "source_dir": str(source),
        "image_embedding_path": str(image_path),
        "rna_embedding_path": str(rna_path),
        "output_aligned_image": str(out_image),
        "output_aligned_rna": str(out_rna),
        "output_aligned_fused": str(out_fused),
        "input_shape_image": [int(zi_np.shape[0]), int(zi_np.shape[1])],
        "input_shape_rna": [int(zg_np.shape[0]), int(zg_np.shape[1])],
        "aligned_shape": [int(fused.shape[0]), int(fused.shape[1])],
        "best_val_loss": float(best_val),
        "history": history,
        "hyperparameters": {
            "latent_dim": int(args.latent_dim),
            "dropout": float(args.dropout),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.max_epochs),
            "early_stop_patience": int(args.early_stop_patience),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "lambda_image": float(args.lambda_image),
            "alpha": float(args.alpha),
            "target_temperature": float(args.target_temperature),
            "init_tau": float(args.init_tau),
            "learnable_logit_temperature": bool(args.learnable_logit_temperature),
        },
        "runtime_seconds": float(time.time() - t0),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[bleep-align] wrote={out_image}, {out_rna}, {out_fused}")


if __name__ == "__main__":
    main()
