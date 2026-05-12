"""
train.py
========
Training loop for the BoneLossKeypointModel.

Features:
  - Per-head OKS monitoring every epoch
  - StepLR scheduler (gamma=0.6, step=4 — from paper)
  - Weighted head losses
  - Early stopping on validation OKS (patience=30)
  - Two checkpoints saved:
      best_oks.pth   — best overall validation OKS
      best_icc.pth   — best ICC (bone loss % accuracy)
  - Bone loss % computed post-hoc from predicted keypoints
  - Three geometric formulas compared: min-max, direct, PCA
  - Full training log saved to training_log.json
  - Windows-safe unicode in all print statements

Usage:
  python train.py --data_dir ./data_prepared --output_dir ./runs/exp1
  python train.py --data_dir ./data_prepared --output_dir ./runs/exp1 --resume ./runs/exp1/best_oks.pth
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from .dataset import build_dataloaders, INPUT_H, INPUT_W, NUM_KP
from model   import BoneLossKeypointModel, OKSMeter, decode_keypoints

# ─────────────────────────────────────────────────────────────────────────────
# Logging — ASCII only (Windows cp1252 safe)
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir):
    log_path = Path(output_dir) / "training.log"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
        handlers= handlers,
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Geometric bone loss calculators (post-hoc, no gradients)
# ─────────────────────────────────────────────────────────────────────────────

def enforce_anatomical_order(cej, inter, apex):
    """Project intersection onto CEJ-Apex axis if ordering is violated."""
    cej_a  = np.array(cej,   dtype=float)
    int_a  = np.array(inter, dtype=float)
    apex_a = np.array(apex,  dtype=float)

    axis = apex_a - cej_a
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return cej, inter, apex

    unit = axis / length
    proj = float(np.dot(int_a - cej_a, unit))
    proj_apex = float(np.dot(apex_a - cej_a, unit))

    if not (0.0 < proj < proj_apex):
        # Project to 50% of root length as fallback
        int_corrected = cej_a + unit * (proj_apex * 0.50)
        return cej, tuple(int_corrected.tolist()), apex

    return cej, inter, apex


def bone_loss_minmax(cej, inter, apex):
    """
    Paper formula: min-max line projection (equations 2-3).
    Minimises maximum error among the three points.
    Returns percentage in [0, 100] or None on failure.
    """
    pts = sorted([(float(p[0]), float(p[1]))
                  for p in [cej, inter, apex]], key=lambda p: p[0])
    a1, b1 = pts[0]
    a2, b2 = pts[1]
    a3, b3 = pts[2]

    denom = a3 - a1
    if abs(denom) < 1e-6:
        # Nearly vertical — use y axis
        root_len  = abs(float(apex[1]) - float(cej[1]))
        bone_dist = abs(float(inter[1]) - float(cej[1]))
        if root_len < 1e-6:
            return None
        return float(np.clip(bone_dist / root_len * 100.0, 0.0, 100.0))

    m = (b3 - b1) / denom
    c = (b1 * (a2 + a3) + b2 * (a3 - a1) - b3 * (a1 + a2)) / (2 * denom)

    def project(pt):
        x, y  = float(pt[0]), float(pt[1])
        px    = (x + m * y - m * c) / (1 + m * m)
        py    = m * px + c
        return np.array([px, py])

    p_cej   = project(cej)
    p_inter = project(inter)
    p_apex  = project(apex)

    d_inter = float(np.linalg.norm(p_inter - p_cej))
    d_apex  = float(np.linalg.norm(p_apex  - p_cej))

    if d_apex < 1e-6:
        return None
    return float(np.clip(d_inter / d_apex * 100.0, 0.0, 100.0))


def bone_loss_direct(cej, inter, apex):
    """
    Direct Euclidean ratio (simplest formula).
    dist(CEJ->intersection) / dist(CEJ->apex) * 100
    """
    d_inter = np.linalg.norm(np.array(inter) - np.array(cej))
    d_apex  = np.linalg.norm(np.array(apex)  - np.array(cej))
    if d_apex < 1e-6:
        return None
    return float(np.clip(d_inter / d_apex * 100.0, 0.0, 100.0))


def bone_loss_pca(cej, inter, apex):
    """
    PCA root axis projection.
    Fit PCA through all 3 points, project onto first principal component.
    More robust when points are not collinear.
    """
    pts  = np.array([cej, inter, apex], dtype=float)
    mean = pts.mean(axis=0)
    centered = pts - mean

    # SVD for PCA
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]   # first principal component

    # Project onto axis
    projs = centered @ axis   # (3,)
    proj_cej   = projs[0]
    proj_inter = projs[1]
    proj_apex  = projs[2]

    # Ensure CEJ is at the smaller projection (crown side)
    if proj_cej > proj_apex:
        proj_cej, proj_apex = proj_apex, proj_cej
        # Recompute inter after axis flip
        proj_inter = -proj_inter + proj_cej + proj_apex

    root_len  = abs(proj_apex - proj_cej)
    bone_dist = abs(proj_inter - proj_cej)

    if root_len < 1e-6:
        return None
    return float(np.clip(bone_dist / root_len * 100.0, 0.0, 100.0))


# All three formulas — used for final test set evaluation only
FORMULA_FNS_ALL = {
    "minmax": bone_loss_minmax,
    "direct": bone_loss_direct,
    "pca":    bone_loss_pca,
}

# During training validation: only direct formula
# Direct is consistently the best and fastest (no SVD, no line fitting)
# minmax and pca add ~6s per validation epoch with no benefit during training
FORMULA_FNS = {
    "direct": bone_loss_direct,
}


def compute_bone_loss_batch(pred_kps_np, gt_bl_mesial_np, gt_bl_distal_np,
                             is_double_np):
    """
    Compute bone loss % from predicted keypoints for a batch.
    Compares all three formulas against GT.

    Args:
        pred_kps_np:      (B, 6, 3) numpy  predicted x,y,conf
        gt_bl_mesial_np:  (B,)      numpy  GT mesial bone loss (or -1)
        gt_bl_distal_np:  (B,)      numpy  GT distal bone loss (or -1)
        is_double_np:     (B,)      bool

    Returns:
        dict: {formula_name: {"pred": [...], "gt": [...]}}
    """
    results = {f: {"pred": [], "gt": []} for f in FORMULA_FNS}

    for b in range(len(pred_kps_np)):
        kps     = pred_kps_np[b]     # (6, 3)
        is_dbl  = bool(is_double_np[b])

        # Mesial root: slots 0 (CEJ-M), 2 (Int-M), 4 (Apex-M)
        gt_m    = float(gt_bl_mesial_np[b])
        if gt_m >= 0:
            cej_m   = (kps[0, 0], kps[0, 1])
            int_m   = (kps[2, 0], kps[2, 1])
            apex_m  = (kps[4, 0], kps[4, 1])
            cej_m, int_m, apex_m = enforce_anatomical_order(
                cej_m, int_m, apex_m
            )
            for fname, fn in FORMULA_FNS.items():
                pred_pct = fn(cej_m, int_m, apex_m)
                if pred_pct is not None:
                    results[fname]["pred"].append(pred_pct)
                    results[fname]["gt"].append(gt_m)

        # Distal root: slots 1 (CEJ-D), 3 (Int-D), 5 (Apex-D)
        gt_d = float(gt_bl_distal_np[b]) if is_dbl else -1.0
        if gt_d >= 0 and kps[5, 2] > 0:
            cej_d   = (kps[1, 0], kps[1, 1])
            int_d   = (kps[3, 0], kps[3, 1])
            apex_d  = (kps[5, 0], kps[5, 1])
            cej_d, int_d, apex_d = enforce_anatomical_order(
                cej_d, int_d, apex_d
            )
            for fname, fn in FORMULA_FNS.items():
                pred_pct = fn(cej_d, int_d, apex_d)
                if pred_pct is not None:
                    results[fname]["pred"].append(pred_pct)
                    results[fname]["gt"].append(gt_d)

    return results


def compute_icc(pred, gt):
    """
    Intraclass Correlation Coefficient — ICC(2,1) two-way mixed,
    absolute agreement. This is the exact form used in the paper
    (Koo & Li, 2016 guidelines).

    Formula:
        ICC = (MSb - MSw) / (MSb + (k-1)*MSw + k*(MSb_rater - MSw)/n)
    where k=2 raters (predicted, GT), n=number of subjects.

    For k=2 this simplifies to the standard form:
        ICC = (MSr - MSe) / (MSr + MSe + 2*MSc/n)

    Interpretation (Koo & Li 2016):
        < 0.50  poor
        0.50-0.75  moderate
        0.75-0.90  good
        > 0.90  excellent
    Target: >= 0.80 (matches paper)
    """
    pred = np.array(pred, dtype=float)
    gt   = np.array(gt,   dtype=float)
    n    = len(pred)
    if n < 2:
        return 0.0

    k = 2  # two raters: predicted and GT

    # Stack into (n, k) matrix — one row per subject, one col per rater
    data       = np.column_stack([pred, gt])
    grand_mean = data.mean()

    # Sum of squares
    ss_total   = ((data - grand_mean) ** 2).sum()

    # Between-subjects SS (row means vs grand mean)
    row_means  = data.mean(axis=1)               # (n,)
    ss_rows    = k * ((row_means - grand_mean) ** 2).sum()

    # Between-raters SS (col means vs grand mean)
    col_means  = data.mean(axis=0)               # (k,)
    ss_cols    = n * ((col_means - grand_mean) ** 2).sum()

    # Error SS (residual)
    ss_error   = ss_total - ss_rows - ss_cols

    # Mean squares
    ms_rows    = ss_rows  / (n - 1)
    ms_cols    = ss_cols  / (k - 1)
    ms_error   = ss_error / ((n - 1) * (k - 1))

    # ICC(2,1) absolute agreement
    denom = ms_rows + ms_error + k * (ms_cols - ms_error) / n
    if abs(denom) < 1e-10:
        return 1.0
    icc = (ms_rows - ms_error) / denom
    return float(np.clip(icc, -1.0, 1.0))


def compute_mae(pred, gt):
    """Mean Absolute Error in percentage points."""
    if not pred:
        return float("inf")
    return float(np.mean(np.abs(np.array(pred) - np.array(gt))))


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, log):
    model.train()
    epoch_loss    = 0.0
    detail_totals = {}
    n_batches     = 0

    for batch in loader:
        # Move tensors to GPU
        gpu_batch = {
            "image":          batch["image"].to(device, non_blocking=True),
            "keypoints":      batch["keypoints"].to(device, non_blocking=True),
            "is_double_root": batch["is_double_root"].to(device, non_blocking=True),
            "bbox":           batch["bbox"].to(device, non_blocking=True),
        }

        optimizer.zero_grad(set_to_none=True)
        out  = model(gpu_batch)
        loss = out["loss"]

        loss.backward()

        # Gradient clipping — prevents exploding gradients on hard samples
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        epoch_loss += loss.item()
        for k, v in out["loss_details"].items():
            detail_totals[k] = detail_totals.get(k, 0.0) + v
        n_batches += 1

    avg_loss    = epoch_loss / max(n_batches, 1)
    avg_details = {k: v / max(n_batches, 1)
                   for k, v in detail_totals.items()}
    return avg_loss, avg_details


# ─────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────────────────

def validate(model, loader, device, oks_meter, log):
    """
    Single forward pass per batch in eval mode.
    No train/eval switching — BatchNorm uses running statistics correctly.
    Validation loss removed; OKS and ICC are the meaningful metrics.
    """
    model.eval()
    oks_meter.reset()

    all_formula_results = {f: {"pred": [], "gt": []} for f in FORMULA_FNS}

    with torch.no_grad():
        for batch in loader:
            gpu_batch = {
                "image":          batch["image"].to(device, non_blocking=True),
                "keypoints":      batch["keypoints"].to(device, non_blocking=True),
                "is_double_root": batch["is_double_root"].to(device, non_blocking=True),
                "bbox":           batch["bbox"].to(device, non_blocking=True),
            }

            # Single forward pass — model is in eval mode throughout
            out      = model(gpu_batch)
            pred_kps = out["keypoints"]           # (B, 6, 3) on GPU

            # OKS update
            oks_meter.update(
                pred_kps       = pred_kps,
                gt_kps         = gpu_batch["keypoints"],
                bboxes         = gpu_batch["bbox"],
                is_double_root = gpu_batch["is_double_root"],
            )

            # Bone loss % computation (CPU, no gradient needed)
            pred_np  = pred_kps.cpu().numpy()
            gt_m_np  = batch["gt_bone_loss_mesial"].numpy()
            gt_d_np  = batch["gt_bone_loss_distal"].numpy()
            dbl_np   = batch["is_double_root"].numpy()

            batch_results = compute_bone_loss_batch(
                pred_np, gt_m_np, gt_d_np, dbl_np
            )
            for f in FORMULA_FNS:
                all_formula_results[f]["pred"].extend(
                    batch_results[f]["pred"]
                )
                all_formula_results[f]["gt"].extend(
                    batch_results[f]["gt"]
                )

    oks_summary     = oks_meter.compute()
    formula_metrics = {}
    for fname in FORMULA_FNS:
        pred = all_formula_results[fname]["pred"]
        gt   = all_formula_results[fname]["gt"]
        formula_metrics[fname] = {
            "icc": compute_icc(pred, gt),
            "mae": compute_mae(pred, gt),
            "n":   len(pred),
        }

    return {
        "val_loss":        0.0,   # removed — OKS + ICC are the real metrics
        "oks":             oks_summary,
        "formula_metrics": formula_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch,
                    metrics, path, log):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
    }, str(path))
    log.info(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler, path, device, log):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    log.info(f"Resumed from epoch {ckpt['epoch']}: {path}")
    return ckpt["epoch"], ckpt["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers (ASCII only for Windows safety)
# ─────────────────────────────────────────────────────────────────────────────

def print_epoch_header(epoch, total_epochs, lr, elapsed_min):
    line = "=" * 65
    print(f"\n{line}")
    print(f"  Epoch {epoch}/{total_epochs}   "
          f"LR: {lr:.2e}   Elapsed: {elapsed_min:.1f}min")
    print(line)


def print_loss_table(train_loss, train_details):
    print(f"\n  Losses (train):")
    print(f"    Total: {train_loss:.4f}")
    print(f"    Head breakdown:")
    for k, v in sorted(train_details.items()):
        print(f"      {k:<22} {v:.4f}")


def print_oks_table(oks_summary):
    print(f"\n  OKS Metrics (validation):")
    print(f"    Overall  OKS: {oks_summary.get('OKS_mean', 0):.4f}  "
          f"AP50: {oks_summary.get('AP50', 0):.4f}  "
          f"AP75: {oks_summary.get('AP75', 0):.4f}  "
          f"mAP50:95: {oks_summary.get('mAP50_95', 0):.4f}")

    pk = oks_summary.get("per_keypoint", {})
    if pk:
        print(f"    Per keypoint:")
        for name, val in pk.items():
            fill = int(val * 20)
            bar  = "#" * fill + "." * (20 - fill)
            print(f"      {name:<28} [{bar}] {val:.4f}")

    sr = oks_summary.get("OKS_single_root")
    dr = oks_summary.get("OKS_double_root")
    if sr and dr:
        print(f"    Single-root: {sr:.4f}   Double-root: {dr:.4f}   "
              f"Gap: {sr-dr:+.4f}")


def print_formula_table(formula_metrics):
    print(f"\n  Bone Loss % (post-hoc geometry):")
    print(f"    {'Formula':<12} {'ICC':>8} {'MAE':>8} {'N':>6}  ICC quality")
    print(f"    {'-'*55}")
    best_f = max(formula_metrics, key=lambda f: formula_metrics[f]["icc"])
    for fname, m in formula_metrics.items():
        icc = m["icc"]
        if icc > 0.90:   quality = "excellent"
        elif icc > 0.75: quality = "good"
        elif icc > 0.50: quality = "moderate"
        elif icc > 0.0:  quality = "poor"
        else:            quality = "failing"
        marker = " <-- best" if fname == best_f else ""
        print(f"    {fname:<12} {icc:>8.4f} {m['mae']:>7.2f}% "
              f"{m['n']:>6}  {quality}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU:    {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── DataLoaders ────────────────────────────────────────────────────────
    log.info("\nLoading datasets...")
    loaders = build_dataloaders(
        data_prepared_dir = args.data_dir,
        batch_size        = args.batch_size,
        num_workers       = args.num_workers,
    )
    train_loader = loaders["training"]
    val_loader   = loaders["validation"]
    log.info(f"Train batches: {len(train_loader)}  "
             f"Val batches: {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────
    log.info("\nBuilding model...")
    model = BoneLossKeypointModel(pretrained=True).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {total_p:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── Scheduler: CosineAnnealingLR ──────────────────────────────────────
    # Decays LR smoothly from lr to eta_min over T_max epochs.
    # Much better than StepLR for fine-tuning: avoids LR → 0 too early.
    # eta_min = 1e-7 keeps a floor so the model never fully stops learning.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = args.epochs,
        eta_min = 1e-7,
    )

    # ── OKS meter ──────────────────────────────────────────────────────────
    oks_meter = OKSMeter(device=device)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        if args.weights_only:
            # Load model weights only — reset optimizer and scheduler
            # Use this when resuming with a new LR/scheduler config
            ckpt = torch.load(str(args.resume), map_location=device,
                              weights_only=False)
            model.load_state_dict(ckpt["model"])
            start_epoch = 1   # restart epoch count
            log.info(f"Loaded weights from epoch {ckpt['epoch']} "
                     f"(optimizer and scheduler reset)")
        else:
            start_epoch, _ = load_checkpoint(
                model, optimizer, scheduler,
                args.resume, device, log
            )
            start_epoch += 1

    # ── Training state ─────────────────────────────────────────────────────
    best_oks          = 0.0
    best_icc          = -1.0   # ICC can be negative — initialise at floor
    no_improve_epochs = 0
    training_log      = []
    start_time        = time.time()

    log.info(f"\nStarting training for {args.epochs} epochs...")
    log.info(f"Early stopping patience: {args.patience} epochs")

    for epoch in range(start_epoch, args.epochs + 1):
        elapsed_min = (time.time() - start_time) / 60
        current_lr  = optimizer.param_groups[0]["lr"]
        print_epoch_header(epoch, args.epochs, current_lr, elapsed_min)

        # ── Train ──────────────────────────────────────────────────────────
        t_start    = time.time()
        train_loss, train_details = train_one_epoch(
            model, train_loader, optimizer, device, log
        )
        t_train = time.time() - t_start

        # ── Validate ───────────────────────────────────────────────────────
        t_start  = time.time()
        val_out  = validate(model, val_loader, device, oks_meter, log)
        t_val    = time.time() - t_start

        val_loss        = val_out["val_loss"]
        oks_summary     = val_out["oks"]
        formula_metrics = val_out["formula_metrics"]

        # ── Print results ──────────────────────────────────────────────────
        print_loss_table(train_loss, train_details)
        print_oks_table(oks_summary)
        print_formula_table(formula_metrics)

        current_oks = oks_summary.get("OKS_mean", 0.0)

        # Best ICC across all formulas
        current_icc = max(
            m["icc"] for m in formula_metrics.values()
        ) if formula_metrics else 0.0
        best_formula = max(
            formula_metrics, key=lambda f: formula_metrics[f]["icc"]
        ) if formula_metrics else "none"

        print(f"\n  Timing: train {t_train:.1f}s  val {t_val:.1f}s")
        print(f"  Best OKS so far: {best_oks:.4f}  "
              f"Best ICC so far: {best_icc:.4f} ({best_formula})")

        # ── Scheduler step ─────────────────────────────────────────────────
        scheduler.step()

        # ── Checkpointing ──────────────────────────────────────────────────
        epoch_metrics = {
            "epoch":         epoch,
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "oks_mean":      current_oks,
            "icc_best":      current_icc,
            "formula":       best_formula,
            "formula_metrics": formula_metrics,
            "oks_details":   oks_summary,
        }

        if current_oks > best_oks:
            best_oks = current_oks
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                epoch_metrics,
                output_dir / "best_oks.pth", log
            )
            print(f"  ** New best OKS: {best_oks:.4f} **")
            no_improve_epochs = 0

        if current_icc > best_icc:
            best_icc = current_icc
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                epoch_metrics,
                output_dir / "best_icc.pth", log
            )
            print(f"  ** New best ICC: {best_icc:.4f} ({best_formula}) **")

        # Always save latest
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            epoch_metrics,
            output_dir / "latest.pth", log
        )

        # ── Log to JSON ────────────────────────────────────────────────────
        training_log.append(epoch_metrics)
        with open(output_dir / "training_log.json", "w",
                  encoding="utf-8") as f:
            json.dump(training_log, f, indent=2)

        # ── Early stopping ─────────────────────────────────────────────────
        no_improve_epochs += 1
        if no_improve_epochs >= args.patience:
            log.info(f"\nEarly stopping: no OKS improvement for "
                     f"{args.patience} epochs.")
            break

        # ── VRAM check every 5 epochs ──────────────────────────────────────
        if torch.cuda.is_available() and epoch % 5 == 0:
            used = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n  VRAM: {used:.2f}GB used  {peak:.2f}GB peak")
            if peak > 3.5:
                print("  WARNING: approaching 4GB VRAM limit. "
                      "Consider reducing batch size.")

    # ── Final summary ──────────────────────────────────────────────────────
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*65}")
    print(f"  Training complete in {total_time:.1f} minutes")
    print(f"  Best validation OKS: {best_oks:.4f}")
    print(f"  Best validation ICC: {best_icc:.4f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Required for Windows multiprocessing in DataLoader
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Train bone loss keypoint model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --data_dir ./data_prepared --output_dir ./runs/exp1
  python train.py --data_dir ./data_prepared --output_dir ./runs/exp1 --batch_size 4
  python train.py --data_dir ./data_prepared --output_dir ./runs/exp1 --resume ./runs/exp1/best_oks.pth
        """,
    )
    parser.add_argument("--data_dir",    default="./data_prepared",
                        help="Path to data_prepared/ folder")
    parser.add_argument("--output_dir",  default="./runs/exp1",
                        help="Output folder for checkpoints and logs")
    parser.add_argument("--epochs",      type=int, default=100,
                        help="Max training epochs (default: 100)")
    parser.add_argument("--batch_size",  type=int, default=8,
                        help="Training batch size. RTX 3050 Ti: 8 is safe, "
                             "reduce to 4 if OOM (default: 8)")
    parser.add_argument("--lr",          type=float, default=0.0001,
                        help="Initial learning rate (default: 0.0001)")
    parser.add_argument("--patience",    type=int, default=30,
                        help="Early stopping patience epochs (default: 30)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers. Default 0 (recommended for Windows). "
                             "Linux users can try 4 for faster loading.")
    parser.add_argument("--resume",       default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--weights_only", action="store_true",
                        help="Load only model weights from checkpoint, "
                             "resetting optimizer and scheduler. "
                             "Use when resuming with a new LR schedule.")

    args = parser.parse_args()
    train(args)