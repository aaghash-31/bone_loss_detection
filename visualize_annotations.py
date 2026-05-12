"""
visualize_annotations.py
========================
Visualize ground truth keypoints on tooth crop images from data_prepared/.

Three modes:
  Mode 1 (--mode gt):   Ground truth keypoints only — data verification
  Mode 2 (--mode pred): Predicted keypoints only — model output inspection
  Mode 3 (--mode both): GT + predicted overlay — error analysis

Run Mode 1 BEFORE more training to verify data correctness.
If keypoints are in wrong positions on crops, fix data prep — don't train.

Color scheme:
  CEJ:          GREEN  (slots 0, 1)
  Intersection: BLUE   (slots 2, 3)
  Apex:         RED    (slots 4, 5)
  Mesial:       solid filled circle
  Distal:       hollow circle (outline only)
  v=1 uncertain: YELLOW ring around the dot
  Root axis line: thin line connecting CEJ -> Intersection -> Apex

Usage:
  # Verify ground truth data (always run this first)
  python visualize_annotations.py --data_dir ./data_prepared --mode gt

  # Visualize predicted keypoints from a checkpoint
  python visualize_annotations.py --data_dir ./data_prepared --mode pred
      --checkpoint ./runs/exp6/best_oks.pth

  # Overlay GT and predicted for error analysis
  python visualize_annotations.py --data_dir ./data_prepared --mode both
      --checkpoint ./runs/exp6/best_oks.pth

  # Control which split and how many images
  python visualize_annotations.py --data_dir ./data_prepared --mode gt
      --split validation --n 20

Output: visualizations/<mode>/<split>/<crop_filename>.jpg
"""

import os
import json
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Color constants (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────

COLOR_CEJ          = (50,  200,  50)    # green
COLOR_INTERSECTION = (200,  80,  80)    # blue
COLOR_APEX         = (50,   80, 220)    # red
COLOR_UNCERTAIN    = (30,  220, 220)    # yellow ring for v=1
COLOR_PRED         = (0,   165, 255)    # orange for predicted points
COLOR_GT_OVERLAY   = (50,  200,  50)    # green for GT in overlay mode
COLOR_ERROR_LINE   = (100, 100, 255)    # light red for GT->pred error lines
COLOR_ROOT_LINE    = (180, 180, 180)    # grey for root axis lines
COLOR_TEXT_BG      = (20,   20,  20)    # dark background for text labels

# Keypoint display properties indexed by slot
# (color, is_mesial, name)
SLOT_PROPS = [
    (COLOR_CEJ,          True,  "CEJ-M"),    # slot 0
    (COLOR_CEJ,          False, "CEJ-D"),    # slot 1
    (COLOR_INTERSECTION, True,  "Int-M"),    # slot 2
    (COLOR_INTERSECTION, False, "Int-D"),    # slot 3
    (COLOR_APEX,         True,  "Apx-M"),   # slot 4
    (COLOR_APEX,         False, "Apx-D"),   # slot 5
]

# Root triplets: (cej_slot, inter_slot, apex_slot)
ROOT_TRIPLETS = [(0, 2, 4), (1, 3, 5)]

RADIUS_FILLED  = 6   # mesial: solid filled circle
RADIUS_HOLLOW  = 6   # distal: hollow circle
THICKNESS_FILL = -1  # filled
THICKNESS_RING = 2   # hollow

FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.38
FONT_THICKNESS  = 1


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_keypoint(img, x, y, color, is_mesial, visibility, label=None):
    """
    Draw a single keypoint on the image.

    v=2 visible:   solid (mesial) or hollow (distal) circle in color
    v=1 uncertain: same shape with yellow uncertainty ring
    v=0 absent:    nothing drawn
    """
    if visibility == 0:
        return

    cx, cy = int(round(x)), int(round(y))

    # Yellow uncertainty ring for v=1 (estimated) keypoints
    if visibility == 1:
        cv2.circle(img, (cx, cy), RADIUS_FILLED + 4,
                   COLOR_UNCERTAIN, 2)

    if is_mesial:
        # Filled circle for mesial
        cv2.circle(img, (cx, cy), RADIUS_FILLED, color, THICKNESS_FILL)
        # White border to separate from background
        cv2.circle(img, (cx, cy), RADIUS_FILLED, (255, 255, 255), 1)
    else:
        # Hollow circle for distal
        cv2.circle(img, (cx, cy), RADIUS_HOLLOW, color, THICKNESS_RING)
        # Small inner dot for visibility
        cv2.circle(img, (cx, cy), 2, color, THICKNESS_FILL)

    # Optional label
    if label:
        tx, ty = cx + 8, cy - 4
        # Background rectangle for readability
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(img, (tx - 1, ty - th - 1),
                      (tx + tw + 1, ty + 1), COLOR_TEXT_BG, -1)
        cv2.putText(img, label, (tx, ty), FONT,
                    FONT_SCALE, (255, 255, 255), FONT_THICKNESS)


def draw_root_lines(img, kps, alpha=0.5):
    """
    Draw thin lines connecting CEJ -> Intersection -> Apex for each root.
    kps: numpy array (6, 3) [x, y, v]
    """
    overlay = img.copy()
    for cej_s, int_s, apex_s in ROOT_TRIPLETS:
        pts = []
        for s in [cej_s, int_s, apex_s]:
            if kps[s, 2] > 0:
                pts.append((int(round(kps[s, 0])), int(round(kps[s, 1]))))
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], COLOR_ROOT_LINE, 1,
                     cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_error_lines(img, gt_kps, pred_kps):
    """
    Draw lines between GT and predicted positions to show localisation error.
    Only draws for visible GT keypoints.
    """
    for s in range(6):
        if gt_kps[s, 2] > 0:
            gx = int(round(gt_kps[s, 0]))
            gy = int(round(gt_kps[s, 1]))
            px = int(round(pred_kps[s, 0]))
            py = int(round(pred_kps[s, 1]))
            cv2.line(img, (gx, gy), (px, py), COLOR_ERROR_LINE, 1,
                     cv2.LINE_AA)


def add_header(img, text, color=(220, 220, 220)):
    """Add a text header bar at the top of the image."""
    h, w = img.shape[:2]
    bar_h = 22
    header = np.zeros((bar_h, w, 3), dtype=np.uint8)
    cv2.putText(header, text, (4, 15), FONT, 0.42,
                color, 1, cv2.LINE_AA)
    return np.vstack([header, img])


def add_legend(img):
    """Add a legend strip at the bottom explaining the color scheme."""
    h, w = img.shape[:2]
    leg_h = 28
    legend = np.zeros((leg_h, w, 3), dtype=np.uint8)

    items = [
        (COLOR_CEJ,          "CEJ"),
        (COLOR_INTERSECTION, "Intersect"),
        (COLOR_APEX,         "Apex"),
        (COLOR_UNCERTAIN,    "uncertain"),
    ]
    x = 6
    for color, label in items:
        cv2.circle(legend, (x + 6, 14), 5, color, -1)
        cv2.putText(legend, label, (x + 14, 18), FONT, 0.36,
                    (200, 200, 200), 1)
        x += 80

    cv2.putText(legend, "filled=mesial  hollow=distal",
                (x + 10, 18), FONT, 0.34, (150, 150, 150), 1)

    return np.vstack([img, legend])


# ─────────────────────────────────────────────────────────────────────────────
# Compute OKS for a single tooth (for overlay mode display)
# ─────────────────────────────────────────────────────────────────────────────

def compute_single_oks(pred_kps, gt_kps, bbox):
    """Quick OKS for one tooth — displayed on overlay images."""
    sigmas = np.array([0.025, 0.025, 0.035, 0.035, 0.040, 0.040])
    vars_  = (2 * sigmas) ** 2

    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    scale_sq = max(bw * bh, 1.0)

    vis     = gt_kps[:, 2]
    visible = (vis > 0).astype(float)
    n_vis   = visible.sum()
    if n_vis == 0:
        return 0.0

    d_sq   = ((pred_kps[:, :2] - gt_kps[:, :2]) ** 2).sum(axis=1)
    exp    = np.exp(-d_sq / (2 * scale_sq * vars_))
    oks    = (exp * visible).sum() / n_vis
    return float(oks)


# ─────────────────────────────────────────────────────────────────────────────
# Run inference on a batch of crop images
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_predict(checkpoint_path, annotations, crop_dir, device):
    """
    Load model from checkpoint and run inference on the given annotations.

    Returns:
        dict: {crop_filename: np.array (6, 3) predicted keypoints in crop space}
    """
    from model   import BoneLossKeypointModel
    from .dataset import INPUT_H, INPUT_W, IMG_MEAN, IMG_STD

    print(f"Loading model from {checkpoint_path}...")
    model = BoneLossKeypointModel(pretrained=False).to(device)
    ckpt  = torch.load(str(checkpoint_path), map_location=device,
                       weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")

    mean = np.array(IMG_MEAN, dtype=np.float32)
    std  = np.array(IMG_STD,  dtype=np.float32)

    predictions = {}

    with torch.no_grad():
        for ann in annotations:
            crop_path = crop_dir / ann["crop_filename"]
            img = cv2.imread(str(crop_path))
            if img is None:
                continue

            # Preprocess — same as dataset.py
            img_resized = cv2.resize(img, (INPUT_W, INPUT_H),
                                     interpolation=cv2.INTER_LINEAR)
            img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm    = (img_rgb.astype(np.float32) / 255.0 - mean) / std
            img_tensor  = torch.from_numpy(
                img_norm.transpose(2, 0, 1)
            ).unsqueeze(0).to(device)

            out  = model({"image": img_tensor,
                          "keypoints":      torch.zeros(1, 6, 3),
                          "is_double_root": torch.zeros(1).bool()})
            pred = out["keypoints"][0].cpu().numpy()   # (6, 3)
            predictions[ann["crop_filename"]] = pred

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Main visualization function
# ─────────────────────────────────────────────────────────────────────────────

def resolve_split_folder(data_dir: Path, split: str) -> Path:
    """Resolve actual split folder with case-insensitive matching."""
    candidates = [
        split,
        split.lower(),
        split.upper(),
        split.capitalize(),
        split.title(),
    ]
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Split folder not found under {data_dir}: tried {candidates}"
    )


def visualize(args):
    data_dir  = Path(args.data_dir)
    split     = args.split
    mode      = args.mode
    n_images  = args.n
    out_root  = Path(args.output_dir) / mode / split
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve split folder path from prepared data directory.
    split_dir = resolve_split_folder(data_dir, split)

    # Load annotations
    ann_path = split_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_annotations = data["annotations"]

    crop_dir = split_dir / "crops"

    # Sample subset
    if n_images < len(all_annotations):
        sample = random.sample(all_annotations, n_images)
    else:
        sample = all_annotations

    print(f"Mode: {mode}  Split: {split}  Samples: {len(sample)}")
    print(f"Output: {out_root}")

    # Run inference if needed
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = {}
    if mode in ("pred", "both"):
        if not args.checkpoint:
            print("ERROR: --checkpoint required for mode 'pred' or 'both'")
            return
        predictions = load_model_and_predict(
            args.checkpoint, sample, crop_dir, device
        )

    # Draw each annotation
    saved = 0
    for ann in sample:
        crop_path = crop_dir / ann["crop_filename"]
        img = cv2.imread(str(crop_path))
        if img is None:
            print(f"  MISSING: {crop_path}")
            continue

        # Resize to INPUT size for consistent display
        from .dataset import INPUT_H, INPUT_W
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)

        # Parse GT keypoints
        gt_kps = np.array(ann["keypoints"], dtype=np.float32)   # (6, 3)
        warns  = ann.get("annotation_warnings", [])
        is_dbl = ann["is_double_root"]
        gt_bl  = ann.get("gt_bone_loss_mesial")

        # Header text
        root_str   = "double" if is_dbl else "single"
        bl_str     = f"  BL: {gt_bl:.1f}%" if gt_bl is not None else ""
        header_txt = (f"{ann['image_id']} tooth{ann['tooth_index']} "
                      f"({root_str}){bl_str}")

        # ── Mode: GT only ──────────────────────────────────────────────────
        if mode == "gt":
            draw_root_lines(img, gt_kps)
            for s, (color, is_mesial, name) in enumerate(SLOT_PROPS):
                x, y, v = gt_kps[s]
                draw_keypoint(img, x, y, color, is_mesial, int(v), name)
            img = add_header(img, f"GT: {header_txt}")

        # ── Mode: Predicted only ───────────────────────────────────────────
        elif mode == "pred":
            pred_kps = predictions.get(ann["crop_filename"])
            if pred_kps is None:
                continue
            draw_root_lines(img, pred_kps)
            for s, (_, is_mesial, name) in enumerate(SLOT_PROPS):
                x, y, conf = pred_kps[s]
                draw_keypoint(img, x, y, COLOR_PRED, is_mesial, 2,
                              f"{name}:{conf:.2f}")
            img = add_header(img, f"PRED: {header_txt}", (200, 140, 80))

        # ── Mode: GT + Predicted overlay ───────────────────────────────────
        elif mode == "both":
            pred_kps = predictions.get(ann["crop_filename"])
            if pred_kps is None:
                continue

            # Draw error lines first (under the points)
            draw_error_lines(img, gt_kps, pred_kps)

            # Draw root lines for both
            draw_root_lines(img, gt_kps, alpha=0.4)

            # Draw GT points (green, slightly larger)
            for s, (color, is_mesial, name) in enumerate(SLOT_PROPS):
                x, y, v = gt_kps[s]
                if v > 0:
                    draw_keypoint(img, x, y, color, is_mesial, int(v))

            # Draw predicted points (orange, slightly smaller)
            for s, (_, is_mesial, name) in enumerate(SLOT_PROPS):
                x, y, conf = pred_kps[s]
                draw_keypoint(img, x, y, COLOR_PRED, is_mesial, 2)

            # OKS score
            bbox = ann.get("bbox_orig", [0, 0, INPUT_W, INPUT_H])
            oks  = compute_single_oks(pred_kps, gt_kps,
                                      [0, 0, INPUT_W, INPUT_H])
            img  = add_header(
                img,
                f"GT(grn)+PRED(org): {header_txt}  OKS:{oks:.3f}",
                (180, 180, 255)
            )

        # Warning indicator
        if warns:
            short_warns = [w.split(":")[0][:15] for w in warns[:2]]
            warn_txt    = " | ".join(short_warns)
            cv2.putText(img, f"WARN: {warn_txt}",
                        (4, img.shape[0] - 6), FONT, 0.32,
                        (30, 200, 220), 1)

        img = add_legend(img)

        # Save
        out_path = out_root / ann["crop_filename"]
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1

    print(f"Saved {saved} images to {out_root}")
    print(f"Open in Windows Explorer: explorer {out_root}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize keypoint annotations on tooth crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify ground truth data — always run this first
  python visualize_annotations.py --data_dir ./data_prepared --mode gt

  # Check validation split predictions
  python visualize_annotations.py --data_dir ./data_prepared --mode pred
      --checkpoint ./runs/exp6/best_oks.pth --split validation

  # Overlay GT and predicted for error analysis
  python visualize_annotations.py --data_dir ./data_prepared --mode both
      --checkpoint ./runs/exp6/best_oks.pth --split validation --n 30
        """
    )
    parser.add_argument("--data_dir",   required=True,
                        help="Path to data_prepared/ folder")
    parser.add_argument("--mode",       choices=["gt", "pred", "both"],
                        default="gt",
                        help="Visualization mode (default: gt)")
    parser.add_argument("--split",      default="training",
                        choices=["training", "validation", "testing"],
                        help="Which split to visualize (default: training)")
    parser.add_argument("--n",          type=int, default=30,
                        help="Number of crops to visualize (default: 30)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (required for mode pred/both)")
    parser.add_argument("--output_dir", default="./visualizations",
                        help="Output directory (default: ./visualizations)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for sample selection (default: 42)")

    args = parser.parse_args()
    random.seed(args.seed)
    visualize(args)