"""
dataset.py
==========
PyTorch Dataset and DataLoader for DenPAR bone loss keypoint training.

Reads from the prepared data_prepared/ directory output by prepare_data.py.

Key responsibilities:
  - Load tooth crop images and 6-slot keypoint annotations
  - Apply visibility downgrade for warned keypoints (estimated points)
  - Apply augmentation (training only)
  - Return tensors in the exact format expected by the model
  - Handle single-root vs double-root teeth correctly

Usage:
    from dataset import build_dataloaders
    loaders = build_dataloaders(data_prepared_dir="./data_prepared")
    train_loader = loaders["training"]
    val_loader   = loaders["validation"]
    test_loader  = loaders["testing"]
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match prepare_data.py exactly
# ─────────────────────────────────────────────────────────────────────────────

NUM_KP = 6

KEYPOINT_NAMES = [
    "cej_mesial",           # slot 0
    "cej_distal",           # slot 1
    "intersection_mesial",  # slot 2
    "intersection_distal",  # slot 3
    "apex_mesial",          # slot 4
    "apex_distal",          # slot 5 — absent (v=0) for single-root
]

# Visibility levels
VIS_ABSENT    = 0   # keypoint not applicable (single-root apex_distal)
VIS_UNCERTAIN = 1   # keypoint present but estimated / uncertain
VIS_VISIBLE   = 2   # keypoint present and annotated with confidence

# Warning tags that should trigger visibility downgrade
# Maps warning substring -> (slot_indices_affected, new_visibility)
VISIBILITY_DOWNGRADES = {
    "cej_distal_estimated":          ([1],    VIS_UNCERTAIN),
    "both_cej_estimated":            ([0, 1], VIS_ABSENT),
    "intersection_distal_mirrored":  ([3],    VIS_UNCERTAIN),
}

# Severity class weights for sampler (inverse frequency)
# None=629, Mild=347, Moderate=456, Severe=43 in training
# Higher weight = sampled more frequently
SEVERITY_WEIGHTS = {
    "none":     1.0,
    "mild":     1.8,
    "moderate": 1.4,
    "severe":   14.6,   # 629/43 ≈ 14.6x underrepresented
}

# Image normalisation — ImageNet mean/std (standard for COCO-pretrained backbones)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# Fixed input size for the model
# Tall crops to preserve root anatomy (teeth are taller than wide)
INPUT_H = 512
INPUT_W = 256


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers (pure numpy/cv2 — no albumentations dependency)
# ─────────────────────────────────────────────────────────────────────────────

def random_brightness_contrast(img, brightness=0.15, contrast=0.15):
    """Random brightness and contrast shift for X-ray images."""
    alpha = 1.0 + random.uniform(-contrast, contrast)   # contrast
    beta  = random.uniform(-brightness, brightness) * 255  # brightness
    img   = img.astype(np.float32)
    img   = img * alpha + beta
    return np.clip(img, 0, 255).astype(np.uint8)


def random_horizontal_flip(img, kps):
    """
    Flip image and keypoints horizontally.
    After flipping, swap mesial/distal pairs so anatomy stays correct:
      slot 0 (cej_mesial)  <-> slot 1 (cej_distal)
      slot 2 (int_mesial)  <-> slot 3 (int_distal)
      slot 4 (apex_mesial) <-> slot 5 (apex_distal)
    """
    w    = img.shape[1]
    img  = cv2.flip(img, 1)
    kps  = kps.copy()

    # Flip x coordinate for visible keypoints
    for i in range(NUM_KP):
        if kps[i, 2] > VIS_ABSENT:
            kps[i, 0] = w - 1 - kps[i, 0]

    # Swap mesial/distal pairs
    for m_idx, d_idx in [(0, 1), (2, 3), (4, 5)]:
        kps[[m_idx, d_idx]] = kps[[d_idx, m_idx]]

    return img, kps


def random_rotation(img, kps, max_angle=8):
    """Small rotation — dental X-rays can be slightly tilted."""
    h, w  = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    cx, cy = w / 2, h / 2

    M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    kps = kps.copy()
    for i in range(NUM_KP):
        if kps[i, 2] > VIS_ABSENT:
            x, y   = kps[i, 0], kps[i, 1]
            pt     = np.array([x, y, 1.0])
            new_pt = M @ pt
            kps[i, 0] = float(new_pt[0])
            kps[i, 1] = float(new_pt[1])

    return img, kps


def random_scale_shift(img, kps, scale_range=(0.9, 1.1), shift_ratio=0.05):
    """Mild scale and shift augmentation."""
    h, w   = img.shape[:2]
    scale  = random.uniform(*scale_range)
    shift_x = random.uniform(-shift_ratio, shift_ratio) * w
    shift_y = random.uniform(-shift_ratio, shift_ratio) * h

    M   = np.float32([[scale, 0, shift_x],
                      [0, scale, shift_y]])
    img = cv2.warpAffine(img, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    kps = kps.copy()
    for i in range(NUM_KP):
        if kps[i, 2] > VIS_ABSENT:
            x, y = kps[i, 0], kps[i, 1]
            kps[i, 0] = x * scale + shift_x
            kps[i, 1] = y * scale + shift_y

    return img, kps


def gaussian_noise(img, var_range=(5, 25)):
    """Add Gaussian noise to simulate X-ray sensor variation."""
    var   = random.uniform(*var_range)
    noise = np.random.normal(0, var ** 0.5, img.shape).astype(np.float32)
    img   = img.astype(np.float32) + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def clamp_keypoints(kps, w, h):
    """Clamp keypoint coordinates to image bounds after augmentation."""
    kps = kps.copy()
    for i in range(NUM_KP):
        if kps[i, 2] > VIS_ABSENT:
            kps[i, 0] = float(np.clip(kps[i, 0], 0, w - 1))
            kps[i, 1] = float(np.clip(kps[i, 1], 0, h - 1))
    return kps


def resize_image_and_keypoints(img, kps, target_h, target_w):
    """Resize image and scale keypoints proportionally."""
    orig_h, orig_w = img.shape[:2]
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    kps     = kps.copy()
    for i in range(NUM_KP):
        if kps[i, 2] > VIS_ABSENT:
            kps[i, 0] *= scale_x
            kps[i, 1] *= scale_y

    return img, kps


def normalise_to_tensor(img):
    """Convert HxWxC uint8 BGR image to normalised CxHxW float tensor."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMG_MEAN, dtype=np.float32)
    std  = np.array(IMG_STD,  dtype=np.float32)
    img  = (img - mean) / std
    img  = img.transpose(2, 0, 1)   # HWC -> CHW
    return torch.from_numpy(img)


def apply_augmentation(img, kps):
    """Apply full training augmentation pipeline."""
    # Brightness / contrast (always, mild)
    img = random_brightness_contrast(img, brightness=0.12, contrast=0.12)

    # Gaussian noise (30% chance)
    if random.random() < 0.30:
        img = gaussian_noise(img, var_range=(5, 20))

    # Horizontal flip (50% chance)
    if random.random() < 0.50:
        img, kps = random_horizontal_flip(img, kps)

    # Rotation (60% chance, small angles only)
    if random.random() < 0.60:
        img, kps = random_rotation(img, kps, max_angle=8)

    # Scale + shift (40% chance)
    if random.random() < 0.40:
        img, kps = random_scale_shift(img, kps,
                                      scale_range=(0.92, 1.08),
                                      shift_ratio=0.04)
    return img, kps


# ─────────────────────────────────────────────────────────────────────────────
# Visibility downgrade
# ─────────────────────────────────────────────────────────────────────────────

def apply_visibility_downgrades(kps, warnings):
    """
    Downgrade keypoint visibility for estimated/uncertain annotations.

    Args:
        kps:      np.ndarray (6, 3)  float32  [x, y, v]
        warnings: list of warning strings from annotations.json

    Returns:
        kps: updated array with corrected visibility flags
    """
    kps = kps.copy()
    for warn_tag, (slots, new_vis) in VISIBILITY_DOWNGRADES.items():
        if any(warn_tag in w for w in warnings):
            for slot in slots:
                if kps[slot, 2] > new_vis:
                    kps[slot, 2] = float(new_vis)
    return kps


# ─────────────────────────────────────────────────────────────────────────────
# Severity label helper
# ─────────────────────────────────────────────────────────────────────────────

def bone_loss_to_severity(pct):
    """Map bone loss percentage to integer severity class."""
    if pct is None:
        return -1   # unknown
    if pct < 15:
        return 0    # none
    elif pct < 33:
        return 1    # mild
    elif pct < 66:
        return 2    # moderate
    else:
        return 3    # severe


# ─────────────────────────────────────────────────────────────────────────────
# Main Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class DenPARDataset(Dataset):
    """
    PyTorch Dataset for DenPAR bone loss keypoint detection.

    Each sample returns a dict:
    {
        'image':          FloatTensor [3, INPUT_H, INPUT_W]
        'keypoints':      FloatTensor [6, 3]   (x, y, visibility)
        'bbox':           FloatTensor [4]       [0, 0, W, H] in crop space
        'is_double_root': BoolTensor  []
        'severity':       LongTensor  []        mesial severity class 0-3
        'image_id':       str
        'tooth_index':    int
        'crop_filename':  str
    }

    Keypoint coordinates are in INPUT_H x INPUT_W space after resize.
    Visibility flags:  0=absent, 1=uncertain(downgraded), 2=visible
    """

    def __init__(self, data_prepared_dir, split, augment=False):
        """
        Args:
            data_prepared_dir: path to data_prepared/ root folder
            split:   "training" | "validation" | "testing"
            augment: apply augmentation (True for training only)
        """
        self.split   = split
        self.augment = augment
        self.crop_dir = Path(data_prepared_dir) / split / "crops"

        ann_path = Path(data_prepared_dir) / split / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                f"Run prepare_data.py first."
            )

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.annotations = data["annotations"]
        print(f"  [{split}] Loaded {len(self.annotations)} tooth samples")
        self._print_summary()

    def _print_summary(self):
        n          = len(self.annotations)
        n_double   = sum(1 for a in self.annotations if a["is_double_root"])
        n_single   = n - n_double
        warned     = sum(1 for a in self.annotations
                         if a["annotation_warnings"])
        print(f"           Single-root: {n_single}  "
              f"Double-root: {n_double}  "
              f"With warnings: {warned}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # ── Load crop image ────────────────────────────────────────────────
        crop_path = self.crop_dir / ann["crop_filename"]
        img = cv2.imread(str(crop_path))     # BGR uint8

        if img is None:
            # Corrupt or missing crop — return a zeroed sample
            # (should not happen if prepare_data ran successfully)
            img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
            ann_warnings = ann.get("annotation_warnings", []) + ["missing_crop"]
        else:
            ann_warnings = ann.get("annotation_warnings", [])

        # ── Load keypoints ─────────────────────────────────────────────────
        kps = np.array(ann["keypoints"], dtype=np.float32)  # (6, 3)

        # ── Apply visibility downgrades for estimated keypoints ────────────
        kps = apply_visibility_downgrades(kps, ann_warnings)

        # NOTE: No resize step needed here.
        # prepare_data.py now saves crops already resized to (INPUT_H, INPUT_W)
        # and keypoints are already in that coordinate space.
        # Resizing here would double-scale the coordinates and break alignment.

        # ── Augmentation (training only) ───────────────────────────────────
        if self.augment:
            img, kps = apply_augmentation(img, kps)

        # ── Clamp keypoints to image bounds after augmentation ─────────────
        kps = clamp_keypoints(kps, w=INPUT_W, h=INPUT_H)

        # ── Normalise and convert to tensor ───────────────────────────────
        image_tensor = normalise_to_tensor(img)   # (3, H, W) float32

        # ── Build bbox tensor [x1, y1, x2, y2] in crop space ──────────────
        bbox_tensor = torch.tensor(
            [0.0, 0.0, float(INPUT_W), float(INPUT_H)],
            dtype=torch.float32
        )

        # ── Severity label from mesial GT bone loss ────────────────────────
        severity = bone_loss_to_severity(ann.get("gt_bone_loss_mesial"))

        return {
            "image":          image_tensor,
            "keypoints":      torch.tensor(kps, dtype=torch.float32),
            "bbox":           bbox_tensor,
            "is_double_root": torch.tensor(ann["is_double_root"],
                                           dtype=torch.bool),
            "severity":       torch.tensor(severity, dtype=torch.long),
            "gt_bone_loss_mesial": torch.tensor(
                ann["gt_bone_loss_mesial"] if ann["gt_bone_loss_mesial"] is not None else -1.0,
                dtype=torch.float32
            ),
            "gt_bone_loss_distal": torch.tensor(
                ann["gt_bone_loss_distal"] if ann["gt_bone_loss_distal"] is not None else -1.0,
                dtype=torch.float32
            ),
            # Metadata (not used in loss, useful for debugging)
            "image_id":      ann["image_id"],
            "tooth_index":   ann["tooth_index"],
            "crop_filename": ann["crop_filename"],
        }

    def get_sample_weights(self):
        """
        Compute per-sample weights for WeightedRandomSampler.
        Upweights severe cases to compensate for class imbalance.
        """
        weights = []
        for ann in self.annotations:
            sev = bone_loss_to_severity(ann.get("gt_bone_loss_mesial"))
            sev_name = ["none", "mild", "moderate", "severe"][max(sev, 0)]
            weights.append(SEVERITY_WEIGHTS[sev_name])
        return weights


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Custom collate function.
    All images are the same size (INPUT_H x INPUT_W) so standard
    stacking works. Metadata is kept as lists.
    """
    images      = torch.stack([b["image"]     for b in batch])
    keypoints   = torch.stack([b["keypoints"] for b in batch])
    bboxes      = torch.stack([b["bbox"]      for b in batch])
    double_root = torch.stack([b["is_double_root"] for b in batch])
    severity    = torch.stack([b["severity"]  for b in batch])
    bl_mesial   = torch.stack([b["gt_bone_loss_mesial"] for b in batch])
    bl_distal   = torch.stack([b["gt_bone_loss_distal"] for b in batch])

    meta = [{
        "image_id":      b["image_id"],
        "tooth_index":   b["tooth_index"],
        "crop_filename": b["crop_filename"],
    } for b in batch]

    return {
        "image":               images,       # (B, 3, H, W)
        "keypoints":           keypoints,    # (B, 6, 3)
        "bbox":                bboxes,       # (B, 4)
        "is_double_root":      double_root,  # (B,) bool
        "severity":            severity,     # (B,) long
        "gt_bone_loss_mesial": bl_mesial,    # (B,) float
        "gt_bone_loss_distal": bl_distal,    # (B,) float
        "meta":                meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_prepared_dir,
    batch_size   = 8,
    num_workers  = 0,
    splits       = ("training", "validation", "testing"),
):
    """
    Build DataLoaders for all splits.

    Uses uniform random sampling for training — no severity weighting.

    Rationale: WeightedRandomSampler upweighted severe bone loss cases
    14.6x. Severe cases often have heavily resorbed bone making
    anatomical landmarks harder to see and less reliably annotated.
    Oversampling the noisiest 3% of data 14x actively hurts keypoint
    training. Uniform sampling gives the model a clean, consistent
    learning signal across all severity classes.

    Severity weighting can be reintroduced in Phase 3 if needed
    for bone loss severity classification, not keypoint detection.

    Args:
        data_prepared_dir: path to data_prepared/ folder
        batch_size:        training batch size (val/test use 2x for speed)
        num_workers:       DataLoader workers (0 = main process, safe on Windows)
        splits:            which splits to build loaders for

    Returns:
        dict: {split_name: DataLoader}
    """
    pin_memory = torch.cuda.is_available()
    loaders    = {}

    for split in splits:
        is_train = (split == "training")

        dataset = DenPARDataset(
            data_prepared_dir = data_prepared_dir,
            split   = split,
            augment = is_train,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size         = batch_size if is_train else batch_size * 2,
            shuffle            = is_train,   # uniform shuffle for training
            sampler            = None,       # no weighted sampler
            num_workers        = num_workers,
            collate_fn         = collate_fn,
            pin_memory         = pin_memory,
            persistent_workers = False,      # disabled — deadlocks on Windows
        )

    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check — run this file directly to verify the dataset
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify dataset loading")
    parser.add_argument("--data_dir", default="./data_prepared",
                        help="Path to data_prepared/ folder")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print("\nBuilding DataLoaders...")
    loaders = build_dataloaders(
        data_prepared_dir = args.data_dir,
        batch_size        = args.batch_size,
        num_workers       = 0,   # 0 for quick test (avoids multiprocessing overhead)
        use_sampler       = True,
    )

    for split, loader in loaders.items():
        print(f"\n[{split}] — {len(loader.dataset)} samples, "
              f"{len(loader)} batches")
        batch = next(iter(loader))

        print(f"  image shape:     {batch['image'].shape}")
        print(f"  keypoints shape: {batch['keypoints'].shape}")
        print(f"  bbox shape:      {batch['bbox'].shape}")
        print(f"  is_double_root:  {batch['is_double_root']}")
        print(f"  severity:        {batch['severity']}")

        # Verify keypoint visibility distribution
        kps  = batch["keypoints"]     # (B, 6, 3)
        vis  = kps[:, :, 2]           # (B, 6)
        for slot_idx, name in enumerate(KEYPOINT_NAMES):
            v0 = (vis[:, slot_idx] == 0).sum().item()
            v1 = (vis[:, slot_idx] == 1).sum().item()
            v2 = (vis[:, slot_idx] == 2).sum().item()
            print(f"  {name:<28} absent={v0}  uncertain={v1}  visible={v2}")

        # Verify image tensor range
        img = batch["image"]
        print(f"  image min/max:   {img.min():.3f} / {img.max():.3f}")
        print(f"  GT mesial loss:  {batch['gt_bone_loss_mesial']}")

        break   # only check first split for quick test
    print("\nDataset verification complete.")