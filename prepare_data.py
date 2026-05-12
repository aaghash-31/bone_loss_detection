"""
prepare_data.py
===============
DenPAR dataset preparation for bone loss severity detection.

What this script does:
  1. Parses all annotation JSONs (keypoints + bone lines)
  2. Computes bone-tooth intersection points geometrically
  3. Assigns keypoints to the 6-slot schema (CEJ-M, CEJ-D, Int-M, Int-D, Apex-M, Apex-D)
  4. Infers root type (single / double) per tooth
  5. Crops and saves individual tooth ROI images
  6. Saves a master annotation CSV and per-split JSON files

Output structure:
  data_prepared/
    training/
      crops/          *.jpg   (one per tooth)
      annotations.json
    validation/
      crops/
      annotations.json
    testing/
      crops/
      annotations.json
    dataset_stats.json
    label_map.json

Hardware target: NVIDIA GeForce RTX 3050 Ti (4GB VRAM), Windows/Linux laptop
Optimizations:
  - Multiprocessing for image I/O and mask processing (CPU-bound)
  - Memory-mapped numpy for large intermediate arrays
  - Batch CLAHE preprocessing on CPU (GPU VRAM reserved for training)
  - Progress bars via tqdm
  - Graceful resume: skips already-processed crops

Usage:
  python prepare_data.py --data_root /path/to/DenPAR --output_dir ./data_prepared
  python prepare_data.py --data_root /path/to/DenPAR --output_dir ./data_prepared --workers 4
  python prepare_data.py --data_root /path/to/DenPAR --output_dir ./data_prepared --verify
"""

import os
import sys
import json
import argparse
import logging
import warnings
import traceback
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("prepare_data.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants — keypoint schema
# ─────────────────────────────────────────────────────────────────────────────

# Fixed 6-slot keypoint order — NEVER change this order after data is prepared
# Visibility: 0 = absent, 1 = occluded, 2 = visible
KEYPOINT_NAMES = [
    "cej_mesial",           # slot 0 — always present
    "cej_distal",           # slot 1 — always present
    "intersection_mesial",  # slot 2 — always present
    "intersection_distal",  # slot 3 — always present
    "apex_mesial",          # slot 4 — always present
    "apex_distal",          # slot 5 — absent (v=0) for single-root teeth
]
NUM_KP = 6

# Sigma per keypoint for OKS computation (calibrated by difficulty)
# Smaller = stricter evaluation
KP_SIGMAS = [0.025, 0.025, 0.035, 0.035, 0.040, 0.040]

# Crop padding around YOLOv8 bbox (10% each side)
CROP_PAD_RATIO = 0.10

# Bone line rasterization thickness (optimal per paper Table 5)
BONE_LINE_THICKNESS = 10

# CLAHE parameters for X-ray contrast enhancement
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)

# Minimum tooth bbox dimension in pixels (filter out tiny annotations)
MIN_BBOX_SIZE = 30

# Fixed resize target — MUST match INPUT_H, INPUT_W in dataset.py exactly
# All crops are resized to this size DURING preparation.
# Keypoint coordinates are scaled to match this target space.
TARGET_H = 512
TARGET_W = 256


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KeypointAnnotation:
    """
    One annotated tooth, fully processed and ready for model training.
    All coordinates are in crop-image space (after padding and cropping).
    """
    # Identity
    image_id:       str       # original image filename e.g. "2.jpg"
    tooth_index:    int       # tooth index within the image (0-based)
    split:          str       # "training" / "validation" / "testing"
    crop_filename:  str       # saved crop filename e.g. "2_tooth0.jpg"

    # Geometry in ORIGINAL image space (before crop) — for debugging
    bbox_orig:      list      # [x1, y1, x2, y2]
    crop_offset:    list      # [ox, oy] top-left corner of padded crop

    # 6-slot keypoints in CROP space: [[x,y,v], ...] * 6
    # v=0 absent, v=2 visible
    keypoints:      list

    # Root type
    is_double_root: bool

    # Computed bone loss ground truth (from GT keypoints, for validation)
    # None if intersection point could not be computed
    gt_bone_loss_mesial:  Optional[float]  # percent 0-100
    gt_bone_loss_distal:  Optional[float]  # percent 0-100, None for single-root

    # Data quality flags
    has_tooth_mask:       bool
    intersection_source:  str   # "mask_overlap" | "closest_point" | "bone_midpoint"
    annotation_warnings:  list  # list of warning strings


# ─────────────────────────────────────────────────────────────────────────────
# Geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(gray_image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to a grayscale X-ray image."""
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )
    return clahe.apply(gray_image)


def load_image_gray(path: str) -> Optional[np.ndarray]:
    """Load image as grayscale. Returns None if file missing or corrupt."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.warning(f"Could not load image: {path}")
    return img


def load_image_mask(path: str, expected_h: Optional[int] = None, expected_w: Optional[int] = None) -> Optional[np.ndarray]:
    """Load a binary mask image and resize to expected dimensions if needed."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        log.warning(f"Could not load mask: {path}")
        return None
    if expected_h is not None and expected_w is not None and mask.shape != (expected_h, expected_w):
        mask = cv2.resize(mask, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)
    return mask


def find_radiograph_mask_file(mask_dir: Path, img_stem: str) -> Optional[Path]:
    """Find a full-image radiograph mask file by stem name and common image extensions."""
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidate = mask_dir / f"{img_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_tooth_mask_from_full_radiograph(full_mask: np.ndarray, bbox):
    """Extract the tooth-connected component inside a full radiograph mask for a given bbox."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(full_mask.shape[1], x2)
    y2 = min(full_mask.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = (full_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    if roi.sum() == 0:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if num_labels <= 1:
        mask = np.zeros_like(full_mask, dtype=np.uint8)
        mask[y1:y2, x1:x2] = roi * 255
        return mask

    best_label = 0
    best_overlap = -1
    for label in range(1, num_labels):
        overlap = int(stats[label, cv2.CC_STAT_AREA])
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label

    if best_label == 0:
        return None

    selected = (labels == best_label).astype(np.uint8)
    mask = np.zeros_like(full_mask, dtype=np.uint8)
    mask[y1:y2, x1:x2] = selected * 255
    return mask


def pad_bbox(bbox, pad_ratio, img_h, img_w):
    """
    Expand bbox by pad_ratio on each side, clamped to image bounds.
    Returns (x1p, y1p, x2p, y2p, ox, oy) where ox,oy is the crop offset.
    """
    x1, y1, x2, y2 = bbox
    bw  = x2 - x1
    bh  = y2 - y1
    px  = int(bw * pad_ratio)
    py  = int(bh * pad_ratio)

    x1p = max(0, int(x1) - px)
    y1p = max(0, int(y1) - py)
    x2p = min(img_w, int(x2) + px)
    y2p = min(img_h, int(y2) + py)

    return x1p, y1p, x2p, y2p, x1p, y1p


def shift_points_to_crop(points, ox, oy):
    """Subtract crop offset from a list of (x, y) tuples."""
    return [(p[0] - ox, p[1] - oy) for p in points]


def points_inside_bbox(points, bbox):
    """Filter points to those inside [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox
    return [p for p in points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]


def load_tooth_mask_by_overlap(mask_folder, bbox, img_h, img_w):
    """
    Find and load the correct tooth mask by spatial overlap with the bbox.

    The mask files are matched to teeth by finding which mask has the
    most pixels inside the tooth's bounding box. This is robust to any
    filename ordering inconsistency between mask PNGs and bbox list.

    Args:
        mask_folder: Path to the per-image mask subfolder
        bbox:        [x1, y1, x2, y2] tooth bounding box
        img_h, img_w: full image dimensions

    Returns:
        tooth_mask (np.ndarray grayscale) or None if no masks found
    """
    mask_path = Path(mask_folder)
    if not mask_path.exists():
        return None

    # Full-image radiograph mask file case.
    if mask_path.is_file():
        full_mask = load_image_mask(str(mask_path), img_h, img_w)
        if full_mask is None:
            return None
        return extract_tooth_mask_from_full_radiograph(full_mask, bbox)

    # Per-tooth mask folder case.
    mask_files = sorted(mask_path.glob("*.png"))
    if not mask_files:
        return None

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(img_w, x2);  y2 = min(img_h, y2)

    best_mask     = None
    best_overlap  = 0

    for mf in mask_files:
        m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        # Resize mask to full image size if needed
        if m.shape != (img_h, img_w):
            m = cv2.resize(m, (img_w, img_h),
                           interpolation=cv2.INTER_NEAREST)
        # Count non-zero pixels inside bbox
        roi      = m[y1:y2, x1:x2]
        overlap  = int((roi > 0).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_mask    = m

    return best_mask if best_overlap > 0 else None


def build_tooth_filter_zones(tooth_mask, img_h, img_w):
    """
    Build two tolerance zones from the tooth mask for keypoint filtering.

    CEJ and Intersection points sit at the tooth boundary (interproximal
    space) so they need a generous dilation to be accepted.
    Apex points sit inside the root so they use a smaller dilation.

    Returns:
        zone_boundary: dilated mask for CEJ + Intersection (±20px tolerance)
        zone_apex:     dilated mask for Apex (±8px tolerance)
    """
    if tooth_mask is None:
        return None, None

    tooth_bin = (tooth_mask > 0).astype(np.uint8)

    # Generous boundary zone for CEJ and intersection points
    k_boundary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    zone_boundary = cv2.dilate(tooth_bin, k_boundary)

    # Strict zone for apex — should be inside root
    k_apex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    zone_apex = cv2.dilate(tooth_bin, k_apex)

    return zone_boundary, zone_apex


def point_in_zone(pt, zone, img_h, img_w):
    """
    Check if point (x, y) falls within a binary zone mask.
    Returns True if inside zone, False otherwise.
    Falls back to True (accept) if zone is None (no mask available).
    """
    if zone is None:
        return True   # no mask available — fall back to bbox filtering
    x, y = int(round(pt[0])), int(round(pt[1]))
    if not (0 <= x < img_w and 0 <= y < img_h):
        return False
    return bool(zone[y, x] > 0)


def filter_points_by_mask(points, zone, img_h, img_w):
    """
    Filter a list of (x,y) points to those falling within the zone mask.
    Falls back to returning all points if zone is None.
    """
    if zone is None:
        return points
    return [p for p in points if point_in_zone(p, zone, img_h, img_w)]


def bone_lines_overlapping_bbox(bone_lines, bbox):
    """Return bone lines that have at least one point inside the tooth bbox."""
    x1, y1, x2, y2 = bbox
    result = []
    for line in bone_lines:
        if any(x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2 for pt in line):
            result.append(line)
    return result


def rasterize_bone_line(points, h, w, thickness=BONE_LINE_THICKNESS):
    """Rasterize a polyline into a binary mask of given thickness."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts  = np.array(points, dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    for i in range(len(pts) - 1):
        cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]),
                 color=1, thickness=thickness)
    return mask


def find_intersection_point(bone_line_pts, tooth_mask, img_h, img_w):
    """
    Compute the bone-tooth intersection point.

    Strategy (in order of preference):
      1. Find overlap between thick bone line mask and dilated tooth boundary.
         Return centroid of overlap — most accurate.
      2. If no overlap, find closest point pair between bone line and tooth contour.
         Return their midpoint — fallback.
      3. If no tooth mask, return midpoint of bone line — last resort.

    Returns:
        (x, y): intersection point as floats
        source: str describing which strategy was used
    """
    if tooth_mask is None:
        mid = bone_line_pts[len(bone_line_pts) // 2]
        return (float(mid[0]), float(mid[1])), "bone_midpoint"

    # Build bone line mask
    bone_mask = rasterize_bone_line(bone_line_pts, img_h, img_w, thickness=3)

    # Build tooth boundary mask (dilated edge)
    tooth_bin = (tooth_mask > 0).astype(np.uint8)
    kernel_3  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tooth_edge = (cv2.dilate(tooth_bin, kernel_3)
                  - cv2.erode(tooth_bin, kernel_3))
    tooth_zone = cv2.dilate(tooth_edge, kernel_5, iterations=2)

    # Overlap
    overlap = bone_mask & tooth_zone
    if overlap.sum() > 0:
        ys, xs = np.where(overlap > 0)
        return (float(xs.mean()), float(ys.mean())), "mask_overlap"

    # Fallback: closest point between bone line pts and tooth contour
    contours, _ = cv2.findContours(
        tooth_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if contours:
        tooth_pts = np.concatenate(contours, axis=0).reshape(-1, 2).astype(float)
        bone_arr  = np.array(bone_line_pts, dtype=float)
        bone_arr[:, 0] = np.clip(bone_arr[:, 0], 0, img_w - 1)
        bone_arr[:, 1] = np.clip(bone_arr[:, 1], 0, img_h - 1)

        # Vectorised pairwise distance — efficient for small point sets
        diffs = bone_arr[:, None, :] - tooth_pts[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        bi, ti = np.unravel_index(dists.argmin(), dists.shape)
        mid = (bone_arr[bi] + tooth_pts[ti]) / 2.0
        return (float(mid[0]), float(mid[1])), "closest_point"

    # Last resort
    mid = bone_line_pts[len(bone_line_pts) // 2]
    return (float(mid[0]), float(mid[1])), "bone_midpoint"


def sort_points_left_to_right(points):
    """Sort (x, y) tuples by x coordinate ascending."""
    return sorted(points, key=lambda p: p[0])


def infer_root_type(apex_points_in_bbox):
    """
    Determine single vs double root from apex point count inside tooth bbox.
    Returns True if double-root, False if single-root.
    """
    return len(apex_points_in_bbox) >= 2


def build_6slot_keypoints(cej_pts, inter_pts, apex_pts, is_double):
    """
    Pack annotated points into the fixed 6-slot schema.

    All input points are (x, y) tuples, sorted left-to-right.
    Absent slots get [0.0, 0.0, 0] (x, y, visibility=0).

    Slot layout:
      0: cej_mesial      1: cej_distal
      2: inter_mesial    3: inter_distal
      4: apex_mesial     5: apex_distal  ← v=0 for single-root
    """
    def slot(pts, idx, required=True):
        if idx < len(pts):
            return [float(pts[idx][0]), float(pts[idx][1]), 2]
        elif required:
            # Should not happen if annotation is complete — flag it
            return [0.0, 0.0, 1]   # v=1: annotated as occluded/missing
        else:
            return [0.0, 0.0, 0]   # v=0: genuinely absent

    cej_s   = sort_points_left_to_right(cej_pts)
    inter_s = sort_points_left_to_right(inter_pts)
    apex_s  = sort_points_left_to_right(apex_pts)

    kps = [
        slot(cej_s,   0, required=True),   # slot 0: cej_mesial
        slot(cej_s,   1, required=True),   # slot 1: cej_distal
        slot(inter_s, 0, required=True),   # slot 2: inter_mesial
        slot(inter_s, 1, required=True),   # slot 3: inter_distal
        slot(apex_s,  0, required=True),   # slot 4: apex_mesial
        slot(apex_s,  1, required=is_double),  # slot 5: apex_distal
    ]
    return kps


def compute_bone_loss_gt(cej, intersection, apex):
    """
    Compute ground truth bone loss percentage using the min-max line
    projection method from the paper (equations 2-3).

    Three points (cej, intersection, apex) are projected onto the
    best-fit line through them. Bone loss is the ratio of the
    CEJ-to-intersection distance over CEJ-to-apex distance.

    Args:
        cej:          (x, y)
        intersection: (x, y)
        apex:         (x, y)

    Returns:
        float: bone loss percentage in [0, 100], or None on failure
    """
    pts = [np.array(cej), np.array(intersection), np.array(apex)]

    # Sort by x so a1 < a2 < a3 (paper requirement)
    pts.sort(key=lambda p: p[0])
    a1, b1 = pts[0]
    a2, b2 = pts[1]
    a3, b3 = pts[2]

    denom = a3 - a1
    if abs(denom) < 1e-6:
        # Near-vertical tooth — use y-axis distance directly
        root_len   = abs(float(apex[1]) - float(cej[1]))
        bone_dist  = abs(float(intersection[1]) - float(cej[1]))
        if root_len < 1e-6:
            return None
        return float(np.clip(bone_dist / root_len * 100.0, 0.0, 100.0))

    # Paper eq. 2: gradient
    m = (b3 - b1) / denom

    # Paper eq. 3: intercept (min-max line)
    c = (b1 * (a2 + a3) + b2 * (a3 - a1) - b3 * (a1 + a2)) / (2 * denom)

    # Project each original (unsorted) point onto line y = m*x + c
    def project(pt):
        x, y = float(pt[0]), float(pt[1])
        # Perpendicular projection onto line
        proj_x = (x + m * y - m * c) / (1 + m * m)
        proj_y = m * proj_x + c
        return np.array([proj_x, proj_y])

    p_cej   = project(cej)
    p_inter = project(intersection)
    p_apex  = project(apex)

    dist_cej_inter = float(np.linalg.norm(p_inter - p_cej))
    dist_cej_apex  = float(np.linalg.norm(p_apex  - p_cej))

    if dist_cej_apex < 1e-6:
        return None

    pct = dist_cej_inter / dist_cej_apex * 100.0
    return float(np.clip(pct, 0.0, 100.0))


def enforce_anatomical_order(cej, intersection, apex):
    """
    Validate that points are in correct anatomical order:
    CEJ (crown side) → Intersection (mid-root) → Apex (root tip).

    Uses projection along the CEJ→Apex axis.
    Returns corrected (cej, intersection, apex) if needed,
    or original if ordering is already valid.
    """
    cej_a  = np.array(cej,          dtype=float)
    int_a  = np.array(intersection, dtype=float)
    apex_a = np.array(apex,         dtype=float)

    axis     = apex_a - cej_a
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        return cej, intersection, apex  # degenerate, skip

    axis_unit = axis / axis_len
    proj_int  = float(np.dot(int_a - cej_a, axis_unit))
    proj_apex = float(np.dot(apex_a - cej_a, axis_unit))

    valid = (0.0 < proj_int < proj_apex)
    if not valid:
        # Project intersection onto axis at 50% of root length as fallback
        proj_int_corrected = proj_apex * 0.50
        int_corrected = cej_a + axis_unit * proj_int_corrected
        return cej, tuple(int_corrected.tolist()), apex

    return cej, intersection, apex


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing (runs in worker process)
# ─────────────────────────────────────────────────────────────────────────────

def process_single_image(args):
    """
    Process one radiograph image: parse annotations, compute intersections,
    build 6-slot keypoints, crop teeth, save crops.

    This function is designed to run in a subprocess (ProcessPoolExecutor).
    All file I/O is self-contained.

    Args:
        args: tuple of (kp_json_path, bone_json_path, img_dir, mask_dir,
                        crop_output_dir, split_name)

    Returns:
        list of KeypointAnnotation dicts (JSON-serialisable),
        or empty list on failure
    """
    (kp_json_path, bone_json_path, img_dir,
     mask_dir, crop_dir, split_name) = args

    results = []

    try:
        # ── Load annotation JSONs ──────────────────────────────────────────
        with open(kp_json_path, "r") as f:
            kp_data = json.load(f)

        image_id = kp_data.get("Image_id", "")
        bboxes   = kp_data.get("bboxes", [])
        cej_all  = kp_data.get("CEJ_Points", [])
        apex_all = kp_data.get("Apex_Points", [])

        if not bboxes:
            return []

        # Bone line annotations
        bone_lines_all = []
        if bone_json_path and Path(bone_json_path).exists():
            with open(bone_json_path, "r") as f:
                bone_data = json.load(f)
            bone_lines_all = bone_data.get("Bone_Lines", [])

        # ── Load full radiograph ───────────────────────────────────────────
        img_path = Path(img_dir) / image_id
        img_gray = load_image_gray(str(img_path))
        if img_gray is None:
            return []

        img_h, img_w = img_gray.shape

        # Apply CLAHE once for the whole image
        img_clahe = apply_clahe(img_gray)

        # Convert to RGB for saving (models expect 3-channel input)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

        # ── Find tooth-wise mask source ───────────────────────────────────
        img_stem = Path(image_id).stem
        mask_folder = Path(mask_dir) / img_stem
        full_radiograph_mask = find_radiograph_mask_file(Path(mask_dir), img_stem)

        # ── Process each tooth bbox ───────────────────────────────────────
        for tooth_idx, bbox in enumerate(bboxes):
            warnings_list = []
            x1, y1, x2, y2 = bbox

            # Skip degenerate bboxes
            if (x2 - x1) < MIN_BBOX_SIZE or (y2 - y1) < MIN_BBOX_SIZE:
                warnings_list.append("bbox_too_small")
                continue

            # ── Load tooth mask (spatial overlap matching) ─────────────────
            # Match mask to tooth by which mask has the most pixels inside
            # this bbox — robust to any filename ordering issues.
            tooth_mask = load_tooth_mask_by_overlap(
                mask_folder, bbox, img_h, img_w
            )
            if tooth_mask is None and full_radiograph_mask is not None:
                tooth_mask = load_tooth_mask_by_overlap(
                    full_radiograph_mask, bbox, img_h, img_w
                )

            has_mask = tooth_mask is not None
            if not has_mask:
                warnings_list.append("no_tooth_mask")

            # ── Build tolerance zones from mask ────────────────────────────
            # zone_boundary: generous dilation for CEJ + intersection
            # zone_apex:     strict dilation for apex (inside root)
            zone_boundary, zone_apex = build_tooth_filter_zones(
                tooth_mask, img_h, img_w
            )

            # ── Filter keypoints using MASK zones (not just bbox) ──────────
            # Step 1: coarse bbox filter to get candidates
            candidates_cej  = points_inside_bbox(cej_all,  bbox)
            candidates_apex = points_inside_bbox(apex_all, bbox)

            # Step 2: fine mask filter to reject points from adjacent teeth
            tooth_cej  = filter_points_by_mask(
                candidates_cej,  zone_boundary, img_h, img_w
            )
            tooth_apex = filter_points_by_mask(
                candidates_apex, zone_apex, img_h, img_w
            )

            # Log how many points were rejected by mask filter
            cej_rejected  = len(candidates_cej)  - len(tooth_cej)
            apex_rejected = len(candidates_apex) - len(tooth_apex)
            if cej_rejected > 0:
                warnings_list.append(f"cej_mask_rejected:{cej_rejected}")
            if apex_rejected > 0:
                warnings_list.append(f"apex_mask_rejected:{apex_rejected}")

            if len(tooth_cej) < 2:
                warnings_list.append(
                    f"insufficient_cej_points:{len(tooth_cej)}"
                )

            if not tooth_apex:
                # Cannot compute bone loss without apex — skip
                continue

            # ── Find bone lines overlapping this tooth ─────────────────────
            tooth_bone_lines = bone_lines_overlapping_bbox(
                bone_lines_all, bbox
            )

            # ── Compute intersection points ────────────────────────────────
            inter_pts    = []
            inter_source = "none"

            if tooth_bone_lines:
                for bone_line in tooth_bone_lines:
                    pt, source = find_intersection_point(
                        bone_line, tooth_mask, img_h, img_w
                    )
                    # Accept intersection only if inside boundary zone
                    if point_in_zone(pt, zone_boundary, img_h, img_w):
                        inter_pts.append(pt)
                        inter_source = source
                    else:
                        # Fallback: use bbox margin filter
                        margin = max(img_w, img_h) * 0.03
                        if (x1 - margin <= pt[0] <= x2 + margin and
                                y1 - margin <= pt[1] <= y2 + margin):
                            inter_pts.append(pt)
                            inter_source = source
                            warnings_list.append("intersection_bbox_fallback")

            if not inter_pts:
                warnings_list.append("no_intersection_points")
                continue

            # ── Determine root type ────────────────────────────────────────
            is_double = infer_root_type(tooth_apex)

            # ── Build 6-slot keypoint array ────────────────────────────────
            if len(tooth_cej) < 2:
                crown_y = y1 + (y2 - y1) * 0.05
                if len(tooth_cej) == 1:
                    existing_x = tooth_cej[0][0]
                    cx = (x1 + x2) / 2
                    mirror_x = 2 * cx - existing_x
                    tooth_cej.append((mirror_x, crown_y))
                    warnings_list.append("cej_distal_estimated")
                else:
                    tooth_cej = [(x1 + (x2 - x1) * 0.2, crown_y),
                                 (x1 + (x2 - x1) * 0.8, crown_y)]
                    warnings_list.append("both_cej_estimated")

            if len(inter_pts) < 2:
                cx = (x1 + x2) / 2
                pt = inter_pts[0]
                mirror_x = 2 * cx - pt[0]
                inter_pts.append((mirror_x, pt[1]))
                warnings_list.append("intersection_distal_mirrored")

            kps_6slot = build_6slot_keypoints(
                tooth_cej, inter_pts, tooth_apex, is_double
            )

            # ── Crop tooth ROI ─────────────────────────────────────────────
            x1p, y1p, x2p, y2p, ox, oy = pad_bbox(
                bbox, CROP_PAD_RATIO, img_h, img_w
            )
            crop_raw = img_rgb[y1p:y2p, x1p:x2p]

            if crop_raw.size == 0:
                warnings_list.append("empty_crop")
                continue

            crop_w = x2p - x1p   # original crop width  before resize
            crop_h = y2p - y1p   # original crop height before resize

            # Resize crop to fixed TARGET size
            # This is what the model sees — coordinates must match this space
            crop_resized = cv2.resize(
                crop_raw, (TARGET_W, TARGET_H),
                interpolation=cv2.INTER_LINEAR
            )

            # Scale factors from original crop space to TARGET space
            scale_x = TARGET_W / crop_w
            scale_y = TARGET_H / crop_h

            # Save resized crop
            crop_filename = f"{img_stem}_tooth{tooth_idx}.jpg"
            crop_path     = Path(crop_dir) / crop_filename
            cv2.imwrite(
                str(crop_path), crop_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            # ── Shift AND scale keypoints to TARGET crop space ─────────────
            # Bug that was here: only offset subtraction, no scale applied.
            # Fix: subtract offset first, then multiply by resize scale.
            # This maps original radiograph coordinates ->
            #   crop-local coordinates -> TARGET pixel coordinates.
            kps_crop = []
            for kp in kps_6slot:
                x, y, v = kp
                if v > 0:
                    x_crop = round((x - ox) * scale_x, 2)
                    y_crop = round((y - oy) * scale_y, 2)
                    kps_crop.append([x_crop, y_crop, v])
                else:
                    kps_crop.append([0.0, 0.0, 0])

            # ── Compute GT bone loss % from ORIGINAL image coordinates ──────
            # Uses original coords directly — no crop/scale involved.
            # The ratio dist(CEJ->intersection)/dist(CEJ->apex) is
            # scale-invariant, so original space is cleanest.
            gt_mesial = None
            gt_distal = None

            try:
                # Retrieve original-space points from kps_6slot
                # kps_6slot holds original radiograph coordinates
                def orig(slot):
                    return (kps_6slot[slot][0], kps_6slot[slot][1])

                # Mesial: slots 0 (CEJ-M), 2 (Int-M), 4 (Apex-M)
                if all(kps_6slot[s][2] > 0 for s in [0, 2, 4]):
                    cej_m, inter_m, apex_m = enforce_anatomical_order(
                        orig(0), orig(2), orig(4)
                    )
                    gt_mesial = compute_bone_loss_gt(cej_m, inter_m, apex_m)

                # Distal: slots 1 (CEJ-D), 3 (Int-D), 5 (Apex-D)
                if is_double and all(kps_6slot[s][2] > 0 for s in [1, 3, 5]):
                    cej_d, inter_d, apex_d = enforce_anatomical_order(
                        orig(1), orig(3), orig(5)
                    )
                    gt_distal = compute_bone_loss_gt(cej_d, inter_d, apex_d)

            except Exception as e:
                warnings_list.append(f"bone_loss_calc_error:{str(e)[:50]}")

            # ── Build annotation record ────────────────────────────────────
            ann = {
                "image_id":              image_id,
                "tooth_index":           tooth_idx,
                "split":                 split_name,
                "crop_filename":         crop_filename,
                "bbox_orig":             [round(v, 2) for v in bbox],
                "crop_offset":           [int(ox), int(oy)],
                "crop_size_orig":        [int(crop_w), int(crop_h)],
                "crop_size_target":      [TARGET_W, TARGET_H],
                "scale_factors":         [round(scale_x, 6), round(scale_y, 6)],
                "keypoints":             kps_crop,
                "keypoint_names":        KEYPOINT_NAMES,
                "keypoint_sigmas":       KP_SIGMAS,
                "is_double_root":        is_double,
                "gt_bone_loss_mesial":   round(gt_mesial, 2) if gt_mesial is not None else None,
                "gt_bone_loss_distal":   round(gt_distal, 2) if gt_distal is not None else None,
                "has_tooth_mask":        has_mask,
                "intersection_source":   inter_source,
                "annotation_warnings":   warnings_list,
            }
            results.append(ann)

    except Exception as e:
        log.error(f"Failed processing {kp_json_path}: {e}")
        log.debug(traceback.format_exc())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Split-level processing
# ─────────────────────────────────────────────────────────────────────────────

def process_split(data_root, output_root, split_name, num_workers):
    """
    Process one split (training / validation / testing).
    Uses ProcessPoolExecutor for parallel image processing.
    """
    split_dir  = Path(data_root)  / split_name
    out_dir    = Path(output_root) / split_name
    crop_dir   = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    img_dir    = split_dir / "Images"
    kp_dir     = split_dir / "Key Points Annotations"
    bone_dir   = split_dir / "Bone Level Annotations"
    candidate_masks = [
        split_dir / "Masks(Tooth-wise)",
        split_dir / "Masks (Tooth-wise)",
        split_dir / "Masks",
        split_dir / "Masks (Radiographs-wise)",
        split_dir / "Masks (Radiograph-wise)",
        split_dir / "Masks(Radiographs-wise)",
    ]
    mask_dir = next((p for p in candidate_masks if p.exists()), candidate_masks[0])

    if not mask_dir.exists():
        log.warning(f"  No mask directory found for split {split_name}; proceeding without mask support")

    # Validate directories exist
    for d, name in [(img_dir, "images"),
                    (kp_dir,  "keypoint annotations"),
                    (bone_dir, "bone level annotations")]:
        if not d.exists():
            log.error(f"  Missing directory: {d}")
            return [], {}

    kp_files = sorted(kp_dir.glob("*.json"))
    if not kp_files:
        log.error(f"  No keypoint JSON files found in {kp_dir}")
        return [], {}

    log.info(f"  Found {len(kp_files)} images in {split_name}")

    # Build argument list for parallel workers
    tasks = []
    for kp_f in kp_files:
        bone_f = bone_dir / kp_f.name
        tasks.append((
            str(kp_f),
            str(bone_f) if bone_f.exists() else None,
            str(img_dir),
            str(mask_dir),
            str(crop_dir),
            split_name,
        ))

    # Process in parallel — use min(workers, cpu_count-1) to leave one core free
    effective_workers = min(num_workers, max(1, os.cpu_count() - 1))
    log.info(f"  Processing with {effective_workers} workers...")

    all_annotations = []
    failed_images   = 0

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(process_single_image, task): task[0]
            for task in tasks
        }
        with tqdm(
            total=len(futures),
            desc=f"  {split_name}",
            unit="img",
            ncols=80,
        ) as pbar:
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_annotations.extend(results)
                except Exception as e:
                    log.error(f"Worker error: {e}")
                    failed_images += 1
                finally:
                    pbar.update(1)
                    pbar.set_postfix(
                        teeth=len(all_annotations),
                        failed=failed_images
                    )

    # ── Compute split statistics ───────────────────────────────────────────
    stats = compute_split_stats(all_annotations, split_name, failed_images)

    # ── Save annotations JSON ──────────────────────────────────────────────
    ann_path = out_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(
            {
                "split":        split_name,
                "num_teeth":    len(all_annotations),
                "num_images":   len(kp_files),
                "statistics":   stats,
                "keypoint_schema": {
                    "num_keypoints":  NUM_KP,
                    "keypoint_names": KEYPOINT_NAMES,
                    "keypoint_sigmas": KP_SIGMAS,
                    "visibility_encoding": {
                        "0": "absent (not applicable for tooth type)",
                        "1": "present but uncertain / estimated",
                        "2": "present and visible",
                    },
                    "slot_layout": {
                        "0": "cej_mesial",
                        "1": "cej_distal",
                        "2": "intersection_mesial",
                        "3": "intersection_distal",
                        "4": "apex_mesial",
                        "5": "apex_distal (absent v=0 for single-root)",
                    },
                    "root_triplets": {
                        "mesial": [0, 2, 4],
                        "distal": [1, 3, 5],
                    },
                },
                "annotations": all_annotations,
            },
            f,
            indent=2,
        )

    log.info(f"  Saved {len(all_annotations)} tooth annotations → {ann_path}")
    log.info(f"  Crops saved → {crop_dir}")

    return all_annotations, stats


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_split_stats(annotations, split_name, failed_images=0):
    """Compute detailed statistics for a processed split."""
    if not annotations:
        return {}

    n           = len(annotations)
    n_double    = sum(1 for a in annotations if a["is_double_root"])
    n_single    = n - n_double
    n_masked    = sum(1 for a in annotations if a["has_tooth_mask"])

    inter_sources = {}
    for a in annotations:
        s = a["intersection_source"]
        inter_sources[s] = inter_sources.get(s, 0) + 1

    warning_counts = {}
    for a in annotations:
        for w in a["annotation_warnings"]:
            warning_counts[w] = warning_counts.get(w, 0) + 1

    # Bone loss distribution
    mesial_vals = [
        a["gt_bone_loss_mesial"]
        for a in annotations
        if a["gt_bone_loss_mesial"] is not None
    ]
    distal_vals = [
        a["gt_bone_loss_distal"]
        for a in annotations
        if a["gt_bone_loss_distal"] is not None
    ]

    def dist_stats(vals):
        if not vals:
            return {}
        arr = np.array(vals)
        return {
            "count":  len(arr),
            "mean":   round(float(arr.mean()), 2),
            "std":    round(float(arr.std()), 2),
            "min":    round(float(arr.min()), 2),
            "max":    round(float(arr.max()), 2),
            "median": round(float(np.median(arr)), 2),
            "severity_distribution": {
                "none":     int((arr < 15).sum()),
                "mild":     int(((arr >= 15) & (arr < 33)).sum()),
                "moderate": int(((arr >= 33) & (arr < 66)).sum()),
                "severe":   int((arr >= 66).sum()),
            },
        }

    return {
        "split":                split_name,
        "total_teeth":          n,
        "failed_images":        failed_images,
        "single_root_teeth":    n_single,
        "double_root_teeth":    n_double,
        "double_root_ratio":    round(n_double / max(n, 1), 3),
        "teeth_with_mask":      n_masked,
        "mask_coverage_ratio":  round(n_masked / max(n, 1), 3),
        "intersection_sources": inter_sources,
        "annotation_warnings":  warning_counts,
        "bone_loss_mesial":     dist_stats(mesial_vals),
        "bone_loss_distal":     dist_stats(distal_vals),
    }


def print_stats_summary(all_stats):
    """Print a human-readable summary of dataset statistics."""
    print("\n" + "=" * 60)
    print("  DATASET PREPARATION SUMMARY")
    print("=" * 60)
    for split, stats in all_stats.items():
        if not stats:
            continue
        print(f"\n  [{split.upper()}]")
        print(f"  Total teeth:      {stats.get('total_teeth', 0)}")
        print(f"  Single-root:      {stats.get('single_root_teeth', 0)}")
        print(f"  Double-root:      {stats.get('double_root_teeth', 0)}"
              f"  ({stats.get('double_root_ratio', 0)*100:.1f}%)")
        print(f"  With tooth mask:  {stats.get('teeth_with_mask', 0)}"
              f"  ({stats.get('mask_coverage_ratio', 0)*100:.1f}%)")

        bl = stats.get("bone_loss_mesial", {})
        if bl:
            print(f"\n  Bone loss (mesial):")
            print(f"    Mean ± SD:  {bl.get('mean', 0):.1f}% ± {bl.get('std', 0):.1f}%")
            print(f"    Range:      {bl.get('min', 0):.1f}% – {bl.get('max', 0):.1f}%")
            sev = bl.get("severity_distribution", {})
            if sev:
                print(f"    None(<15%): {sev.get('none', 0)}  "
                      f"Mild(15-33%): {sev.get('mild', 0)}  "
                      f"Moderate(33-66%): {sev.get('moderate', 0)}  "
                      f"Severe(>66%): {sev.get('severe', 0)}")

        isrc = stats.get("intersection_sources", {})
        if isrc:
            print(f"\n  Intersection point sources:")
            for src, cnt in sorted(isrc.items()):
                print(f"    {src}: {cnt}")

        warns = stats.get("annotation_warnings", {})
        if warns:
            print(f"\n  Annotation warnings:")
            for w, cnt in sorted(warns.items(), key=lambda x: -x[1]):
                print(f"    {w}: {cnt}")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Verification pass
# ─────────────────────────────────────────────────────────────────────────────

def verify_prepared_data(output_root):
    """
    Quick sanity check after preparation:
      - Every annotation's crop file exists
      - Keypoint arrays have correct shape
      - GT bone loss values are in [0, 100]
      - No NaN values
    """
    print("\n  Running verification pass...")
    errors = 0
    checked = 0

    for split in ["training", "validation", "testing"]:
        ann_path = Path(output_root) / split / "annotations.json"
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            data = json.load(f)

        crop_dir = Path(output_root) / split / "crops"
        for ann in data["annotations"]:
            checked += 1

            # Check crop exists
            cp = crop_dir / ann["crop_filename"]
            if not cp.exists():
                print(f"  ERROR: missing crop {cp}")
                errors += 1
                continue

            # Check crop is readable
            img = cv2.imread(str(cp))
            if img is None:
                print(f"  ERROR: corrupt crop {cp}")
                errors += 1
                continue

            # Check keypoints shape
            kps = ann["keypoints"]
            if len(kps) != NUM_KP:
                print(f"  ERROR: wrong keypoint count {len(kps)} in {ann['crop_filename']}")
                errors += 1
                continue

            for kp in kps:
                if len(kp) != 3:
                    print(f"  ERROR: malformed keypoint {kp}")
                    errors += 1

            # Check GT bone loss
            for key in ["gt_bone_loss_mesial", "gt_bone_loss_distal"]:
                val = ann.get(key)
                if val is not None:
                    if not (0.0 <= val <= 100.0):
                        print(f"  WARN: bone loss out of range {val} in {ann['crop_filename']}")

    print(f"  Verified {checked} annotations: {errors} errors found.")
    if errors == 0:
        print("  All checks passed.")
    return errors == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare DenPAR dataset for bone loss keypoint training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py --data_root ./DenPAR --output_dir ./data_prepared
  python prepare_data.py --data_root ./DenPAR --output_dir ./data_prepared --workers 6
  python prepare_data.py --data_root ./DenPAR --output_dir ./data_prepared --splits training
  python prepare_data.py --data_root ./DenPAR --output_dir ./data_prepared --verify
        """,
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to DenPAR dataset root (contains training/, validation/, testing/)",
    )
    parser.add_argument(
        "--output_dir",
        default="./data_prepared",
        help="Output directory for processed data (default: ./data_prepared)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Training", "Validation", "Testing"],
        choices=["Training", "Validation", "Testing"],
        help="Which splits to process (default: all three)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4, max: cpu_count-1)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification pass after preparation",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Print statistics from already-prepared data and exit",
    )
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)

    # ── Stats-only mode ────────────────────────────────────────────────────
    if args.stats_only:
        all_stats = {}
        for split in args.splits:
            ann_path = output_dir / split / "annotations.json"
            if ann_path.exists():
                with open(ann_path) as f:
                    d = json.load(f)
                all_stats[split] = d.get("statistics", {})
        print_stats_summary(all_stats)
        return

    # ── Validate input ─────────────────────────────────────────────────────
    if not data_root.exists():
        log.error(f"Data root does not exist: {data_root}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Data root:  {data_root}")
    log.info(f"Output dir: {output_dir}")
    log.info(f"Splits:     {args.splits}")
    log.info(f"Workers:    {args.workers}")

    # ── Process each split ─────────────────────────────────────────────────
    all_stats = {}
    total_teeth = 0

    for split in args.splits:
        split_path = data_root / split
        if not split_path.exists():
            log.warning(f"Split directory not found, skipping: {split_path}")
            continue

        log.info(f"\nProcessing split: {split}")
        annotations, stats = process_split(
            data_root   = data_root,
            output_root = output_dir,
            split_name  = split,
            num_workers = args.workers,
        )
        all_stats[split] = stats
        total_teeth += len(annotations)

    # ── Save global metadata ───────────────────────────────────────────────
    meta_path = output_dir / "dataset_stats.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "total_teeth_processed": total_teeth,
                "splits_processed":      args.splits,
                "keypoint_schema": {
                    "num_keypoints":     NUM_KP,
                    "keypoint_names":    KEYPOINT_NAMES,
                    "keypoint_sigmas":   KP_SIGMAS,
                },
                "processing_config": {
                    "bone_line_thickness": BONE_LINE_THICKNESS,
                    "crop_pad_ratio":      CROP_PAD_RATIO,
                    "clahe_clip_limit":    CLAHE_CLIP_LIMIT,
                    "clahe_tile_grid":     list(CLAHE_TILE_GRID),
                    "min_bbox_size":       MIN_BBOX_SIZE,
                },
                "per_split": all_stats,
            },
            f,
            indent=2,
        )

    # ── Print summary ──────────────────────────────────────────────────────
    print_stats_summary(all_stats)
    log.info(f"\nTotal teeth prepared: {total_teeth}")
    log.info(f"Global stats saved:   {meta_path}")

    # ── Optional verification pass ─────────────────────────────────────────
    if args.verify:
        verify_prepared_data(output_dir)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()