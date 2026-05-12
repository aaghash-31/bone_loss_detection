"""
model.py
========
Unified Keypoint R-CNN for bone loss severity keypoint detection.

Architecture:
  - Backbone:  ResNet-50 + FPN  (COCO pretrained)
  - 4 specialised keypoint heads:
      Head 0: CEJ         (slots 0,1 — always present)
      Head 1: Intersection (slots 2,3 — always present)
      Head 2: Apex-M      (slot 4   — always present)
      Head 3: Apex-D      (slot 5   — absent for single-root)
  - Auxiliary root-type classifier (binary: single vs double root)

All heads share the same ResNet-50 + FPN backbone.
Each head is a small FCN producing a heatmap per keypoint.
Keypoint coordinates are decoded as the argmax of the heatmap.

OKS metric is computed per head and overall after each forward pass
during validation, giving you per-keypoint training progress visibility.

Usage:
    from model import BoneLossKeypointModel, decode_keypoints
    model = BoneLossKeypointModel(pretrained=True)
    model = model.cuda()
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

import numpy as np

from .dataset import INPUT_H, INPUT_W, NUM_KP, KEYPOINT_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Heatmap output size: INPUT/4 (stride 4 from ResNet P2 feature map)
# 512/4=128 height, 256/4=64 width — good balance of resolution vs speed
# No deconvolution upsampling: keeps training fast on RTX 3050 Ti
HMAP_H = INPUT_H // 4   # 128
HMAP_W = INPUT_W // 4   # 64

# FPN output channels
FPN_OUT_CHANNELS = 256

# Sigma values for OKS computation (must match prepare_data.py)
KP_SIGMAS = torch.tensor(
    [0.025, 0.025, 0.035, 0.035, 0.040, 0.040],
    dtype=torch.float32
)

# Head definitions: (head_name, keypoint_slots, loss_weight)
# CEJ upweighted to 1.4x — it's the bottleneck (stuck at OKS ~0.55)
# Intersection reduced to 1.0x — already learning well (OKS ~0.73)
# Apex heads stay at 1.5x — hardest anatomically
HEAD_DEFS = [
    ("cej",           [0, 1], 1.4),
    ("intersection",  [2, 3], 1.0),
    ("apex_mesial",   [4],    1.5),
    ("apex_distal",   [5],    1.5),
]

# Root-type classifier loss weight (auxiliary — shouldn't dominate)
ROOT_CLS_WEIGHT = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Backbone: ResNet-50 + FPN
# ─────────────────────────────────────────────────────────────────────────────

class ResNet50FPN(nn.Module):
    """
    ResNet-50 backbone with Feature Pyramid Network.
    Returns multi-scale feature maps P2, P3, P4, P5.
    We use P2 (highest resolution, stride 4) for keypoint heatmaps.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base    = resnet50(weights=weights)

        # Extract layer groups (remove avgpool and fc)
        self.layer0 = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )                                   # stride 4,  64 ch
        self.layer1 = base.layer1           # stride 4,  256 ch
        self.layer2 = base.layer2           # stride 8,  512 ch
        self.layer3 = base.layer3           # stride 16, 1024 ch
        self.layer4 = base.layer4           # stride 32, 2048 ch

        # FPN: takes C2, C3, C4, C5 and outputs P2-P5
        self.fpn = FeaturePyramidNetwork(
            in_channels_list = [256, 512, 1024, 2048],
            out_channels     = FPN_OUT_CHANNELS,
            extra_blocks     = LastLevelMaxPool(),
        )

    def forward(self, x):
        # Bottom-up pathway
        c0 = self.layer0(x)     # stride 4
        c2 = self.layer1(c0)    # stride 4,  256 ch
        c3 = self.layer2(c2)    # stride 8,  512 ch
        c4 = self.layer3(c3)    # stride 16, 1024 ch
        c5 = self.layer4(c4)    # stride 32, 2048 ch

        # FPN top-down pathway
        fpn_input  = {"0": c2, "1": c3, "2": c4, "3": c5}
        fpn_output = self.fpn(fpn_input)

        # Return P2 (stride 4, highest resolution) for heatmap decoding
        # P2 spatial size: (INPUT_H/4, INPUT_W/4) = (128, 64)
        return fpn_output["0"]


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint head (shared architecture, different weights per head)
# ─────────────────────────────────────────────────────────────────────────────

class KeypointHead(nn.Module):
    """
    Lightweight FCN head: FPN P2 features -> heatmap per keypoint slot.

    Architecture: 3x Conv(256->256, BN, ReLU) + 1x Conv(256->num_slots)
    Output size: (HMAP_H, HMAP_W) = (INPUT_H/4, INPUT_W/4) = (128, 64)

    No deconvolution — keeps memory and compute within RTX 3050 Ti budget.
    The 128x64 heatmap gives 4px resolution per image pixel which is
    sufficient for tooth landmark localisation at 512x256 input size.
    """

    def __init__(self, in_channels, num_slots, head_name):
        super().__init__()
        self.head_name = head_name
        self.num_slots = num_slots

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final 1x1 conv — no BN, no activation
        self.heatmap_conv = nn.Conv2d(256, num_slots, 1)
        nn.init.normal_(self.heatmap_conv.weight, std=0.001)
        nn.init.constant_(self.heatmap_conv.bias, 0)

    def forward(self, features):
        """
        Args:
            features: (B, FPN_OUT_CHANNELS, HMAP_H, HMAP_W)
        Returns:
            heatmaps: (B, num_slots, HMAP_H, HMAP_W) raw logits
        """
        x = self.conv_layers(features)
        return self.heatmap_conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Root-type classifier (auxiliary head)
# ─────────────────────────────────────────────────────────────────────────────

class RootTypeClassifier(nn.Module):
    """
    Binary classifier: single-root (0) vs double-root (1).
    Branches off the global average-pooled P2 features.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),   # logits for [single, double]
        )

    def forward(self, features):
        """
        Args:
            features: (B, C, H, W)
        Returns:
            logits: (B, 2)
        """
        x = self.pool(features)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class BoneLossKeypointModel(nn.Module):
    """
    Unified Keypoint R-CNN for dental bone loss keypoint detection.

    Forward pass returns:
        During training:
            {'loss': total_loss, 'loss_details': {head_name: loss_value}}

        During eval (torch.no_grad()):
            {
              'heatmaps':    {head_name: (B, n_slots, H, W)},
              'keypoints':   (B, 6, 3)  decoded (x, y, confidence),
              'root_logits': (B, 2)
            }
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = ResNet50FPN(pretrained=pretrained)

        # Build one head per HEAD_DEF entry
        self.heads = nn.ModuleDict()
        for head_name, slots, _ in HEAD_DEFS:
            self.heads[head_name] = KeypointHead(
                in_channels = FPN_OUT_CHANNELS,
                num_slots   = len(slots),
                head_name   = head_name,
            )

        self.root_classifier = RootTypeClassifier(FPN_OUT_CHANNELS)

        # Store slot assignments for decoding
        self.head_slots = {
            head_name: slots for head_name, slots, _ in HEAD_DEFS
        }
        self.head_weights = {
            head_name: weight for head_name, _, weight in HEAD_DEFS
        }

        # ── Pre-compute heatmap coordinate grids (avoid rebuilding per batch) ──
        # These are constant for the lifetime of the model.
        # Registered as buffers so they move to GPU with model.to(device).
        scale_x = HMAP_W / INPUT_W
        scale_y = HMAP_H / INPUT_H
        self.register_buffer("hmap_scale_x",
                              torch.tensor(scale_x, dtype=torch.float32))
        self.register_buffer("hmap_scale_y",
                              torch.tensor(scale_y, dtype=torch.float32))

        yy = torch.arange(HMAP_H, dtype=torch.float32)
        xx = torch.arange(HMAP_W, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        # Shape: (1, HMAP_H, HMAP_W) — broadcast-ready for batch dimension
        self.register_buffer("grid_x", grid_x.unsqueeze(0))
        self.register_buffer("grid_y", grid_y.unsqueeze(0))

        # Pre-compute decode scale factors (heatmap -> INPUT space)
        self.register_buffer("decode_scale_x",
                              torch.tensor(INPUT_W / HMAP_W, dtype=torch.float32))
        self.register_buffer("decode_scale_y",
                              torch.tensor(INPUT_H / HMAP_H, dtype=torch.float32))

    def forward(self, batch):
        """
        Args:
            batch: dict from DataLoader collate_fn containing:
                   'image'     (B, 3, H, W)
                   'keypoints' (B, 6, 3)   — only needed during training
                   'is_double_root' (B,)   — only needed during training

        Returns:
            During training: loss dict
            During eval:     prediction dict
        """
        images = batch["image"]                         # (B, 3, H, W)
        B      = images.shape[0]

        # ── Shared backbone ────────────────────────────────────────────────
        p2_features = self.backbone(images)             # (B, 256, H/4, W/4)

        # ── Root type classifier ───────────────────────────────────────────
        root_logits = self.root_classifier(p2_features) # (B, 2)

        # ── Keypoint heads ─────────────────────────────────────────────────
        all_heatmaps = {}
        for head_name, head_module in self.heads.items():
            all_heatmaps[head_name] = head_module(p2_features)

        if self.training:
            return self._compute_loss(
                all_heatmaps  = all_heatmaps,
                root_logits   = root_logits,
                gt_keypoints  = batch["keypoints"],
                gt_root_type  = batch["is_double_root"].long(),
            )
        else:
            keypoints = decode_keypoints(
                all_heatmaps,
                self.head_slots,
                B,
                decode_scale_x = self.decode_scale_x,
                decode_scale_y = self.decode_scale_y,
            )
            return {
                "heatmaps":    all_heatmaps,
                "keypoints":   keypoints,
                "root_logits": root_logits,
            }

    def _compute_loss(self, all_heatmaps, root_logits,
                      gt_keypoints, gt_root_type):
        """
        Compute weighted heatmap loss for each head.

        Uses Mean Squared Error between predicted heatmap and
        target Gaussian heatmap centred at GT keypoint location.
        Absent keypoints (v=0) contribute zero loss.
        Uncertain keypoints (v=1) contribute 0.5x loss.

        Args:
            all_heatmaps: dict {head_name: (B, n_slots, Hm, Wm)}
            root_logits:  (B, 2)
            gt_keypoints: (B, 6, 3)  float  [x, y, v] in INPUT space
            gt_root_type: (B,)       long   0=single, 1=double

        Returns:
            dict: {'loss': scalar, 'loss_details': {name: scalar}}
        """
        device   = root_logits.device
        Hm, Wm   = HMAP_H, HMAP_W
        # Use pre-computed scale factors (buffers, already on correct device)
        scale_x  = self.hmap_scale_x
        scale_y  = self.hmap_scale_y

        total_loss   = torch.tensor(0.0, device=device)
        loss_details = {}

        for head_name, slots in self.head_slots.items():
            pred_heatmaps = all_heatmaps[head_name]   # (B, n_slots, Hm, Wm)
            weight        = self.head_weights[head_name]
            head_loss     = torch.tensor(0.0, device=device)
            n_active      = 0

            for local_idx, slot_idx in enumerate(slots):
                # GT for this slot: (B, 3)  [x, y, v]
                gt_slot = gt_keypoints[:, slot_idx, :]  # (B, 3)
                vis     = gt_slot[:, 2]                  # (B,)

                # Only train on confident annotations (v=2)
                # v=1 (estimated/uncertain) excluded to avoid noisy gradients
                present_mask = (vis >= 2.0)   # (B,) bool
                if not present_mask.any():
                    continue

                # Build target Gaussian heatmaps using pre-computed grids
                gt_hmap = build_target_heatmaps(
                    gt_xy   = gt_slot[:, :2],
                    vis     = vis,
                    scale_x = self.hmap_scale_x,
                    scale_y = self.hmap_scale_y,
                    grid_x  = self.grid_x,
                    grid_y  = self.grid_y,
                )

                pred_slot = pred_heatmaps[:, local_idx, :]  # (B, Hm, Wm)

                # Visibility-weighted MSE
                # v=2 (confident):  full weight 1.0 — train on these
                # v=1 (uncertain):  weight 0.0 — EXCLUDE estimated points
                # v=0 (absent):     weight 0.0 — excluded by present_mask
                # Rationale: ~18% of CEJ distal points are estimated by
                # mirroring. These noisy targets degrade CEJ head learning.
                # Training only on confident annotations gives cleaner signal.
                vis_weight = (vis >= 2.0).float()   # 1.0 for v=2, 0.0 otherwise

                mse = F.mse_loss(
                    pred_slot,
                    gt_hmap,
                    reduction="none"
                ).mean(dim=[1, 2])                   # (B,)

                slot_loss  = (mse * vis_weight).sum() / (vis_weight.sum() + 1e-6)
                head_loss  = head_loss + slot_loss
                n_active  += 1

            if n_active > 0:
                head_loss = head_loss / n_active

            weighted_head_loss     = weight * head_loss
            total_loss             = total_loss + weighted_head_loss
            loss_details[head_name] = head_loss.detach().item()

        # ── Root type classifier loss ──────────────────────────────────────
        root_loss = F.cross_entropy(root_logits, gt_root_type)
        total_loss = total_loss + ROOT_CLS_WEIGHT * root_loss
        loss_details["root_classifier"] = root_loss.detach().item()
        loss_details["total"]           = total_loss.detach().item()

        return {"loss": total_loss, "loss_details": loss_details}


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian heatmap target builder
# ─────────────────────────────────────────────────────────────────────────────

def build_target_heatmaps(gt_xy, vis, scale_x, scale_y,
                           grid_x, grid_y, sigma=4.0):
    """
    Build Gaussian heatmap targets for a batch of keypoints.
    Fully vectorised. Uses pre-computed coordinate grids (no allocation per call).

    Args:
        gt_xy:   (B, 2)  keypoint (x, y) in INPUT image space
        vis:     (B,)    visibility flags  (0=absent, 1=uncertain, 2=visible)
        scale_x: scalar  INPUT_W -> heatmap width scale factor
        scale_y: scalar  INPUT_H -> heatmap height scale factor
        grid_x:  (1, Hm, Wm)  pre-computed x-coordinate grid
        grid_y:  (1, Hm, Wm)  pre-computed y-coordinate grid
        sigma:   Gaussian spread in heatmap pixels

    Returns:
        heatmaps: (B, Hm, Wm) float32, Gaussian peak at GT location.
                  Zero for absent keypoints (vis == 0).
    """
    B = gt_xy.shape[0]

    # Scale GT coordinates to heatmap space — (B, 1, 1) for broadcasting
    kp_x = (gt_xy[:, 0] * scale_x).view(B, 1, 1)
    kp_y = (gt_xy[:, 1] * scale_y).view(B, 1, 1)

    # grid_x / grid_y are (1, Hm, Wm) — broadcasts to (B, Hm, Wm)
    dx = grid_x - kp_x
    dy = grid_y - kp_y

    heatmaps = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

    # Zero out absent keypoints — (B, 1, 1) broadcast mask
    present  = (vis > 0).float().view(B, 1, 1)
    heatmaps = heatmaps * present

    return heatmaps   # (B, Hm, Wm)


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint decoding (heatmap argmax -> (x, y, confidence))
# ─────────────────────────────────────────────────────────────────────────────

def decode_keypoints(all_heatmaps, head_slots, B,
                      decode_scale_x=None, decode_scale_y=None):
    """
    Decode heatmaps into (x, y, confidence) keypoint predictions.
    Fully vectorised — no Python loops over batch dimension.

    Args:
        all_heatmaps:    dict {head_name: (B, n_slots, Hm, Wm)}
        head_slots:      dict {head_name: [slot_indices]}
        B:               batch size
        decode_scale_x:  scalar tensor  heatmap_w -> INPUT_W scale (pre-computed)
        decode_scale_y:  scalar tensor  heatmap_h -> INPUT_H scale (pre-computed)

    Returns:
        keypoints: (B, 6, 3) float32 [x, y, confidence] in INPUT space
    """
    device    = next(iter(all_heatmaps.values())).device
    keypoints = torch.zeros(B, NUM_KP, 3, device=device)

    # Use pre-computed scales if provided, else compute from first heatmap
    if decode_scale_x is None or decode_scale_y is None:
        first_hmap  = next(iter(all_heatmaps.values()))
        decode_scale_x = torch.tensor(INPUT_W / first_hmap.shape[3],
                                      device=device, dtype=torch.float32)
        decode_scale_y = torch.tensor(INPUT_H / first_hmap.shape[2],
                                      device=device, dtype=torch.float32)

    for head_name, heatmaps in all_heatmaps.items():
        slots = head_slots[head_name]
        Hm    = heatmaps.shape[2]
        Wm    = heatmaps.shape[3]

        for local_idx, slot_idx in enumerate(slots):
            slot_hmap = heatmaps[:, local_idx, :, :]      # (B, Hm, Wm)

            # ── Hard argmax — fully vectorised ────────────────────────────
            flat       = slot_hmap.reshape(B, -1)          # (B, Hm*Wm)
            peak_idx   = flat.argmax(dim=-1)               # (B,)
            confidence = flat.gather(1, peak_idx.unsqueeze(1)).squeeze(1)

            peak_row = (peak_idx // Wm).float()            # (B,)
            peak_col = (peak_idx %  Wm).float()            # (B,)

            # ── Vectorised sub-pixel Taylor refinement ────────────────────
            # Clamp peak indices so neighbours are always in-bounds
            row_i = peak_row.long().clamp(1, Hm - 2)      # (B,)
            col_i = peak_col.long().clamp(1, Wm - 2)      # (B,)
            b_idx = torch.arange(B, device=device)

            # Gather neighbour values for all samples simultaneously
            # dx > 0 means right neighbour is higher -> shift right +0.25
            dx = (slot_hmap[b_idx, row_i, col_i + 1]
                  - slot_hmap[b_idx, row_i, col_i - 1])   # (B,)
            dy = (slot_hmap[b_idx, row_i + 1, col_i]
                  - slot_hmap[b_idx, row_i - 1, col_i])   # (B,)

            refined_col = peak_col + 0.25 * dx.sign()
            refined_row = peak_row + 0.25 * dy.sign()

            # ── Scale to INPUT space ──────────────────────────────────────
            keypoints[:, slot_idx, 0] = refined_col * decode_scale_x
            keypoints[:, slot_idx, 1] = refined_row * decode_scale_y
            keypoints[:, slot_idx, 2] = torch.sigmoid(confidence)

    return keypoints


# ─────────────────────────────────────────────────────────────────────────────
# OKS metric (computed during validation — no gradients)
# ─────────────────────────────────────────────────────────────────────────────

class OKSMeter:
    """
    Accumulates OKS scores across batches for per-head epoch reporting.

    Usage:
        meter = OKSMeter(device)
        for batch in val_loader:
            with torch.no_grad():
                out   = model(batch)
                preds = out["keypoints"]       # (B, 6, 3)  x,y,conf
                meter.update(preds, batch["keypoints"], batch["bbox"],
                             batch["is_double_root"])
        summary = meter.compute()
        meter.pretty_print(summary)
        meter.reset()
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.sigmas = KP_SIGMAS.to(device)
        self.reset()

    def reset(self):
        self._all_oks        = []
        self._per_slot_oks   = [[] for _ in range(NUM_KP)]
        self._single_oks     = []
        self._double_oks     = []

    def update(self, pred_kps, gt_kps, bboxes, is_double_root=None):
        """
        Args:
            pred_kps:      (B, 6, 3)  predicted (x, y, confidence)
            gt_kps:        (B, 6, 3)  ground truth (x, y, visibility)
            bboxes:        (B, 4)     [x1,y1,x2,y2] in INPUT space
            is_double_root:(B,)       bool — optional, for breakdown
        """
        with torch.no_grad():
            B      = pred_kps.shape[0]
            device = pred_kps.device

            # Object scale: sqrt(bbox area)
            bw       = bboxes[:, 2] - bboxes[:, 0]
            bh       = bboxes[:, 3] - bboxes[:, 1]
            scale_sq = (bw * bh).clamp(min=1.0)    # (B,)

            vars_    = (2 * self.sigmas.to(device)) ** 2   # (6,)

            gt_xy    = gt_kps[:, :, :2]             # (B, 6, 2)
            gt_vis   = gt_kps[:, :, 2]              # (B, 6)
            visible  = (gt_vis > 0).float()

            pred_xy  = pred_kps[:, :, :2]           # (B, 6, 2)
            d_sq     = ((pred_xy - gt_xy) ** 2).sum(dim=-1)  # (B, 6)

            exponent = -d_sq / (
                2.0 * scale_sq.unsqueeze(1) * vars_.unsqueeze(0)
            )
            per_slot_oks = torch.exp(exponent)       # (B, 6)

            # Overall OKS per sample (average over visible slots)
            n_vis    = visible.sum(dim=1).clamp(min=1e-6)
            oks_per  = (per_slot_oks * visible).sum(dim=1) / n_vis  # (B,)

            self._all_oks.extend(oks_per.cpu().numpy().tolist())

            # Per-slot OKS
            for s in range(NUM_KP):
                vis_s = visible[:, s]
                if vis_s.sum() > 0:
                    slot_oks = (per_slot_oks[:, s] * vis_s).sum() / vis_s.sum()
                    self._per_slot_oks[s].append(slot_oks.cpu().item())

            # Single vs double root breakdown
            if is_double_root is not None:
                dbl     = is_double_root.bool().cpu().numpy()
                oks_np  = oks_per.cpu().numpy()
                self._single_oks.extend(oks_np[~dbl].tolist())
                self._double_oks.extend(oks_np[dbl].tolist())

    def compute(self):
        """Return full OKS summary dictionary."""
        if not self._all_oks:
            return {}

        arr = np.array(self._all_oks)

        result = {
            "OKS_mean":  float(np.mean(arr)),
            "AP50":      float((arr >= 0.50).mean()),
            "AP75":      float((arr >= 0.75).mean()),
            "mAP50_95":  float(np.mean([(arr >= t).mean()
                                         for t in np.arange(0.5, 1.0, 0.05)])),
        }

        result["per_keypoint"] = {}
        for s, name in enumerate(KEYPOINT_NAMES):
            vals = self._per_slot_oks[s]
            if vals:
                result["per_keypoint"][name] = float(np.mean(vals))

        if self._single_oks:
            result["OKS_single_root"] = float(np.mean(self._single_oks))
        if self._double_oks:
            result["OKS_double_root"] = float(np.mean(self._double_oks))

        return result

    def pretty_print(self, result=None, prefix="[Val]"):
        """Print a human-readable OKS summary for training logs."""
        if result is None:
            result = self.compute()
        if not result:
            print(f"{prefix} No OKS data.")
            return

        print(f"\n{prefix} OKS Summary:")
        print(f"  Overall  — OKS: {result.get('OKS_mean', 0):.4f}  "
              f"AP50: {result.get('AP50', 0):.4f}  "
              f"AP75: {result.get('AP75', 0):.4f}  "
              f"mAP50:95: {result.get('mAP50_95', 0):.4f}")

        pk = result.get("per_keypoint", {})
        if pk:
            print("  Per keypoint:")
            for name, val in pk.items():
                bar_len = int(val * 20)
                bar     = "#" * bar_len + "-" * (20 - bar_len)
                print(f"    {name:<28} [{bar}] {val:.4f}")

        sr = result.get("OKS_single_root")
        dr = result.get("OKS_double_root")
        if sr is not None and dr is not None:
            gap = sr - dr
            print(f"  Root type — Single: {sr:.4f}  "
                  f"Double: {dr:.4f}  Gap: {gap:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick model verification
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nBuilding model...")
    model = BoneLossKeypointModel(pretrained=True).to(device)

    # Count parameters
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Dummy forward pass — training mode
    B = 4
    dummy_batch = {
        "image":          torch.randn(B, 3, INPUT_H, INPUT_W).to(device),
        "keypoints":      torch.rand(B, NUM_KP, 3).to(device),
        "is_double_root": torch.randint(0, 2, (B,)).bool().to(device),
        "bbox":           torch.tensor([[0, 0, INPUT_W, INPUT_H]] * B,
                                        dtype=torch.float32).to(device),
    }
    # Set some keypoints as absent for realism
    dummy_batch["keypoints"][:, 5, 2] = 0.0    # apex_distal absent for half

    print(f"\nForward pass (train, batch={B})...")
    model.train()
    t0  = time.time()
    out = model(dummy_batch)
    t1  = time.time()
    print(f"  Loss:         {out['loss'].item():.4f}")
    print(f"  Loss details: {out['loss_details']}")
    print(f"  Time:         {(t1-t0)*1000:.1f}ms")

    # Eval mode
    print(f"\nForward pass (eval, batch={B})...")
    model.eval()
    with torch.no_grad():
        t0  = time.time()
        out = model(dummy_batch)
        t1  = time.time()
    print(f"  Keypoints shape: {out['keypoints'].shape}")
    print(f"  Root logits:     {out['root_logits'].shape}")
    print(f"  Time:            {(t1-t0)*1000:.1f}ms")

    # Test OKS meter
    print("\nOKS meter test...")
    meter = OKSMeter(device=device)
    gt_kps = dummy_batch["keypoints"].clone()
    gt_kps[:, :, 2] = 2.0   # all visible for test
    meter.update(out["keypoints"], gt_kps, dummy_batch["bbox"],
                 dummy_batch["is_double_root"])
    summary = meter.compute()
    meter.pretty_print(summary, prefix="[Test]")

    # VRAM usage
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nVRAM used: {used:.2f} GB  peak: {peak:.2f} GB")
        print(f"RTX 3050 Ti budget: 4.0 GB  "
              f"{'OK' if peak < 3.5 else 'WARNING: near limit'}")

    print("\nModel verification complete.")