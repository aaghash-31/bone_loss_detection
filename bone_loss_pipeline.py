"""
bone_loss_pipeline.py
======================
Pipeline for bone loss assessment on full RVG images with multiple teeth.

This script:
1. Loads a pre-trained keypoint detection model
2. Takes a full RVG image and tooth bounding boxes
3. For each tooth:
   - Crops the tooth ROI
   - Predicts keypoints (CEJ, intersection, apex)
   - Computes bone loss using multiple strategies
4. Annotates the original image with keypoints and bone loss values
5. Saves the annotated result

Bone loss computation strategies:
- Direct: Euclidean distance between CEJ and intersection points
- Best fit: Projection of intersection along the CEJ-apex line
- Min/Max/Average across mesial and distal sides

Usage:
    python bone_loss_pipeline.py --image path/to/rvg.jpg --bboxes '[[x1,y1,x2,y2], ...]' --checkpoint path/to/model.pth --output annotated.jpg

Requirements:
- PyTorch, OpenCV, NumPy
- Pre-trained model checkpoint
- Tooth bounding boxes (can be obtained from tooth detection model)
"""

import argparse
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Import from our project
from bone_loss.model import BoneLossKeypointModel, decode_keypoints
from bone_loss.dataset import INPUT_H, INPUT_W, IMG_MEAN, IMG_STD, NUM_KP

# Color scheme for visualization (same as visualize_annotations.py)
COLOR_CEJ = (50, 200, 50)      # green
COLOR_INTERSECTION = (200, 80, 80)    # blue
COLOR_APEX = (50, 80, 220)     # red
COLOR_TEXT = (255, 255, 255)   # white for text
COLOR_BG = (0, 0, 0)           # black background for text

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1

def detect_teeth_yolo(image_path, yolo_model, conf_threshold=0.5):
    """
    Detect teeth in the full RVG image using YOLO model.

    Returns: list of bboxes [[x1,y1,x2,y2], ...]
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Run YOLO detection
    results = yolo_model(img, conf=conf_threshold)

    bboxes = []
    for result in results:
        for box in result.boxes:
            # Get bbox coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])

    print(f"Detected {len(bboxes)} teeth")
    return bboxes

def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = BoneLossKeypointModel(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded from epoch {ckpt.get('epoch', '?')}")
    return model

def preprocess_crop(crop_img):
    """Preprocess crop image for model input."""
    # Resize to model input size
    img = cv2.resize(crop_img, (INPUT_W, INPUT_H))

    # Convert to tensor and normalize
    img = img.astype(np.float32) / 255.0
    img = (img - IMG_MEAN) / IMG_STD
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)  # Ensure float32

    return img

def predict_keypoints(model, crop_img, device):
    """Predict keypoints for a single tooth crop."""
    img_tensor = preprocess_crop(crop_img).to(device)

    with torch.no_grad():
        batch = {"image": img_tensor}
        output = model(batch)
        keypoints = output["keypoints"][0]  # (6, 3) - x, y, confidence

    return keypoints.cpu().numpy()

def compute_bone_loss_direct(cej_points, inter_points, apex_points):
    """
    Compute bone loss as direct Euclidean ratio between CEJ and intersection, expressed as % of root length.

    Returns: (mesial_pct, distal_pct)
    """
    mesial_root_len = np.linalg.norm(apex_points[0] - cej_points[0])
    distal_root_len = np.linalg.norm(apex_points[1] - cej_points[1])

    mesial_pct = 0.0 if mesial_root_len < 1e-6 else (np.linalg.norm(cej_points[0] - inter_points[0]) / mesial_root_len) * 100.0
    distal_pct = 0.0 if distal_root_len < 1e-6 else (np.linalg.norm(cej_points[1] - inter_points[1]) / distal_root_len) * 100.0

    return mesial_pct, distal_pct

def compute_bone_loss_best_fit(cej_points, inter_points, apex_points):
    """
    Compute bone loss using best fit line approach.
    Fit a line to CEJ and apex points, then measure projection of intersection along that line.

    Returns: (mesial_pct, distal_pct)
    """
    losses = []

    for side in [0, 1]:  # mesial, distal
        cej = cej_points[side]
        inter = inter_points[side]
        apex = apex_points[side]

        # Points for line fitting: CEJ and apex
        points = np.array([cej, apex])

        # Fit line: ax + by + c = 0
        if np.allclose(cej, apex):
            # Degenerate case: CEJ and apex coincide
            distance = 0.0
        else:
            # Line from CEJ to apex
            dx = apex[0] - cej[0]
            dy = apex[1] - cej[1]
            root_len = np.sqrt(dx * dx + dy * dy)

            # Project intersection onto the CEJ->apex line
            inter_vec = inter - cej
            root_len_sq = dx * dx + dy * dy
            projected = (inter_vec[0] * dx + inter_vec[1] * dy) / root_len_sq
            projected = np.clip(projected, 0.0, 1.0)
            distance = projected * 100.0

        losses.append(distance)

    return losses[0], losses[1]

def compute_bone_loss_all_strategies(keypoints):
    """
    Compute bone loss using all strategies.

    keypoints: (6, 3) array [x, y, confidence]

    Returns: dict with all strategies
    """
    # Extract points (only x,y, ignore confidence for computation)
    cej_points = keypoints[0:2, :2]    # slots 0,1
    inter_points = keypoints[2:4, :2]  # slots 2,3
    apex_points = keypoints[4:6, :2]   # slots 4,5

    results = {}

    # Direct distance
    results['direct'] = compute_bone_loss_direct(cej_points, inter_points, apex_points)

    # Best fit line
    results['best_fit'] = compute_bone_loss_best_fit(cej_points, inter_points, apex_points)

    return results

def rects_overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def draw_bone_loss_text(img, bbox, bone_loss_results, placed_rects):
    """
    Draw bone loss percentages on the image for all calculated methods.

    bbox: [x1, y1, x2, y2]
    bone_loss_results: dict from compute_bone_loss_all_strategies
    placed_rects: list of already drawn text box rects
    """
    x1, y1, x2, y2 = bbox

    mesial_direct, distal_direct = bone_loss_results['direct']
    mesial_fit, distal_fit = bone_loss_results['best_fit']

    avg_direct = (mesial_direct + distal_direct) / 2.0
    avg_fit = (mesial_fit + distal_fit) / 2.0

    text = (
        f"BF%: M:{mesial_fit:.1f}% D:{distal_fit:.1f}% Avg:{avg_fit:.1f}%  "
        f"Dir%: M:{mesial_direct:.1f}% D:{distal_direct:.1f}% Avg:{avg_direct:.1f}%"
    )
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    margin = 6
    box_width = text_width + 2 * margin
    box_height = text_height + 2 * margin

    rect_left = max(0, min(img.shape[1] - box_width, x1))
    preferred_top = y1 - box_height
    preferred_bottom = y2

    candidates = []
    if preferred_top >= 0:
        candidates.append((rect_left, preferred_top, rect_left + box_width, preferred_top + box_height))
    if preferred_bottom + box_height <= img.shape[0]:
        candidates.append((rect_left, preferred_bottom, rect_left + box_width, preferred_bottom + box_height))

    # If top and bottom both collide, try shifting vertically below bbox.
    if not candidates:
        y = max(0, min(img.shape[0] - box_height, preferred_bottom))
        candidates.append((rect_left, y, rect_left + box_width, y + box_height))

    chosen = None
    for candidate in candidates:
        if not any(rects_overlap(candidate, existing) for existing in placed_rects):
            chosen = candidate
            break

    # If still colliding, shift downwards until free, then upwards if needed.
    if chosen is None:
        y = preferred_bottom
        while y + box_height <= img.shape[0]:
            candidate = (rect_left, y, rect_left + box_width, y + box_height)
            if not any(rects_overlap(candidate, existing) for existing in placed_rects):
                chosen = candidate
                break
            y += box_height + margin

    if chosen is None:
        y = max(0, preferred_top)
        while y >= 0:
            candidate = (rect_left, y, rect_left + box_width, y + box_height)
            if not any(rects_overlap(candidate, existing) for existing in placed_rects):
                chosen = candidate
                break
            y -= box_height + margin

    if chosen is None:
        chosen = candidates[0]

    rect_left, rect_top, rect_right, rect_bottom = chosen
    placed_rects.append(chosen)

    cv2.rectangle(img, (rect_left, rect_top), (rect_right, rect_bottom), COLOR_BG, -1)
    text_pos = (rect_left + margin, rect_bottom - margin - baseline)
    cv2.putText(img, text, text_pos, FONT, FONT_SCALE, COLOR_TEXT, FONT_THICKNESS)
def draw_keypoints(img, keypoints, bbox):
    """
    Draw keypoints on the image.

    keypoints: (6, 3) array with coordinates already in original image space
    bbox: [x1, y1, x2, y2] in original image coordinates
    """
    for slot in range(6):
        x, y, conf = keypoints[slot]
        if conf > 0:  # Only draw visible keypoints
            orig_x = int(x)
            orig_y = int(y)

            # Choose color based on keypoint type
            if slot in [0, 1]:  # CEJ
                color = COLOR_CEJ
                label = f"CEJ{'-M' if slot == 0 else '-D'}"
            elif slot in [2, 3]:  # Intersection
                color = COLOR_INTERSECTION
                label = f"Int{'-M' if slot == 2 else '-D'}"
            else:  # Apex
                color = COLOR_APEX
                label = f"Apx{'-M' if slot == 4 else '-D'}"

            # Draw point
            if slot % 2 == 0:  # Mesial: filled
                cv2.circle(img, (orig_x, orig_y), 4, color, -1)
            else:  # Distal: hollow
                cv2.circle(img, (orig_x, orig_y), 4, color, 2)

            # Draw label
            cv2.putText(img, label, (orig_x + 5, orig_y - 5), FONT, 0.4, color, 1)

def draw_bbox(img, bbox):
    """Draw detected tooth bounding box."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle

def process_image(image_path, yolo_checkpoint, bone_checkpoint, output_path, conf_threshold=0.5):
    """Main processing pipeline."""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load YOLO model for tooth detection
    print("Loading YOLO tooth detection model...")
    yolo_model = YOLO(yolo_checkpoint)

    # Detect teeth
    bboxes = detect_teeth_yolo(image_path, yolo_model, conf_threshold)

    # Load bone loss model
    bone_model = load_model(bone_checkpoint, device)

    placed_rects = []
    # Process each detected tooth
    for i, bbox in enumerate(bboxes):
        print(f"Processing tooth {i+1}/{len(bboxes)}")

        x1, y1, x2, y2 = bbox

        # Crop tooth ROI with padding
        pad_ratio = 0.1
        img_h, img_w = img.shape[:2]
        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * pad_ratio)
        py = int(bh * pad_ratio)

        crop_x1 = max(0, x1 - px)
        crop_y1 = max(0, y1 - py)
        crop_x2 = min(img_w, x2 + px)
        crop_y2 = min(img_h, y2 + py)

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Predict keypoints
        keypoints = predict_keypoints(bone_model, crop, device)

        # Scale keypoints from crop space to original image coordinates
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        scale_x = crop_width / INPUT_W
        scale_y = crop_height / INPUT_H

        keypoints[:, 0] = crop_x1 + keypoints[:, 0] * scale_x
        keypoints[:, 1] = crop_y1 + keypoints[:, 1] * scale_y

        # Compute bone loss
        bone_loss_results = compute_bone_loss_all_strategies(keypoints)

        # Draw on original image
        draw_bbox(img, bbox)  # Draw detected bbox
        draw_keypoints(img, keypoints, bbox)
        draw_bone_loss_text(img, bbox, bone_loss_results, placed_rects)

    # Save result
    cv2.imwrite(str(output_path), img)
    print(f"Annotated image saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Complete bone loss assessment pipeline for RVG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bone_loss_pipeline.py --image rvg.jpg --yolo_checkpoint ./yolo_runs/tooth_detection/weights/best.pt --bone_checkpoint ./runs/exp7/best_oks.pth --output annotated.jpg

  # With custom confidence threshold
  python bone_loss_pipeline.py --image rvg.jpg --yolo_checkpoint yolo_model.pt --bone_checkpoint bone_model.pth --output result.jpg --conf 0.3
        """
    )
    parser.add_argument("--image", required=True, help="Path to input RVG image")
    parser.add_argument("--yolo_checkpoint", required=True, help="Path to YOLO tooth detection model")
    parser.add_argument("--bone_checkpoint", required=True, help="Path to bone loss keypoint model")
    parser.add_argument("--output", required=True, help="Path to save annotated output image")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold (default: 0.5)")

    args = parser.parse_args()

    # Run pipeline
    process_image(args.image, args.yolo_checkpoint, args.bone_checkpoint, args.output, args.conf)

if __name__ == "__main__":
    main()