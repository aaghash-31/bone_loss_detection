# 🦷 AI-Assisted Alveolar Bone Loss Detection

Automated detection and quantification of alveolar bone loss from dental IOPA radiographs using deep learning keypoint detection.



## 📌 Overview
This repository implements an end-to-end AI pipeline that takes a full intraoral periapical (IOPA) radiograph and automatically:

Detects each tooth using YOLOv8x
Predicts 6 anatomical keypoints per tooth using a unified Keypoint R-CNN model
Computes bone loss percentage per tooth using validated geometric strategies
Annotates the original radiograph with keypoints and bone loss values

The model achieves OKS 0.720, AP50 0.891, ICC 0.510, and MAE 10.92% on the DenPAR validation set.


## 🧠 Clinical Background
Alveolar bone loss is the key diagnostic indicator of periodontitis — a chronic inflammatory disease affecting ~19% of adults globally.

Traditionally, clinicians measure bone loss manually on radiographs by identifying three anatomical landmarks:

| Landmark | Description |
|----------|-------------|
| CEJ (Cemento-Enamel Junction) | Crown-root boundary — the healthy baseline |
| Intersection | Where alveolar bone crest meets tooth surface |
| Apex | Root tip — defines total root length |

Bone Loss % = `dist(CEJ → Intersection) / dist(CEJ → Apex) × 100`

This project automates that measurement using deep learning.

## 🗂️ Repository Structure

```
├── prepare_data.py                 # Dataset preparation with mask-based keypoint filtering
├── dataset.py                      # PyTorch Dataset and DataLoader
├── model.py                        # Unified Keypoint R-CNN model + OKS metric
├── train.py                        # Training loop with evaluation
├── visualize_annotations.py        # Visualize GT / predicted keypoints on crops
├── bone_loss_pipeline.py           # Full RVG inference pipeline (end-to-end)
├── convert_coco_to_yolo.py         # Convert DenPAR annotations to YOLO format
├── train_yolo_tooth_detection.py   # Train YOLOv8x tooth detector
└── README.md
```



## 📦 Dataset
This project uses the DenPAR dataset: 1000 expert-annotated IOPA radiographs from the University of Peradeniya, Sri Lanka.

Dataset split:

- Training: 650 images, 1,475 teeth
- Validation: 150 images, 335 teeth
- Testing: 200 images, 451 teeth

Download:

- https://doi.org/10.5281/zenodo.14181645

Expected folder structure after download:

```
DenPAR Radiographs Dataset/
└── Dataset/
    ├── training/
    │   ├── images/
    │   ├── key point annotations/
    │   ├── bone level annotations/
    │   ├── masks(radiograph wise)/
    │   └── masks(tooth wise)/
    ├── validation/
    │   └── (same structure)
    └── testing/
        └── (same structure)
```

## ⚙️ Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU (recommended 4GB+ VRAM)
- CUDA 11.8+

### Setup

```bash
# Clone the repository
git clone https://github.com/aaghash-31/bone-loss-detection.git
cd bone-loss-detection

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy tqdm pillow
```


## 🚀 Quick Start

### Step 1 — Prepare the data

```bash
python prepare_data.py \
  --data_root "path/to/DenPAR Radiographs Dataset/Dataset" \
  --output_dir ./data_prepared \
  --workers 4
```

This will:

- Extract per-tooth blobs from radiograph-wise masks using connected components.
- Filter keypoints by mask zones to prevent cross-tooth leakage.
- Resize all crops to `256×512` with correctly scaled coordinates.
- Save annotations to `data_prepared/{split}/annotations.json`.

Verify the output:

```bash
python prepare_data.py \
  --data_root "path/to/Dataset" \
  --output_dir ./data_prepared \
  --stats_only
```

### Step 2 — Verify ground truth annotations

```bash
python visualize_annotations.py \
  --data_dir ./data_prepared \
  --mode gt \
  --split training \
  --n 30
```

Check the visualization output in `visualizations/gt/training/`:

- 🟢 Green dots: CEJ at the crown-root junction.
- 🔵 Blue dots: Intersection near the bone line.
- 🔴 Red dots: Apex at the root tip.

### Step 3 — Train the keypoint model

```bash
python train.py \
  --data_dir ./data_prepared \
  --output_dir ./runs/exp1 \
  --batch_size 4 \
  --lr 0.0001 \
  --epochs 100
```

> Windows tip: If training is slow or hangs, set `--num_workers 0`.

Checkpoints saved:

- `runs/exp1/best_oks.pth` — best validation OKS
- `runs/exp1/best_icc.pth` — best ICC

### Step 4 — Visualize predictions

```bash
python visualize_annotations.py \
  --data_dir ./data_prepared \
  --mode pred \
  --split validation \
  --checkpoint ./runs/exp1/best_oks.pth \
  --n 30
```

```bash
python visualize_annotations.py \
  --data_dir ./data_prepared \
  --mode both \
  --split validation \
  --checkpoint ./runs/exp1/best_oks.pth \
  --n 30
```

### Step 5 — Run full inference on an RVG image

```bash
python bone_loss_pipeline.py \
  --image path/to/rvg.jpg \
  --yolo_checkpoint path/to/tooth_detector.pt \
  --bone_checkpoint ./runs/exp1/best_oks.pth \
  --output annotated_result.jpg
```

## 🏗️ Model Architecture
The keypoint detection model is a Unified Keypoint R-CNN with:

- Backbone: ResNet-50 + Feature Pyramid Network (pretrained on COCO).
- Four specialized keypoint heads.
- Auxiliary binary root-type classifier for single vs. double roots.
- Decoder: hard argmax + Taylor sub-pixel refinement.

### Keypoint schema

Each tooth uses a fixed 6-slot schema regardless of root type:

| Slot | Name                  | Description |
|------|-----------------------|-------------|
| 0    | `cej_mesial`          | always present |
| 1    | `cej_distal`          | always present |
| 2    | `intersection_mesial` | always present |
| 3    | `intersection_distal` | always present |
| 4    | `apex_mesial`         | always present |
| 5    | `apex_distal`         | absent for single-root, present for double-root |

Visibility values:

- `0` = absent
- `1` = estimated / uncertain
- `2` = confident annotation

## 📊 Evaluation Metrics

| Metric   | Description                                                 | Goal               |
|----------|-------------------------------------------------------------|--------------------|
| OKS      | Object Keypoint Similarity                                  | higher is better   |
| AP50     | % of teeth with OKS ≥ 0.50                                  | higher is better   |
| AP75     | % of teeth with OKS ≥ 0.75                                  | higher is better   |
| mAP50:95 | Mean AP across OKS thresholds 0.50 to 0.95                   | higher is better   |
| ICC      | Intraclass Correlation Coefficient for bone loss reliability | > 0.75 = good      |
| MAE      | Mean absolute error in percentage points                    | lower is better    |

### Best results (Epoch 85)

| Metric   | Value  |
|----------|--------|
| OKS Mean | 0.7197 |
| AP50     | 0.891  |
| AP75     | 0.513  |
| mAP50:95 | 0.521  |
| ICC      | 0.510  |
| MAE      | 10.92% |

### Per-keypoint OKS at best epoch

| Keypoint            | OKS   |
|---------------------|-------|
| Intersection Mesial | 0.769 |
| Intersection Distal | 0.761 |
| Apex Mesial         | 0.741 |
| CEJ Distal          | 0.684 |
| CEJ Mesial          | 0.681 |
| Apex Distal         | 0.634 |


## 🔑 Key Design Decisions

1. **Radiograph-wise mask + connected components**
   - Early versions used bounding-box-only filtering, causing cross-tooth leakage.
   - Fix: extract individual tooth blobs from the radiograph mask and filter keypoints by mask overlap.
   - Mask zones:
     - CEJ and Intersection: ±25 px dilation
     - Apex: ±10 px dilation

2. **Correct coordinate scaling**
   - Bug: only crop offset was subtracted, but resize scale was not applied.
   - Correct formula:

     ```python
     x_crop = (x_original - offset_x) * scale_x
     y_crop = (y_original - offset_y) * scale_y
     ```

3. **Hard argmax vs soft argmax**
   - Soft argmax was initially used for sub-pixel accuracy.
   - On near-uniform heatmaps, it can collapse to the crop center.
   - Hard argmax is more stable, with Taylor sub-pixel refinement added afterward.

4. **Bone loss computation is post-hoc**
   - Bone loss percentage is computed after inference, keeping the model objective focused on OKS.
   - Strategies:
     - Direct ratio: `dist(CEJ→Intersection) / dist(CEJ→Apex) × 100`
     - Min-max projection: a tilt-aware geometric alternative



## 🦷 Bone Loss Severity Classification

| Severity  | Bone Loss Range | Clinical Meaning       |
|-----------|------------------|------------------------|
| None      | < 15%            | Healthy                |
| Mild      | 15% – 33%        | Early periodontitis    |
| Moderate  | 33% – 66%        | Moderate periodontitis |
| Severe    | > 66%            | Severe periodontitis   |

## 🔧 Training Configuration Reference

| Parameter    | Value                              |
|--------------|------------------------------------|
| Optimizer    | Adam                               |
| Learning Rate| 0.0001                             |
| Scheduler    | CosineAnnealingLR (T_max=100)      |
| Batch Size   | 4                                  |
| Epochs       | 100                                |
| Input Size   | 256 × 512 (W × H)                  |
| Heatmap Size | 64 × 128 (stride 4)                |
| Gaussian σ   | 4.0 px                             |
| GPU          | NVIDIA RTX 3050 Ti (4.3GB VRAM)    |
| Peak VRAM    | 2.26GB                             |
| Epoch Time   | ~163 seconds                       |

### Resume training from checkpoint

```bash
python train.py \
  --data_dir ./data_prepared \
  --output_dir ./runs/exp1 \
  --batch_size 4 \
  --resume ./runs/exp1/latest.pth
```

```bash
python train.py \
  --data_dir ./data_prepared \
  --output_dir ./runs/exp2 \
  --batch_size 4 \
  --lr 0.00003 \
  --resume ./runs/exp1/best_oks.pth \
  --weights_only
```

## 🦷 Tooth Detection (YOLOv8x)

The full inference pipeline requires a separate YOLOv8x tooth detection model. To train it on DenPAR:

```bash
python convert_coco_to_yolo.py
```

```bash
python train_yolo_tooth_detection.py \
  --data_yaml ./yolo_dataset/data.yaml \
  --output_dir ./yolo_runs \
  --epochs 50
```

The best tooth detection checkpoint will be at `yolo_runs/tooth_detection/weights/best.pt`.

## 🐛 Common Issues

| Issue | Cause | Fix |
|------|-------|-----|
| `no_tooth_mask` warning for all teeth | Wrong folder name or missing mask | Ensure `masks(radiograph wise)/` exists in each split |
| Epoch time > 10 minutes (Windows) | DataLoader multiprocessing deadlock | Use `--num_workers 0` |
| OKS flat at 0.01 for many epochs | Wrong decoder (soft argmax) or wrong data | Use updated `model.py` with hard argmax |
| Keypoints misplaced on crops | Coordinate scaling bug | Delete `data_prepared/` and rerun `prepare_data.py` |
| CUDA out of memory | Batch size too large | Reduce `--batch_size` to 2 |
| ICC always negative | Old ICC formula | Use current `train.py` with ICC(2,1) |

## 📁 Output Files
After training, your `runs/exp1/` folder will contain:

```
runs/exp1/
├── best_oks.pth       # Checkpoint with best validation OKS
├── best_icc.pth       # Checkpoint with best ICC
├── latest.pth         # Most recent epoch checkpoint
├── training_log.json  # Full per-epoch metrics (OKS, ICC, MAE, per-keypoint)
└── training.log       # Text log of training progress
```

## 📖 References

- Wimalasiri C. et al., "AI-assisted radiographic analysis in detecting alveolar bone loss severity and patterns", arXiv:2506.20522, 2025
- Rasnayaka S. et al., "DenPAR: Annotated Intra-oral Periapical Radiographs Dataset", Zenodo, DOI: 10.5281/zenodo.14181645, 2024
- He K. et al., "Mask R-CNN", IEEE ICCV, 2017
- He K. et al., "Deep Residual Learning for Image Recognition", CVPR, 2016
- Lin T.Y. et al., "Feature Pyramid Networks for Object Detection", CVPR, 2017
- Koo T.K. and Li M.Y., "A Guideline of Selecting and Reporting Intraclass Correlation Coefficients", Journal of Chiropractic Medicine, 2016




## 📄 License
This project is released for academic and research use. The DenPAR dataset is subject to its own license terms — see Zenodo for details.


## 👤 Author
Aaghash A S
