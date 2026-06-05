🦷 AI-Assisted Alveolar Bone Loss Detection

Automated detection and quantification of alveolar bone loss from dental IOPA radiographs using deep learning keypoint detection.


📌 Overview
This repository implements an end-to-end AI pipeline that takes a full intraoral periapical (IOPA) radiograph and automatically:

Detects each tooth using YOLOv8x
Predicts 6 anatomical keypoints per tooth using a unified Keypoint R-CNN model
Computes bone loss percentage per tooth using validated geometric strategies
Annotates the original radiograph with keypoints and bone loss values

The model achieves OKS 0.720, AP50 0.891, ICC 0.510, and MAE 10.92% on the DenPAR validation set.

🧠 Clinical Background
Alveolar bone loss is the key diagnostic indicator of periodontitis — a chronic inflammatory disease affecting ~19% of adults globally. Traditionally, clinicians measure bone loss manually on radiographs by identifying three anatomical landmarks:
LandmarkDescriptionCEJ (Cemento-Enamel Junction)Crown-root boundary — the healthy baselineIntersectionWhere alveolar bone crest meets tooth surfaceApexRoot tip — defines total root length
Bone Loss % = dist(CEJ → Intersection) / dist(CEJ → Apex) × 100
This project automates that measurement using deep learning.

🗂️ Repository Structure
├── prepare_data.py           # Dataset preparation with mask-based keypoint filtering
├── dataset.py                # PyTorch Dataset and DataLoader
├── model.py                  # Unified Keypoint R-CNN model + OKS metric
├── train.py                  # Training loop with evaluation
├── visualize_annotations.py  # Visualize GT / predicted keypoints on crops
├── bone_loss_pipeline.py     # Full RVG inference pipeline (end-to-end)
├── convert_coco_to_yolo.py   # Convert DenPAR annotations to YOLO format
├── train_yolo_tooth_detection.py  # Train YOLOv8x tooth detector
└── README.md

📦 Dataset
This project uses the DenPAR dataset — 1000 expert-annotated IOPA radiographs from University of Peradeniya, Sri Lanka.
SplitImagesTeethTraining6501,475Validation150335Testing200451
Download: https://doi.org/10.5281/zenodo.14181645
Expected Folder Structure After Download
DenPAR Radiographs Dataset/
└── Dataset/
    ├── training/
    │   ├── images/
    │   ├── key point annotations/       ← JSON files with CEJ and Apex points
    │   ├── bone level annotations/      ← JSON files with bone line polylines
    │   ├── masks(radiograph wise)/      ← One PNG mask per radiograph (all teeth)
    │   └── masks(tooth wise)/           ← Per-tooth PNG masks in subfolders
    ├── validation/
    │   └── (same structure)
    └── testing/
        └── (same structure)

⚙️ Installation
Requirements

Python 3.9+
CUDA-capable GPU (minimum 4GB VRAM recommended)
CUDA 11.8+

Setup
bash# Clone the repository
git clone https://github.com/aaghash-31/bone-loss-detection.git
cd bone-loss-detection

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy tqdm pillow

🚀 Quick Start
Step 1 — Prepare the Data
bashpython prepare_data.py \
  --data_root "path/to/DenPAR Radiographs Dataset/Dataset" \
  --output_dir ./data_prepared \
  --workers 4
This will:

Extract per-tooth blobs from radiograph-wise masks using connected components
Filter keypoints by mask zones to eliminate cross-tooth leakage
Resize all crops to 256×512 with correctly scaled coordinates
Save annotations to data_prepared/{split}/annotations.json

Verify the output:
bashpython prepare_data.py \
  --data_root "path/to/Dataset" \
  --output_dir ./data_prepared \
  --stats_only
Step 2 — Verify Ground Truth Annotations
Always run this before training to confirm keypoints are correctly positioned on crops:
bashpython visualize_annotations.py \
  --data_dir ./data_prepared \
  --mode gt \
  --split training \
  --n 30
Output images saved to visualizations/gt/training/. Open in file explorer and confirm:

🟢 Green dots (CEJ) at the crown-root junction
🔵 Blue dots (Intersection) mid-root near bone line
🔴 Red dots (Apex) at root tips

Step 3 — Train the Keypoint Model
bashpython train.py \
  --data_dir ./data_prepared \
  --output_dir ./runs/exp1 \
  --batch_size 4 \
  --lr 0.0001 \
  --epochs 100

Note for Windows users: If training is slow (>10 min/epoch), ensure --num_workers 0 is set (default).

Training saves two checkpoints:

runs/exp1/best_oks.pth — best validation OKS
runs/exp1/best_icc.pth — best ICC (bone loss accuracy)

Step 4 — Visualize Predictions
bash# Predicted keypoints only
python visualize_annotations.py \
  --data_dir ./data_prepared \
  --mode pred \
  --split validation \
  --checkpoint ./runs/exp1/best_oks.pth \
  --n 30

# Ground truth vs predicted overlay (error analysis)
python visualize_annotations.py \
  --data_dir ./data_prepared \
  --mode both \
  --split validation \
  --checkpoint ./runs/exp1/best_oks.pth \
  --n 30
Step 5 — Run Full RVG Inference
Run the complete pipeline on a full radiograph (multiple teeth, uncropped):
bashpython bone_loss_pipeline.py \
  --image path/to/rvg.jpg \
  --yolo_checkpoint path/to/tooth_detector.pt \
  --bone_checkpoint ./runs/exp1/best_oks.pth \
  --output annotated_result.jpg

🏗️ Model Architecture
The keypoint detection model is a Unified Keypoint R-CNN with:

Backbone: ResNet-50 + Feature Pyramid Network (pretrained on COCO)
4 Specialised Keypoint Heads:

HeadSlotsKeypointsLoss WeightSigmaCEJ0, 1CEJ Mesial, CEJ Distal1.4×0.025Intersection2, 3Intersection Mesial, Intersection Distal1.0×0.035Apex Mesial4Apex Mesial1.5×0.040Apex Distal5Apex Distal (absent for single-root)1.5×0.040

Auxiliary Root-Type Classifier: Binary head predicting single vs. double root — gates the apex_distal slot
Decoding: Hard argmax + Taylor sub-pixel refinement (not soft-argmax — see Key Design Decisions)

Keypoint Schema
Every tooth uses a fixed 6-slot schema regardless of root type:
Slot 0: cej_mesial          (always present, v=2)
Slot 1: cej_distal          (always present, v=2)
Slot 2: intersection_mesial (always present, v=2)
Slot 3: intersection_distal (always present, v=2)
Slot 4: apex_mesial         (always present, v=2)
Slot 5: apex_distal         (v=0 for single-root, v=2 for double-root)
Visibility: 0 = absent, 1 = estimated/uncertain, 2 = confident annotation

📊 Evaluation Metrics
MetricDescriptionTargetOKS MeanObject Keypoint Similarity — keypoint equivalent of IoUHigher is betterAP50% of teeth with OKS ≥ 0.50Higher is betterAP75% of teeth with OKS ≥ 0.75Higher is bettermAP50:95Mean AP across OKS thresholds 0.50 to 0.95Higher is betterICCIntraclass Correlation Coefficient — bone loss reliability> 0.75 = goodMAEMean Absolute Error in percentage pointsLower is better
Best Results (Epoch 85)
MetricValueOKS Mean0.7197AP500.891AP750.513mAP50:950.521ICC0.510MAE10.92%
Per-Keypoint OKS at Best Epoch
KeypointOKSIntersection Mesial0.769Intersection Distal0.761Apex Mesial0.741CEJ Distal0.684CEJ Mesial0.681Apex Distal0.634

🔑 Key Design Decisions
1. Radiograph-wise Mask + Connected Components for Keypoint Filtering
Early versions used bounding-box-only filtering to assign keypoints to teeth. This caused cross-tooth leakage in ~40% of crops — CEJ points from adjacent teeth were incorrectly assigned to the target tooth.
Fix: Load the radiograph-wise mask PNG, run cv2.connectedComponentsWithStats to extract individual tooth blobs, match each bbox to its blob by maximum pixel overlap, then filter keypoints using per-tooth mask zones:

CEJ and Intersection: ±25px dilation (boundary landmarks)
Apex: ±10px dilation (inside root)

2. Coordinate Scaling in Crop Preparation
A critical bug in early versions: keypoints only had the crop offset subtracted, without applying the resize scale factor.
python# WRONG (early version)
x_crop = x_original - offset_x

# CORRECT
x_crop = (x_original - offset_x) * scale_x   # scale_x = TARGET_W / crop_width
y_crop = (y_original - offset_y) * scale_y   # scale_y = TARGET_H / crop_height
3. Hard Argmax vs Soft-Argmax
Soft-argmax was initially used for sub-pixel accuracy. On near-uniform heatmaps (cold start), soft-argmax returns the weighted centroid of the entire map — always the crop centre — causing OKS to collapse to ~0.011 and never recover.
Fix: Hard argmax always returns the true heatmap peak, allowing training to bootstrap correctly. Sub-pixel refinement adds 0.25px shift toward the higher neighbour pixel.
4. Bone Loss Computation is Post-Hoc (No Gradient)
Bone loss percentage is computed from predicted keypoints after inference using deterministic geometry — not as a training target. This keeps the model objective clean (OKS only) and allows trying different geometric formulas without retraining.
Two strategies implemented:

Direct ratio: dist(CEJ→Intersection) / dist(CEJ→Apex) × 100 — consistently best
Min-max projection: Paper's Equations 2-3 — handles tilted teeth but more sensitive to noise


🦷 Bone Loss Severity Classification
SeverityBone Loss RangeClinical MeaningNone< 15%HealthyMild15% – 33%Early periodontitisModerate33% – 66%Moderate periodontitisSevere> 66%Severe periodontitis

🔧 Training Configuration Reference
ParameterValueOptimizerAdamLearning Rate0.0001SchedulerCosineAnnealingLR (T_max=100, eta_min=1e-7)Batch Size4Epochs100Input Size256 × 512 (W × H)Heatmap Size64 × 128 (stride 4)Gaussian Sigma4.0 pxGPUNVIDIA RTX 3050 Ti (4.3GB VRAM)Peak VRAM2.26GBEpoch Time~163 seconds
Resume Training from Checkpoint
bash# Resume with same optimizer and scheduler
python train.py \
  --data_dir ./data_prepared \
  --output_dir ./runs/exp1 \
  --batch_size 4 \
  --resume ./runs/exp1/latest.pth

# Load weights only, reset optimizer (new LR or scheduler)
python train.py \
  --data_dir ./data_prepared \
  --output_dir ./runs/exp2 \
  --batch_size 4 \
  --lr 0.00003 \
  --resume ./runs/exp1/best_oks.pth \
  --weights_only

🦷 Tooth Detection (YOLOv8x)
The full inference pipeline requires a separate YOLOv8x tooth detection model. To train it on DenPAR:
bash# Step 1: Convert DenPAR annotations to YOLO format
python convert_coco_to_yolo.py

# Step 2: Train YOLOv8x
python train_yolo_tooth_detection.py \
  --data_yaml ./yolo_dataset/data.yaml \
  --output_dir ./yolo_runs \
  --epochs 50
The best tooth detection checkpoint will be at yolo_runs/tooth_detection/weights/best.pt.

🐛 Common Issues
IssueCauseFixno_tooth_mask warning for all teethWrong folder name or mask not foundCheck masks(radiograph wise)/ folder exists in each splitEpoch time > 10 minutes (Windows)DataLoader multiprocessing deadlockUse --num_workers 0 (default)OKS flat at 0.01 for many epochsWrong decoder (soft-argmax) or wrong dataEnsure you're using updated model.py with hard argmaxKeypoints in wrong positions on cropsCoordinate scaling bugDelete data_prepared/ and rerun prepare_data.py with current versionCUDA out of memoryBatch size too largeReduce --batch_size to 2ICC always negativeOld ICC formula (group-level, not paired)Use current train.py with ICC(2,1) formula

📁 Output Files
After training, your runs/exp1/ folder will contain:
runs/exp1/
├── best_oks.pth       # Checkpoint with best validation OKS
├── best_icc.pth       # Checkpoint with best ICC
├── latest.pth         # Most recent epoch checkpoint
├── training_log.json  # Full per-epoch metrics (OKS, ICC, MAE, per-keypoint)
└── training.log       # Text log of training progress

📖 References

Wimalasiri C. et al., "AI-assisted radiographic analysis in detecting alveolar bone loss severity and patterns", arXiv:2506.20522, 2025
Rasnayaka S. et al., "DenPAR: Annotated Intra-oral Periapical Radiographs Dataset", Zenodo, DOI: 10.5281/zenodo.14181645, 2024
He K. et al., "Mask R-CNN", IEEE ICCV, 2017
He K. et al., "Deep Residual Learning for Image Recognition", CVPR, 2016
Lin T.Y. et al., "Feature Pyramid Networks for Object Detection", CVPR, 2017
Koo T.K. and Li M.Y., "A Guideline of Selecting and Reporting Intraclass Correlation Coefficients", Journal of Chiropractic Medicine, 2016


📄 License
This project is released for academic and research use. The DenPAR dataset is subject to its own license terms — see Zenodo for details.

👤 Author
Aaghash A S