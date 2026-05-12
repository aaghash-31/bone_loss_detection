"""
train_yolo_tooth_detection.py
=============================
Train YOLOv8 model for tooth detection using DenPAR dataset.

Usage:
    python train_yolo_tooth_detection.py --data_yaml ./yolo_data/data.yaml --output_dir ./yolo_runs
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for tooth detection")
    parser.add_argument("--data_yaml", required=True, help="Path to data.yaml")
    parser.add_argument("--output_dir", default="./yolo_runs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")

    args = parser.parse_args()

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model

    # Train the model
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        project=args.output_dir,
        name="tooth_detection"
    )

    print("Training complete!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()