"""
convert_coco_to_yolo.py
========================
Convert DenPAR COCO annotations to YOLO format for tooth detection training.

This script:
1. Reads COCO format annotations (coco_format_bonelines_*.json)
2. Converts bounding boxes to YOLO format (normalized coordinates)
3. Creates YOLO dataset structure with images and labels
4. Generates data.yaml for YOLO training

Output structure:
  yolo_dataset/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    data.yaml
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, w, h] to YOLO format [x_center, y_center, w, h] normalized.
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm

def process_split(coco_file, images_dir, output_labels_dir, output_images_dir, class_id=0):
    """
    Process one split (train/val/test) from COCO to YOLO.
    """
    print(f"Processing {coco_file}...")

    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Create image_id to image info mapping
    images_info = {img['id']: img for img in coco_data['images']}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Process each image
    for img_id, annotations in tqdm(annotations_by_image.items()):
        img_info = images_info[img_id]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']

        # Source image path
        src_img_path = images_dir / img_filename

        # Destination paths
        dst_img_path = output_images_dir / img_filename
        label_filename = Path(img_filename).stem + '.txt'
        dst_label_path = output_labels_dir / label_filename

        # Copy image
        if src_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image {src_img_path} not found")
            continue

        # Convert annotations to YOLO format
        yolo_lines = []
        for ann in annotations:
            if 'bbox' in ann:
                x_center, y_center, w_norm, h_norm = convert_bbox_coco_to_yolo(
                    ann['bbox'], img_width, img_height
                )
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Write label file
        with open(dst_label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

def create_data_yaml(output_dir, num_classes=1, class_names=['tooth']):
    """
    Create data.yaml file for YOLO training.
    """
    yaml_content = f"""train: {output_dir}/images/train
val: {output_dir}/images/val
test: {output_dir}/images/test

nc: {num_classes}
names: {class_names}
"""

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created data.yaml at {yaml_path}")

def main():
    # Paths
    base_dir = Path(r"d:\oralvis\denpar_dataset\DenPAR Radiographs Dataset\Dataset")
    output_dir = Path(r"d:\oralvis\yolo_dataset")

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process each split
    splits = [
        ('Training', 'coco_format_bonelines_train.json'),
        ('Validation', 'coco_format_bonelines_val.json'),
        ('Testing', 'coco_format_bonelines_test.json')
    ]

    for split_name, coco_filename in splits:
        coco_file = base_dir / split_name / 'Bone Level Annotations' / coco_filename
        images_dir = base_dir / split_name / 'Images'
        output_labels_dir = output_dir / 'labels' / split_name.lower()
        output_images_dir = output_dir / 'images' / split_name.lower()

        if coco_file.exists():
            process_split(coco_file, images_dir, output_labels_dir, output_images_dir)
        else:
            print(f"Warning: {coco_file} not found")

    # Create data.yaml
    create_data_yaml(output_dir)

    print("Conversion complete!")
    print(f"YOLO dataset created at: {output_dir}")

if __name__ == "__main__":
    main()