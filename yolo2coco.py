#! /usr/bin/env python3
# Convert YOLO annotations to COCO format
# Usage: python yolo2coco.py --input ./YOLO_dataset --output ./coco_dataset

import os
import glob
import shutil
import json
import argparse
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm
import PIL.Image as Image
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert YOLO annotations to COCO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python yolo2coco.py -i ./YOLO_dataset -o ./coco_dataset
  python yolo2coco.py -i ./YOLO_dataset -o ./coco_dataset --size 640 640
  python yolo2coco.py -i ./YOLO_dataset -o ./coco_dataset -s 640 640 -k
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to YOLO dataset root directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output COCO dataset directory"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resize images to WIDTH HEIGHT (e.g., --size 640 640). If not specified, original size is kept."
    )
    parser.add_argument(
        "--keep-aspect-ratio", "-k",
        action="store_true",
        help="Keep aspect ratio when resizing (pad with black if necessary)"
    )
    
    # 引数がない場合はヘルプを表示
    import sys
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    return parser.parse_args()


def detect_split(img_path, yolo_yaml):
    """Decide split by checking configured paths in yolo_yaml"""
    train_sub = yolo_yaml.get("train", "")
    val_sub = yolo_yaml.get("val", "")
    # Normalize separators
    p = img_path.replace(os.sep, '/')
    if train_sub and train_sub in p:
        return "train"
    if val_sub and val_sub in p:
        return "val"
    return "other"


def convert_yolo_to_coco(yolos_path: str, output_path: str, target_size: tuple = None, keep_aspect_ratio: bool = False):
    """Main conversion function
    
    Args:
        yolos_path: Path to YOLO dataset root directory
        output_path: Path to output COCO dataset directory
        target_size: Target image size as (width, height). None to keep original size.
        keep_aspect_ratio: If True, keep aspect ratio and pad with black
    """
    
    # =========================================================
    # Load YOLO dataset information from YAML
    yaml_path = os.path.join(yolos_path, "can_1step_data_collection.yaml")
    if not os.path.exists(yaml_path):
        # Try to find any yaml file in the directory
        yaml_files = glob.glob(os.path.join(yolos_path, "*.yaml"))
        if yaml_files:
            yaml_path = yaml_files[0]
        else:
            raise FileNotFoundError(f"No YAML file found in {yolos_path}")
    
    with open(yaml_path, 'r') as f:
        yolo_yaml = yaml.safe_load(f)
    print(f"Loaded YAML config from: {yaml_path}")
    # =========================================================

    img_files = glob.glob(os.path.join(yolos_path, "images", "**", "*.*"), recursive=True)
    img_files = [f.replace(os.sep, '/') for f in img_files if f.lower().endswith(('.png',))]

    # Prepare output dirs
    os.makedirs(os.path.join(output_path, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)

    # Build category list from yolo_yaml['names']
    categories = []
    for cid, name in sorted(yolo_yaml.get('names', {}).items(), key=lambda x: int(x[0])):
        categories.append({"id": int(cid), "name": str(name), "supercategory": "none"})

    # Containers for splits
    coco_data = {
        "train": {"images": [], "annotations": [], "categories": categories},
        "val": {"images": [], "annotations": [], "categories": categories},
        "other": {"images": [], "annotations": [], "categories": categories},
    }

    ann_id = 1
    img_id = 1

    for img_path in tqdm(img_files, desc="Converting images"):
        split = detect_split(img_path, yolo_yaml)
        try:
            img = Image.open(img_path)
            orig_w, orig_h = img.size
        except Exception:
            # skip broken images
            continue

        # Resize image if target_size is specified
        if target_size:
            target_w, target_h = target_size
            if keep_aspect_ratio:
                # Calculate scale to fit within target size while keeping aspect ratio
                scale = min(target_w / orig_w, target_h / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                resized_img = img.resize((new_w, new_h), Image.LANCZOS)
                
                # Create new image with padding
                final_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
                paste_x = (target_w - new_w) // 2
                paste_y = (target_h - new_h) // 2
                final_img.paste(resized_img, (paste_x, paste_y))
                
                # Calculate offset and scale for bbox adjustment
                bbox_scale = scale
                bbox_offset_x = paste_x
                bbox_offset_y = paste_y
            else:
                # Simple resize without keeping aspect ratio
                final_img = img.resize((target_w, target_h), Image.LANCZOS)
                bbox_scale_x = target_w / orig_w
                bbox_scale_y = target_h / orig_h
                bbox_offset_x = 0
                bbox_offset_y = 0
            
            w, h = target_w, target_h
        else:
            final_img = img
            w, h = orig_w, orig_h
            bbox_scale = 1.0
            bbox_scale_x = 1.0
            bbox_scale_y = 1.0
            bbox_offset_x = 0
            bbox_offset_y = 0

        # Copy/save image into output_path/images/<split>/
        dst_dir = os.path.join(output_path, 'images', split)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(img_path))
        try:
            if target_size:
                final_img.save(dst_path)
            else:
                shutil.copy2(img_path, dst_path)
        except Exception:
            # continue even if save/copy fails
            pass

        # Add image entry
        img_entry = {
            "id": img_id,
            "file_name": os.path.basename(img_path),
            # "file_name": os.path.relpath(dst_path, start=output_path).replace(os.sep, "/"),
            "width": w,
            "height": h
        }
        coco_data[split]["images"].append(img_entry)

        # Corresponding label path: replace images/ with labels/ and .png -> .txt
        label_path = img_path.replace("/images/", "/labels/")
        label_path = os.path.splitext(label_path)[0] + ".txt"

        if os.path.exists(label_path):
            with open(label_path, "r") as lf:
                for line in lf:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    nw = float(parts[3])
                    nh = float(parts[4])

                    # Calculate bbox in original image coordinates
                    bbox_w_orig = nw * orig_w
                    bbox_h_orig = nh * orig_h
                    x_tl_orig = xc * orig_w - bbox_w_orig / 2.0
                    y_tl_orig = yc * orig_h - bbox_h_orig / 2.0

                    # Adjust bbox for resized image
                    if target_size:
                        if keep_aspect_ratio:
                            x_tl = x_tl_orig * bbox_scale + bbox_offset_x
                            y_tl = y_tl_orig * bbox_scale + bbox_offset_y
                            bbox_w = bbox_w_orig * bbox_scale
                            bbox_h = bbox_h_orig * bbox_scale
                        else:
                            x_tl = x_tl_orig * bbox_scale_x
                            y_tl = y_tl_orig * bbox_scale_y
                            bbox_w = bbox_w_orig * bbox_scale_x
                            bbox_h = bbox_h_orig * bbox_scale_y
                    else:
                        x_tl = x_tl_orig
                        y_tl = y_tl_orig
                        bbox_w = bbox_w_orig
                        bbox_h = bbox_h_orig

                    # Ensure bbox inside image (clamp)
                    x_tl = int(max(0.0, x_tl))
                    y_tl = int(max(0.0, y_tl))
                    bbox_w = int(max(0.0, min(bbox_w, w - x_tl)))
                    bbox_h = int(max(0.0, min(bbox_h, h - y_tl)))

                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [x_tl, y_tl, bbox_w, bbox_h],
                        "area": bbox_w * bbox_h,
                        "iscrowd": 0,
                        "segmentation": []  # empty; polygon segmentation not available from YOLO box
                    }
                    coco_data[split]["annotations"].append(ann)
                    ann_id += 1

        img_id += 1

    # Write out COCO json files for each split that has images
    for split_name, data in coco_data.items():
        if len(data["images"]) == 0:
            continue
        out_json_path = os.path.join(output_path, 'annotations', f'instances_{split_name}.json')
        with open(out_json_path, "w", encoding="utf-8") as out_f:
            json.dump({
                "images": data["images"],
                "annotations": data["annotations"],
                "categories": data["categories"]
            }, out_f, ensure_ascii=False, indent=2)

    # Print summary
    print("COCO conversion finished.")
    print(f"Output directory: {output_path}")
    for split_name in ("train", "val", "other"):
        imgs = len(coco_data[split_name]["images"])
        anns = len(coco_data[split_name]["annotations"])
        if imgs:
            print(f"  {split_name}: images={imgs}, annotations={anns}, -> {output_path}/annotations/instances_{split_name}.json")


if __name__ == "__main__":
    args = parse_args()
    print(f"Input YOLO dataset: {args.input}")
    print(f"Output COCO dataset: {args.output}")
    if args.size:
        print(f"Target image size: {args.size[0]}x{args.size[1]}")
        if args.keep_aspect_ratio:
            print("Keeping aspect ratio (padding with black)")
    convert_yolo_to_coco(args.input, args.output, 
                         target_size=tuple(args.size) if args.size else None,
                         keep_aspect_ratio=args.keep_aspect_ratio)