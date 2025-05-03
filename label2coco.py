# --- START OF MODIFIED SCRIPT label2coco.py (v3 - 0-based IDs) ---

import json
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm.notebook import tqdm # Use notebook version for Kaggle/Jupyter
import argparse # Import argparse
import datetime
import shutil # For directory operations

# --- Constants and Configuration ---

# VisDrone original categories (for reference)
VISDRONE_CATEGORIES_MAP = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}

# --- MODIFICATION: COCO categories with IDs 0-9 ---
# Generate the list mapping VisDrone names to COCO IDs 0-9
COCO_CATEGORIES = []
# Keep track of the mapping from original VisDrone ID (1-10) to new COCO ID (0-9)
VISDRONE_ID_TO_COCO_ID = {} # This will be {1: 0, 2: 1, ..., 10: 9}
coco_id_counter = 0
for vis_id, name in VISDRONE_CATEGORIES_MAP.items():
    # Assign supercategories based on your needs, example below:
    supercategory = "person" if name in ["pedestrian", "people"] else "vehicle"
    COCO_CATEGORIES.append({"id": coco_id_counter, "name": name, "supercategory": supercategory})
    VISDRONE_ID_TO_COCO_ID[vis_id] = coco_id_counter
    coco_id_counter += 1
# --- END MODIFICATION ---


# Mapping VisDrone ID -> YOLO Class ID (0-9) - Remains the same
VISDRONE_ID_TO_YOLO_ID = {k: i for i, k in enumerate(VISDRONE_CATEGORIES_MAP.keys())}

# --- MODIFICATION: Mapping YOLO Class ID (0-9) -> COCO Category ID (0-9) ---
# Now directly map YOLO ID (0-9) to the new COCO ID (0-9)
YOLO_ID_TO_COCO_ID = {i: i for i in range(len(VISDRONE_CATEGORIES_MAP))}
# --- END MODIFICATION ---


# --- Step 1: VisDrone TXT to YOLO TXT Conversion (NO CHANGES NEEDED HERE) ---
# (Function definition remains the same as previous version)
def convert_box_visdrone_to_yolo(size, box):
    """Converts VisDrone bbox to YOLO format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_min, y_min, width, height = box
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0
    x_center_norm = x_center * dw
    y_center_norm = y_center * dh
    width_norm = width * dw
    height_norm = height * dh
    return (x_center_norm, y_center_norm, width_norm, height_norm)

def convert_visdrone_txt_to_yolo(visdrone_annot_dir, visdrone_image_dir, yolo_label_dir):
    """Converts VisDrone .txt annotations to YOLO .txt format."""
    vis_annot_p = Path(visdrone_annot_dir)
    vis_img_p = Path(visdrone_image_dir)
    yolo_label_p = Path(yolo_label_dir)

    print(f"\n--- Running Step 1: VisDrone TXT -> YOLO TXT ---")
    print(f"  VisDrone Annotations: {vis_annot_p}")
    print(f"  VisDrone Images: {vis_img_p}")
    print(f"  YOLO Labels Output: {yolo_label_p}")

    if not vis_annot_p.is_dir():
        print(f"ERROR: VisDrone annotation directory not found: {vis_annot_p}")
        return False, 0
    if not vis_img_p.is_dir():
        print(f"ERROR: VisDrone image directory not found: {vis_img_p}")
        return False, 0

    yolo_label_p.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(list(vis_annot_p.glob("*.txt")))
    if not annotation_files:
        print(f"WARNING: No .txt annotation files found in {vis_annot_p}")
        return True, 0

    print(f"  Found {len(annotation_files)} annotation files. Converting...")
    skipped_files = 0
    processed_files = 0
    total_yolo_lines = 0

    pbar = tqdm(annotation_files, desc="  VisDrone->YOLO", leave=False)
    for f_annot in pbar:
        img_name = f_annot.stem + ".jpg"
        f_img = vis_img_p / img_name
        f_yolo = yolo_label_p / f_annot.name

        if not f_img.is_file():
            skipped_files += 1
            continue

        try:
            img_size = Image.open(f_img).size
            lines_out = []
            with open(f_annot, 'r', encoding='utf-8') as file_in:
                for row_str in file_in.read().strip().splitlines():
                    parts = row_str.split(',')
                    if len(parts) == 9 and parts[8] == '': parts = parts[:8]
                    if len(parts) != 8: continue
                    try:
                        score = int(parts[4])
                        if score == 0: continue
                        visdrone_cls = int(parts[5])
                        # Use VISDRONE_ID_TO_YOLO_ID map for filtering valid original classes
                        if visdrone_cls not in VISDRONE_ID_TO_YOLO_ID: continue
                        yolo_cls = VISDRONE_ID_TO_YOLO_ID[visdrone_cls] # Get YOLO ID (0-9)
                        box_vis = tuple(map(float, parts[:4]))
                        if box_vis[2] <= 0 or box_vis[3] <= 0: continue
                        box_yolo = convert_box_visdrone_to_yolo(img_size, box_vis)
                        lines_out.append(f"{yolo_cls} {' '.join(f'{x:.6f}' for x in box_yolo)}\n")
                    except (ValueError, IndexError): continue
            if lines_out:
                with open(f_yolo, "w", encoding="utf-8") as fl: fl.writelines(lines_out)
                total_yolo_lines += len(lines_out)
            processed_files += 1
        except UnidentifiedImageError: skipped_files += 1
        except Exception as e:
            print(f"Warning: Error processing {f_annot.name}/{f_img}: {e}, skipping.")
            skipped_files += 1
    print(f"  Step 1 Finished: Processed {processed_files}, Generated {total_yolo_lines} YOLO lines, Skipped {skipped_files} files.")
    return True, total_yolo_lines


# --- Step 2: YOLO TXT to COCO JSON Conversion (MODIFIED for 0-based IDs, fisheye style) ---

def initialize_coco_dict_fisheye_style_0based(): # Renamed for clarity
    """Creates the basic structure of a COCO JSON dictionary, 0-based IDs."""
    now = datetime.datetime.now()
    return {
        "info": { "year": now.year, "version": "1.0", "description": "VisDrone YOLO->COCO (0-based IDs, fisheye style)",
                  "contributor": "Conversion Script", "url": "", "date_created": now.strftime("%Y-%m-%d %H:%M:%S")},
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": COCO_CATEGORIES, # Use the globally defined COCO_CATEGORIES (now 0-based)
        "images": [],
        "annotations": [] # Annotations will use 0-based category_id and omit segmentation
    }

def convert_yolo_to_coco(yolo_label_dir, image_dir, output_coco_json):
    """Converts YOLO .txt annotations to COCO JSON format (0-based IDs, fisheye style)."""
    yolo_p = Path(yolo_label_dir)
    img_p = Path(image_dir)
    coco_out_p = Path(output_coco_json)

    print(f"\n--- Running Step 2: YOLO TXT -> COCO JSON (0-based, Fisheye Style) ---")
    print(f"  YOLO Labels Input: {yolo_p}")
    print(f"  Images Source: {img_p}")
    print(f"  COCO JSON Output: {coco_out_p}")

    if not img_p.is_dir():
        print(f"ERROR: Image directory not found: {img_p}")
        return False

    coco_data = initialize_coco_dict_fisheye_style_0based() # Use the 0-based initializer
    image_id_counter = 1
    annotation_id_counter = 1
    processed_images = 0
    processed_annotations = 0
    skipped_label_files = 0
    skipped_image_errors = 0

    image_files = sorted([f for f in img_p.glob("*.jpg")])
    if not image_files:
        print(f"WARNING: No .jpg images found in {img_p}")
    else:
        print(f"  Found {len(image_files)} images. Processing for COCO...")

    pbar = tqdm(image_files, desc="  YOLO->COCO", leave=False)
    for img_file in pbar:
        label_filename = img_file.stem + ".txt"
        label_file = yolo_p / label_filename

        try:
            with Image.open(img_file) as img: img_width, img_height = img.size
            coco_image_id = image_id_counter
            coco_data["images"].append({
                "id": coco_image_id, "width": img_width, "height": img_height, "file_name": img_file.name,
                "license": 1, "flickr_url": "", "coco_url": "", "date_captured": "",
            })
            processed_images += 1
            image_id_counter += 1

            if label_file.is_file():
                with open(label_file, 'r', encoding='utf-8') as f_label:
                    for line in f_label:
                        parts = line.strip().split()
                        if len(parts) != 5: continue
                        try:
                            yolo_cls_id = int(parts[0])
                            x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
                            abs_width_f = width_norm * img_width
                            abs_height_f = height_norm * img_height
                            x_min_f = (x_center_norm * img_width) - (abs_width_f / 2.0)
                            y_min_f = (y_center_norm * img_height) - (abs_height_f / 2.0)

                            x_min = int(round(x_min_f))
                            y_min = int(round(y_min_f))
                            abs_width = int(round(abs_width_f))
                            abs_height = int(round(abs_height_f))

                            if abs_width <= 0 or abs_height <= 0: continue
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            if x_min + abs_width > img_width: abs_width = img_width - x_min
                            if y_min + abs_height > img_height: abs_height = img_height - y_min
                            if abs_width <= 0 or abs_height <= 0: continue
                            area = abs_width * abs_height # Use int area

                            # --- CRITICAL CHANGE: Use the 0-based mapping ---
                            if yolo_cls_id not in YOLO_ID_TO_COCO_ID: continue # Should map 0-9
                            coco_cat_id = YOLO_ID_TO_COCO_ID[yolo_cls_id] # Gets 0-9 directly now
                            # --- END CRITICAL CHANGE ---

                            coco_annotation = {
                                "id": annotation_id_counter,
                                "image_id": coco_image_id,
                                "category_id": coco_cat_id, # Store 0-9 ID
                                "bbox": [x_min, y_min, abs_width, abs_height],
                                "area": area,
                                "iscrowd": 0,
                            }
                            coco_data["annotations"].append(coco_annotation)
                            annotation_id_counter += 1
                            processed_annotations += 1
                        except ValueError: continue
            else: skipped_label_files += 1

        except UnidentifiedImageError: skipped_image_errors += 1
        except Exception as e:
            print(f"Warning: Error processing {img_file}/{label_file}: {e}, skipping image.")
            skipped_image_errors += 1

    print(f"  Step 2 Finished: Processed {processed_images} images, Added {processed_annotations} COCO annotations.")
    if skipped_label_files > 0: print(f"  Note: {skipped_label_files} images had no corresponding label file.")
    if skipped_image_errors > 0: print(f"  Warning: Skipped {skipped_image_errors} images due to errors.")

    print(f"  Saving COCO data to {coco_out_p}...")
    try:
        coco_out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(coco_out_p, 'w', encoding='utf-8') as f: json.dump(coco_data, f, indent=2)
        print("  COCO JSON saved successfully!")
        return True
    except Exception as e:
        print(f"ERROR saving JSON file: {e}")
        return False


# --- Main Execution Logic (NO CHANGES NEEDED HERE) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VisDrone DET dataset (train/val) to COCO format (0-based IDs, fisheye style).") # Updated description

    # Input Arguments
    parser.add_argument('--train-annot-dir', required=True, help="Path to VisDrone TRAINING annotations directory (.txt files)")
    parser.add_argument('--train-img-dir', required=True, help="Path to VisDrone TRAINING images directory (.jpg files)")
    parser.add_argument('--val-annot-dir', required=True, help="Path to VisDrone VALIDATION annotations directory (.txt files)")
    parser.add_argument('--val-img-dir', required=True, help="Path to VisDrone VALIDATION images directory (.jpg files)")

    # Output Arguments
    parser.add_argument('--output-dir', required=True, help="Directory to save the final COCO JSON files (train_coco.json, val_coco.json)")
    parser.add_argument('--intermediate-dir', required=True, help="Directory to store intermediate YOLO label files (will be created)")

    args = parser.parse_args()

    # --- Prepare Directories ---
    output_dir = Path(args.output_dir)
    intermediate_dir = Path(args.intermediate_dir)
    intermediate_train_dir = intermediate_dir / "train" / "labels"
    intermediate_val_dir = intermediate_dir / "val" / "labels"

    output_dir.mkdir(parents=True, exist_ok=True)
    if intermediate_dir.exists():
        print(f"Intermediate directory {intermediate_dir} exists. Files may be overwritten.")
    intermediate_train_dir.mkdir(parents=True, exist_ok=True)
    intermediate_val_dir.mkdir(parents=True, exist_ok=True)

    print("--- Conversion Pipeline Started ---")

    # --- Process Training Set ---
    print("\n===== Processing Training Set =====")
    train_step1_success, train_yolo_lines = convert_visdrone_txt_to_yolo(
        args.train_annot_dir, args.train_img_dir, intermediate_train_dir )
    if train_step1_success:
        train_coco_path = output_dir / "visdrone_train_coco.json" # Renamed output file
        train_step2_success = convert_yolo_to_coco( intermediate_train_dir, args.train_img_dir, train_coco_path )
        if not train_step2_success: print("ERROR: Failed during Step 2 (YOLO->COCO) for TRAINING set.")
    else: print("ERROR: Failed during Step 1 (VisDrone->YOLO) for TRAINING set.")

    # --- Process Validation Set ---
    print("\n===== Processing Validation Set =====")
    val_step1_success, val_yolo_lines = convert_visdrone_txt_to_yolo(
        args.val_annot_dir, args.val_img_dir, intermediate_val_dir )
    if val_step1_success:
        val_coco_path = output_dir / "visdrone_val_coco.json" # Renamed output file
        val_step2_success = convert_yolo_to_coco( intermediate_val_dir, args.val_img_dir, val_coco_path )
        if not val_step2_success: print("ERROR: Failed during Step 2 (YOLO->COCO) for VALIDATION set.")
    else: print("ERROR: Failed during Step 1 (VisDrone->YOLO) for VALIDATION set.")

    print("\n--- Conversion Pipeline Finished ---")

# --- END OF MODIFIED SCRIPT ---