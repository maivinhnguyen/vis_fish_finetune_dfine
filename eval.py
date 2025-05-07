from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
import numpy as np
import argparse # Import argparse

def main(gt_file_path, dt_file_path, eval_type_str):
    """
    Main function to perform COCO evaluation.
    """
    # --- Configuration (now from arguments) ---
    gt_file = gt_file_path
    dt_file = dt_file_path
    # Use a more descriptive temporary file name, potentially unique if needed
    # For simplicity, we'll keep it fixed relative to the script or current dir
    temp_dt_file = './temp_filtered_detections.json'
    eval_type = eval_type_str

    print(f"--- Configuration ---")
    print(f"Ground Truth File: {gt_file}")
    print(f"Detections File: {dt_file}")
    print(f"Temporary Filtered Detections: {temp_dt_file}")
    print(f"Evaluation Type: {eval_type}")
    print(f"---------------------\n")

    # --- Load Ground Truth Annotations ---
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file not found: {gt_file}")
        return # Use return instead of exit() in a function
    try:
        coco_gt = COCO(gt_file)
    except Exception as e:
        print(f"Error loading ground truth file {gt_file}: {e}")
        return

    gt_image_ids = set(coco_gt.getImgIds())
    print(f"Loaded ground truth. Found {len(gt_image_ids)} image IDs.")
    if not gt_image_ids:
        print("Warning: No image IDs found in the ground truth file.")

    # --- Load Detection Results ---
    if not os.path.exists(dt_file):
        print(f"Error: Detections file not found: {dt_file}")
        return
    try:
        with open(dt_file, 'r') as f:
            detection_data_full = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {dt_file}: {e}")
        return
    except Exception as e:
        print(f"Error loading {dt_file}: {e}")
        return

    # --- Extract and Filter Detections ---
    if not isinstance(detection_data_full, dict) or 'annotations' not in detection_data_full:
        print(f"Warning: {dt_file} does not appear to be in the expected COCO format dictionary (missing 'annotations' key).")
        if isinstance(detection_data_full, list):
             print("Detection file seems to be a list of annotations directly. Using it as is.")
             detection_annotations = detection_data_full
        else:
             print("Trying to handle as a differently structured COCO format.")
             detection_annotations = detection_data_full.get('annotations', [])
             if not detection_annotations and isinstance(detection_data_full, list):
                 detection_annotations = detection_data_full
    else:
        detection_annotations = detection_data_full.get('annotations', [])

    print(f"Total detections loaded: {len(detection_annotations)}")
    if not detection_annotations:
        print("Warning: No detection annotations loaded. Evaluation will likely be empty.")
        # No need to proceed if there are no detections
        # return # Or let it proceed and show zero results

    # --- Create Image ID Mapping ---
    gt_images = {}
    if 'images' in coco_gt.dataset:
        for img in coco_gt.dataset['images']:
            if 'file_name' in img:
                gt_images[img['file_name']] = img['id']
            else:
                print(f"Warning: Ground truth image missing file_name: {img}")

    dt_images = {}
    dt_image_ids_from_file_images_section = set() # IDs from 'images' section of dt_file
    if isinstance(detection_data_full, dict) and 'images' in detection_data_full:
        for img in detection_data_full['images']:
            if 'file_name' in img:
                dt_images[img['file_name']] = img['id']
                dt_image_ids_from_file_images_section.add(img['id'])
            else:
                print(f"Warning: Detection image missing file_name: {img}")

    # Collect unique image IDs from detection annotations as a fallback
    # or if 'images' section is missing in dt_file
    dt_image_ids_from_annotations = set()
    if detection_annotations:
        for anno in detection_annotations:
            if 'image_id' in anno:
                dt_image_ids_from_annotations.add(anno['image_id'])

    # Prefer IDs from 'images' section if available, otherwise from annotations
    final_dt_image_ids = dt_image_ids_from_file_images_section if dt_image_ids_from_file_images_section else dt_image_ids_from_annotations
    print(f"Found {len(final_dt_image_ids)} unique image IDs in detections.")

    id_mapping = {}
    if gt_images and dt_images:
        print("Creating image ID mapping based on file names...")
        for filename, dt_id in dt_images.items():
            if filename in gt_images:
                id_mapping[dt_id] = gt_images[filename]
    elif len(final_dt_image_ids) > 0 and len(final_dt_image_ids) == len(gt_image_ids):
        print("Warning: Can't map by filename. Trying sequential mapping (assumes same order and same ID set)...")
        sorted_dt_ids = sorted(list(final_dt_image_ids))
        sorted_gt_ids = sorted(list(gt_image_ids))
        for dt_id, gt_id in zip(sorted_dt_ids, sorted_gt_ids):
            id_mapping[dt_id] = gt_id
    else:
        print("Warning: Can't create reliable mapping by filename or sequential. Using identity mapping where IDs match...")
        for dt_id in final_dt_image_ids:
            if dt_id in gt_image_ids:
                id_mapping[dt_id] = dt_id

    print(f"Created mapping for {len(id_mapping)} image IDs between detections and ground truth.")
    if not id_mapping and final_dt_image_ids and gt_image_ids: # Only error if both sides had IDs
        print("ERROR: Could not create any mapping between detection and ground truth image IDs.")
        print("Please check that your detection and ground truth files reference the same images,")
        print("either by file_name (if 'images' section exists in both) or by having identical 'image_id' sets.")
        return

    # --- Map and Filter Detections ---
    mapped_detection_annotations = []
    unmapped_count = 0
    for item in detection_annotations:
        if isinstance(item, dict) and 'image_id' in item:
            orig_image_id = item['image_id']
            if orig_image_id in id_mapping:
                mapped_item = item.copy()
                mapped_item['image_id'] = id_mapping[orig_image_id]
                mapped_detection_annotations.append(mapped_item)
            elif orig_image_id in gt_image_ids: # Fallback: if ID exists in GT, assume it's already correct
                mapped_detection_annotations.append(item) # Add as is
            else:
                unmapped_count += 1
    if unmapped_count > 0:
        print(f"Warning: {unmapped_count} detections could not be mapped to a ground truth image ID and were discarded.")


    print(f"Detections after ID mapping: {len(mapped_detection_annotations)}")

    filtered_detection_annotations = [
        item for item in mapped_detection_annotations
        if item.get('image_id') in gt_image_ids
    ]
    print(f"Detections after filtering for GT image IDs: {len(filtered_detection_annotations)}")

    if not filtered_detection_annotations:
        print("Warning: No detections remaining after filtering by ground truth image IDs. Evaluation might yield zero results.")
        # We can let it proceed to show zero results or return
        # return

    # --- Save Filtered Detections to a Temporary File ---
    try:
        with open(temp_dt_file, 'w') as f:
            json.dump(filtered_detection_annotations, f)
        print(f"Saved filtered and mapped detections to temporary file: {temp_dt_file}")
    except Exception as e:
        print(f"Error writing temporary filtered detections file {temp_dt_file}: {e}")
        return

    # --- Load Filtered Detections into COCO API ---
    try:
        coco_dt = coco_gt.loadRes(temp_dt_file)
    except Exception as e:
        print(f"Error loading filtered results using coco_gt.loadRes({temp_dt_file}): {e}")
        try:
            print("Trying to load filtered results list directly into loadRes...")
            coco_dt = coco_gt.loadRes(filtered_detection_annotations)
            print("Direct list loading successful.")
        except Exception as e2:
            print(f"Direct list loading also failed: {e2}")
            return

    # --- Initialize and Run Evaluation ---
    print(f"\nRunning COCO Evaluation (type: '{eval_type}')...")
    try:
        coco_eval = COCOeval(coco_gt, coco_dt, eval_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        print(f"Error during COCO evaluation steps: {e}")
        return

    # --- Compute F1 Score ---
    # (Your compute_f1_scores function remains unchanged)
    def compute_f1_scores(coco_eval, iou_threshold_idx=0):
        precision = coco_eval.eval['precision']
        # recall = coco_eval.eval['recall'] # Original recall array
        # In newer pycocotools, recall might have a different shape.
        # It's often (T, K, A, M) for specific recall thresholds.
        # For max F1, we are interested in recall achieved at the point of max F1.
        # Let's use coco_eval.params.recThrs to iterate recall points.

        s = coco_eval.eval['scores'] # per-category scores

        # precision shape: (T, R, K, A, M)
        # T: IoU thresholds (e.g., 10)
        # R: Recall thresholds (e.g., 101)
        # K: Categories (e.g., num_classes)
        # A: Area ranges (e.g., 4)
        # M: Max detections (e.g., 3)

        f1_scores_dict = {}
        
        # Ensure coco_gt.cats is populated
        if not coco_gt.cats:
            print("Warning: coco_gt.cats is empty. Cannot get category names for F1 scores.")
            # Fallback to generic category names if coco_gt.cats is not available
            category_names = {cat_id: f"cat_{cat_id}" for cat_id in coco_eval.params.catIds}
        else:
            category_names = {cat_id: coco_gt.cats[cat_id]['name'] for cat_id in coco_eval.params.catIds if cat_id in coco_gt.cats}
            # Handle cases where a catId might not be in coco_gt.cats (should not happen if GT is consistent)
            for cat_id in coco_eval.params.catIds:
                if cat_id not in category_names:
                    category_names[cat_id] = f"cat_{cat_id}_missing_name"


        # Overall F1 (mF1@0.5)
        # We want precision at a specific IoU threshold (iou_threshold_idx),
        # across all recall thresholds, for all categories, 'all' area, maxDets=100.
        # Default M index for maxDets=100 is 2. Default A index for 'all' area is 0.
        
        # p has shape (R, K) for the selected IoU, Area, MaxDets
        p = precision[iou_threshold_idx, :, :, 0, 2]
        
        # Recall values for which precision is calculated
        # These are the 101 recall thresholds from 0 to 1.0
        recall_thresholds = coco_eval.params.recThrs 

        max_f1_per_category = []
        
        for k_idx, cat_id in enumerate(coco_eval.params.catIds):
            prec_at_recalls = p[:, k_idx] # Precision for this category at all recall thresholds
            
            current_max_f1 = 0
            for r_idx, rec_val in enumerate(recall_thresholds):
                prec_val = prec_at_recalls[r_idx]
                if prec_val + rec_val > 0:
                    f1 = 2 * prec_val * rec_val / (prec_val + rec_val)
                    if f1 > current_max_f1:
                        current_max_f1 = f1
            max_f1_per_category.append(current_max_f1)
            f1_scores_dict[f'f1_{category_names.get(cat_id, f"cat_{cat_id}")}'] = float(current_max_f1)

        if max_f1_per_category:
            mean_f1 = np.mean(max_f1_per_category)
        else:
            mean_f1 = 0.0 # Handle case with no categories or no valid F1s
        f1_scores_dict['mean_f1'] = float(mean_f1)
        
        # F1 at different IoU thresholds (if iou_threshold_idx is 0, i.e., for IoU=0.5 initially)
        if iou_threshold_idx == 0:
            iou_thrs_values = coco_eval.params.iouThrs
            for t_idx, iou_val in enumerate(iou_thrs_values):
                p_at_iou = precision[t_idx, :, :, 0, 2] # (R, K)
                
                current_max_f1_at_iou_per_category = []
                for k_idx, _ in enumerate(coco_eval.params.catIds):
                    prec_at_recalls_for_iou = p_at_iou[:, k_idx]
                    
                    cat_max_f1 = 0
                    for r_idx, rec_val in enumerate(recall_thresholds):
                        prec_val = prec_at_recalls_for_iou[r_idx]
                        if prec_val + rec_val > 0:
                            f1 = 2 * prec_val * rec_val / (prec_val + rec_val)
                            if f1 > cat_max_f1:
                                cat_max_f1 = f1
                    current_max_f1_at_iou_per_category.append(cat_max_f1)
                
                if current_max_f1_at_iou_per_category:
                    mean_f1_at_iou = np.mean(current_max_f1_at_iou_per_category)
                else:
                    mean_f1_at_iou = 0.0
                f1_scores_dict[f'f1_iou_{iou_val:.2f}'] = float(mean_f1_at_iou)
                
        return f1_scores_dict

    f1_scores = compute_f1_scores(coco_eval, iou_threshold_idx=0) # Index 0 for IoU=0.5

    # --- Print Specific Evaluation Metrics ---
    print('\n--- Specific Evaluation Metrics ---')
    if hasattr(coco_eval, 'stats') and len(coco_eval.stats) >= 12:
        stats = coco_eval.stats
        print(f"AP_0.50:0.95 (Primary Challenge Metric): {stats[0]:.4f}")
        print(f"AP_0.50: {stats[1]:.4f}")
        print(f"AP_0.75: {stats[2]:.4f}")
        print(f"AP_Small: {stats[3]:.4f}")
        print(f"AP_Medium: {stats[4]:.4f}")
        print(f"AP_Large: {stats[5]:.4f}")
        print(f"AR_maxDets=1: {stats[6]:.4f}")
        print(f"AR_maxDets=10: {stats[7]:.4f}")
        print(f"AR_maxDets=100: {stats[8]:.4f}")
        print(f"AR_Small: {stats[9]:.4f}")
        print(f"AR_Medium: {stats[10]:.4f}")
        print(f"AR_Large: {stats[11]:.4f}")

        print("\n--- F1 Scores ---")
        # The 'mean_f1' calculated corresponds to IoU=0.5 because iou_threshold_idx=0 was used
        print(f"F1_Score (mean over categories @ IoU=0.50): {f1_scores.get('mean_f1', 0.0):.4f}")

        print("\n--- F1 Scores at different IoUs (mean over categories) ---")
        for key, value in f1_scores.items():
            if key.startswith('f1_iou_'):
                print(f"{key.replace('f1_iou_', 'F1@IoU='):}: {value:.4f}")
        
        print("\n--- Per-Category F1 Scores (@ IoU=0.50) ---")
        for key, value in f1_scores.items():
            if key.startswith('f1_') and not key.startswith('f1_iou_') and key != 'mean_f1':
                print(f"{key}: {value:.4f}")
    else:
        print("Evaluation stats are not available or incomplete.")

    print('---------------------------------')

    # --- Clean up ---
    try:
        if os.path.exists(temp_dt_file): # Check if it exists before removing
            os.remove(temp_dt_file)
            print(f"Removed temporary file: {temp_dt_file}")
    except OSError as e:
        print(f"Error removing temporary file {temp_dt_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate COCO detections against ground truth.")
    parser.add_argument(
        "--gt_file",
        "-g",
        type=str,
        required=True,
        help="Path to the COCO ground truth JSON file."
    )
    parser.add_argument(
        "--dt_file",
        "-d",
        type=str,
        required=True,
        help="Path to the COCO detections JSON file (can be full COCO format or just a list of annotation dicts)."
    )
    parser.add_argument(
        "--eval_type",
        "-e",
        type=str,
        default="bbox",
        choices=["bbox", "segm", "keypoints"],
        help="Type of evaluation (default: 'bbox')."
    )
    args = parser.parse_args()

    main(args.gt_file, args.dt_file, args.eval_type)