from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os # Import os for file checking
import numpy as np # Needed for some COCOeval parameters if you customize them

# --- Configuration ---
gt_file = 'ground_truth.json'
dt_file = 'detections/detections.json'
# Use a more descriptive temporary file name
temp_dt_file = './temp_filtered_detections.json'
eval_type = 'bbox' # Type of evaluation ('bbox', 'segm', 'keypoints')

# --- Load Ground Truth Annotations ---
if not os.path.exists(gt_file):
    print(f"Error: Ground truth file not found: {gt_file}")
    exit()
try:
    coco_gt = COCO(gt_file)
except Exception as e:
    print(f"Error loading ground truth file {gt_file}: {e}")
    exit()

# Get all image IDs from ground truth (use a set for faster lookups)
gt_image_ids = set(coco_gt.getImgIds())
print(f"Loaded ground truth. Found {len(gt_image_ids)} image IDs.")
if not gt_image_ids:
    print("Warning: No image IDs found in the ground truth file.")

# --- Load Detection Results ---
if not os.path.exists(dt_file):
    print(f"Error: Detections file not found: {dt_file}")
    exit()
try:
    with open(dt_file, 'r') as f:
        # Load the entire COCO-style dictionary
        detection_data_full = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from {dt_file}: {e}")
    exit()
except Exception as e:
    print(f"Error loading {dt_file}: {e}")
    exit()

# --- Extract and Filter Detections ---
# Check if the loaded data is a dictionary and has the 'annotations' key
if not isinstance(detection_data_full, dict) or 'annotations' not in detection_data_full:
    print(f"Error: {dt_file} does not appear to be in the expected COCO format dictionary (missing 'annotations' key).")
    # Optional: Check if it's maybe just a list of annotations (older format?)
    if isinstance(detection_data_full, list):
         print("Warning: Detection file seems to be a list of annotations directly. Using it as is.")
         detection_annotations = detection_data_full
    else:
         exit()
else:
    # Access the list of annotation dictionaries
    detection_annotations = detection_data_full.get('annotations', []) # Use .get for safety

print(f"Total detections loaded: {len(detection_annotations)}")

# Filter the list of annotations to match ground truth image IDs
# Add a check for item type just in case
filtered_detection_annotations = [
    item for item in detection_annotations
    if isinstance(item, dict) and item.get('image_id') in gt_image_ids
]
print(f"Detections after filtering for GT image IDs: {len(filtered_detection_annotations)}")

if not filtered_detection_annotations:
    print("Warning: No detections remaining after filtering by ground truth image IDs. Evaluation might yield zero results.")
    # Consider exiting if you require matching detections
    # exit()

# --- Save Filtered Detections to a Temporary File ---
# COCOeval expects loadRes to load a *list* of annotation dicts
try:
    with open(temp_dt_file, 'w') as f:
        json.dump(filtered_detection_annotations, f)
    print(f"Saved filtered detections to temporary file: {temp_dt_file}")
except Exception as e:
    print(f"Error writing temporary filtered detections file {temp_dt_file}: {e}")
    exit()

# --- Load Filtered Detections into COCO API ---
# coco_gt.loadRes expects a file path containing a LIST of annotation dicts,
# or it can take the list directly. Loading from file is common.
try:
    coco_dt = coco_gt.loadRes(temp_dt_file)
except Exception as e:
    print(f"Error loading filtered results using coco_gt.loadRes({temp_dt_file}): {e}")
    # Try loading the list directly as a fallback/debug step
    try:
        print("Trying to load filtered results list directly into loadRes...")
        coco_dt = coco_gt.loadRes(filtered_detection_annotations)
        print("Direct list loading successful.")
    except Exception as e2:
        print(f"Direct list loading also failed: {e2}")
        exit()


# --- Initialize and Run Evaluation ---
print(f"\nRunning COCO Evaluation (type: '{eval_type}')...")
try:
    coco_eval = COCOeval(coco_gt, coco_dt, eval_type)

    # Optional: Set specific evaluation parameters if needed
    # Example: Evaluate on specific IoU thresholds
    # coco_eval.params.iouThrs = np.array([0.5, 0.75])
    # Example: Set area ranges (COCO defaults are usually fine)
    # coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    # coco_eval.params.areaRngLbl = ['all', 'small', 'medium', 'large']
    # Example: Set max detections per image
    # coco_eval.params.maxDets = [1, 10, 100] # Default

    coco_eval.evaluate()    # Run per image evaluation computations
    coco_eval.accumulate()  # Accumulate results over all images
    coco_eval.summarize()   # Print the standard COCO summary
except Exception as e:
    print(f"Error during COCO evaluation steps: {e}")
    exit()

# --- Compute F1 Score ---
def compute_f1_scores(coco_eval, iou_threshold_idx=0):
    """
    Compute F1 scores at different IoU thresholds
    
    Args:
        coco_eval: COCOeval object after evaluate() and accumulate()
        iou_threshold_idx: Index to select IoU threshold (0=0.5, 5=0.95, etc.)
                         Default is 0 which corresponds to IoU=0.5
    
    Returns:
        Dictionary of F1 scores at different evaluation settings
    """
    # Get precision and recall arrays from evaluation
    precision = coco_eval.eval['precision']
    recall = coco_eval.eval['recall']
    
    # Extract F1 scores at specified IoU threshold
    # precision shape: [TxRxKxAxM] where:
    # T: IoU thresholds [0.5:0.05:0.95], 10 thresholds
    # R: recall thresholds [0:0.01:1], 101 points
    # K: category, 80 by default
    # A: area range, 4 by default (all, small, medium, large)
    # M: max detections, 3 by default (1, 10, 100)
    
    f1_scores = {}
    
    # For each category
    categories = coco_eval.params.catIds
    category_names = {cat: coco_gt.cats[cat]['name'] if cat in coco_gt.cats else f"cat_{cat}" 
                     for cat in categories}
    
    # Overall F1 (across all categories)
    # Get precision and recall for all categories, all areas, max dets=100
    # T=IoU threshold index, R=all recall points, K=all categories, A=all areas, M=max dets 100
    p = precision[iou_threshold_idx, :, :, 0, 2]  # [R, K]
    r = recall[iou_threshold_idx, :, 0, 2]  # [K]
    
    # Compute F1 for each category at each recall point
    f1 = np.zeros_like(p)
    for k_idx, _ in enumerate(categories):
        for r_idx in range(len(coco_eval.params.recThrs)):
            if p[r_idx, k_idx] + r[k_idx] > 0:  # Avoid division by zero
                f1[r_idx, k_idx] = 2 * p[r_idx, k_idx] * r[k_idx] / (p[r_idx, k_idx] + r[k_idx])
    
    # Get the maximum F1 score for each category
    max_f1_per_category = np.max(f1, axis=0)
    
    # Average F1 across all categories
    mean_f1 = np.mean(max_f1_per_category)
    f1_scores['mean_f1'] = float(mean_f1)
    
    # F1 for each category
    for k_idx, cat_id in enumerate(categories):
        f1_scores[f'f1_{category_names[cat_id]}'] = float(max_f1_per_category[k_idx])
    
    # F1 at different IoU thresholds
    if iou_threshold_idx == 0:  # Only compute for the first call (IoU=0.5)
        iou_thresholds = coco_eval.params.iouThrs
        f1_at_ious = {}
        for t_idx, iou_thr in enumerate(iou_thresholds):
            # Get precision and recall for this IoU threshold
            p_at_iou = precision[t_idx, :, :, 0, 2]  # [R, K]
            r_at_iou = recall[t_idx, :, 0, 2]  # [K]
            
            # Compute F1 for each category at this IoU
            f1_at_iou = np.zeros_like(p_at_iou)
            for k_idx, _ in enumerate(categories):
                for r_idx in range(len(coco_eval.params.recThrs)):
                    if p_at_iou[r_idx, k_idx] + r_at_iou[k_idx] > 0:
                        f1_at_iou[r_idx, k_idx] = 2 * p_at_iou[r_idx, k_idx] * r_at_iou[k_idx] / (p_at_iou[r_idx, k_idx] + r_at_iou[k_idx])
            
            # Maximum F1 for all categories at this IoU
            max_f1_at_iou = np.max(f1_at_iou, axis=0)
            f1_at_ious[f'f1_iou_{iou_thr:.2f}'] = float(np.mean(max_f1_at_iou))
        
        f1_scores.update(f1_at_ious)
    
    return f1_scores

# Compute F1 scores at IoU threshold 0.5 (index 0)
f1_scores = compute_f1_scores(coco_eval, iou_threshold_idx=0)

# --- Print Specific Evaluation Metrics ---
# Standard COCOeval stats indices:
# 0: AP @ IoU=0.50:0.95 | area=all | maxDets=100
# 1: AP @ IoU=0.50      | area=all | maxDets=100
# 2: AP @ IoU=0.75      | area=all | maxDets=100
# 3: AP @ IoU=0.50:0.95 | area=small | maxDets=100
# 4: AP @ IoU=0.50:0.95 | area=medium | maxDets=100
# 5: AP @ IoU=0.50:0.95 | area=large | maxDets=100
# 6: AR @ IoU=0.50:0.95 | area=all | maxDets=1
# 7: AR @ IoU=0.50:0.95 | area=all | maxDets=10
# 8: AR @ IoU=0.50:0.95 | area=all | maxDets=100
# 9: AR @ IoU=0.50:0.95 | area=small | maxDets=100
# 10: AR @ IoU=0.50:0.95 | area=medium | maxDets=100
# 11: AR @ IoU=0.50:0.95 | area=large | maxDets=100
print('\n--- Specific Evaluation Metrics ---')
if hasattr(coco_eval, 'stats') and len(coco_eval.stats) >= 12: # Check if stats exist and have expected length
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
    
    # Print F1 scores
    print("\n--- F1 Scores ---")
    print(f"F1_Score (IoU=0.50): {f1_scores['mean_f1']:.4f}")
    
    # Print F1 at different IoU thresholds if available
    for key, value in f1_scores.items():
        if key.startswith('f1_iou_') and key != 'f1_iou_0.50':
            print(f"{key}: {value:.4f}")
    
    # Print per-category F1 scores if available
    print("\n--- Per-Category F1 Scores (IoU=0.50) ---")
    for key, value in f1_scores.items():
        if key.startswith('f1_') and key != 'mean_f1' and not key.startswith('f1_iou_'):
            print(f"{key}: {value:.4f}")
else:
    print("Evaluation stats are not available or incomplete.")

print('---------------------------------')

# --- Clean up ---
# Optional: Remove the temporary file
# try:
#     os.remove(temp_dt_file)
#     print(f"Removed temporary file: {temp_dt_file}")
# except OSError as e:
#     print(f"Error removing temporary file {temp_dt_file}: {e}")