from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os # os.path.basename will be used
import numpy as np
import argparse
import datetime # Added for COCOeval eval['date'] if needed, though not directly used by F1

def main(gt_file_path, dt_file_path, eval_type_str):
    """
    Main function to perform COCO evaluation.
    """
    # --- Configuration (now from arguments) ---
    gt_file = gt_file_path
    dt_file = dt_file_path
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
        return
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

    # --- Create Image ID Mapping ---
    gt_images_by_basename = {} 
    if 'images' in coco_gt.dataset:
        for img in coco_gt.dataset['images']:
            if 'file_name' in img:
                base_name = os.path.basename(img['file_name'])
                if base_name in gt_images_by_basename:
                    print(f"Warning: Duplicate basename '{base_name}' in ground truth images. Mapping might be ambiguous. Keeping id: {gt_images_by_basename[base_name]}")
                else:
                    gt_images_by_basename[base_name] = img['id']
            else:
                print(f"Warning: Ground truth image missing file_name: {img}")

    dt_images_by_basename = {} 
    dt_image_ids_from_file_images_section = set()
    if isinstance(detection_data_full, dict) and 'images' in detection_data_full:
        for img in detection_data_full['images']:
            if 'file_name' in img:
                base_name = os.path.basename(img['file_name'])
                if base_name in dt_images_by_basename:
                     print(f"Warning: Duplicate basename '{base_name}' in detection images. Mapping might be ambiguous. Keeping id: {dt_images_by_basename[base_name]}")
                else:
                    dt_images_by_basename[base_name] = img['id'] 
                dt_image_ids_from_file_images_section.add(img['id']) 
            else:
                print(f"Warning: Detection image missing file_name: {img}")

    dt_image_ids_from_annotations = set()
    if detection_annotations:
        for anno in detection_annotations:
            if 'image_id' in anno:
                dt_image_ids_from_annotations.add(anno['image_id'])

    final_dt_image_ids = dt_image_ids_from_file_images_section if dt_image_ids_from_file_images_section else dt_image_ids_from_annotations
    print(f"Found {len(final_dt_image_ids)} unique image IDs in detections.")

    id_mapping = {}
    if gt_images_by_basename and dt_images_by_basename:
        print("Creating image ID mapping based on base file names...")
        mapped_count = 0
        unmapped_dt_basenames = 0
        for dt_basename, dt_id in dt_images_by_basename.items():
            if dt_basename in gt_images_by_basename:
                id_mapping[dt_id] = gt_images_by_basename[dt_basename]
                mapped_count +=1
            else:
                unmapped_dt_basenames += 1
        if unmapped_dt_basenames > 0:
            print(f"Warning: {unmapped_dt_basenames} basenames from detection 'images' section not found in ground truth 'images' section basenames.")
        print(f"Mapped {mapped_count} image IDs using basenames.")
    elif len(final_dt_image_ids) > 0 and len(final_dt_image_ids) == len(gt_image_ids) and not (gt_images_by_basename or dt_images_by_basename):
        print("Warning: Cannot map by filename. Trying sequential mapping (assumes same order and count of image IDs)...")
        sorted_dt_ids = sorted(list(final_dt_image_ids))
        sorted_gt_ids = sorted(list(gt_image_ids))
        for dt_id, gt_id in zip(sorted_dt_ids, sorted_gt_ids):
            id_mapping[dt_id] = gt_id
    else: 
        print("Warning: Filename or sequential mapping not fully applicable. Using identity mapping for matching IDs as a fallback...")
        for dt_id in final_dt_image_ids:
            if dt_id in gt_image_ids and dt_id not in id_mapping: 
                id_mapping[dt_id] = dt_id

    print(f"Created mapping for {len(id_mapping)} image IDs between detections and ground truth.")
    if not id_mapping and final_dt_image_ids and gt_image_ids:
        print("ERROR: Could not create any mapping between detection and ground truth image IDs.")
        # ... (error message)
        return

    mapped_detection_annotations = []
    unmapped_detection_ids_count = 0
    successfully_mapped_ids_in_annotations = set()

    for item in detection_annotations:
        if isinstance(item, dict) and 'image_id' in item:
            orig_image_id = item['image_id']
            if orig_image_id in id_mapping:
                mapped_item = item.copy()
                mapped_item['image_id'] = id_mapping[orig_image_id]
                mapped_detection_annotations.append(mapped_item)
                successfully_mapped_ids_in_annotations.add(orig_image_id)
            elif orig_image_id in gt_image_ids: 
                print(f"Info: Detection with image_id {orig_image_id} not in id_mapping but exists in gt_image_ids. Assuming it's already aligned.")
                mapped_detection_annotations.append(item) 
                successfully_mapped_ids_in_annotations.add(orig_image_id)
            else:
                unmapped_detection_ids_count += 1
    
    if unmapped_detection_ids_count > 0:
        print(f"Warning: {unmapped_detection_ids_count} detections had image_ids not found in the created mapping or directly in GT image IDs and were discarded.")
    print(f"Detections after ID mapping (based on {len(successfully_mapped_ids_in_annotations)} unique mapped/aligned detection image_ids): {len(mapped_detection_annotations)}")


    filtered_detection_annotations = [
        item for item in mapped_detection_annotations
        if item.get('image_id') in gt_image_ids 
    ]
    print(f"Detections after filtering for GT image IDs: {len(filtered_detection_annotations)}")

    if not filtered_detection_annotations and detection_annotations : 
        print("Warning: No detections remaining after filtering by ground truth image IDs. Evaluation might yield zero results.")

    try:
        with open(temp_dt_file, 'w') as f:
            json.dump(filtered_detection_annotations, f)
        print(f"Saved filtered and mapped detections to temporary file: {temp_dt_file}")
    except Exception as e:
        print(f"Error writing temporary filtered detections file {temp_dt_file}: {e}")
        return

    if not gt_image_ids: # Moved this check up
         print("No ground truth images. Skipping COCO API loading and evaluation.")
         # Clean up temp file if it was created
         if os.path.exists(temp_dt_file):
            try:
                os.remove(temp_dt_file)
            except OSError:
                pass # Ignore error if removal fails
         return


    if not filtered_detection_annotations: # Check if filtered list is empty
        print("No filtered detections to load. Evaluation will proceed with empty detections.")
        try:
            coco_dt = coco_gt.loadRes([]) 
            print("Loaded empty detections into COCO API.")
        except Exception as e:
            print(f"Error loading empty results: {e}")
            return 
    else:
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

    print(f"\nRunning COCO Evaluation (type: '{eval_type}')...")
    
    try:
        coco_eval = COCOeval(coco_gt, coco_dt, eval_type) 
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        print(f"Error during COCO evaluation steps: {e}")
        if not filtered_detection_annotations:
            print("This error might be due to evaluating with no detections.")
        return

    def compute_f1_scores(coco_eval_obj, current_coco_gt, iou_threshold_idx=0):
        if not hasattr(coco_eval_obj, 'eval') or 'precision' not in coco_eval_obj.eval:
            print("Warning: Precision data not found in coco_eval.eval. Cannot compute F1 scores.")
            return {}
            
        precision = coco_eval_obj.eval['precision']
        # s = coco_eval_obj.eval['scores'] # This variable was unused

        f1_scores_dict = {}
        
        category_names = {}
        if current_coco_gt.cats:
            category_names = {cat_id: current_coco_gt.cats[cat_id]['name'] 
                              for cat_id in coco_eval_obj.params.catIds 
                              if cat_id in current_coco_gt.cats}
            for cat_id in coco_eval_obj.params.catIds:
                if cat_id not in category_names: # Handle categories present in eval but not in GT's cat list (e.g. if GT has no annotations for it)
                    category_names[cat_id] = f"cat_{cat_id}_name_missing_in_gt"
        else:
            print("Warning: coco_gt.cats is empty. Using generic category names for F1 scores.")
            category_names = {cat_id: f"cat_{cat_id}" for cat_id in coco_eval_obj.params.catIds}
        
        if not coco_eval_obj.params.catIds:
            print("Warning: No category IDs found in coco_eval.params. F1 scores will be empty.")
            return {'mean_f1': 0.0}

        recall_thresholds = coco_eval_obj.params.recThrs 
        max_dets_idx = coco_eval_obj.params.maxDets.index(100) if 100 in coco_eval_obj.params.maxDets else -1

        # --- Calculate F1 for the specific iou_threshold_idx (e.g., IoU=0.50) ---
        # p_at_specific_iou shape: (num_recall_thresholds, num_categories)
        p_at_specific_iou = precision[iou_threshold_idx, :, :, 0, max_dets_idx]
        
        max_f1_per_category_at_specific_iou = []
        for k_idx, cat_id in enumerate(coco_eval_obj.params.catIds):
            if k_idx >= p_at_specific_iou.shape[1]: 
                print(f"Warning: Category index {k_idx} (ID: {cat_id}) out of bounds for precision array (IoU idx {iou_threshold_idx}). Skipping F1 for this cat.")
                continue
            
            prec_at_recalls = p_at_specific_iou[:, k_idx] # PR curve for this category
            
            current_max_f1 = 0.0
            for r_idx, rec_val in enumerate(recall_thresholds):
                if r_idx >= prec_at_recalls.shape[0]: 
                    continue
                prec_val = prec_at_recalls[r_idx]
                
                # MODIFICATION START: Handle prec_val = -1
                current_prec_for_f1 = 0.0
                if prec_val > -1: # Valid precision value (can be 0 or positive)
                    current_prec_for_f1 = prec_val
                # If prec_val is -1, current_prec_for_f1 remains 0.0
                
                if current_prec_for_f1 + rec_val > 0: # If either P or R (or both) are > 0
                    f1 = 2 * current_prec_for_f1 * rec_val / (current_prec_for_f1 + rec_val)
                else: # Both P_for_f1 and R are 0. F1 is 0.
                    f1 = 0.0
                # MODIFICATION END
                
                if f1 > current_max_f1:
                    current_max_f1 = f1
            max_f1_per_category_at_specific_iou.append(current_max_f1)
            cat_name_key = category_names.get(cat_id, f"cat_{cat_id}")
            f1_scores_dict[f'f1_{cat_name_key}'] = float(current_max_f1)

        mean_f1_at_specific_iou = np.mean(max_f1_per_category_at_specific_iou) if max_f1_per_category_at_specific_iou else 0.0
        f1_scores_dict['mean_f1'] = float(mean_f1_at_specific_iou) # This is mean F1 @ IoU specified by iou_threshold_idx
        
        # --- Calculate F1 scores for all IoU thresholds specified in coco_eval_obj.params.iouThrs ---
        # This block runs if the initial call to compute_f1_scores was for iou_threshold_idx=0 (typically IoU=0.5)
        # to avoid redundant calculations if this function were called multiple times with different iou_threshold_idx.
        if iou_threshold_idx == 0: # Standard call is with iou_threshold_idx=0
            iou_thrs_values = coco_eval_obj.params.iouThrs

            for t_idx, iou_val in enumerate(iou_thrs_values):
                if t_idx >= precision.shape[0]: continue 
                
                # p_at_current_iter_iou shape: (num_recall_thresholds, num_categories)
                p_at_current_iter_iou = precision[t_idx, :, :, 0, max_dets_idx] 
                
                current_max_f1_at_iter_iou_per_category = []
                for k_idx, cat_id in enumerate(coco_eval_obj.params.catIds):
                    if k_idx >= p_at_current_iter_iou.shape[1]: continue 
                    
                    prec_at_recalls_for_iter_iou = p_at_current_iter_iou[:, k_idx]
                    
                    cat_max_f1_at_iter_iou = 0.0
                    for r_idx, rec_val in enumerate(recall_thresholds):
                        if r_idx >= prec_at_recalls_for_iter_iou.shape[0]: continue 
                        prec_val = prec_at_recalls_for_iter_iou[r_idx]
                        
                        # MODIFICATION START: Handle prec_val = -1
                        current_prec_for_f1 = 0.0
                        if prec_val > -1:
                            current_prec_for_f1 = prec_val
                        
                        if current_prec_for_f1 + rec_val > 0:
                            f1 = 2 * current_prec_for_f1 * rec_val / (current_prec_for_f1 + rec_val)
                        else:
                            f1 = 0.0
                        # MODIFICATION END
                        
                        if f1 > cat_max_f1_at_iter_iou:
                            cat_max_f1_at_iter_iou = f1
                    current_max_f1_at_iter_iou_per_category.append(cat_max_f1_at_iter_iou)
                
                mean_f1_at_iter_iou = np.mean(current_max_f1_at_iter_iou_per_category) if current_max_f1_at_iter_iou_per_category else 0.0
                f1_scores_dict[f'f1_iou_{iou_val:.2f}'] = float(mean_f1_at_iter_iou)
                
        return f1_scores_dict

    f1_scores = compute_f1_scores(coco_eval, coco_gt, iou_threshold_idx=0) # iou_threshold_idx=0 corresponds to IoU=0.5 typically

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
        # This 'mean_f1' is the mean F1 over categories at the specific IoU passed to compute_f1_scores (default 0.50)
        print(f"F1_Score (mean over categories @ IoU={coco_eval.params.iouThrs[0]:.2f}): {f1_scores.get('mean_f1', 0.0):.4f}")

        print("\n--- Mean F1 Scores at different IoUs (mean over categories) ---")
        for key, value in sorted(f1_scores.items()): # Sort for consistent output
            if key.startswith('f1_iou_'):
                print(f"{key.replace('f1_iou_', 'F1@IoU='):<12}: {value:.4f}")
        
        print(f"\n--- Per-Category F1 Scores (@ IoU={coco_eval.params.iouThrs[0]:.2f}) ---")
        for key, value in sorted(f1_scores.items()): # Sort for consistent output
            if key.startswith('f1_') and not key.startswith('f1_iou_') and key != 'mean_f1':
                print(f"{key:<25}: {value:.4f}") # Adjusted spacing for category names
    else:
        print("Evaluation stats are not available or incomplete. This can happen if there are no GT annotations or no detections.")

    print('---------------------------------')

    try:
        if os.path.exists(temp_dt_file):
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