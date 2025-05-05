# --- START OF FILE trt_inf_realtime.py ---

"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Optimized for real-time (Batch Size 1) inference and FPS measurement.
Uses TensorRT 8.x style API (num_io_tensors) based on working script.
Includes fix for dtype mismatch when using --fp16 flag and NameError fix.
"""

import time
import contextlib
import collections
from collections import OrderedDict
import json
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

import torch
import torchvision.transforms as T

import tensorrt as trt
# import cv2
import os
import sys
import traceback
from tqdm import tqdm

# --- Global Constants ---
CATEGORY_MAP = {
    0: "Bus", 1: "Bike", 2: "Car", 3: "Pedestrian", 4: "Truck"
    # Add more categories if needed
}

# --- Helper Functions ---

@torch.no_grad()
def box_cxcywh_to_xyxy(x, size_wh):
    """Converts boxes from [cx, cy, w, h] (relative) to [x1, y1, x2, y2] (absolute)."""
    if x.numel() == 0: return torch.empty((0, 4), dtype=x.dtype, device=x.device)
    w, h = size_wh
    if w <= 0 or h <= 0: return torch.empty((0, 4), dtype=x.dtype, device=x.device)
    if x.shape[-1] != 4: raise ValueError(f"Input tensor last dim should be 4, got {x.shape}")
    cx, cy, w_box, h_box = x.unbind(-1)
    x1 = (cx - 0.5 * w_box) * w; y1 = (cy - 0.5 * h_box) * h
    x2 = (cx + 0.5 * w_box) * w; y2 = (cy + 0.5 * h_box) * h
    return torch.stack((x1, y1, x2, y2), dim=-1)

class TimeProfiler(contextlib.ContextDecorator):
    """Helper for timing CUDA events."""
    def __init__(self, device):
        self.total = 0; self.count = 0; self.device = device
    def __enter__(self):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        self.start = time.perf_counter(); return self
    def __exit__(self, type, value, traceback):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        self.total += time.perf_counter() - self.start; self.count += 1
    def reset(self): self.total = 0; self.count = 0
    def average_time_ms(self): return (self.total / self.count) * 1000 if self.count > 0 else 0

# THIS FUNCTION MUST BE DEFINED BEFORE main() CALLS IT
def create_coco_annotation(image_id, label, box, score):
    """Creates COCO annotation dict from absolute xyxy box."""
    x1, y1, x2, y2 = box
    x1_f, y1_f = max(0.0, float(x1)), max(0.0, float(y1))
    x2_f, y2_f = max(x1_f, float(x2)), max(y1_f, float(y2))
    width = x2_f - x1_f
    height = y2_f - y1_f
    if width < 1e-3 or height < 1e-3: return None
    return {"image_id": image_id, "category_id": int(label),
            "bbox": [x1_f, y1_f, width, height], "score": float(score),
            "area": float(width * height), "iscrowd": 0, "id": -1} # ID assigned later

# --- TRTInference Class Definition ---
class TRTInference(object):
    """Handles TensorRT engine loading and inference. Optimized for BS=1. (Uses num_io_tensors API)"""
    def __init__(self, engine_path, device='cuda:0', use_fp16=False, verbose=False):
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.torch_dtype_flag = torch.float16 if use_fp16 else torch.float32
        self.verbose = verbose
        self.max_batch_size = 1

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        if not self.engine: raise RuntimeError(f"Engine load failed: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None: raise RuntimeError("Failed to create Execution Context.")

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device, use_fp16)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler(self.device)

        self._validate_fp16_engine(use_fp16)

        print(f"--- TRTInference Initialized (Optimized for BS=1, Uses num_io_tensors API) ---")
        print(f"Engine: {os.path.basename(engine_path)}")
        print(f"Device: {self.device}")
        print(f"FP16 Flag Active: {use_fp16}")
        print(f"Input Names: {self.input_names}")
        print(f"Output Names: {self.output_names}")
        print(f"----------------------------------------------------")

    def _validate_fp16_engine(self, use_fp16):
        """Checks if engine likely supports FP16 if use_fp16 is True."""
        if use_fp16 and self.input_names:
            try:
                first_input_name = self.input_names[0]
                if first_input_name in self.bindings:
                    allocated_dtype = self.bindings[first_input_name].data.dtype
                    engine_np_dtype = self.bindings[first_input_name].dtype
                    if allocated_dtype == torch.float16:
                         if self.verbose: print(f"Engine input '{first_input_name}' allocated as FP16.")
                    else:
                        print(f"Warning: --fp16 flag set, but input '{first_input_name}' allocated as {allocated_dtype} (engine reports {engine_np_dtype}).")
                else: print(f"Warning: Binding '{first_input_name}' not found for FP16 validation.")
            except Exception as e: print(f"Warning: Error during FP16 validation: {e}")

    def load_engine(self, path):
        """Loads a TensorRT engine from file."""
        if not os.path.exists(path): print(f"Error: Engine file not found: {path}"); return None
        print(f"Loading TRT engine from {path}...")
        trt.init_libnvinfer_plugins(self.logger, '')
        try:
            with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime: engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None: print(f"Error: Failed to deserialize engine from {path}"); return None
            print("TRT engine loaded successfully."); return engine
        except Exception as e: print(f"Error loading TRT engine: {e}"); traceback.print_exc(); return None

    def get_input_names(self):
        """Gets input tensor names (using num_io_tensors API primarily)."""
        names = []
        if not hasattr(self.engine, 'num_io_tensors'): raise AttributeError("Engine object missing 'num_io_tensors'.")
        num_tensors = self.engine.num_io_tensors
        for i in range(num_tensors):
            try:
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT: names.append(name)
            except Exception as e: print(f"Error getting input info at index {i}: {e}")
        return names

    def get_output_names(self):
        """Gets output tensor names (using num_io_tensors API primarily)."""
        names = []
        if not hasattr(self.engine, 'num_io_tensors'): raise AttributeError("Engine object missing 'num_io_tensors'.")
        num_tensors = self.engine.num_io_tensors
        for i in range(num_tensors):
             try:
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT: names.append(name)
             except Exception as e: print(f"Error getting output info at index {i}: {e}")
        return names

    def get_bindings(self, engine, context, max_batch_size, device, use_fp16) -> OrderedDict:
        """Allocates memory for inputs and outputs (using num_io_tensors API, BS=1)."""
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        if not hasattr(engine, 'num_io_tensors'): raise AttributeError("Engine object missing 'num_io_tensors'.")
        num_tensors = engine.num_io_tensors

        for i in range(num_tensors):
            try:
                name = engine.get_tensor_name(i)
                trt_dtype = engine.get_tensor_dtype(name)
                raw_shape = engine.get_tensor_shape(name); shape = list(raw_shape)
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                is_dynamic = -1 in raw_shape
            except Exception as e: print(f"Error getting tensor info for index {i}: {e}"); continue

            np_dtype = trt.nptype(trt_dtype)

            if len(shape) > 0:
                 if shape[0] == -1 or shape[0] > max_batch_size: shape[0] = max_batch_size

            if is_input and is_dynamic:
                try: context.set_input_shape(name, tuple(shape))
                except Exception as e: print(f"Warning: Failed set input shape for '{name}': {e}.")

            try:
                alloc_shape = tuple(shape)
                if is_input and use_fp16 and trt_dtype == trt.DataType.HALF: torch_dtype = torch.float16
                else: torch_dtype = torch.from_numpy(np.array([], dtype=np_dtype)).dtype
                data = torch.empty(alloc_shape, dtype=torch_dtype, device=device)
                bindings[name] = Binding(name, np_dtype, alloc_shape, data, data.data_ptr())
                if self.verbose: print(f"Binding alloc: {name}, Shape: {alloc_shape}, Alloc Dtype: {torch_dtype}, Engine Dtype: {np_dtype}")
            except Exception as e: print(f"Error alloc memory for '{name}': {e}"); raise
        return bindings

    @torch.no_grad()
    def __call__(self, blob):
        """Runs inference for a single image (BS=1). Handles dtype conversion."""
        for n in self.input_names:
            if n not in blob: raise ValueError(f"Missing input: {n}")
            input_tensor = blob[n]
            if input_tensor.device != self.device: input_tensor = input_tensor.to(self.device)
            expected_engine_dtype = self.bindings[n].data.dtype
            if expected_engine_dtype == torch.float16 and input_tensor.dtype == torch.float32: input_tensor = input_tensor.half()
            elif input_tensor.dtype != expected_engine_dtype: raise TypeError(f"Input '{n}' dtype mismatch: Engine expected {expected_engine_dtype}, Got {input_tensor.dtype}")
            if not input_tensor.is_contiguous(): input_tensor = input_tensor.contiguous()
            self.bindings_addr[n] = input_tensor.data_ptr()
        try:
            success = self.context.execute_v2(bindings=list(self.bindings_addr.values()))
            if not success: raise RuntimeError("TensorRT execute_v2 failed.")
        except Exception as e: print(f"Error during TRT execution: {e}"); traceback.print_exc(); raise
        outputs = {n: self.bindings[n].data for n in self.output_names}
        return outputs
# --- End of TRTInference Class ---


# --- More Helper Functions ---

@torch.no_grad()
def process_single_image(model: TRTInference, image_path: str, transforms: T.Compose, device: torch.device, conf_thresh: float):
    """Loads, preprocesses, infers, and postprocesses a single image."""
    try:
        im_pil = Image.open(image_path).convert('RGB')
        w, h = im_pil.size
        if w <= 0 or h <= 0: print(f"Warning: Skipping image with invalid dims W={w}, H={h}: {image_path}"); return None, None
        im_data = transforms(im_pil).unsqueeze(0).to(device) # FP32 tensor
        orig_size_tensor = torch.tensor([[h, w]], dtype=torch.int64).to(device)
        blob = {}
        missing_inputs = []
        # --- Blob Preparation (Verify Keys) ---
        required_inputs = set(model.input_names)
        if 'images' in required_inputs: blob['images'] = im_data; required_inputs.remove('images')
        if 'orig_target_sizes' in required_inputs: blob['orig_target_sizes'] = orig_size_tensor; required_inputs.remove('orig_target_sizes')
        if required_inputs: raise ValueError(f"Engine requires unhandled inputs: {required_inputs}.")
        # --- End Blob Preparation ---
        outputs = model(blob)
        required_outputs = {'scores', 'labels', 'boxes'} # Use set for efficient check
        if not required_outputs.issubset(outputs.keys()): print(f"Warning: Missing outputs for {image_path}"); return None, None
        scores = outputs['scores'][0]; labels = outputs['labels'][0]; boxes_relative = outputs['boxes'][0]
        mask = scores > conf_thresh; filtered_scores = scores[mask]
        num_detections = len(filtered_scores)
        if num_detections == 0: return [], im_pil
        filtered_labels = labels[mask]; filtered_boxes_rel = boxes_relative[mask]
        orig_size_wh_tensor = torch.tensor([w, h], dtype=torch.float32, device=device)
        absolute_boxes_xyxy = box_cxcywh_to_xyxy(filtered_boxes_rel, orig_size_wh_tensor)
        processed_results = {'labels': filtered_labels.cpu().numpy(), 'boxes_abs_xyxy': absolute_boxes_xyxy.cpu().numpy(), 'scores': filtered_scores.cpu().numpy()}
        return processed_results, im_pil
    except FileNotFoundError: print(f"Error: Image file not found: {image_path}"); return None, None
    except Exception as e: print(f"Error processing {os.path.basename(image_path)}: {e}"); traceback.print_exc(); return None, None

def get_image_files(folder_path, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
    """Gets image files."""
    image_files = []
    if not isinstance(folder_path, str) or not os.path.isdir(folder_path): print(f"Error: Invalid dir: {folder_path}"); return []
    print(f"Searching images in: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions): image_files.append(os.path.join(root, file))
    print(f"Found {len(image_files)} images."); return sorted(image_files)

def build_coco_format_json(all_annotations, image_files_info, category_map):
    """Builds COCO JSON."""
    if not isinstance(all_annotations, list) or not isinstance(image_files_info, dict) or not isinstance(category_map, dict): print("Error: Invalid inputs for build_coco."); return None
    print("Building COCO JSON output...")
    images_list = []; categories_list = []
    for img_id, info in image_files_info.items():
         if isinstance(info, dict) and all(k in info for k in ['filename', 'width', 'height']): images_list.append({"id": img_id, "file_name": info['filename'], "height": info['height'], "width": info['width']})
         else: print(f"Warning: Skipping invalid image info for img_id {img_id}")
    categories_list = [{"id": k, "name": v, "supercategory": "object"} for k, v in category_map.items()]
    # Assign unique IDs to annotations (starting from 1 is common)
    for i, anno in enumerate(all_annotations):
        # Check if anno is a dictionary before assigning id
        if isinstance(anno, dict):
            anno["id"] = i + 1
        else:
             print(f"Warning: Found non-dict item in annotations list at index {i}. Skipping ID assignment.")

    coco_format = {
        "info": {"description": "TRT Realtime Inference Results (num_io_tensors API)", "version": "1.0", "year": datetime.now().year, "contributor":"script", "date_created": datetime.now().strftime("%Y/%m/%d %H:%M:%S")},
        "licenses": [], "images": images_list, "annotations": all_annotations, "categories": categories_list }
    print(f"COCO JSON built: {len(images_list)} images, {len(all_annotations)} annotations."); return coco_format


# --- Main Function ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run TensorRT inference (BS=1, num_io_tensors API) for max FPS.")
    parser.add_argument('-trt', '--trt', type=str, required=True, help='Path to TensorRT engine file (.engine)')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input image file or directory')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device for inference (e.g., cuda:0)')
    parser.add_argument('-o', '--output', type=str, default='results_coco_trt_realtime.json', help='Output COCO JSON file name')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision (engine must support it)')
    parser.add_argument('--vis', type=str, default=None, help='Optional: Dir to save visualization images (first 5 processed)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose TRT logging')
    parser.add_argument('--skip_coco', action='store_true', help='Skip generating COCO JSON output (for pure FPS measure)')
    args = parser.parse_args()

    # --- Validation ---
    if not os.path.exists(args.trt): print(f"Error: TRT engine not found: {args.trt}"); return 1
    if not os.path.exists(args.input): print(f"Error: Input path not found: {args.input}"); return 1
    try:
        device = torch.device(args.device)
        if device.type == 'cuda':
             if not torch.cuda.is_available(): print(f"Error: CUDA not available."); return 1
             torch.cuda.set_device(device); print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        else: print(f"Using CPU device: {device}")
    except Exception as e: print(f"Error setting device '{args.device}': {e}"); return 1
    if args.vis:
        if not os.path.isdir(args.vis):
            try: os.makedirs(args.vis); print(f"Created vis dir: {args.vis}")
            except Exception as e: print(f"Warning: Error creating vis dir {args.vis}: {e}. Vis disabled."); args.vis = None
        else: print(f"Using vis dir: {args.vis}")

    # --- Initialize Model ---
    try: model = TRTInference(args.trt, device=args.device, use_fp16=args.fp16, verbose=args.verbose)
    except Exception as e: print(f"Error initializing TRTInference: {e}"); return 1

    # --- Prepare Transforms ---
    try:
        transforms = T.Compose([
            T.Resize((640, 640)), # Must match model input size
            T.ToTensor(),
            # Correct Python comment:
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    except Exception as e: print(f"Error creating transforms: {e}"); return 1

    # --- Get Image Files ---
    valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp'); image_paths = []
    if os.path.isdir(args.input): image_paths = get_image_files(args.input, extensions=valid_image_extensions)
    elif os.path.isfile(args.input) and args.input.lower().endswith(valid_image_extensions): image_paths = [args.input]
    else: print(f"Error: Invalid input path: {args.input}"); return 1
    if not image_paths: print("No valid images found."); return 0

    # --- Inference and Postprocessing Loop ---
    all_coco_annotations = []; image_files_info = {}; total_processed_count = 0; vis_count = 0; max_vis_images = 5
    print(f"Starting inference for {len(image_paths)} images...")
    overall_timer = TimeProfiler(device)

    # --- Optional Warmup ---
    if len(image_paths) > 10:
         print("Performing warmup..."); warmup_path = image_paths[0]
         try:
             for _ in range(5): _, _ = process_single_image(model, warmup_path, transforms, device, args.threshold)
             print("Warmup complete.")
         except Exception as w_e: print(f"Warmup failed: {w_e}")

    overall_timer.reset()
    with overall_timer:
        for img_id, img_path in enumerate(tqdm(image_paths, desc="Processing Images", unit="image")):
            processed_results, pil_image = process_single_image(model, img_path, transforms, device, args.threshold)
            if processed_results is not None:
                total_processed_count += 1
                # --- Store results for COCO (if not skipped) ---
                if not args.skip_coco:
                    if pil_image: w, h = pil_image.size; image_files_info[img_id] = {'filename': os.path.basename(img_path), 'width': w, 'height': h}
                    else: image_files_info[img_id] = {'filename': os.path.basename(img_path), 'width': -1, 'height': -1}

                    # PROBLEM IS HERE: Need to call the globally defined function
                    if isinstance(processed_results, dict): # Check if detections exist
                        labels = processed_results['labels']; boxes = processed_results['boxes_abs_xyxy']; scores = processed_results['scores']
                        for i in range(len(scores)):
                            label_id = labels[i].item()
                            if label_id in CATEGORY_MAP:
                                 # *** CORRECT CALL ***
                                 coco_anno = create_coco_annotation(img_id, label_id, boxes[i], scores[i].item())
                                 if coco_anno: all_coco_annotations.append(coco_anno)

                # --- Optional Visualization ---
                if args.vis and pil_image and vis_count < max_vis_images:
                     try:
                        if isinstance(processed_results, dict) and len(processed_results['scores']) > 0:
                            draw = ImageDraw.Draw(pil_image); boxes = processed_results['boxes_abs_xyxy']; labels = processed_results['labels']; scores = processed_results['scores']
                            for k in range(len(scores)):
                                label_id = labels[k].item(); label_name = CATEGORY_MAP.get(label_id, f"ID:{label_id}"); score_txt = f"{scores[k]:.2f}"
                                box_k = boxes[k]; draw_box = [max(0, box_k[0]), max(0, box_k[1]), min(pil_image.width, box_k[2]), min(pil_image.height, box_k[3])]
                                draw.rectangle(draw_box, outline='lime', width=2)
                                draw.text((max(0, draw_box[0]), max(0, draw_box[1] - 10)), f"{label_name}: {score_txt}", fill='red')
                        safe_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in os.path.basename(img_path))
                        vis_filename = os.path.join(args.vis, f"vis_{img_id}_{safe_filename}"); vis_filename += '.jpg' if not vis_filename.lower().endswith(('.png','.jpg','.jpeg')) else ''
                        pil_image.save(vis_filename); vis_count += 1
                     except Exception as vis_e: print(f"Warning: Vis saving error: {vis_e}")

    # --- Timing & FPS ---
    total_processing_time = overall_timer.total
    fps = total_processed_count / total_processing_time if total_processing_time > 0 and total_processed_count > 0 else 0

    # --- COCO Output ---
    if not args.skip_coco and total_processed_count > 0:
        print("Starting COCO JSON generation...")
        try:
            coco_json_output = build_coco_format_json(all_coco_annotations, image_files_info, CATEGORY_MAP)
            if coco_json_output:
                coco_json_output["performance"] = {"total_items_found": len(image_paths), "total_items_processed": total_processed_count,"total_annotations_generated": len(all_coco_annotations), "confidence_threshold": args.threshold,"batch_size": 1, "fp16_enabled": args.fp16, "total_processing_time_seconds": round(total_processing_time, 4),"average_fps": round(fps, 2)}
                output_dir = os.path.dirname(args.output);
                if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
                with open(args.output, 'w') as f: json.dump(coco_json_output, f, indent=2)
                print(f"\nCOCO annotations saved to: {args.output}")
            else: print("\nCOCO JSON generation failed.")
        except Exception as e: print(f"\nError during COCO processing: {e}"); traceback.print_exc()
    elif args.skip_coco: print("\nSkipped COCO JSON generation.")
    else: print("\nNo images processed. Skipping COCO JSON.")

    # --- Summary ---
    print(f"\n--- Processing Summary (BS=1 Optimized, num_io_tensors API) ---")
    print(f"Engine: {os.path.basename(args.trt)}")
    print(f"Input: {args.input}")
    print(f"Device: {args.device}, FP16 Flag: {args.fp16}")
    print(f"Images Found: {len(image_paths)}")
    print(f"Images Processed: {total_processed_count}")
    print(f"Total Processing Time (loop): {total_processing_time:.3f} seconds")
    print(f"Average FPS: {fps:.2f}")
    if not args.skip_coco: print(f"Annotations Generated: {len(all_coco_annotations)}")
    if args.vis: print(f"Visualizations (up to {max_vis_images}) saved in: {args.vis}")
    print(f"---------------------------------------------")

    return 0

# --- Execution Guard ---
if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)

# --- END OF FILE trt_inf_realtime.py ---