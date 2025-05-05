# --- START OF FILE trt_inf.py ---

"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import time
import contextlib
import collections
from collections import OrderedDict
import json
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw # Keep ImageDraw for potential future visualization

import torch
import torchvision.transforms as T

import tensorrt as trt
import cv2
import os
from tqdm import tqdm # Added for progress bars

# --- Category Map ---
# As requested: Bus: 0, Bike: 1, Car: 2, Pedestrian: 3, Truck: 4
# Ensure your model outputs these category IDs directly.
# If your model outputs different IDs, adjust this map accordingly
# or map the model's output IDs to these standard IDs within the processing loop.
CATEGORY_MAP = {
    0: "Bus",
    1: "Bike",
    2: "Car",
    3: "Pedestrian",
    4: "Truck"
    # Add more if your model supports them and you want them in the output JSON
}

class TimeProfiler(contextlib.ContextDecorator):
    """Helper for timing CUDA events."""
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

class TRTInference(object):
    """Handles TensorRT engine loading and inference."""
    def __init__(self, engine_path, device='cuda:0', backend='torch', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend # Only 'torch' backend is fully implemented here
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        # Check if the context was created successfully
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT Execution Context. Check engine compatibility and GPU resources.")

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler() # For potential per-inference timing if needed

        print(f"TRT Engine Input Names: {self.input_names}")
        print(f"TRT Engine Output Names: {self.output_names}")
        # Expecting inputs like 'images' and 'orig_target_sizes'
        # Expecting outputs like 'scores', 'labels', 'boxes'

    def load_engine(self, path):
        """Loads a TensorRT engine from file."""
        print(f"Loading TRT engine from {path}...")
        trt.init_libnvinfer_plugins(self.logger, '')
        try:
            with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to deserialize engine from {path}")
            print("TRT engine loaded successfully.")
            return engine
        except Exception as e:
            print(f"Error loading TensorRT engine: {e}")
            raise

    def get_input_names(self):
        """Gets the names of the engine's input tensors."""
        names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        """Gets the names of the engine's output tensors."""
        names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        """Allocates memory for inputs and outputs."""
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        dynamic_inputs = False
        has_dynamic_shapes = False # Flag to check if any binding is dynamic

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            # Use get_binding_shape for consistency if context is available, fallback to engine shape
            # Note: get_binding_shape requires the binding index, not name
            # binding_idx = engine.get_binding_index(name) # TensorRT 8+ might use get_tensor_name(index) consistently
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            # Use engine shape primarily, context shape might be needed for dynamic setting
            shape = list(engine.get_tensor_shape(name)) # Use list for mutability

            # Check for dynamic dimensions (-1)
            if -1 in shape:
                has_dynamic_shapes = True
                print(f"Binding '{name}' has dynamic dimensions: {tuple(shape)}")
                # Set default profile shape if dynamic, typically involving batch size
                if shape[0] == -1:
                    shape[0] = max_batch_size # Set max batch size for allocation
                    if is_input:
                        dynamic_inputs = True
                        # Set initial shape context for dynamic input using set_input_shape
                        try:
                            context.set_input_shape(name, tuple(shape))
                            print(f"Dynamic input '{name}' detected, setting initial context shape to {tuple(shape)}")
                        except Exception as e:
                            print(f"Warning: Failed to set initial input shape for dynamic input '{name}': {e}. "
                                  f"Ensure engine profile supports this shape or handle shape setting during inference.")
                # Potentially handle other dynamic dims if needed, e.g., sequence length
                # else:
                #    print(f"Warning: Dynamic dimension found in non-batch axis for '{name}'. Ensure handling is correct.")

            # Allocate memory on the target device
            try:
                # Ensure shape is a tuple for torch.empty
                alloc_shape = tuple(shape)
                # Convert numpy dtype to torch dtype
                torch_dtype = torch.from_numpy(np.array([], dtype=dtype)).dtype
                data = torch.empty(alloc_shape, dtype=torch_dtype, device=device)
                bindings[name] = Binding(name, dtype, alloc_shape, data, data.data_ptr())
                print(f"Binding allocated: {name}, Shape: {alloc_shape}, Dtype: {dtype}")
            except Exception as e:
                 print(f"Error allocating memory for binding '{name}' with shape {alloc_shape} and dtype {dtype} on device {device}: {e}")
                 raise

        if not has_dynamic_shapes:
             print("Engine reports static shapes for all bindings.")
        elif not dynamic_inputs and has_dynamic_shapes:
             print("Engine has dynamic shapes, but only in outputs (or handled by profile).")

        return bindings

    def run_torch(self, blob):
        """Runs inference using PyTorch tensors."""
        # Keep track of the actual batch size used in this run
        actual_batch_size = 0

        # --- 1. Prepare Inputs ---
        for n in self.input_names:
            if n not in blob:
                 raise ValueError(f"Missing required input in blob: {n}")

            input_tensor = blob[n]
            # Ensure data is on the correct device
            if str(input_tensor.device) != str(self.device):
                 input_tensor = input_tensor.to(self.device)
                 # print(f"Warning: Input '{n}' moved to device {self.device}") # Optional warning

            # Ensure data is contiguous in memory
            if not input_tensor.is_contiguous():
                 input_tensor = input_tensor.contiguous()
                 # print(f"Warning: Input tensor '{n}' was not contiguous. Made it contiguous.") # Optional warning

            # Track the batch size (assuming first dim is batch)
            current_batch_size = input_tensor.shape[0]
            if actual_batch_size == 0:
                 actual_batch_size = current_batch_size
            elif actual_batch_size != current_batch_size:
                 # This case might occur if inputs have different batch dims, which is unusual
                 print(f"Warning: Inconsistent batch sizes detected among inputs ('{n}' shape {input_tensor.shape[0]}, expected {actual_batch_size})")
                 # Decide how to handle: error out, or use the first one? Using first for now.

            # --- Shape Handling ---
            current_binding = self.bindings[n]
            # For dynamic input shapes, update the context if the current tensor shape differs
            # from the shape currently set in the context (or the initial max shape).
            # Check if this input binding is dynamic (had -1 initially)
            is_dynamic_input = -1 in engine.get_tensor_shape(n) # Check original engine shape def

            if is_dynamic_input and current_binding.shape != input_tensor.shape:
                # Only set if the shape is actually different from the last time or initial set
                # This check might be redundant if TRT optimizes internally, but safer
                try:
                    self.context.set_input_shape(n, input_tensor.shape)
                    # Update our internal tracking of the binding shape - ptr remains the same until next step
                    self.bindings[n] = current_binding._replace(shape=input_tensor.shape)
                    # print(f"Updated context shape for dynamic input '{n}' to: {input_tensor.shape}") # Optional info
                except Exception as e:
                     print(f"Error setting input shape for '{n}' to {input_tensor.shape}: {e}. Engine profile might not support this shape.")
                     raise # Re-raise as this is likely a fatal error for dynamic shapes

            # --- Dtype Check ---
            expected_torch_dtype = torch.from_numpy(np.array([], dtype=current_binding.dtype)).dtype
            if input_tensor.dtype != expected_torch_dtype:
                 raise TypeError(f"Input '{n}' dtype mismatch: Expected {expected_torch_dtype}, Got {input_tensor.dtype}")

            # --- Update Pointer ---
            # Assign the data pointer of the *potentially modified* input_tensor to the binding address dictionary
            self.bindings_addr[n] = input_tensor.data_ptr()


        # --- 2. Execute Inference ---
        try:
            # Ensure all input pointers are set before executing
            # The list conversion ensures order for older Python versions
            # Use execute_v2 for unified sync/async based on context properties (though we use sync here)
            success = self.context.execute_v2(list(self.bindings_addr.values()))
            if not success:
                 raise RuntimeError("TensorRT execution failed. Check logs for details.")
        except Exception as e:
            print(f"Error during TensorRT execute_v2: {e}")
            # Consider dumping context state or shapes here for debugging if possible
            raise # Re-raise the execution error

        # --- 3. Process Outputs ---
        outputs = {n: self.bindings[n].data for n in self.output_names}

        # Slice the output tensors to the actual batch size processed.
        # This is crucial if the allocated buffer (using max_batch_size) is larger than the actual_batch_size.
        sliced_outputs = {}
        for n, output_tensor in outputs.items():
             # Assume outputs have batch dimension first. Check if tensor has enough dimensions.
             if len(output_tensor.shape) > 0 and output_tensor.shape[0] == self.max_batch_size:
                 # Only slice if the allocated dimension matches max_batch_size (indicating potential padding)
                 # and the actual batch size is smaller.
                 if actual_batch_size < self.max_batch_size:
                    # print(f"Slicing output '{n}' from shape {output_tensor.shape} to batch size {actual_batch_size}") # Optional info
                    try:
                        sliced_outputs[n] = output_tensor[:actual_batch_size]
                    except Exception as slice_err:
                        print(f"Error slicing output '{n}' (shape {output_tensor.shape}) to size {actual_batch_size}: {slice_err}")
                        # Fallback to unsliced tensor if slicing fails unexpectedly
                        sliced_outputs[n] = output_tensor
                 else:
                     # Actual batch size matches max batch size, no slicing needed
                     sliced_outputs[n] = output_tensor
             else:
                 # Output tensor might not have a batch dim, or its first dim doesn't match max_batch_size
                 # (e.g., output shape depends on input values, or it's already correctly sized).
                 # Assume it's correctly sized by TRT.
                 sliced_outputs[n] = output_tensor


        return sliced_outputs # Return potentially sliced outputs

    def __call__(self, blob):
        """Performs inference based on the configured backend."""
        if self.backend == 'torch':
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        """Synchronizes the CUDA stream if using PyTorch backend."""
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()

def create_coco_annotation(image_id, label, box, score):
    """
    Create a COCO format annotation dictionary.
    Assumes 'label' is the integer category ID defined in CATEGORY_MAP.
    """
    x1, y1, x2, y2 = box
    # Ensure coordinates are reasonable (non-negative, x2>x1, y2>y1)
    x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
    x2, y2 = max(x1, float(x2)), max(y1, float(y2)) # Ensure x2>=x1, y2>=y1

    width = x2 - x1
    height = y2 - y1

    # Basic validation for area
    if width <= 0 or height <= 0:
        # print(f"Warning: Invalid box dimensions after clamping for image {image_id}: [{x1}, {y1}, {x2}, {y2}] -> w={width}, h={height}. Skipping.")
        return None

    return {
        "image_id": image_id,
        "category_id": int(label), # Directly use the integer label from model
        "bbox": [x1, y1, width, height], # COCO format [x, y, width, height]
        "score": float(score),
        "area": float(width * height),
        "iscrowd": 0,
        "id": -1 # Placeholder, will be assigned uniquely later
    }

def process_images_batch(model: TRTInference, image_paths: list, device: str, conf_thresh=0.4, batch_size=16):
    """
    Process a list of images in batches and return COCO format annotations.
    Uses image index as image_id.
    """
    # Define transforms (consistent with common detection models like DETR/YOLO)
    # Resize should match model training if possible, 640x640 is common
    transforms = T.Compose([
        T.Resize((640, 640)), # Resize to fixed size expected by the model
        T.ToTensor(),
        # Add normalization if required by the model
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_coco_annotations = []
    annotation_counter = 0 # For unique annotation IDs

    # Process images in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing image batches"):
            batch_paths = image_paths[i:i+batch_size]
            current_batch_actual_size = len(batch_paths) # Store the actual number of images in this batch
            images_batch = []
            orig_sizes_batch = []
            image_ids_batch = [] # Store original index as image_id

            for idx, path in enumerate(batch_paths):
                image_id = i + idx # Use global index as image_id
                image_ids_batch.append(image_id)

                try:
                    im_pil = Image.open(path).convert('RGB')
                    w, h = im_pil.size
                    orig_sizes_batch.append([h, w]) # Store as [height, width] - common for DETR-like models
                    images_batch.append(transforms(im_pil))
                except Exception as e:
                    print(f"Error loading or transforming image {path}: {e}. Skipping.")
                    # This image will be skipped, affecting batch size consistency downstream if not handled
                    # For now, we proceed but the batch sent to model might be smaller than intended max
                    continue # Simple skip for now

            if not images_batch: # If all images in batch failed or batch was empty
                continue

            # Update actual batch size if images were skipped
            current_batch_actual_size = len(images_batch)
            if current_batch_actual_size == 0: # Double check after loop
                continue

            # Stack tensors for the batch
            try:
                im_data = torch.stack(images_batch).to(device)
                # NOTE: D-FINE models (like DETR) often expect original sizes as [height, width] tensor
                orig_size_tensor = torch.tensor(orig_sizes_batch, dtype=torch.int64).to(device) # Ensure int64 dtype if needed by model
            except Exception as e:
                print(f"Error stacking tensors for batch starting at index {i}: {e}")
                continue # Skip this batch

            # Prepare blob for the model - **ADJUST KEYS BASED ON YOUR MODEL'S EXPECTED INPUT NAMES**
            # Common names: 'images', 'pixel_values', 'input', 'orig_target_sizes', 'original_sizes'
            blob = {
                'images': im_data, # Adjust key if model expects 'pixel_values' etc.
                'orig_target_sizes': orig_size_tensor, # Adjust key if model expects 'original_sizes' etc.
            }
            # Verify required input names exist in blob
            for name in model.input_names:
                if name not in blob:
                    print(f"FATAL ERROR: Model requires input '{name}' but it's not in the prepared blob. Check blob preparation logic.")
                    # Decide whether to raise error or skip batch
                    raise KeyError(f"Missing required model input: {name}")


            # Run inference
            try:
                outputs = model(blob)
                # Ensure CUDA synchronizes after inference if needed for accurate timing or debugging
                # model.synchronize() # Usually handled by TimeProfiler or naturally by data access
            except Exception as e:
                print(f"Error during inference for batch starting at index {i}: {e}")
                # Decide how to handle: skip batch, partial results etc.
                continue # Skip batch on error

            # Ensure expected outputs are present - **ADJUST KEYS BASED ON YOUR MODEL'S OUTPUT NAMES**
            # Common names: 'scores', 'logits', 'pred_logits', 'labels', 'pred_classes', 'boxes', 'pred_boxes'
            required_outputs = ['scores', 'labels', 'boxes'] # Based on your logs
            if not all(k in outputs for k in required_outputs):
                 print(f"Error: Model output missing required keys. Found: {list(outputs.keys())}, Expected: {required_outputs}. Skipping batch results.")
                 continue

            # Process outputs for each image in the batch
            # Output tensors have shape [actual_batch_size, num_detections, ...] after slicing in run_torch
            batch_scores = outputs['scores'] # Shape: [actual_batch_size, num_queries]
            batch_labels = outputs['labels'] # Shape: [actual_batch_size, num_queries]
            batch_boxes = outputs['boxes']   # Shape: [actual_batch_size, num_queries, 4] (xyxy format expected)

            # Check if batch size matches output - this should be correct after slicing in run_torch
            if not (len(batch_scores) == len(batch_labels) == len(batch_boxes) == current_batch_actual_size):
                 print(f"Warning: Mismatch between processed batch size ({current_batch_actual_size}) and output tensor size ({len(batch_scores)}). This might indicate an issue in output slicing or model execution. Skipping batch.")
                 continue


            for b_idx in range(current_batch_actual_size): # Iterate up to the actual number of images processed
                img_scores = batch_scores[b_idx]
                img_labels = batch_labels[b_idx]
                img_boxes = batch_boxes[b_idx]
                # Get the correct image ID corresponding to this batch item
                # Need to handle skipped images if image_ids_batch wasn't adjusted
                # Assuming image_ids_batch corresponds directly to the items in images_batch that made it
                current_image_id = image_ids_batch[b_idx] # This assumes no images were skipped *within* the batch loading loop

                # Filter detections by confidence threshold
                try:
                    mask = img_scores > conf_thresh
                    filtered_scores = img_scores[mask]
                    filtered_labels = img_labels[mask]
                    filtered_boxes = img_boxes[mask]
                except Exception as e:
                    print(f"Error filtering results for image_id {current_image_id}: {e}")
                    continue # Skip processing this image's results

                for j in range(len(filtered_scores)):
                    label = filtered_labels[j].item()
                    # Optional: Check if label is in our defined CATEGORY_MAP
                    if label not in CATEGORY_MAP:
                        # print(f"Warning: Model produced unknown category ID {label} for image {current_image_id}. Skipping detection.")
                        continue

                    box = filtered_boxes[j].tolist() # Convert tensor to list [x1, y1, x2, y2]
                    score = filtered_scores[j].item()

                    # Create COCO annotation
                    coco_anno = create_coco_annotation(
                        image_id=current_image_id,
                        label=label,
                        box=box,
                        score=score
                    )

                    if coco_anno:
                         # Assign unique ID across all annotations
                         coco_anno["id"] = annotation_counter
                         all_coco_annotations.append(coco_anno)
                         annotation_counter += 1

    return all_coco_annotations

def get_image_files(folder_path, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
    """Get all image files from a folder with specified extensions"""
    image_files = []
    print(f"Searching for images in: {folder_path}")
    if not os.path.isdir(folder_path):
        print(f"Error: Provided path is not a directory: {folder_path}")
        return []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    print(f"Found {len(image_files)} images with extensions {extensions}.")
    return sorted(image_files) # Sort for consistent order

def build_coco_format_json(annotations, image_files, category_map):
    """Build the complete COCO format JSON with images, categories, and annotations"""
    print("Building COCO JSON output...")
    if not image_files:
        print("Warning: No image files provided to build_coco_format_json. 'images' list will be empty.")
        # Decide if this is an error or acceptable
        # return None # Or return structure with empty images list

    # Create images list using the actual filenames and indices as IDs
    images_list = []
    processed_image_ids = set(anno['image_id'] for anno in annotations) # Get IDs of images that actually have annotations

    for img_id, img_path in enumerate(image_files):
        # Only include image entries for images that were successfully processed and might have annotations
        # Or include all found images? Current approach: Include all found images, even if no annotations.
        width, height = -1, -1 # Default values
        try:
            # Using PIL to get dimensions accurately
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
             print(f"Warning: Could not read dimensions for {img_path}: {e}. Using defaults (-1, -1).")

        images_list.append({
            "id": img_id, # Use index as ID, matching process_images_batch
            "file_name": os.path.basename(img_path),
            "path": img_path, # Keep full path for reference if needed
            "height": height,
            "width": width
        })


    # Create categories list from the provided map
    categories_list = [{"id": k, "name": v, "supercategory": "object"} for k, v in category_map.items()]

    # Build final COCO format dictionary
    coco_format = {
        "info": {
            "description": "D-FINE TRT Inference Results",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "trt_inf.py script",
            "date_created": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        },
        "licenses": [], # Add license info if applicable
        "images": images_list,
        "annotations": annotations, # The annotations passed in
        "categories": categories_list
    }
    print("COCO JSON structure built.")
    return coco_format


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run TensorRT inference and output COCO annotations.")
    parser.add_argument('-trt', '--trt', type=str, required=True, help='Path to TensorRT engine file (.engine)')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input image file or directory containing images')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device for inference (e.g., cuda:0)')
    parser.add_argument('-o', '--output', type=str, default='results_coco.json', help='Output COCO JSON file name')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Confidence threshold for detections')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose TRT logging')
    # Removed video specific args for clarity, can be added back if needed

    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.trt):
        print(f"Error: TensorRT engine file not found at {args.trt}")
        return 1 # Indicate error exit code
    if not os.path.exists(args.input):
        print(f"Error: Input path not found at {args.input}")
        return 1
    if not torch.cuda.is_available():
        print(f"Error: CUDA device '{args.device}' requested, but CUDA is not available.")
        return 1


    # --- Initialize Model ---
    model = None # Initialize to None
    try:
        # Assign global engine variable used in run_torch's dynamic check
        global engine
        engine = TRTInference(
            args.trt,
            device=args.device,
            max_batch_size=args.batch_size,
            verbose=args.verbose
        ).engine # Store the engine instance itself if needed globally (e.g., for the dynamic check)

        # Create the model instance
        model = TRTInference(
            args.trt,
            device=args.device,
            max_batch_size=args.batch_size,
            verbose=args.verbose
        )

    except Exception as e:
        print(f"Error initializing TRTInference: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return 1

    all_coco_annotations = []
    image_files = []
    total_processed_count = 0
    total_items_found = 0
    processing_type = ""

    # --- Start Timing ---
    overall_start_time = time.time()

    # --- Process Input ---
    valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if os.path.isdir(args.input):
        processing_type = "directory"
        print(f"Processing directory: {args.input}")
        image_files = get_image_files(args.input, extensions=valid_image_extensions)
        total_items_found = len(image_files)

        if total_items_found == 0:
            print("No images found in the directory.")
            # No error, just nothing to do
        else:
            print(f"Found {total_items_found} images. Starting batch processing...")
            try:
                all_coco_annotations = process_images_batch(
                    model,
                    image_files,
                    args.device,
                    conf_thresh=args.threshold,
                    batch_size=args.batch_size
                )
                # Assume all found images were attempted if no error stopped the process
                total_processed_count = total_items_found
            except Exception as e:
                print(f"\nError during batch processing: {e}")
                import traceback
                traceback.print_exc()
                # Annotations might be partially filled, proceed to save what we have
                print("Attempting to save any annotations generated before the error.")
                # Decide if processed count should be updated based on progress bar or error point
                # For simplicity, keep it as total_items_found, implying an attempt was made

    elif os.path.isfile(args.input) and args.input.lower().endswith(valid_image_extensions):
        processing_type = "single_image"
        print(f"Processing single image: {args.input}")
        image_files = [args.input]
        total_items_found = 1

        try:
            # Use the same batch processing function for simplicity
            all_coco_annotations = process_images_batch(
                model,
                image_files,
                args.device,
                conf_thresh=args.threshold,
                batch_size=1 # Force batch size 1 for single image
            )
            total_processed_count = 1
        except Exception as e:
            print(f"Error processing single image: {e}")
            import traceback
            traceback.print_exc()

    else:
        print(f"Error: Input path '{args.input}' is not a valid image file ({valid_image_extensions}) or directory.")
        return 1


    # --- End Timing ---
    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time

    # --- Calculate FPS ---
    fps = 0
    if overall_elapsed_time > 0 and total_processed_count > 0:
        fps = total_processed_count / overall_elapsed_time
    elif total_processed_count == 0:
        print("No items were processed, FPS is not applicable.")
    else: # elapsed_time is 0 or negative
        print("Warning: Elapsed time is zero or negative, cannot calculate FPS.")

    # --- Build and Save COCO JSON ---
    # Only attempt to save if we processed something and have image file info
    if total_processed_count > 0 and image_files:
        try:
            coco_json_output = build_coco_format_json(all_coco_annotations, image_files, CATEGORY_MAP)

            # Add performance metrics to the output JSON
            coco_json_output["performance"] = {
                "processing_type": processing_type,
                "total_items_found": total_items_found,
                "total_items_processed": total_processed_count, # Reflects attempts or successful completions
                "total_annotations_generated": len(all_coco_annotations),
                "confidence_threshold": args.threshold,
                "batch_size": args.batch_size,
                "total_time_seconds": round(overall_elapsed_time, 4),
                "fps": round(fps, 2)
            }

            # Save results
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                print(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir)

            with open(args.output, 'w') as f:
                json.dump(coco_json_output, f, indent=2)
            print(f"\nCOCO format annotations saved to: {args.output}")

        except Exception as e:
            print(f"\nError building or saving COCO JSON output to {args.output}: {e}")
            import traceback
            traceback.print_exc()
            # Don't terminate script if JSON saving fails, summary is still useful

    elif not image_files:
         print("\nNo image files were found or provided. Skipping COCO JSON output.")
    else: # Processed count is 0, but files were found
        print("\nNo items were successfully processed (or processing failed early). Skipping COCO JSON output.")


    # --- Print Summary ---
    print(f"\n--- Processing Summary ---")
    print(f"Input Path: {args.input}")
    print(f"Input Type: {processing_type}")
    print(f"Engine File: {args.trt}")
    print(f"Output File: {args.output}")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Confidence Threshold: {args.threshold}")
    print(f"Total Items Found: {total_items_found}")
    print(f"Total Items Processed (attempted/completed): {total_processed_count}")
    print(f"Total Annotations Generated: {len(all_coco_annotations)}")
    print(f"Total Processing Time: {overall_elapsed_time:.2f} seconds")
    print(f"Overall FPS (based on processed items): {fps:.2f}")
    print(f"--------------------------")

    return 0 # Indicate success


if __name__ == '__main__':
    # Assign global engine variable for use in run_torch if needed
    # This is slightly awkward; alternatives include passing engine instance or using a class structure for main logic
    engine = None
    exit_code = main()
    exit(exit_code)
# --- END OF FILE trt_inf.py ---