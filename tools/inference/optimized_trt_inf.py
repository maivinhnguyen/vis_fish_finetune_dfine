"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
(Adapted for TensorRT inference with directory processing and COCO output)
"""

import time
import contextlib
import collections
from collections import OrderedDict
import json
import datetime
import os
import glob
import sys

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import torch.utils.data

import tensorrt as trt

# Assuming engine.core and YAMLConfig are not needed for TRT direct inference
# If they were for post-processing steps not part of the TRT model,
# they might need to be re-introduced.
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from engine.core import YAMLConfig


class TimeProfiler(contextlib.ContextDecorator):
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
    def __init__(self, engine_path, device='cuda:0', max_batch_size=1, verbose=False): # Default max_batch_size to 1
        self.engine_path = engine_path
        self.device = torch.device(device) # Use torch.device object
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size, device) -> OrderedDict:
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = list(engine.get_tensor_shape(name)) # Use list for mutable shape
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1: # Dynamic batch size
                shape[0] = max_batch_size
            
            # Set binding dimensions for inputs if dynamic
            # For TRT 8.0+ this might be done differently with optimization profiles.
            # Assuming explicit shape setting for inputs if dynamic.
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                 # Check if context has an active optimization profile that needs to be configured
                if engine.num_optimization_profiles > 0 and context.active_optimization_profile < 0:
                     context.active_optimization_profile = 0 # Default to profile 0
                
                # For dynamic shapes, you might need to set input dimensions here
                # if not already handled by an optimization profile.
                # context.set_input_shape(name, tuple(shape)) # Ensure it's a tuple

                # Try setting binding dimensions (more common for older TRT or specific setups)
                # context.set_binding_shape(engine.get_binding_index(name), tuple(shape))

                # For newer TRT versions, explicit context.set_input_shape might be preferred
                # This seems to be done in run_torch now, which is more flexible.
                pass


            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, tuple(shape), data, data.data_ptr())
        return bindings

    def run_torch(self, blob):
        # Ensure input tensors are contiguous
        contiguous_blob = {n: blob[n].contiguous() for n in self.input_names}

        for n in self.input_names:
            # Update context input shape if current blob's shape is different from binding's initial shape
            # This is crucial for dynamic batch sizes or other dynamic input dimensions.
            if self.bindings[n].shape != contiguous_blob[n].shape:
                # print(f"Updating input shape for {n} from {self.bindings[n].shape} to {contiguous_blob[n].shape}")
                if not self.context.set_input_shape(n, contiguous_blob[n].shape):
                    print(f"Warning: Failed to set input shape for {n}", file=sys.stderr)
                # Update internal binding shape representation if needed, though data ptr and shape are from blob
                self.bindings[n] = self.bindings[n]._replace(shape=contiguous_blob[n].shape)


            assert self.bindings[n].data.dtype == contiguous_blob[n].dtype, f'{n} dtype mismatch: binding {self.bindings[n].data.dtype}, blob {contiguous_blob[n].dtype}'
            # Forcing the binding data ptr to the input blob's data ptr
            self.bindings_addr[n] = contiguous_blob[n].data_ptr()
        
        # Update output bindings if their shapes depend on input shapes and are dynamic
        # This is more complex and depends on how the TRT engine is built and if outputs are dynamic.
        # For now, assume output shapes allocated at init are sufficient or handled by TRT.

        self.context.execute_v2(list(self.bindings_addr.values()))
        
        # Create output dictionary ensuring correct shapes from context if possible
        outputs = {}
        for n in self.output_names:
            # Get the actual output shape from the context after execution
            output_shape = tuple(self.context.get_tensor_shape(n))
            
            # Slice the pre-allocated buffer to the actual output size
            # This is important if max_batch_size was larger than actual batch_size
            # or if other output dimensions are dynamic.
            num_dims = len(output_shape)
            slicer = tuple(slice(0, output_shape[d]) for d in range(num_dims))
            outputs[n] = self.bindings[n].data[slicer]

        return outputs


    def __call__(self, blob):
        # Ensure input tensors are on the correct device
        blob_on_device = {n: v.to(self.device) for n, v in blob.items()}
        return self.run_torch(blob_on_device)

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)


class CocoImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_files_list, transforms):
        self.image_files = image_files_list
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        coco_image_id = idx + 1  # 1-based ID

        try:
            im_pil = Image.open(image_path).convert('RGB')
            original_width, original_height = im_pil.size
            im_data = self.transforms(im_pil)
            img_filename = os.path.basename(image_path)

            return {
                "im_data": im_data,
                "original_size_for_model": torch.tensor([original_width, original_height], dtype=torch.float32),  # W, H
                "coco_image_id": coco_image_id,
                "original_width": original_width,
                "original_height": original_height,
                "file_name": img_filename,
                "status": "ok"
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {e}", file=sys.stderr)
            # Return a dummy item or skip, here returning status to filter later
            # Shape must match expected output for collate_fn
            return {
                "im_data": torch.zeros((3, 640, 640)), # Dummy data (adjust size if your transform changes it)
                "original_size_for_model": torch.tensor([0,0], dtype=torch.float32),
                "coco_image_id": coco_image_id,
                "original_width": 0,
                "original_height": 0,
                "file_name": os.path.basename(image_path),
                "status": "error"
            }


def coco_collate_fn_revised(batch):
    # Filter out items that had errors during loading
    batch = [item for item in batch if item["status"] == "ok"]
    if not batch: # If all items in batch had errors
        return {
            "im_data_batch": torch.empty(0),
            "original_sizes_for_model_batch": torch.empty(0), # Renamed for clarity
            "coco_image_ids": [],
            "original_widths": [],
            "original_heights": [],
            "file_names": [],
            "empty_batch": True
        }

    return {
        "im_data_batch": torch.stack([item['im_data'] for item in batch]),
        "original_sizes_for_model_batch": torch.stack([item['original_size_for_model'] for item in batch]),
        "coco_image_ids": [item['coco_image_id'] for item in batch],
        "original_widths": [item['original_width'] for item in batch],
        "original_heights": [item['original_height'] for item in batch],
        "file_names": [item['file_name'] for item in batch],
        "empty_batch": False
    }


def process_directory_to_coco_trt(trt_model: TRTInference, device_str: str, input_dir: str, 
                                  output_json: str, conf_threshold: float = 0.4, 
                                  batch_size_dataloader: int = 1, num_workers_dataloader: int = 0):
    """
    Process all images in a directory using TensorRT model and output results in COCO format.
    """
    coco_output = {
        "info": {
            "description": "COCO format results from D-FINE TensorRT model",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "D-FINE Authors (TensorRT adaptation)",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [], "images": [], "annotations": [], "categories": []
    }
    
    category_mapping = {}
    annotation_id_counter = 1
    
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + \
                         glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(input_dir, "*.png")) + \
                         glob.glob(os.path.join(input_dir, "*.bmp")))
    
    if not image_files:
        print("No images found in the directory.")
        with open(output_json, 'w') as f:
            json.dump(coco_output, f, indent=2)
        print(f"Empty COCO format results saved to {output_json}")
        return

    # Standard D-FINE transforms (adjust if your TRT model expects different input size)
    transforms_val = T.Compose([
        T.Resize((640, 640)), 
        T.ToTensor(),
    ])
    
    dataset = CocoImageDataset(image_files, transforms_val)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_dataloader,
        shuffle=False,
        num_workers=num_workers_dataloader,
        collate_fn=coco_collate_fn_revised,
        pin_memory=(True if torch.device(device_str).type == 'cuda' else False)
    )
    
    print(f"Found {len(image_files)} images to process with TensorRT.")
    start_time_total = datetime.datetime.now()
    
    processed_image_count = 0
    
    # Ensure engine input names are what we expect.
    # Typically 'images' and 'orig_target_sizes' for D-FINE like models.
    # Verify these against your actual TRT engine's input names.
    # You can print trt_model.input_names to check.
    # print(f"TRT Engine Input Names: {trt_model.input_names}")
    # print(f"TRT Engine Output Names: {trt_model.output_names}")
    # Assuming standard names:
    image_input_name = 'images' # Or whatever your TRT engine expects for image data
    size_input_name = 'orig_target_sizes' # Or for original sizes

    with torch.no_grad(): # PyTorch context, though TRT manages its own graph
        for batch_data in dataloader:
            if batch_data.get("empty_batch", False) or batch_data["im_data_batch"].numel() == 0:
                num_failed_in_batch = batch_size_dataloader 
                print(f"Skipping an empty or failed batch of approx {num_failed_in_batch} images.", file=sys.stderr)
                continue

            im_data_batch = batch_data['im_data_batch'] # Already on device if pin_memory + CUDA
            orig_sizes_batch = batch_data['original_sizes_for_model_batch']

            # Prepare blob for TRT model
            # Ensure names match your TRT engine's input tensor names
            blob = {
                image_input_name: im_data_batch.to(trt_model.device),
                size_input_name: orig_sizes_batch.to(trt_model.device)
            }
            
            outputs_trt = trt_model(blob) # Perform inference
            
            # Outputs from TRT are tensors: e.g., outputs_trt['labels'], outputs_trt['boxes'], outputs_trt['scores']
            # Shapes are typically (batch_size, num_detections, ...)
            labels_batch_trt = outputs_trt.get('labels', torch.empty(0)) # Provide default if key missing
            boxes_batch_trt = outputs_trt.get('boxes', torch.empty(0))
            scores_batch_trt = outputs_trt.get('scores', torch.empty(0))
            
            for i in range(len(batch_data['coco_image_ids'])): # Iterate through items in the batch
                processed_image_count += 1
                coco_image_id = batch_data['coco_image_ids'][i]
                original_width = batch_data['original_widths'][i]
                original_height = batch_data['original_heights'][i]
                file_name = batch_data['file_names'][i]

                coco_output["images"].append({
                    "id": coco_image_id, "width": original_width,
                    "height": original_height, "file_name": file_name,
                })
                
                # Get detections for the i-th image in the batch
                # Detach from graph and move to CPU for numpy conversion if needed
                current_labels = labels_batch_trt[i].detach().cpu() if labels_batch_trt.numel() > 0 else torch.empty(0)
                current_boxes = boxes_batch_trt[i].detach().cpu() if boxes_batch_trt.numel() > 0 else torch.empty(0)
                current_scores = scores_batch_trt[i].detach().cpu() if scores_batch_trt.numel() > 0 else torch.empty(0)
                
                if current_scores.numel() > 0:
                    mask = current_scores > conf_threshold
                    filtered_labels = current_labels[mask].numpy()
                    filtered_boxes = current_boxes[mask].numpy()
                    filtered_scores = current_scores[mask].numpy()
                else: # No detections or scores tensor was empty
                    filtered_labels, filtered_boxes, filtered_scores = [], [], []

                for k_detection in range(len(filtered_labels)):
                    label_id = int(filtered_labels[k_detection])
                    
                    if label_id not in category_mapping:
                        category_mapping[label_id] = {
                            "id": label_id, "name": f"class_{label_id}", "supercategory": "none"
                        }
                    
                    x1, y1, x2, y2 = filtered_boxes[k_detection]
                    width = x2 - x1
                    height = y2 - y1
                    
                    coco_output["annotations"].append({
                        "id": annotation_id_counter, "image_id": coco_image_id,
                        "category_id": label_id,
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "area": float(width * height), "iscrowd": 0,
                        "score": float(filtered_scores[k_detection])
                    })
                    annotation_id_counter += 1
    
    end_time_total = datetime.datetime.now()
    elapsed_time = (end_time_total - start_time_total).total_seconds()
    
    coco_output["images"].sort(key=lambda x: x["id"])
    coco_output["categories"] = sorted(list(category_mapping.values()), key=lambda x: x["id"]) # Sort categories
    
    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    print(f"COCO format results saved to {output_json}")
    print(f"Successfully processed {processed_image_count} images with {annotation_id_counter-1} total detections.")
    if processed_image_count < len(image_files):
        print(f"Warning: {len(image_files) - processed_image_count} images could not be processed or were skipped.")

    if elapsed_time > 0 and processed_image_count > 0:
        fps = processed_image_count / elapsed_time
        print(f"Total time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average processing speed: {fps:.2f} FPS (including I/O and TRT inference)")
    else:
        print(f"Time elapsed: {elapsed_time:.2f} seconds. No images processed or time too short for FPS.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="TensorRT D-FINE Inference for Image Directories")
    parser.add_argument('-trt', '--trt_engine', type=str, required=True, help="Path to TensorRT engine file")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Input directory of images")
    parser.add_argument('-o', '--output_json', type=str, default='trt_coco_results.json',
                        help='Output JSON file in COCO format')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help="Device to use ('cuda:0', 'cpu' - Note: TRT typically CUDA)")
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Confidence threshold for detections')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for DataLoader (and max_batch_size for TRT engine if it supports dynamic batching)')
    parser.add_argument('--num_workers', type=int, default=0,  # For batch_size=1, num_workers=0 or 1 is common
                        help='Number of workers for data loading')
    parser.add_argument('--verbose_trt', action='store_true', help='Enable verbose TensorRT logging')
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.trt_engine):
        print(f"Error: TensorRT engine file '{args.trt_engine}' not found.", file=sys.stderr)
        sys.exit(1)

    # Initialize TRTInference. max_batch_size should match or exceed dataloader batch_size.
    # If your TRT engine was built for a fixed batch size, args.batch_size must match that.
    # If built with dynamic batch size, max_batch_size here is an upper limit for allocation.
    trt_model_instance = TRTInference(
        args.trt_engine, 
        device=args.device, 
        max_batch_size=args.batch_size, # TRT engine will be configured for this max
        verbose=args.verbose_trt
    )

    process_directory_to_coco_trt(
        trt_model_instance, 
        args.device,
        args.input_dir, 
        args.output_json, 
        args.threshold,
        args.batch_size, # This is for the DataLoader
        args.num_workers
    )