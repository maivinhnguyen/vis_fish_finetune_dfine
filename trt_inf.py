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
from PIL import Image, ImageDraw

import torch
import torchvision.transforms as T

import tensorrt as trt
import cv2
import os
from tqdm import tqdm  # Added for progress bars

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
    def __init__(self, engine_path, device='cuda:0', backend='torch', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
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
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, '{} dtype mismatch'.format(n)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()

def create_coco_annotation(image_id, label, box, score, category_map=None):
    """
    Create a COCO format annotation dictionary.
    """
    # Default category map if none is provided
    if category_map is None:
        # This is a placeholder - replace with your actual category mapping
        category_map = {i: i for i in range(91)}
    
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    return {
        "image_id": image_id,
        "category_id": int(label),
        "bbox": [float(x1), float(y1), float(width), float(height)],
        "score": float(score),
        "area": float(width * height),
        "iscrowd": 0
    }

def process_images_batch(model, image_paths, device, conf_thresh=0.4, batch_size=16):
    """
    Process a batch of images and return COCO format annotations
    """
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    results = []
    coco_annotations = []
    annotation_id = 0
    
    # Use torch.no_grad() to save memory and speed up inference
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            orig_sizes = []
            image_ids = []
            
            # Prepare batch data
            for idx, path in enumerate(batch_paths):
                image_id = i + idx
                image_ids.append(image_id)
                
                im_pil = Image.open(path).convert('RGB')
                w, h = im_pil.size
                orig_sizes.append([w, h])
                images.append(transforms(im_pil))
            
            # Stack all images and sizes in batch
            im_data = torch.stack(images).to(device)
            orig_size = torch.tensor(orig_sizes).to(device)
            
            blob = {
                'images': im_data,
                'orig_target_sizes': orig_size,
            }
            
            # Run inference
            outputs = model(blob)
            
            # Process outputs for each image in batch
            for b in range(len(batch_paths)):
                img_labels = outputs['labels'][b]
                img_boxes = outputs['boxes'][b]
                img_scores = outputs['scores'][b]
                
                mask = img_scores > conf_thresh
                filtered_labels = img_labels[mask]
                filtered_boxes = img_boxes[mask]
                filtered_scores = img_scores[mask]
                
                for j in range(len(filtered_scores)):
                    anno = create_coco_annotation(
                        image_ids[b], 
                        filtered_labels[j].item(), 
                        filtered_boxes[j].tolist(), 
                        filtered_scores[j].item()
                    )
                    coco_annotations.append(anno)
                    annotation_id += 1
    
    return coco_annotations

def process_video_frames(model, video_path, device, conf_thresh=0.4, output_images=False, output_dir='frames', 
                         save_every=1, batch_size=8):
    """
    Process video frames and return COCO format annotations
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if output_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    coco_annotations = []
    frame_id = 0
    total_frames_processed = 0
    
    # Create batches for efficient processing
    batch_images = []
    batch_orig_sizes = []
    batch_frame_ids = []
    
    with torch.no_grad():
        pbar = tqdm(total=frame_count, desc="Processing video frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every nth frame if specified
            if frame_id % save_every == 0:
                # Convert to PIL Image
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                w, h = frame_pil.size
                
                batch_images.append(transforms(frame_pil))
                batch_orig_sizes.append([w, h])
                batch_frame_ids.append(frame_id)
                
                # Save the frame if requested
                if output_images:
                    frame_filename = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                
                total_frames_processed += 1
                
                # Process batch when it reaches the specified size
                if len(batch_images) == batch_size:
                    # Stack the batch inputs
                    im_data = torch.stack(batch_images).to(device)
                    orig_size = torch.tensor(batch_orig_sizes).to(device)
                    
                    blob = {
                        'images': im_data,
                        'orig_target_sizes': orig_size,
                    }
                    
                    # Run inference
                    outputs = model(blob)
                    
                    # Process outputs for each frame in batch
                    for b in range(len(batch_images)):
                        img_labels = outputs['labels'][b]
                        img_boxes = outputs['boxes'][b]
                        img_scores = outputs['scores'][b]
                        
                        mask = img_scores > conf_thresh
                        filtered_labels = img_labels[mask]
                        filtered_boxes = img_boxes[mask]
                        filtered_scores = img_scores[mask]
                        
                        for j in range(len(filtered_scores)):
                            anno = create_coco_annotation(
                                batch_frame_ids[b], 
                                filtered_labels[j].item(), 
                                filtered_boxes[j].tolist(), 
                                filtered_scores[j].item()
                            )
                            coco_annotations.append(anno)
                    
                    # Clear the batch lists
                    batch_images = []
                    batch_orig_sizes = []
                    batch_frame_ids = []
            
            frame_id += 1
            pbar.update(1)
        
        # Process any remaining frames in the last batch
        if batch_images:
            im_data = torch.stack(batch_images).to(device)
            orig_size = torch.tensor(batch_orig_sizes).to(device)
            
            blob = {
                'images': im_data,
                'orig_target_sizes': orig_size,
            }
            
            outputs = model(blob)
            
            for b in range(len(batch_images)):
                img_labels = outputs['labels'][b]
                img_boxes = outputs['boxes'][b]
                img_scores = outputs['scores'][b]
                
                mask = img_scores > conf_thresh
                filtered_labels = img_labels[mask]
                filtered_boxes = img_boxes[mask]
                filtered_scores = img_scores[mask]
                
                for j in range(len(filtered_scores)):
                    anno = create_coco_annotation(
                        batch_frame_ids[b], 
                        filtered_labels[j].item(), 
                        filtered_boxes[j].tolist(), 
                        filtered_scores[j].item()
                    )
                    coco_annotations.append(anno)
        
        pbar.close()
        cap.release()
    
    return coco_annotations, total_frames_processed

def get_image_files(folder_path, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Get all image files from a folder with specified extensions"""
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)  # Sort to ensure consistent order

def build_coco_format_json(annotations, category_map=None):
    """Build the complete COCO format JSON with images, categories, and annotations"""
    # Default category map if none is provided
    if category_map is None:
        # This is a placeholder - replace with actual categories
        category_map = {
            # Standard COCO categories - update as needed
            0: "Bus", 1: "Bike", 2: "Car", 3: "Pedestrian", 4: "Truck"
            # ... add more categories as needed
        }
    
    # Get unique image IDs from annotations
    image_ids = set([a["image_id"] for a in annotations])
    
    # Create images list
    images = [{"id": img_id, "file_name": f"image_{img_id}.jpg"} for img_id in image_ids]
    
    # Create categories list
    categories = [{"id": k, "name": v} for k, v in category_map.items()]
    
    # Build final COCO format
    coco_format = {
        "info": {
            "description": "D-FINE Inference Results",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    return coco_format

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt', type=str, required=True, help='Path to TensorRT engine')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input image, video or directory')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device for inference')
    parser.add_argument('-o', '--output', type=str, default='results.json', help='Output JSON file name')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--save_frames', action='store_true', help='Save video frames as images')
    parser.add_argument('--frame_dir', type=str, default='frames', help='Directory to save frames')
    parser.add_argument('--save_every', type=int, default=1, help='Save every nth frame')
    
    args = parser.parse_args()

    # Initialize model
    print(f"Loading TensorRT engine from {args.trt}...")
    model = TRTInference(args.trt, device=args.device, max_batch_size=args.batch_size)
    
    # Start overall timing (from first to last image)
    overall_start_time = time.time()
    coco_annotations = []
    
    if os.path.isdir(args.input):
        # Process directory of images
        print(f"Processing directory of images: {args.input}")
        image_files = get_image_files(args.input)
        total_images = len(image_files)
        
        if total_images == 0:
            print("No images found in the directory!")
            return
            
        print(f"Found {total_images} images")
        coco_annotations = process_images_batch(
            model, 
            image_files, 
            args.device, 
            conf_thresh=args.threshold,
            batch_size=args.batch_size
        )
        total_processed = total_images
        
    elif os.path.splitext(args.input)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process single image
        print(f"Processing single image: {args.input}")
        image_files = [args.input]
        coco_annotations = process_images_batch(
            model, 
            image_files, 
            args.device, 
            conf_thresh=args.threshold
        )
        total_processed = 1
        
    else:
        # Process as video
        print(f"Processing video: {args.input}")
        coco_annotations, total_processed = process_video_frames(
            model, 
            args.input, 
            args.device, 
            conf_thresh=args.threshold,
            output_images=args.save_frames,
            output_dir=args.frame_dir,
            save_every=args.save_every,
            batch_size=args.batch_size
        )
    
    # Calculate overall elapsed time and FPS
    overall_elapsed_time = time.time() - overall_start_time
    fps = total_processed / overall_elapsed_time
    
    # Build complete COCO format JSON
    coco_json = build_coco_format_json(coco_annotations)
    
    # Add performance metrics to the output
    coco_json["performance"] = {
        "total_time_seconds": overall_elapsed_time,
        "fps": fps,
        "total_frames_or_images": total_processed
    }
    
    # Save results to JSON file
    with open(args.output, 'w') as f:
        json.dump(coco_json, f, indent=2)
    
    print(f"\nProcessing completed!")
    print(f"Total time: {overall_elapsed_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames/images processed: {total_processed}")
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()