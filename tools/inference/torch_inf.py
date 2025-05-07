"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw
import json
import datetime
import os
import glob
import sys
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def process_directory_to_coco(model, device, input_dir, output_json, conf_threshold=0.4):
    """
    Process all images in a directory and output results in COCO format
    """
    # Create COCO format structure
    coco_output = {
        "info": {
            "description": "COCO format results from D-FINE model",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # You can customize category definitions based on your model's classes
    # For now using placeholder indices
    category_mapping = {
        0: "pedestrian",
        1: "people",
        2: 'bicycle',
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8:  "bus",
        9: "motor"
    }  # Will be populated from detection results
    annotation_id = 1
    
    # Get all image files in the directory
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.bmp"))
    
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    print(f"Found {len(image_files)} images to process")
    
    # Start timer
    start_time = datetime.datetime.now()
    
    for image_id, image_path in enumerate(image_files, 1):
        img_filename = os.path.basename(image_path)
        
        # Load and process image
        im_pil = Image.open(image_path).convert('RGB')
        original_width, original_height = im_pil.size
        
        # Add image info to COCO format
        coco_output["images"].append({
            "id": image_id,
            "width": original_width,
            "height": original_height,
            "file_name": img_filename,
        })
        
        # Prepare image for model
        orig_size = torch.tensor([[original_width, original_height]]).to(device)
        im_data = transforms(im_pil).unsqueeze(0).to(device)
        
        # Run model inference
        outputs = model(im_data, orig_size)
        labels, boxes, scores = outputs
        
        # Only consider predictions above threshold
        mask = scores[0] > conf_threshold
        filtered_labels = labels[0][mask].detach().cpu().numpy()
        filtered_boxes = boxes[0][mask].detach().cpu().numpy()
        filtered_scores = scores[0][mask].detach().cpu().numpy()
        
        # Process each detection
        for i in range(len(filtered_labels)):
            label_id = int(filtered_labels[i])
            
            # Add category if not seen before
            if label_id not in category_mapping:
                category_mapping[label_id] = {
                    "id": label_id,
                    "name": f"class_{label_id}",  # Replace with actual class names if available
                    "supercategory": "none"
                }
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = filtered_boxes[i]
            width = x2 - x1
            height = y2 - y1
            
            # Add annotation in COCO format
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": label_id,
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "area": float(width * height),
                "iscrowd": 0,
                "score": float(filtered_scores[i])
            })
            annotation_id += 1
    
    # Stop timer and calculate FPS
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    fps = len(image_files) / elapsed_time
    
    # Add categories to COCO output
    for cat_id, cat_info in category_mapping.items():
        coco_output["categories"].append(cat_info)
    
    # Write output to JSON file
    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    print(f"COCO format results saved to {output_json}")
    print(f"Processed {len(image_files)} images with {annotation_id-1} total detections")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {fps:.2f} FPS")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    model.eval()  # Ensure model is in evaluation mode

    if os.path.isdir(args.input):
        # Process directory of images and output COCO format
        process_directory_to_coco(model, device, args.input, args.output, args.threshold)
    elif os.path.isfile(args.input):
        # Handle single file (original functionality)
        file_path = args.input
        if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Process as image
            process_image(model, device, file_path)
            print("Image processing complete.")
        else:
            # Process as video
            process_video(model, device, file_path)
    else:
        print(f"Input path {args.input} does not exist")


def process_image(model, device, file_path):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)


def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue', )

        im.save('torch_results.jpg')


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('torch_results.mp4', fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True, 
                      help='Input image file, video file, or directory of images')
    parser.add_argument('-o', '--output', type=str, default='coco_results.json',
                      help='Output JSON file in COCO format')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Confidence threshold for detections')
    args = parser.parse_args()
    main(args)