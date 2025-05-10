import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import argparse

def get_image_Id(img_name):
    """Calculate image ID based on the challenge specification"""
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId

def run_inference(
    weights_path,
    test_images_dir,
    output_json_path,
    conf_threshold=0.25,
    iou_threshold=0.45,
    img_size=640,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Run YOLOv8 inference on test images and save results in COCO format
    
    Args:
        weights_path: Path to the YOLOv8 model weights (.pth or .pt)
        test_images_dir: Directory containing test images
        output_json_path: Path to save the output JSON file
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        img_size: Input image size for the model
        device: Device to run inference on ('cuda' or 'cpu')
    """
    print(f"Loading YOLOv8n model from {weights_path}...")
    
    # Load model
    model = YOLO(weights_path)
    
    # Category ID mapping (as per the challenge - 5 classes)
    # Adjust these according to your model's class mapping
    category_mapping = {
        0: 0,  # Bus
        1: 1,  # Bike
        2: 2,  # Car
        3: 3,  # Pedestrian
        4: 4,  # Truck
    }
    
    # Get all image files
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for ext in valid_extensions:
        image_files.extend(list(Path(test_images_dir).glob(f'*{ext}')))
    
    if not image_files:
        raise ValueError(f"No images found in {test_images_dir}")
    
    print(f"Found {len(image_files)} images. Running inference...")
    
    # Initialize results list
    results = []
    detection_id = 0
    
    # Process each image
    for img_path in tqdm(image_files):
        img_filename = os.path.basename(img_path)
        
        try:
            # Calculate image ID
            image_id = get_image_Id(img_filename)
            
            # Run inference
            detections = model(img_path, conf=conf_threshold, iou=iou_threshold, device=device)
            
            # Process detections
            for det in detections:
                boxes = det.boxes
                
                for i in range(len(boxes)):
                    box = boxes[i].xyxy[0].cpu().numpy()  # x1, y1, x2, y2 format
                    conf = float(boxes[i].conf[0].cpu().numpy())
                    cls = int(boxes[i].cls[0].cpu().numpy())
                    
                    # Convert to COCO format [x, y, width, height]
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Calculate area
                    area = width * height
                    
                    # Map to the correct category ID if needed
                    if cls in category_mapping:
                        category_id = category_mapping[cls]
                    else:
                        continue  # Skip if class not in mapping
                    
                    # Create detection entry
                    detection = {
                        "id": detection_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "area": float(area),
                        "iscrowd": 0,
                        "score": float(conf)
                    }
                    
                    results.append(detection)
                    detection_id += 1
                    
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    print(f"Inference completed. Total detections: {len(results)}")
    
    # Write results to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Results saved to {output_json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 inference to COCO JSON format')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv8 weights (.pth or .pt)')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory with test images')
    parser.add_argument('--output', type=str, default='detections.json', help='Output JSON file path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IoU threshold')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    args = parser.parse_args()
    
    run_inference(
        weights_path=args.weights,
        test_images_dir=args.test_dir,
        output_json_path=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size,
        device=args.device
    )