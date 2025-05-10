import os
import json
import argparse
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path


def get_image_name_from_id(image_id):
    """Reverse of get_image_Id function - try to reconstruct image name from ID"""
    # This is an approximation as we can't fully reverse the exact filename
    # Format: cameraX_Y_Z.png where X=camera index, Y=scene code, Z=frame number
    
    # Extract parts from image_id
    id_str = str(image_id)
    
    if len(id_str) >= 2:
        # First digit is camera index
        camera_idx = id_str[0]
        
        # Second digit is scene index
        scene_idx = int(id_str[1])
        scene_list = ['M', 'A', 'E', 'N']
        if 0 <= scene_idx < len(scene_list):
            scene_code = scene_list[scene_idx]
        else:
            scene_code = 'X'  # Unknown scene
        
        # Rest of digits is frame index
        if len(id_str) > 2:
            frame_idx = id_str[2:]
        else:
            frame_idx = "0"
        
        return f"camera{camera_idx}_{scene_code}_{frame_idx}.png"
    
    return f"unknown_{image_id}.png"


def find_image_path(image_filename, image_dir, image_extensions=['.png', '.jpg', '.jpeg']):
    """Try to find the image file with the given filename or a similar one"""
    # Case 1: Exact match
    for ext in image_extensions:
        if os.path.exists(os.path.join(image_dir, image_filename)):
            return os.path.join(image_dir, image_filename)
    
    # Case 2: Try with different extensions
    basename = os.path.splitext(image_filename)[0]
    for ext in image_extensions:
        potential_path = os.path.join(image_dir, basename + ext)
        if os.path.exists(potential_path):
            return potential_path
    
    # Case 3: Try to find by image ID embedded in filename
    # This is a fallback if the exact filename can't be determined
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
    
    # Extract ID from potential filename
    id_part = basename.split('_')[-1]
    for img_path in image_files:
        if id_part in img_path.stem:
            return str(img_path)
    
    return None


def visualize_detections(json_file, image_dir, output_dir=None, confidence_threshold=0.0, 
                         show_images=False, save_images=True, max_images=None):
    """
    Visualize object detections from a COCO-format JSON file
    
    Args:
        json_file: Path to JSON file with detections
        image_dir: Directory containing the original images
        output_dir: Directory to save visualization results (if save_images=True)
        confidence_threshold: Minimum confidence score for visualizing a detection
        show_images: Whether to display images (will pause script execution)
        save_images: Whether to save visualized images
        max_images: Maximum number of images to process (None for all)
    """
    # Create output directory if needed
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load detections from JSON file
    with open(json_file, 'r') as f:
        detections = json.load(f)
    
    print(f"Loaded {len(detections)} detections from {json_file}")
    
    # Group detections by image_id
    images_detections = defaultdict(list)
    for detection in detections:
        if detection.get('score', 1.0) >= confidence_threshold:
            images_detections[detection['image_id']].append(detection)
    
    print(f"Found detections for {len(images_detections)} unique images")
    
    # Class labels and colors (adjust based on your classes)
    class_names = {
        0: "Bus",
        1: "Bike",
        2: "Car", 
        3: "Pedestrian",
        4: "Truck"
    }
    
    # Generate distinct colors for each class
    np.random.seed(42)  # For reproducibility
    colors = {}
    for class_id in class_names.keys():
        colors[class_id] = (
            int(np.random.randint(0, 256)),
            int(np.random.randint(0, 256)), 
            int(np.random.randint(0, 256))
        )
    
    # Set max images to process
    image_ids = list(images_detections.keys())
    if max_images is not None:
        image_ids = image_ids[:max_images]
    
    # Process each image
    for image_id in tqdm(image_ids):
        # Get image name from ID
        image_filename = get_image_name_from_id(image_id)
        
        # Find the actual image path
        image_path = find_image_path(image_filename, image_dir)
        
        if not image_path or not os.path.exists(image_path):
            print(f"Image not found for ID {image_id} (estimated filename: {image_filename})")
            continue
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        # Make a copy for visualization
        vis_image = image.copy()
        
        # Draw each detection
        for det in images_detections[image_id]:
            x, y, w, h = [int(v) for v in det['bbox']]
            category_id = det['category_id']
            confidence = det.get('score', 1.0)
            
            # Get class name and color
            class_name = class_names.get(category_id, f"Class {category_id}")
            color = colors.get(category_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size and background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw text background
            cv2.rectangle(
                vis_image, 
                (x, y - text_size[1] - 10), 
                (x + text_size[0], y), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                vis_image, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        # Save the visualized image
        if save_images and output_dir:
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, vis_image)
        
        # Show the image if requested
        if show_images:
            # Resize large images for display
            height, width = vis_image.shape[:2]
            max_dim = 1280
            if height > max_dim or width > max_dim:
                scale = max_dim / max(height, width)
                new_height, new_width = int(height * scale), int(width * scale)
                display_img = cv2.resize(vis_image, (new_width, new_height))
            else:
                display_img = vis_image.copy()
                
            # Display image
            cv2.imshow(f"Detection ID: {image_id}", display_img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Press 'q' to quit
            if key == ord('q'):
                break
    
    print("Visualization complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize detections from COCO JSON file")
    parser.add_argument('--json', type=str, required=True, help='Path to JSON file with detections')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing original images')
    parser.add_argument('--output_dir', type=str, default='visualization_results', help='Directory to save visualization results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Show visualized images')
    parser.add_argument('--no_save', action='store_true', help='Do not save visualized images')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    visualize_detections(
        json_file=args.json,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.conf,
        show_images=args.show,
        save_images=not args.no_save,
        max_images=args.max_images
    )