import os
import json
import argparse
import cv2
import random


def visualize_annotations_cv2(json_path, images_dir, output_dir, num_images_to_visualize):
    """
    Load COCO-format JSON annotations and save annotated images to output_dir using OpenCV.
    Includes class name and score in the visualization.

    Args:
        json_path (str): Path to COCO JSON file.
        images_dir (str): Directory containing image files referenced in JSON.
        output_dir (str): Directory to save annotated images.
        num_images_to_visualize (int): Number of images to visualize (randomly selected).
    """
    # Load annotation data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images_data = data.get('images', [])
    annotations_data = data.get('annotations', [])
    
    # Define your class mapping (ID to Name)
    # This should match the 'categories' section in your COCO JSON if it exists,
    # or be defined based on your model's output.
    # For example, if your model outputs class IDs 0-4:
    defined_categories = {
        0: 'bus',
        1: 'bike',
        2: 'car',
        3: 'pedestrian',
        4: 'truck'
    }
    # If your JSON already has a 'categories' section, you can use that instead:
    # json_categories = {c['id']: c['name'] for c in data.get('categories', [])}
    # And then decide which mapping to use (defined_categories or json_categories)
    # For this example, we'll prioritize defined_categories if a category_id is found there.
    
    categories_from_json = {c['id']: c['name'] for c in data.get('categories', [])}


    # Randomly select images if the list is larger than num_images_to_visualize
    if len(images_data) > num_images_to_visualize:
        selected_images_info = random.sample(images_data, num_images_to_visualize)
    else:
        selected_images_info = images_data # Visualize all if fewer than requested

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Group annotations by image_id for faster lookup
    annotations_by_image_id = {}
    for ann in annotations_data:
        img_id = ann['image_id']
        if img_id not in annotations_by_image_id:
            annotations_by_image_id[img_id] = []
        annotations_by_image_id[img_id].append(ann)


    # Process each selected image
    for img_info in selected_images_info:
        img_file = img_info['file_name']
        img_path = os.path.join(images_dir, img_file)

        if not os.path.isfile(img_path):
            print(f"Image not found, skipping: {img_path}")
            continue

        # Read image with OpenCV
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image, skipping: {img_path}")
            continue
        
        print(f"Processing: {img_path}")

        # Get annotations for the current image
        current_image_annotations = annotations_by_image_id.get(img_info['id'], [])

        for ann in current_image_annotations:
            x, y, w, h = map(int, ann['bbox'])
            category_id = ann['category_id']
            score = ann.get('score', None)  # Get score if available, otherwise None

            # Determine class name
            # Prioritize your defined mapping, then try from JSON, then 'N/A'
            class_name = defined_categories.get(category_id)
            if class_name is None: # If not in your defined map, try the JSON's categories
                class_name = categories_from_json.get(category_id, 'N/A_ID:'+str(category_id))


            # Create label string
            label_parts = [class_name]
            if score is not None:
                label_parts.append(f"{score:.2f}") # Format score to 2 decimal places
            
            label_text = ": ".join(label_parts)

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red box

            # Prepare for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (0, 255, 0) # Green text for visibility against red box

            # Get text size to position it nicely
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # Position text above the top-left corner of the bounding box
            # Ensure text background is opaque for better readability
            text_x = x
            text_y = y - 5 - baseline # Adjust y to be above the box
            if text_y < text_height : # If text goes off screen on top, put it inside
                text_y = y + text_height + 5


            # Draw a filled rectangle as background for the text
            cv2.rectangle(img, (text_x, text_y - text_height - baseline//2), 
                               (text_x + text_width, text_y + baseline//2), 
                               (0,0,0), cv2.FILLED) # Black background
            # Put text
            cv2.putText(img, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


        # Save annotated image
        # Sanitize filename if needed, though os.path.join should handle most cases
        safe_img_file = os.path.basename(img_file) # Ensure only filename part
        out_path = os.path.join(output_dir, safe_img_file)
        try:
            cv2.imwrite(out_path, img)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Error saving image {out_path}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO annotations with class names and scores.')
    parser.add_argument('-j', '--json', required=True, help='Path to COCO JSON file (predictions or ground truth with scores)')
    parser.add_argument('-i', '--images_dir', required=True, help='Directory containing images') # Changed -d to -i for clarity
    parser.add_argument('-o', '--output_dir', required=True, help='Directory to save annotated images') # Changed -outdir to -output_dir
    parser.add_argument('-n', '--num_images', type=int, default=10, help='Number of images to visualize (default: 10)') # Changed required to default
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualize_annotations_cv2(args.json, args.images_dir, args.output_dir, args.num_images)