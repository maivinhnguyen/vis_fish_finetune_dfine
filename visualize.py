import json
import os
import random
from PIL import Image, ImageDraw, ImageFont
import argparse

def visualize_random_detection(json_path, image_folder, output_path="visualization_output.jpg"):
    """
    Loads detections, picks a random image with detections, draws the bounding boxes,
    and saves the result.
    """
    print(f"Loading detections from: {json_path}")
    if not os.path.exists(json_path):
        print(f"Error: Detection file not found at {json_path}")
        return

    print(f"Image folder specified: {image_folder}")
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_path}: {e}")
        return
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return

    # --- Extract necessary data ---
    if not isinstance(data, dict):
        print("Error: Expected JSON root to be a dictionary (COCO format).")
        return

    images_info = data.get('images', [])
    annotations = data.get('annotations', [])
    categories_info = data.get('categories', [])

    if not images_info:
        print("Error: No 'images' found in the JSON data.")
        return
    if not annotations:
        print("Warning: No 'annotations' found in the JSON data. Cannot visualize.")
        # You might still want to load and save a random image without boxes
        # return

    print(f"Found {len(images_info)} images and {len(annotations)} annotations in JSON.")

    # --- Create mappings for easy lookup ---
    image_id_to_info = {img['id']: img for img in images_info}
    category_id_to_name = {cat['id']: cat.get('name', f"ID:{cat['id']}") for cat in categories_info}

    image_id_to_annotations = {}
    for ann in annotations:
        img_id = ann.get('image_id')
        if img_id is not None:
            if img_id not in image_id_to_annotations:
                image_id_to_annotations[img_id] = []
            image_id_to_annotations[img_id].append(ann)

    # --- Select a random image ID that has annotations ---
    image_ids_with_annotations = list(image_id_to_annotations.keys())

    if not image_ids_with_annotations:
        print("Error: No image IDs found with corresponding annotations.")
        # Check if images exist even without annotations
        if image_id_to_info:
             print("Trying to pick a random image ID from the 'images' list instead.")
             image_ids_with_annotations = list(image_id_to_info.keys())
             if not image_ids_with_annotations:
                 print("Error: No image IDs found in 'images' list either.")
                 return
        else:
             return


    random_image_id = random.choice(image_ids_with_annotations)
    print(f"\nSelected random image ID: {random_image_id}")

    if random_image_id not in image_id_to_info:
        print(f"Error: Image ID {random_image_id} found in annotations but not in 'images' list.")
        return

    selected_image_info = image_id_to_info[random_image_id]
    image_filename = selected_image_info.get('file_name')

    if not image_filename:
        print(f"Error: Image ID {random_image_id} has no 'file_name' attribute in JSON.")
        return

    image_path = os.path.join(image_folder, image_filename)
    print(f"Corresponding image file: {image_path}")

    # --- Load the image ---
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        # Try searching common subdirectories as a fallback
        common_subdirs = ['images', 'train', 'val', 'test'] # Add others if needed
        found = False
        for subdir in common_subdirs:
            alt_path = os.path.join(image_folder, subdir, image_filename)
            if os.path.exists(alt_path):
                image_path = alt_path
                print(f"Found image in subdirectory: {image_path}")
                found = True
                break
        if not found:
             print(f"Could not find {image_filename} in {image_folder} or common subdirectories.")
             return


    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        # Try to load a default font
        try:
            font = ImageFont.truetype("arial.ttf", 15) # Adjust font and size if needed
        except IOError:
            font = ImageFont.load_default()
            print("Arial font not found, using default PIL font.")

    except FileNotFoundError:
        print(f"Error: Cannot open image file (double check path): {image_path}")
        return
    except Exception as e:
        print(f"Error loading image or creating Draw object: {e}")
        return

    # --- Draw annotations for the selected image ---
    annotations_for_image = image_id_to_annotations.get(random_image_id, [])
    print(f"Drawing {len(annotations_for_image)} annotations for this image.")

    for ann in annotations_for_image:
        bbox = ann.get('bbox')          # COCO format: [xmin, ymin, width, height]
        score = ann.get('score')        # Score from your detection
        category_id = ann.get('category_id') # 0-based category ID

        if bbox is None or score is None or category_id is None:
            print(f"Warning: Skipping annotation with missing data: {ann}")
            continue

        # Ensure bbox has 4 elements
        if len(bbox) != 4:
             print(f"Warning: Skipping annotation with invalid bbox length: {bbox}")
             continue

        try:
            # Convert COCO bbox to [xmin, ymin, xmax, ymax] for drawing
            xmin, ymin, w, h = map(float, bbox)
            xmax = xmin + w
            ymax = ymin + h
            draw_bbox = [xmin, ymin, xmax, ymax]

            # Get category name
            category_name = category_id_to_name.get(category_id, f"ID:{category_id}")

            # Create label text
            label = f"{category_name}: {score:.2f}"

            # Draw rectangle (adjust outline color/width if desired)
            draw.rectangle(draw_bbox, outline="red", width=2)

            # Draw text label (adjust position and fill color)
            text_position = (xmin, ymin - 15 if ymin > 15 else ymin) # Position above box
            # Optional: Draw a background rectangle for the text for better visibility
            # text_bbox = draw.textbbox(text_position, label, font=font) # Requires Pillow 9+
            # draw.rectangle(text_bbox, fill="red")
            draw.text(text_position, label, fill="white", font=font) # White text

        except Exception as e:
            print(f"Error drawing annotation {ann.get('id', 'N/A')}: {e}")
            print(f"  bbox: {bbox}")


    # --- Save the image ---
    try:
        image.save(output_path)
        print(f"\nSuccessfully saved visualization to: {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COCO detection results on a random image.")
    parser.add_argument(
        "--json",
        type=str,
        default="detections.json",
        help="Path to the COCO format detections JSON file (default: detections.json)"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to the folder containing the original images."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualization_output.jpg",
        help="Path to save the output visualization JPG file (default: visualization_output.jpg)"
    )

    args = parser.parse_args()

    visualize_random_detection(args.json, args.images, args.output)
