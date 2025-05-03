import json
import os
import random # <--- Import the random module
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import matplotlib.pyplot as plt

# --- Configuration ---
annotation_file = '/kaggle/working/visdrone_coco_output/visdrone_train_coco.json'
image_dir = '/kaggle/input/visdrone-dataset/VisDrone2019-DET-train/VisDrone2019-DET-train/images'

# === No longer need to edit filename here ===
# target_filename = "..." # <-- REMOVED

box_outline_color = 'lime' # Bright color for boxes
box_width = 3              # Slightly thicker boxes
label_text_color = 'black' # Black text
label_background_color = 'lime' # Use the same bright color for background
font_size = 16             # Increased font size for visibility

# --- Load Annotations ---
print(f"Loading annotations from: {annotation_file}")
try:
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Annotation file not found at {annotation_file}")
    exit()
except json.JSONDecodeError:
    print(f"ERROR: Could not decode JSON from {annotation_file}. Is it a valid JSON file?")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading annotations: {e}")
    exit()

print("Annotations loaded successfully.")

# --- Pre-process Annotations and Select Random Image ---
target_image_id = None
target_image_info = None
target_filename = None # Initialize

try:
    all_images = coco_data.get('images', [])
    if not all_images:
        print("ERROR: No 'images' data found in the annotation file.")
        exit()

    # === Select a random image ===
    target_image_info = random.choice(all_images)
    target_filename = target_image_info.get('file_name')
    target_image_id = target_image_info.get('id')

    if target_filename is None or target_image_id is None:
        print(f"ERROR: Randomly selected image data is incomplete: {target_image_info}")
        exit()

    print(f"Randomly selected image: '{target_filename}' (ID: {target_image_id})")
    # ============================

    # Prepare annotations lookup (needed for the selected image)
    annotations_by_image_id = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image_id:
            annotations_by_image_id[img_id] = []
        annotations_by_image_id[img_id].append(ann)

    # Prepare categories lookup
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

    if not categories:
        print("WARNING: No categories found in the annotation file. Labels will show IDs.")

except KeyError as e:
    print(f"ERROR: Missing expected key in COCO JSON: {e}. Check file structure.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during preprocessing: {e}")
    exit()


# --- Load Font (prioritize TrueType for better rendering) ---
try:
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    print(f"Loaded font: DejaVuSans.ttf (size {font_size})")
except IOError:
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        print(f"Loaded font: arial.ttf (size {font_size})")
    except IOError:
        print(f"Warning: DejaVuSans.ttf and arial.ttf not found. Using default PIL font (may be low quality).")
        font = ImageFont.load_default()
except Exception as e:
     print(f"Warning: Could not load preferred font: {e}. Using default PIL font.")
     font = ImageFont.load_default()

# --- Draw and Display Image ---
# Construct the full image path using the randomly selected filename
image_path = os.path.join(image_dir, target_filename)

try:
    # Open image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    print(f"Processing image: {image_path}")

    # Get annotations for this specific image ID
    img_annotations = annotations_by_image_id.get(target_image_id, [])

    if not img_annotations:
         print(f"No annotations found for image: {target_filename} (ID: {target_image_id})")

    # Draw bounding boxes and labels
    for ann in img_annotations:
        bbox = ann.get('bbox')
        category_id = ann.get('category_id')

        # --- Annotation data validation ---
        if bbox is None or category_id is None:
            print(f"Warning: Incomplete annotation skipped (Ann ID: {ann.get('id')}) - Missing bbox or category_id.")
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
             print(f"Warning: Invalid bbox format skipped (Ann ID: {ann.get('id')}) - bbox: {bbox}")
             continue

        # --- Bbox coordinate processing ---
        try:
            x_min, y_min, width, height = map(float, bbox) # Convert to float first for robustness
            x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height) # Then to int
            if width <= 0 or height <= 0:
                print(f"Warning: Non-positive width/height skipped (Ann ID: {ann.get('id')}) - bbox: {bbox}")
                continue
        except (ValueError, TypeError) as e:
            print(f"Warning: Invalid bbox coordinate values skipped (Ann ID: {ann.get('id')}) - bbox: {bbox} Error: {e}")
            continue

        x_max = x_min + width
        y_max = y_min + height

        # --- Drawing ---
        # Draw rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline=box_outline_color, width=box_width)

        # Prepare label text
        category_name = categories.get(category_id, f"ID:{category_id}") # Get name or show ID if not found

        # Draw label with background
        try:
             # Use textbbox for better size calculation if available
             if hasattr(draw, 'textbbox'):
                 text_bbox = draw.textbbox((0, 0), category_name, font=font)
                 text_width = text_bbox[2] - text_bbox[0]
                 text_height = text_bbox[3] - text_bbox[1]
             else: # Fallback for older PIL versions
                 text_width, text_height = draw.textsize(category_name, font=font)


             text_bg_y0 = max(0, y_min - text_height - 4) # Ensure y0 doesn't go < 0
             text_bg_x0 = x_min
             text_bg_x1 = x_min + text_width + 4
             text_bg_y1 = text_bg_y0 + text_height + 4

             # Check if text goes off the right edge, shift left if necessary
             img_width, _ = img.size
             if text_bg_x1 > img_width:
                 shift = text_bg_x1 - img_width
                 text_bg_x0 -= shift
                 text_bg_x1 -= shift

             draw.rectangle(
                [text_bg_x0, text_bg_y0, text_bg_x1, text_bg_y1],
                fill=label_background_color
             )
             draw.text((text_bg_x0 + 2, text_bg_y0 + 2), category_name, fill=label_text_color, font=font)

        except AttributeError:
             print(f"Warning: Could not accurately determine text size for '{category_name}' (AttributeError). Using basic text draw.")
             draw.text((x_min + 2, y_min + 2), category_name, fill=label_text_color, font=font) # Simpler placement
        except Exception as e:
             print(f"Error drawing text for {category_name} on {target_filename}: {e}")
             draw.text((x_min + 2, y_min + 2), category_name, fill=label_text_color, font=font)


    # --- Display the final image ---
    fig, ax = plt.subplots(1, 1, figsize=(15, 10)) # Adjust size as needed for better view
    ax.imshow(img)
    ax.set_title(f"{target_filename}\n(ID: {target_image_id})", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


except FileNotFoundError:
    print(f"ERROR: Image file not found at {image_path}. Check image_dir path and that the file exists.")
except UnidentifiedImageError:
     print(f"ERROR: Could not open or identify image file at {image_path}. It might be corrupted or not a valid image format.")
except Exception as e:
    print(f"ERROR: An unexpected error occurred processing image {image_path}: {e}")
    # import traceback
    # traceback.print_exc()


print("\nFinished displaying image.")