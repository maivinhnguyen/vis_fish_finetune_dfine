import glob
import imagesize
import os
import json
import argparse
from tqdm import tqdm


def convert(dir_data, output_dir=None):
    """
    Convert VisDrone annotations to COCO format
    
    Args:
        dir_data: Root directory containing VisDrone dataset folders
        output_dir: Directory to save the output JSON files (default: current directory)
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    train_data = os.path.join(dir_data, 'VisDrone2019-DET-train', 'VisDrone2019-DET-train')
    val_data = os.path.join(dir_data, 'VisDrone2019-DET-val', 'VisDrone2019-DET-val')
    test_data = os.path.join(dir_data, 'VisDrone2019-DET-test-dev', 'VisDrone2019-DET-test-dev')
    
    subsets = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
    }

    # Define valid category IDs (VisDrone: 1-10) and names
    categories = [
        {'id': 0, 'name': 'pedestrian', 'supercategory': 'human'},
        {'id': 1, 'name': 'people', 'supercategory': 'human'},
        {'id': 2, 'name': 'bicycle', 'supercategory': 'vehicle'},
        {'id': 3, 'name': 'car', 'supercategory': 'vehicle'},
        {'id': 4, 'name': 'van', 'supercategory': 'vehicle'},
        {'id': 5, 'name': 'truck', 'supercategory': 'vehicle'},
        {'id': 6, 'name': 'tricycle', 'supercategory': 'vehicle'},
        {'id': 7, 'name': 'awning-tricycle', 'supercategory': 'vehicle'},
        {'id': 8, 'name': 'bus', 'supercategory': 'vehicle'},
        {'id': 9, 'name': 'motor', 'supercategory': 'vehicle'}
    ]
    
    # Create a mapping from VisDrone IDs to COCO IDs
    visdrone_to_coco_id = {i+1: i for i in range(10)}  # VisDrone uses 1-10, COCO uses 0-9
    
    for split, data_path in subsets.items():
        print(f"Processing {split} at {data_path}")
        
        # Skip if directory doesn't exist
        if not os.path.exists(data_path):
            print(f"  -> Directory {data_path} not found, skipping...")
            continue
            
        coco = {
            'images': [],
            'annotations': [],
            'categories': categories
        }

        # Build image index
        img_folder = os.path.join(data_path, 'images')
        if not os.path.exists(img_folder):
            print(f"  -> Image folder {img_folder} not found, skipping...")
            continue
            
        image_id_map = {}
        img_id = 0
        print('  -> Reading images')
        for img_path in tqdm(glob.glob(os.path.join(img_folder, '*.jpg'))):
            try:
                width, height = imagesize.get(img_path)
                fname = os.path.basename(img_path)
                coco['images'].append({
                    'id': img_id,
                    'file_name': fname,
                    'width': width,
                    'height': height,
                    'license': 1
                })
                image_id_map[fname] = img_id
                img_id += 1
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        # Build annotations
        anno_folder = os.path.join(data_path, 'annotations')
        if not os.path.exists(anno_folder):
            print(f"  -> Annotation folder {anno_folder} not found, skipping...")
            continue
            
        anno_id = 0
        print('  -> Reading annotations')
        for txt_path in tqdm(glob.glob(os.path.join(anno_folder, '*.txt'))):
            img_name = os.path.splitext(os.path.basename(txt_path))[0] + '.jpg'
            if img_name not in image_id_map:
                print(f"  Warning: Found annotation for {img_name} but image not found, skipping...")
                continue
                
            image_id = image_id_map[img_name]
            
            # Read annotation lines
            try:
                with open(txt_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                for line in lines:
                    parts = line.split(',')
                    
                    # Check if we have at least 8 parts (bbox, score, class, truncation, occlusion)
                    if len(parts) < 8:
                        continue
                    
                    try:
                        # Extract the class ID (6th item, 0-indexed)
                        orig_id = int(parts[5])
                        
                        # Skip if not in our category mapping
                        if orig_id not in visdrone_to_coco_id:
                            continue
                            
                        # Map to COCO ID (0-indexed)
                        cat_id = visdrone_to_coco_id[orig_id]
                        
                        # Extract bbox coordinates - ensure they're valid integers
                        x = max(0, int(float(parts[0])))
                        y = max(0, int(float(parts[1])))
                        w = max(1, int(float(parts[2])))  # Ensure width is at least 1
                        h = max(1, int(float(parts[3])))  # Ensure height is at least 1
                        
                        # Get image dimensions
                        img_info = next(img for img in coco['images'] if img['id'] == image_id)
                        img_width, img_height = img_info['width'], img_info['height']
                        
                        # Make sure bbox is within image bounds
                        x = min(x, img_width - 1)
                        y = min(y, img_height - 1)
                        w = min(w, img_width - x)
                        h = min(h, img_height - y)
                        
                        # Skip tiny bboxes
                        if w < 1 or h < 1:
                            continue
                            
                        area = w * h
                        
                        # Add the annotation
                        coco['annotations'].append({
                            'id': anno_id,
                            'image_id': image_id,
                            'category_id': cat_id,
                            'bbox': [x, y, w, h],
                            'area': area,
                            'iscrowd': 0,
                            'segmentation': []  # COCO format requires this
                        })
                        anno_id += 1
                    except (ValueError, IndexError) as e:
                        print(f"  Warning: Error parsing line in {txt_path}: {e}")
                        continue
            
            except Exception as e:
                print(f"Error processing annotation file {txt_path}: {e}")
                continue

        # Save JSON
        out_name = f'annotations_VisDrone_{split}.json'
        out_path = os.path.join(output_dir, out_name)
        print(f"  -> Saving COCO annotations to {out_path}")
        print(f"  -> Statistics: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
        
        with open(out_path, 'w') as f:
            json.dump(coco, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert VisDrone annotations to COCO format')
    parser.add_argument('--data_dir', type=str, required=True, help='Root data directory containing VisDrone folders')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output JSON files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert(args.data_dir, args.output_dir)