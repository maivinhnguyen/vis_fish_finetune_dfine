import glob
import imagesize
import os
import json
import argparse
from tqdm import tqdm


def convert(dir_data):
    train_data = os.path.join(dir_data, 'VisDrone2019-DET-train', 'VisDrone2019-DET-train')
    val_data = os.path.join(dir_data, 'VisDrone2019-DET-val', 'VisDrone2019-DET-val')
    test_data = os.path.join(dir_data, 'VisDrone2019-DET-test-dev', 'VisDrone2019-DET-test-dev')
    subsets = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
    }

    # Define valid category IDs (VisDrone: 1-10) and remap to 0-based
    VALID_IDS = list(range(1, 11))  # exclude 'others' (11)

    for split, data_path in subsets.items():
        print(f"Processing {split} at {data_path}")
        coco = {
            'images': [],
            'annotations': [],
            'categories': []
        }

        # build image index
        img_folder = os.path.join(data_path, 'images')
        image_id_map = {}
        img_id = 0
        print('  -> Reading images')
        for img_path in tqdm(glob.glob(os.path.join(img_folder, '*.jpg'))):
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

        # build annotations
        anno_folder = os.path.join(data_path, 'annotations')
        anno_id = 0
        print('  -> Reading annotations')
        for txt_path in tqdm(glob.glob(os.path.join(anno_folder, '*.txt'))):
            img_name = os.path.splitext(os.path.basename(txt_path))[0] + '.jpg'
            if img_name not in image_id_map:
                continue
            image_id = image_id_map[img_name]
            # read lines
            with open(txt_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                parts = line.split(',')
                orig_id = int(parts[5])
                # skip invalid or 'others'
                if orig_id not in VALID_IDS:
                    continue
                # remap to 0-based
                cat_id = orig_id - 1
                x, y, w, h = map(int, parts[:4])
                area = w * h
                coco['annotations'].append({
                    'id': anno_id,
                    'image_id': image_id,
                    'category_id': cat_id,
                    'bbox': [x, y, w, h],
                    'area': area,
                    'iscrowd': 0,
                    'ignore': 0
                })
                anno_id += 1

        # create categories array
        names = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]
        for cid, name in enumerate(names):
            coco['categories'].append({
                'id': cid,
                'name': name,
                'supercategory': 'none'
            })

        # save JSON
        out_name = f'annotations_VisDrone_{split}.json'
        out_path = os.path.join('/kaggle/working', out_name)
        print(f"  -> Saving COCO annotations to {out_path}")
        with open(out_path, 'w') as f:
            json.dump(coco, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./', help='Root data directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert(args.data_dir)
