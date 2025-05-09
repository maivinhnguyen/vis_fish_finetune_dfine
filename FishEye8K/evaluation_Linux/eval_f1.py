from pycocotools.coco import COCO
from pycocotools.cocoeval_modified import COCOeval
import json

coco_gt = COCO("/home/enn/workspace/contest/aicity/AICITY2024_Track4/dataset/dataset_used_to_experiment/fisheye8K/val/val.json")

print(type(coco_gt))
gt_image_ids = coco_gt.getImgIds()

print("Total images ", len(gt_image_ids))

with open('file_inference/maipeo2.json', 'r') as f:
    detection_data = json.load(f)
with open("/home/enn/workspace/contest/aicity/AICITY2024_Track4/dataset/dataset_used_to_experiment/fisheye8K/val/val.json", "r") as f:
    ground_truth = json.load(f)

mp = dict()
for i in ground_truth["images"]:
    mp[i["file_name"]] = i["id"]

mp2 = dict()
for i in range(len(detection_data["images"])):
    mp2[detection_data["images"][i]["id"]] = mp[detection_data["images"][i]["file_name"]]
    detection_data["images"][i]["id"] = mp[detection_data["images"][i]["file_name"]]
    
for i in range(len(detection_data["annotations"])):
    detection_data["annotations"][i]["image_id"] = mp2[detection_data["annotations"][i]["image_id"]]

detection_data = detection_data["annotations"]

print(detection_data[0])

filtered_detection_data = [
    item for item in detection_data if item['image_id'] in gt_image_ids]

with open('./temp.json', 'w') as f:
    json.dump(filtered_detection_data, f)

coco_dt = coco_gt.loadRes('./temp.json')
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
print('----------------------------------------')
print('AP_0.5-0.95', coco_eval.stats[0])
print('AP_0.5', coco_eval.stats[1])
print('AP_S', coco_eval.stats[3])
print('AP_M', coco_eval.stats[4])
print('AP_L', coco_eval.stats[5])
print('f1_score: ', coco_eval.stats[24])
print('----------------------------------------')
