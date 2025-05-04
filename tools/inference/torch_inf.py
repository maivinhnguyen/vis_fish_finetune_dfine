# """
# Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
# """

# import torch
# import torch.nn as nn
# import torchvision.transforms as T

# import numpy as np
# from PIL import Image, ImageDraw

# import sys
# import os
# import cv2  # Added for video processing

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from engine.core import YAMLConfig


# def draw(images, labels, boxes, scores, thrh=0.4):
#     for i, im in enumerate(images):
#         draw = ImageDraw.Draw(im)

#         scr = scores[i]
#         lab = labels[i][scr > thrh]
#         box = boxes[i][scr > thrh]
#         scrs = scr[scr > thrh]

#         for j, b in enumerate(box):
#             draw.rectangle(list(b), outline='red')
#             draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue', )

#         im.save('torch_results.jpg')


# def process_image(model, device, file_path):
#     im_pil = Image.open(file_path).convert('RGB')
#     w, h = im_pil.size
#     orig_size = torch.tensor([[w, h]]).to(device)

#     transforms = T.Compose([
#         T.Resize((640, 640)),
#         T.ToTensor(),
#     ])
#     im_data = transforms(im_pil).unsqueeze(0).to(device)

#     output = model(im_data, orig_size)
#     labels, boxes, scores = output

#     draw([im_pil], labels, boxes, scores)


# def process_video(model, device, file_path):
#     cap = cv2.VideoCapture(file_path)

#     # Get video properties
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter('torch_results.mp4', fourcc, fps, (orig_w, orig_h))

#     transforms = T.Compose([
#         T.Resize((640, 640)),
#         T.ToTensor(),
#     ])

#     frame_count = 0
#     print("Processing video frames...")
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to PIL image
#         frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         w, h = frame_pil.size
#         orig_size = torch.tensor([[w, h]]).to(device)

#         im_data = transforms(frame_pil).unsqueeze(0).to(device)

#         output = model(im_data, orig_size)
#         labels, boxes, scores = output

#         # Draw detections on the frame
#         draw([frame_pil], labels, boxes, scores)

#         # Convert back to OpenCV image
#         frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

#         # Write the frame
#         out.write(frame)
#         frame_count += 1

#         if frame_count % 10 == 0:
#             print(f"Processed {frame_count} frames...")

#     cap.release()
#     out.release()
#     print("Video processing complete. Result saved as 'results_video.mp4'.")


# def main(args):
#     """Main function"""
#     cfg = YAMLConfig(args.config, resume=args.resume)

#     if 'HGNetv2' in cfg.yaml_cfg:
#         cfg.yaml_cfg['HGNetv2']['pretrained'] = False

#     if args.resume:
#         checkpoint = torch.load(args.resume, map_location='cpu')
#         if 'ema' in checkpoint:
#             state = checkpoint['ema']['module']
#         else:
#             state = checkpoint['model']
#     else:
#         raise AttributeError('Only support resume to load model.state_dict by now.')

#     # Load train mode state and convert to deploy mode
#     cfg.model.load_state_dict(state)

#     class Model(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.model = cfg.model.deploy()
#             self.postprocessor = cfg.postprocessor.deploy()

#         def forward(self, images, orig_target_sizes):
#             outputs = self.model(images)
#             outputs = self.postprocessor(outputs, orig_target_sizes)
#             return outputs

#     device = args.device
#     model = Model().to(device)

#     # Check if the input file is an image or a video
#     file_path = args.input
#     if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
#         # Process as image
#         process_image(model, device, file_path)
#         print("Image processing complete.")
#     else:
#         # Process as video
#         process_video(model, device, file_path)


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, required=True)
#     parser.add_argument('-r', '--resume', type=str, required=True)
#     parser.add_argument('-i', '--input', type=str, required=True)
#     parser.add_argument('-d', '--device', type=str, default='cpu')
#     args = parser.parse_args()
#     main(args)
"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw
import sys
import os
import cv2
import json  # Added for COCO format export

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue')

        im.save(f'torch_results_{i}.jpg')


def process_image(model, device, file_path, image_id=0, coco_annotations=None):
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

    if coco_annotations is not None:
        # Convert predictions to COCO format
        scr = scores[0]
        lab = labels[0][scr > 0.4]
        box = boxes[0][scr > 0.4]
        scrs = scr[scr > 0.4]

        for j, b in enumerate(box):
            # Convert box to [x, y, width, height]
            coco_box = [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
            annotation = {
                "image_id": image_id,
                "category_id": int(lab[j].item()),
                "bbox": coco_box,
                "score": float(scrs[j].item()),
                "id": len(coco_annotations) + 1
            }
            coco_annotations.append(annotation)

    draw([im_pil], labels, boxes, scores)
    return image_id + 1


def process_folder(model, device, folder_path):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []  # Populate based on model classes if available
    }
    
    # Assuming categories are known or can be extracted from model
    # Replace with actual category IDs and names if available
    coco_output["categories"] = [
        {"id": i, "name": f"class_{i}"} for i in range(1, 91)  # Example: 90 classes
    ]

    image_id = 1
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    print(f"Processing images in folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            file_path = os.path.join(folder_path, filename)
            coco_output["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": Image.open(file_path).size[0],
                "height": Image.open(file_path).size[1]
            })
            image_id = process_image(model, device, file_path, image_id, coco_output["annotations"])
    
    # Save COCO annotations to JSON
    with open('coco_results.json', 'w') as f:
        json.dump(coco_output, f, indent=4)
    
    print("Folder processing complete. COCO annotations saved as 'coco_results.json'.")
    return image_id


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        draw([frame_pil], labels, boxes, scores)

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'torch_results.mp4'.")


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

    input_path = args.input
    if os.path.isdir(input_path):
        # Process as folder
        process_folder(model, device, input_path)
    elif os.path.splitext(input_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process as image
        process_image(model, device, input_path)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, input_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)