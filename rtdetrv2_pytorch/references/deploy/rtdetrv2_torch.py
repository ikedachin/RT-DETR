"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
# Add the parent directory to the Python path
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw

from src.core import YAMLConfig

# for dev
json_path = "./coco_dataset/annotations/instances_train.json"

def load_class_names(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    categories = data['categories']
    class_names = {cat['id']: cat['name'] for cat in categories}
    return class_names

print("Class names:", load_class_names(json_path))





def draw(images, labels, boxes, scores, output_dir, base_name, label_names=None, thrh = 0.6):
    """
    Draw bounding boxes and labels on images and save them to the output directory.
    
    Args:
        images: List of PIL Image objects to draw on
        labels: Tensor of predicted class labels for each detection
        boxes: Tensor of bounding box coordinates [x1, y1, x2, y2] for each detection
        scores: Tensor of confidence scores for each detection
        output_dir (str): Directory path where the output images will be saved
        base_name (str): Base filename for the output image
        label_names (dict, optional): Dictionary mapping label IDs to class names. Defaults to None.
        thrh (float, optional): Confidence threshold for filtering detections. Defaults to 0.6.
    
    Returns:
        None: Saves annotated images to the specified output directory
    """
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        # pickup lab, box, scrs if score > thrh
        scr = scores[i].detach().cpu()
        lab = labels[i][scr > thrh].detach().cpu()
        box = boxes[i][scr > thrh].detach().cpu()
        scrs = scores[i][scr > thrh].detach().cpu()

        # draw boxes and labels ここを改造して、画像を切り取ってそれぞれのタスクに渡すとOCRとかできるかも
        # 想定タスクboxとラベルの結果をもって、読み順を推論
        # OCR、画像解説などを行って、読み順通りにテキストを保存
        # 今はとりあえず箱とラベルを描画するだけ
        for j, b in enumerate(box):
            # Draw bounding box
            draw.rectangle(list(b.numpy()), outline='red', width=2)
            
            # Prepare label text
            if label_names is not None:
                label_text = label_names.get(lab[j].item(), str(lab[j].item()))
            else:
                label_text = str(lab[j].item())
            text = f"{label_text} {round(scrs[j].item(), 2)}"
            
            # Calculate text position (above the box)
            text_x = b[0].item()
            text_y = b[1].item() - 15  # 15 pixels above the box
            
            # Draw text with outline (stroke) for hollow effect
            outline_color = 'black'
            text_color = 'white'
            
            # Draw outline by drawing text slightly offset in all directions
            for offset_x in [-1, 0, 1]:
                for offset_y in [-1, 0, 1]:
                    if offset_x != 0 or offset_y != 0:
                        draw.text((text_x + offset_x, text_y + offset_y), 
                                text=text, fill=outline_color)
            
            # Draw main text
            draw.text((text_x, text_y), text=text, fill=text_color)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        im.save(os.path.join(output_dir, f'{base_name}'))

def get_crop_area(image, labels, boxes, scores, base_name, label_names=None, thrh = 0.6) -> list:
    """
    Crop detection areas from an image based on predicted bounding boxes.
    
    Args:
        image (PIL.Image): The original PIL Image to crop from
        labels (torch.Tensor): Tensor of predicted class labels for each detection
        boxes (torch.Tensor): Tensor of bounding box coordinates [x1, y1, x2, y2] for each detection
        scores (torch.Tensor): Tensor of confidence scores for each detection
        base_name (str): Base filename (currently unused but kept for consistency)
        label_names (dict, optional): Dictionary mapping label IDs to class names. Defaults to None.
        thrh (float, optional): Confidence threshold for filtering detections. Defaults to 0.6.
    
    Returns:
        tuple: A tuple containing:
            - crop_areas (list): List of PIL Image objects, each containing a cropped detection area
            - crop_area_datas (list): List of dictionaries containing metadata for each crop:
                - 'label' (str): Class name or label ID
                - 'score' (float): Confidence score rounded to 2 decimal places
                - 'box' (list): Bounding box coordinates [x1, y1, x2, y2]
    """
    
    crop_areas = []
    crop_area_datas = []
    scr = scores.detach().cpu()
    lab = labels[scr > thrh].detach().cpu()
    box = boxes[scr > thrh].detach().cpu()
    scrs = scores[scr > thrh].detach().cpu()

    for j, b in enumerate(box):
        # Ensure coordinates are integers and within image bounds
        x1, y1, x2, y2 = int(b[0].item()), int(b[1].item()), int(b[2].item()), int(b[3].item())
        
        # Crop the original image (not the drawn one)
        # x1, y1: バウンディングボックスの左上の座標
        # x2, y2: バウンディングボックスの右下の座標
        crop_area = image.crop((x1, y1, x2, y2))
        crop_areas.append(crop_area)
        crop_area_datas.append({
            'label': label_names.get(lab[j].item(), str(lab[j].item())) if label_names is not None else str(lab[j].item()),
            'score': round(scrs[j].item(), 2),
            'box': [x1, y1, x2, y2]
        })
    
    return crop_areas, crop_area_datas


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.training_config is not None:
        label_names = load_class_names(args.training_config)
    else:
        label_names = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    model.eval()  # Set model to evaluation mode

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    with torch.no_grad():  # Disable gradient computation for inference
        output = model(im_data, orig_size)
    
    labels, boxes, scores = output
    # print(f'output: {output}')
    print
    print(f'labels: {labels.shape}\n boxes: {boxes.shape}\n scores: {scores.shape}')
    filename_with_ext = os.path.basename(args.im_file)
    b_name = os.path.basename(filename_with_ext)
    
    # First crop the original image (before drawing)
    crop_areas, crop_area_datas = get_crop_area(im_pil, labels[0], boxes[0], scores[0], b_name, label_names)
    
    # Then draw on the image
    draw([im_pil], labels, boxes, scores, args.output, b_name, label_names)
    
    # Save cropped areas
    for idx, crop_area in enumerate(crop_areas):
        crop_area.save(os.path.join(args.output, f'crop_{idx}_{b_name}'))
    # # Save crop area data as JSON
    # with open(os.path.join(args.output, f'crop_data_{b_name}.json'), 'w', encoding='utf-8') as f:
    #     json.dump(crop_area_datas, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='path to model config file', )
    parser.add_argument('-r', '--resume', type=str, help='path to model checkpoint', )
    parser.add_argument('-f', '--im-file', type=str, help='path to input image file', )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-o', '--output', type=str, default='output')
    parser.add_argument('-t', '--training_config', type=str, default=None, help='path to training config file, optional')
    args = parser.parse_args()
    main(args)
