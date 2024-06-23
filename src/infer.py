import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from torch.nn import functional as F

import json
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.data.road import Road

device = 'cpu'
model_name = 'alexnet'
epochs = 1

with open('ckpt/labels_map.json', 'r') as openfile:
    # Reading from json file
    labels_map = json.load(openfile)

def infer(ckpt_path:str, image_path: str, device):
    image = Image.open(image_path).convert("RGB")
    transform = A.Compose([
        A.Resize(360, 240),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),   # image: (Height, Width, Channel) -> (Channel, Height, Width)
    ])  

    model = torch.jit.load(ckpt_path).to(device)

    transformed_image = transform(image=np.array(image))['image'].to(device)
    # (Channel, Height, Width) -> (1, Channel, Height, Width)
    pred = model(transformed_image.unsqueeze(0)).squeeze(0).cpu() 
    pred = F.softmax(pred, dim=0)
    pred_class = pred.argmax()

    idx2label = {v: k for (k, v) in labels_map.items()}
    return idx2label[pred_class.item()]

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and device =='cuda' else "cpu")
    
    # inference
    pred = infer(f"ckpt/{model_name}_{epochs}epoch.pt", "data/dry_asphalt/202201252343338-dry-asphalt-smooth.jpg", DEVICE)
    print(pred)