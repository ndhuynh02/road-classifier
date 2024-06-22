from typing import Optional

import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.data.road import Road

class TransformRoad(Dataset):
    def __init__(self, dataset: Road, transform: Optional[A.Compose] = None):
        self.dataset = dataset
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        transformed_image = self.transform(image=np.array(image))['image']

        return transformed_image, label
