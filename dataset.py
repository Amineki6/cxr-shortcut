import os
import logging
import torch
import pandas as pd
import torchvision
import torchvision.transforms.v2 as transforms
from pathlib import Path

def grayscale_to_rgb(i):
    return torch.cat([i, i, i], dim=0) if i.shape[0] == 1 else i


class CXP_dataset(torchvision.datasets.VisionDataset):

    def __init__(self, root_dir: str, 
                 csv_file: Path | str, 
                 augment: bool = True, 
                 ) -> None:

        if augment:
            transform = transforms.Compose([
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Lambda(grayscale_to_rgb),
                transforms.Normalize(  # params for pretrained resnet, see https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html#torchvision.models.DenseNet121_Weights
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=20),
                #transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(0.75, 1.3))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Lambda(grayscale_to_rgb),
                transforms.Normalize(  # params for pretrained resnet, see https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html#torchvision.models.DenseNet121_Weights
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        super().__init__(root_dir, transform)

        df = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.path = df.Path.str.replace('CheXpert-v1.0/', 'CheXpert-v1.0-small/', regex=False)
        self.idx = df.index
        self.transform = transform

        self.labels = df.Pneumothorax.astype(int)
        self.drain = df.Drain.astype(int)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        try:
            img = torchvision.io.read_image(os.path.join(self.root_dir, self.path[index]))
            img = self.transform(img)
            return img, self.labels[index], self.drain[index]
        except RuntimeError as e:
            logging.error(f"Error loading image at index {index}: {self.path[index]}")
            logging.error(f"Error message: {e}")
            # Return the next valid image
            return self.__getitem__((index + 1) % len(self))        
    
    def __len__(self) -> int:
        return len(self.path)