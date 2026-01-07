import os
import logging
import torch
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms.v2 as transforms
from pathlib import Path

def grayscale_to_rgb(i):
    return torch.cat([i, i, i], dim=0) if i.shape[0] == 1 else i


class CXP_dataset(torchvision.datasets.VisionDataset):

    def __init__(self, root_dir: str, 
                 csv_file: Path | str, 
                 augment: bool = True, 
                 return_weights: bool = False
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

        if return_weights:
            # Compute sample weights
            # We use effective number of samples balancing as per
            # https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html

            # Combine labels and drain into group identifiers
            groups = self.labels * 2 + self.drain  # Creates groups: 0, 1, 2, 3

            # Count samples per group
            counts = np.bincount(groups, minlength=4)

            # Effective number of samples
            beta = 0.99
            effective_num = (1 - beta ** counts) / (1 - beta)

            # Per-class weights (inverse of effective number)
            class_weights = 1.0 / effective_num

            # Per-sample weights
            self.weights = class_weights[groups]            
            
            # Normalize for easier comparison with BCE
            self.weights = self.weights / self.weights.sum() * len(self.weights)
            assert self.weights.sum() == len(self.weights)
            logging.info(f'Sample weights @ beta=0.99 are {np.unique(self.weights)}.')

            # Second version with higher beta = stronger reweighting, closer to naive class weighting
            beta = 0.9999
            effective_num = (1 - beta ** counts) / (1 - beta)
            class_weights = 1.0 / effective_num
            self.weights2 = class_weights[groups]   
            self.weights2 = self.weights2 / self.weights2.sum() * len(self.weights2) 
            assert self.weights2.sum() == len(self.weights2)
            logging.info(f'Sample weights @ beta=0.9999 are {np.unique(self.weights2)}.')

        self.return_weights = return_weights

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int] | tuple[torch.Tensor, int, int, float]:
        try:
            img = torchvision.io.read_image(os.path.join(self.root_dir, self.path[index]))
            img = self.transform(img)
            if self.return_weights:
                return img, self.labels[index], self.drain[index], self.weights[index], self.weights2[index]
            else:
                return img, self.labels[index], self.drain[index]
        except RuntimeError as e:
            logging.error(f"Error loading image at index {index}: {self.path[index]}")
            logging.error(f"Error message: {e}")
            # Return the next valid image
            return self.__getitem__((index + 1) % len(self))        
    
    def __len__(self) -> int:
        return len(self.path)