import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score, recall, precision, average_precision, auroc
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler 
from torchvision import datasets, transforms
from torchvision.models import resnet18
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger
import time

# Data balancing methods:

# Weightedrandomsampler

def Weightedrandomsampler(y_train, class_counts):
    # Compute weights for each class:
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

    # Assign weight to each sample:
    sample_weights = torch.tensor([class_weights[label].item() for label in y_train],
                    dtype=torch.float)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


class MyImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label, image_path
    

# Functions:

def prepare_datasets(cfg):
    
    image_paths = []
    labels = []

    class_counts = {}

    for class_name in cfg.data.classes_to_use:
        class_dir = os.path.join(cfg.data.train_base, class_name)
        class_label = cfg.data.class_to_label[class_name]

        # Count images to report statistics later
        class_counts[class_label] = 0

        # Load all .jpg files recursively
        pattern = os.path.join(class_dir, "**/*.jpg")

        for img_path in glob.glob(pattern, recursive=True):
            image_paths.append(img_path)
            labels.append(class_label)
            class_counts[class_label] += 1

    
    # Extract object IDs:

    object_ids = [os.path.basename(os.path.dirname(p)) for p in image_paths]

    # Each object with its label:

    object_to_label = {}

    for obj, label in zip(object_ids, labels):
        object_to_label[obj] = label

    unique_objects = np.array(list(object_to_label.keys()))
    unique_labels = np.array(list(object_to_label.values()))


    # Split objects by label (stratified):

    obj_train, obj_val, y_train, y_val = train_test_split(
        unique_objects,
        unique_labels,
        test_size=0.20,
        stratify=unique_labels,
        random_state=42
    )

    # Add test data:

    # obj_train, obj_val, y_train, y_val = train_test_split(
    #     obj_trainval,
    #     y_trainval,
    #     test_size=0.15 / 0.90,
    #     stratify=y_trainval,
    #     random_state=42
    # )

    # Assign images based on object IDs:

    X_train = [p for p in image_paths if os.path.basename(os.path.dirname(p)) in obj_train]
    X_val = [p for p in image_paths if os.path.basename(os.path.dirname(p)) in obj_val]
    #X_test = [p for p in image_paths if os.path.basename(os.path.dirname(p)) in obj_test]

    y_train = [labels[image_paths.index(p)] for p in X_train]
    y_val = [labels[image_paths.index(p)] for p in X_val]
    #y_test = [labels[image_paths.index(p)] for p in X_test]

    # Transform data:

    transform = transforms.Compose([
        #transforms.Resize((1024, 1024)),
        #transforms.Resize((256, 256)),
        transforms.Resize((512, 512)),
        #transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    train_dataset = MyImageDataset(X_train, y_train, transform=transform)
    val_dataset   = MyImageDataset(X_val, y_val, transform=transform)
    #test_dataset  = MyImageDataset(X_test, y_test, transform=transform)

    
    class_counts = [int(class_counts[i]) for i in sorted(class_counts.keys())]
  
    return train_dataset, val_dataset, class_counts, image_paths, y_train



def get_data_loaders(cfg):
   
    train_dataset, val_dataset, class_counts, image_paths, y_train = prepare_datasets(cfg)

    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.data.batch_size, 
                              sampler=Weightedrandomsampler(y_train, class_counts) if cfg.data.use_weightedrandomsampler else None, 
                              shuffle=False if cfg.data.use_weightedrandomsampler else True, 
                              num_workers=cfg.data.num_workers)
    val_loader   = DataLoader(val_dataset, 
                              batch_size=cfg.data.batch_size, 
                              shuffle=False, 
                              num_workers=cfg.data.num_workers)
    #test_loader  = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
   
    return train_loader, val_loader, class_counts, image_paths