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

class MyImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label
    

# Functions:

def get_images_for(image_paths, objects):
    return [p for p in image_paths if os.path.basename(os.path.dirname(p)) in objects]


def prepare_datasets(cfg):
    
    image_paths = []
    labels = []
    count_class0 = 0
    count_class1 = 0

    # Assign 0 for downey mildew, 1 for rust:
    for img_path in glob.glob(os.path.join(cfg.data.train_base, "Downy_mildew/**/*.jpg"), recursive=True):
        image_paths.append(img_path)
        labels.append(0)
        count_class0 +=1

    for img_path in glob.glob(os.path.join(cfg.data.train_base, "Chocolate_spot/**/*.jpg"), recursive=True):
        image_paths.append(img_path)
        labels.append(1)
        count_class1 +=1

    class_counts = [count_class0, count_class1]
    
    # Extract object IDs:

    object_ids = [os.path.basename(os.path.dirname(p)) for p in image_paths]

    # Each object with its label:

    object_to_label = {}
    for obj, label in zip(object_ids, labels):
        object_to_label[obj] = label

    unique_objects = np.array(list(object_to_label.keys()))
    unique_labels = np.array(list(object_to_label.values()))


    # Split objects by label (stratified):

    obj_trainval, obj_test, y_trainval, y_test = train_test_split(
        unique_objects,
        unique_labels,
        test_size=0.10,
        stratify=unique_labels,
        random_state=42
    )

    obj_train, obj_val, y_train, y_val = train_test_split(
        obj_trainval,
        y_trainval,
        test_size=0.15 / 0.90,
        stratify=y_trainval,
        random_state=42
    )

    # Assign images based on object IDs:

    X_train = get_images_for(image_paths, obj_train)
    X_val = get_images_for(image_paths, obj_val)
    X_test = get_images_for(image_paths, obj_test)

    y_train = [labels[image_paths.index(p)] for p in X_train]
    y_val = [labels[image_paths.index(p)] for p in X_val]
    y_test = [labels[image_paths.index(p)] for p in X_test]

    # Transform data:

    train_transform = transforms.Compose([
        #transforms.Resize((1024, 1024)),
        transforms.Resize((512, 512)),
        #transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_test_transform = transforms.Compose([
        #transforms.Resize((1024, 1024)),
        transforms.Resize((512, 512)),
        #transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MyImageDataset(X_train, y_train, transform=train_transform)
    val_dataset   = MyImageDataset(X_val, y_val, transform=val_test_transform)
    test_dataset  = MyImageDataset(X_test, y_test, transform=val_test_transform)

    return train_dataset, val_dataset, test_dataset, class_counts


def get_data_loaders(cfg):
   
    train_dataset, val_dataset, test_dataset, class_counts = prepare_datasets(cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    return train_loader, val_loader, test_loader, class_counts