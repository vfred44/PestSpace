import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler 
import numpy as np
from sklearn.model_selection import train_test_split
import glob
from hydra.utils import instantiate
from collections import Counter


# Data balancing methods:

# Weightedrandomsampler

def Weightedrandomsampler(y_train, class_counts, cfg):

    # Compute weights for each class:
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

    # Convert class names to disease ids:
    y_train_ids = [cfg.data.diseases_to_label[c] for c in y_train]

    # Assign weight to each sample:
    sample_weights = torch.tensor([class_weights[label].item() for label in y_train_ids],
                    dtype=torch.float)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


class MyImageDataset(Dataset):
    def __init__(self, file_paths, labels, cfg, transform = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(image_path).convert('RGB')
        disease_label = self.labels[idx]
        class_name = self.labels[idx]
        
        # Disease label
        disease_label = self.cfg.data.diseases_to_label[class_name]
       
        # Plant label:
        plant_name = class_name.split("_")[1]
        plant_label = self.cfg.data.plants_to_label[plant_name]

        if self.transform:
            image = self.transform(image)

        return image, plant_label, disease_label, image_path
    

def prepare_datasets(cfg):
    
    image_paths = []
    labels = []
    class_counts = {}

    for class_name in cfg.data.diseases_to_use:
        class_dir = os.path.join(cfg.data.train_base, class_name)
  
        # Load all .jpg files recursively
        pattern = os.path.join(class_dir, "**/*.[jJ][pP][gG]")
      
        for img_path in glob.glob(pattern, recursive=True):
            # print(img_path)
            image_paths.append(img_path)
            labels.append(class_name)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    
    # Extract object IDs:
    object_ids = [os.path.basename(os.path.dirname(p)) for p in image_paths]

    # Each object with its label:
    object_to_label = {}

    for obj, label in zip(object_ids, labels):
        object_to_label[obj] = label

    unique_objects = np.array(list(object_to_label.keys()))
    unique_labels = np.array(list(object_to_label.values()))

    print("Class counts:")
    print(class_counts)

    # Split objects by label (stratified):
    obj_train, obj_val, y_train, y_val = train_test_split(
        unique_objects,
        unique_labels,
        test_size=0.20,
        stratify=unique_labels,
        random_state=42
    )

    # Assign images based on object IDs:
    X_train = [p for p in image_paths if os.path.basename(os.path.dirname(p)) in obj_train]
    X_val = [p for p in image_paths if os.path.basename(os.path.dirname(p)) in obj_val]
   
    y_train = [labels[image_paths.index(p)] for p in X_train]
    y_val = [labels[image_paths.index(p)] for p in X_val]

    train_transform = instantiate(cfg.data.transforms.train)

    val_transform = instantiate(cfg.data.transforms.val)
  
    
    train_dataset = MyImageDataset(X_train, y_train, cfg=cfg, transform=train_transform)
    val_dataset   = MyImageDataset(X_val, y_val, cfg=cfg, transform=val_transform)
  
    class_counts = [class_counts[c] for c in cfg.data.diseases_to_use]

    return train_dataset, val_dataset, class_counts, image_paths, y_train



def get_data_loaders(cfg):
   
    train_dataset, val_dataset, class_counts, image_paths, y_train = prepare_datasets(cfg)

    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.data.batch_size, 
                              sampler=Weightedrandomsampler(y_train, class_counts, cfg) if cfg.data.use_weightedrandomsampler else None, 
                              shuffle=False if cfg.data.use_weightedrandomsampler else True, 
                              num_workers=cfg.data.num_workers)
    val_loader   = DataLoader(val_dataset, 
                              batch_size=cfg.data.batch_size, 
                              shuffle=False, 
                              num_workers=cfg.data.num_workers)
   
    return train_loader, val_loader, class_counts, image_paths
