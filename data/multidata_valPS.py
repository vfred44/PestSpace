import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler 
import numpy as np
from sklearn.model_selection import train_test_split
import glob
from hydra.utils import instantiate


# Data balancing methods:

# Weightedrandomsampler

def Weightedrandomsampler(y_train, class_counts, cfg):

    # Compute weights for each class:
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

    # Convert class names to disease ids
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
        plant_name = class_name.replace("PS_", "").split("_")[0]
        plant_label = self.cfg.data.plants_to_label[plant_name]

        if self.transform:
            image = self.transform(image)

        return image, plant_label, disease_label, image_path
    

# Functions:

def prepare_datasets(cfg):
    
    image_paths = []
    labels = []
    class_counts = {}

    for class_name in cfg.data.diseases_to_use:
        class_dir = os.path.join(cfg.data.train_base, class_name)
        #class_label = cfg.data.class_to_label[class_name]

        #class_counts[class_label] = 0

        # Load all .jpg files recursively
        pattern = os.path.join(class_dir, "**/*.[jJ][pP][gG]")
      
        for img_path in glob.glob(pattern, recursive=True):
            # print("image_path")
            # print(img_path)
            image_paths.append(img_path)
            labels.append(class_name)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1


    
    # Extract object IDs:

    object_ids = [os.path.basename(os.path.dirname(p)) for p in image_paths]

    object_to_label = {}
    object_to_paths = {}

    for p, obj, label in zip(image_paths, object_ids, labels):
        object_to_label[obj] = label
        object_to_paths.setdefault(obj, []).append(p)

    # Split objects into PS and non-PS
    ps_objects = []
    ps_labels = []

    non_ps_objects = []

    for obj, label in object_to_label.items():
        plant_name = label.split("_")[0]

        if plant_name.startswith("PS"):
            ps_objects.append(obj)
            ps_labels.append(label)
        else:
            non_ps_objects.append(obj)

    ps_objects = np.array(ps_objects)
    ps_labels = np.array(ps_labels)

    # Split only PS objects
    ps_train_obj, ps_val_obj, _, _ = train_test_split(
        ps_objects,
        ps_labels,
        test_size=0.20,
        stratify=ps_labels,
        random_state=42
    )

    # Final object sets
    train_objects = set(non_ps_objects) | set(ps_train_obj)
    val_objects = set(ps_val_obj)

    # Assign images based on object membership
    X_train = [p for obj in train_objects for p in object_to_paths[obj]]
    X_val   = [p for obj in val_objects   for p in object_to_paths[obj]]

    # Labels
    y_train = [object_to_label[os.path.basename(os.path.dirname(p))] for p in X_train]
    y_val   = [object_to_label[os.path.basename(os.path.dirname(p))] for p in X_val]

    print("Train objects:", len(train_objects))
    print("Val objects:", len(val_objects))

    print("Example val paths:")
    print(X_val)


    y_train = [labels[image_paths.index(p)] for p in X_train]
    y_val = [labels[image_paths.index(p)] for p in X_val]
    #y_test = [labels[image_paths.index(p)] for p in X_test]

    # Transform data:

    # transform = transforms.Compose([
    #     #transforms.Resize((1024, 1024)),
    #     #transforms.Resize((256, 256)),
    #     #transforms.Resize((512, 512)),
    #     transforms.Resize(512),
    #     transforms.CenterCrop(512),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406),
    #                          (0.229, 0.224, 0.225))
    # ])

    train_transform = instantiate(cfg.data.transforms.train)

    val_transform = instantiate(cfg.data.transforms.val)
  
    
    train_dataset = MyImageDataset(X_train, y_train, cfg=cfg, transform=train_transform)
    val_dataset   = MyImageDataset(X_val, y_val, cfg=cfg, transform=val_transform)
    #test_dataset  = MyImageDataset(X_test, y_test, transform=transform)


    #class_counts = [int(class_counts[i]) for i in sorted(class_counts.keys())]
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
    #test_loader  = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
   
    return train_loader, val_loader, class_counts, image_paths