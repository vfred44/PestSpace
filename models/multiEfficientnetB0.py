import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
#from torchmetrics.functional import accuracy, f1_score, recall, precision
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler 
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import glob
import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger
import math
from omegaconf.listconfig import ListConfig


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, class_counts=None, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None and class_counts is not None:
            class_count = np.array(class_counts)
            inv_freq = 1/(class_count/class_count.sum())
            alpha = inv_freq/inv_freq.sum()

        # Convert alpha to torch tensor
        if isinstance(alpha, np.ndarray):
            alpha = torch.tensor(alpha, dtype=torch.float32)

        # If None, default to 1
        if alpha is None:
            alpha = 1

        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = probability of true class
        if isinstance(self.alpha, (float, int)):
            alpha = self.alpha
        else:
            # gather class-specific alpha if tensor
            alpha = self.alpha.to(targets.device)[targets]

        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class multiEfficientnetB0(pl.LightningModule):
    def __init__(self, model, classes_to_use=None, plants_to_use=None, lr=1e-4, weight_decay=1e-3, use_focal_loss=False, use_weighted_loss=False, class_counts=None, alpha=None, gamma=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.num_classes = len(classes_to_use)
        self.num_plants = len(plants_to_use)
        self.class_names = classes_to_use
        self.plant_names = plants_to_use

        if use_focal_loss:
            self.loss_fn = FocalLoss(class_counts=class_counts, gamma=gamma, alpha=alpha)
        
        elif use_weighted_loss and class_counts is not None:
            #print("class_counts:", class_counts)
            #print("type:", type(class_counts))
            class_counts = torch.tensor(class_counts, dtype=torch.float)
            class_weights = 1.0 / class_counts
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

 
        # Replace final classification layer
        num_features = self.model.classifier[1].in_features
        #self.model.classifier[1] = nn.Linear(num_features, self.num_classes)

        # Plant head
        self.plant_head = nn.Linear(num_features, self.num_plants)

        # Disease head
        self.disease_head = nn.Linear(num_features, self.num_classes)

        # Store
        self.train_disease_preds = []
        self.train_disease_targets = []
        self.train_plant_preds = []
        self.train_plant_targets = []

        self.val_disease_preds = []
        self.val_disease_targets = []
        self.val_plant_preds = []
        self.val_plant_targets = []

        self.val_images = []
        self.image_ids = []
        #self.val_preds_probs = []

    # def forward(self, x):
    #     return self.model(x)
    
    def forward(self, x):
        features = self.model.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)

        plant_logits = self.plant_head(features)
        disease_logits = self.disease_head(features)

        return plant_logits, disease_logits


    def training_step(self, batch, batch_idx):
        inputs, plant_labels, disease_labels, image_paths = batch
        #outputs = self(inputs)
        plant_logits, disease_logits = self(inputs)
        #train_loss = self.loss_fn(outputs, labels)
        #preds = torch.argmax(outputs, dim=1)
        train_plant_loss = self.loss_fn(plant_logits, plant_labels)
        train_disease_loss = self.loss_fn(disease_logits, disease_labels)
        
        train_loss = train_plant_loss + train_disease_loss

        #self.train_preds.append(preds.cpu())
        #self.train_targets.append(labels.cpu())

        # predictions
        plant_preds = torch.argmax(plant_logits, dim=1)
        disease_preds = torch.argmax(disease_logits, dim=1)

        # store
        self.train_disease_preds.append(disease_preds.detach().cpu())
        self.train_disease_targets.append(disease_labels.detach().cpu())
        self.train_plant_preds.append(plant_preds.detach().cpu())
        self.train_plant_targets.append(plant_labels.detach().cpu())

        self.log("train_loss", train_loss, on_epoch=True, on_step=False, logger=True)
        self.log("train_loss_plant", train_plant_loss, on_epoch=True, on_step=False, logger=True)
        self.log("train_loss_disease", train_disease_loss, on_epoch=True, on_step=False, logger=True)
                    
        return train_loss

    def on_train_epoch_end(self):
        if len(self.train_plant_preds) > 0:
            plant_preds = torch.cat(self.train_plant_preds).numpy()
            plant_targets = torch.cat(self.train_plant_targets).numpy()
            train_plant_bal_acc = balanced_accuracy_score(plant_targets, plant_preds)
            train_plant_precision = precision_score(plant_targets, plant_preds, average="macro", zero_division=0)
            train_plant_recall = recall_score(plant_targets, plant_preds, average="macro", zero_division=0)
            train_plant_f1 = f1_score(plant_targets, plant_preds, average="macro", zero_division=0)
       
            self.log_dict({
                "train_plant_bal_acc": train_plant_bal_acc,
                "train_plant_precision": train_plant_precision,
                "train_plant_recall": train_plant_recall,
                "train_plant_f1": train_plant_f1
            }, on_epoch=True, on_step=False, logger=True)


        if len(self.train_disease_preds) > 0:
            disease_preds = torch.cat(self.train_disease_preds).numpy()
            disease_targets = torch.cat(self.train_disease_targets).numpy()
            train_disease_bal_acc = balanced_accuracy_score(disease_targets, disease_preds)
            train_disease_precision = precision_score(disease_targets, disease_preds, average="macro", zero_division=0)
            train_disease_recall = recall_score(disease_targets, disease_preds, average="macro", zero_division=0)
            train_disease_f1 = f1_score(disease_targets, disease_preds, average="macro", zero_division=0)
            
            self.log_dict({
                "train_disease_bal_acc": train_disease_bal_acc,
                "train_disease_precision": train_disease_precision,
                "train_disease_recall": train_disease_recall,
                "train_disease_f1": train_disease_f1
            }, on_epoch=True, on_step=False, logger=True)


        # clear buffers
        self.train_plant_preds.clear()
        self.train_plant_targets.clear()
        self.train_disease_preds.clear()
        self.train_disease_targets.clear()


    def validation_step (self, batch, batch_idx):
        inputs, plant_labels, disease_labels, image_paths = batch
        #outputs = self(inputs)
        plant_logits, disease_logits = self(inputs)
        #val_loss = self.loss_fn(outputs, labels)
        #preds = torch.argmax(outputs, dim=1)
        val_plant_loss = self.loss_fn(plant_logits, plant_labels)
        val_disease_loss = self.loss_fn(disease_logits, disease_labels)
        
        val_loss = val_plant_loss + val_disease_loss

        #IDs:
        ids = [os.path.basename(os.path.dirname(p)) for p in image_paths]
      
        plant_preds = torch.argmax(plant_logits, dim=1)
        disease_preds = torch.argmax(disease_logits, dim=1)

        # store
        self.val_disease_preds.append(disease_preds.detach().cpu())
        self.val_disease_targets.append(disease_labels.detach().cpu())
        self.val_plant_preds.append(plant_preds.detach().cpu())
        self.val_plant_targets.append(plant_labels.detach().cpu())
        self.val_images.append(inputs.detach().cpu())

        self.image_ids.extend(ids)
        #self.val_preds_probs.append(pos_probs.cpu())
        
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, logger=True)
        self.log("val_loss_plant", val_plant_loss, on_epoch=True, on_step=False, logger=True)
        self.log("val_loss_disease", val_disease_loss, on_epoch=True, on_step=False, logger=True)
    
        return val_loss
    
    
    def on_validation_epoch_start(self):
        self.val_disease_preds.clear()
        self.val_disease_targets.clear()
        self.val_plant_preds.clear()
        self.val_plant_targets.clear()

        self.val_images.clear()
        self.image_ids.clear()
        #self.val_preds_probs.clear()


    def on_validation_epoch_end(self):
        if len(self.val_plant_preds) > 0:
            plant_preds = torch.cat(self.val_plant_preds).numpy()
            plant_targets = torch.cat(self.val_plant_targets).numpy()
            val_plant_bal_acc = balanced_accuracy_score(plant_targets, plant_preds)
            val_plant_precision = precision_score(plant_targets, plant_preds, average="macro", zero_division=0)
            val_plant_recall = recall_score(plant_targets, plant_preds, average="macro", zero_division=0)
            val_plant_f1 = f1_score(plant_targets, plant_preds, average="macro", zero_division=0)
            
            self.log_dict({
                "val_plant_bal_acc": val_plant_bal_acc,
                "val_plant_precision": val_plant_precision,
                "val_plant_recall": val_plant_recall,
                "val_plant_f1": val_plant_f1
            }, on_epoch=True, on_step=False, logger=True)

        if len(self.val_disease_preds) > 0:
            disease_preds = torch.cat(self.val_disease_preds).numpy()
            disease_targets = torch.cat(self.val_disease_targets).numpy()
            val_disease_bal_acc = balanced_accuracy_score(disease_targets, disease_preds)
            val_disease_precision = precision_score(disease_targets, disease_preds, average="macro", zero_division=0)
            val_disease_recall = recall_score(disease_targets, disease_preds, average="macro", zero_division=0)
            val_disease_f1 = f1_score(disease_targets, disease_preds, average="macro", zero_division=0)
            
            self.log_dict({
                "val_disease_bal_acc": val_disease_bal_acc,
                "val_disease_precision": val_disease_precision,
                "val_disease_recall": val_disease_recall,
                "val_disease_f1": val_disease_f1
            }, on_epoch=True, on_step=False, logger=True)


    def on_fit_end(self):
        # For disease
        disease_pred = torch.cat(self.val_disease_preds, dim=0).numpy()
        disease_targets = torch.cat(self.val_disease_targets, dim=0).numpy()

        #For plant
        plant_pred = torch.cat(self.val_plant_preds, dim=0).numpy()
        plant_targets = torch.cat(self.val_plant_targets, dim=0).numpy()

        print("Unique disease targets:", np.unique(disease_targets))
        print("Unique disease preds:", np.unique(disease_pred))

        images = torch.cat(self.val_images, dim=0).permute(0, 2, 3, 1).numpy()
        #y_probs = torch.cat(self.val_preds_probs, dim=0).numpy()
        img_ids = np.array(self.image_ids)

        print("img_ids:")
        print(img_ids)
        
        # log FP and FN

        max_images = 20

        def log_head(preds, targets, class_names, head_name):
            for c, class_name in enumerate(class_names):
                fp_idx = np.where((preds == c) & (targets != c))[0]
                fn_idx = np.where((targets == c) & (preds != c))[0]
                print("class_names:")
                print(class_name)
                print(targets)

                fp_images = images[fp_idx][:max_images]
                fn_images = images[fn_idx][:max_images]

                fp_ids = img_ids[fp_idx][:max_images]
                fn_ids = img_ids[fn_idx][:max_images]

                # unnormalize
                fp_images = [
                    np.clip(img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]),0.0,1.0)
                    for img in fp_images
                ]
                fn_images = [
                    np.clip(img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]),0.0,1.0)
                    for img in fn_images
                ]

                self.logger.experiment.log({
                    f"{head_name}_confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=targets,
                        preds=preds,
                        class_names=class_names
                    ),
                    f"{head_name}_False_positives_{class_name}": [
                        wandb.Image(img, caption=f"ID: {id}") for img,id in zip(fp_images, fp_ids)
                    ],
                    f"{head_name}_False_negatives_{class_name}": [
                        wandb.Image(img, caption=f"ID: {id}") for img,id in zip(fn_images, fn_ids)
                    ]
                }, commit=False)


        log_head(disease_pred, disease_targets, self.class_names, head_name="disease")
        log_head(plant_pred, plant_targets, self.plant_names, head_name="plant")

        

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        test_loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        test_acc = (preds == labels).float().mean()
        self.log_dict({"test_loss": test_loss, "test_acc": test_acc}, logger=True)
        
        return test_loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        #return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
