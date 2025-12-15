import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score, recall, precision
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
from sklearn.metrics import balanced_accuracy_score
import glob
import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger
import math


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = probability of true class
        if isinstance(self.alpha, (float, int)):
            alpha = self.alpha
        else:
            # gather class-specific alpha if tensor
            alpha = self.alpha[targets]
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Resnet18(pl.LightningModule):
    def __init__(self, model, num_classes=2, lr=1e-4, use_focal_loss=False, use_weighted_loss=False, class_counts=None, alpha=1, gamma=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

        if use_focal_loss:
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        
        elif use_weighted_loss and class_counts is not None:
            class_counts = torch.tensor(class_counts, dtype=torch.float)
            class_weights = 1.0 / class_counts
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

 
        # Replace final classification layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)


        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
        self.val_images = []
        self.image_ids = []
        #self.val_preds_probs = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels, image_paths = batch
        outputs = self(inputs)
        train_loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        
        self.train_preds.append(preds.cpu())
        self.train_targets.append(labels.cpu())
        
        #train_bal_acc = balanced_accuracy(preds, labels, task="multiclass", num_classes=2)
        train_precision = precision(preds, labels, task="multiclass", num_classes=2, average="macro")
        train_recall = recall(preds, labels, task="multiclass", num_classes=2, average="macro")
        train_f1score = f1_score(preds, labels, task="multiclass", num_classes=2, average="macro")

        self.log_dict({"train_loss": train_loss, 
                       #"train_acc": train_bal_acc, 
                       "train_precision": train_precision, 
                       "train_recall": train_recall, 
                       "train_f1score": train_f1score
                       }, on_epoch=True, on_step=False, logger=True)
                    
        return train_loss

    def on_train_epoch_end(self):
        if len(self.train_preds) > 0:
            preds = torch.cat(self.train_preds).numpy()
            targets = torch.cat(self.train_targets).numpy()
            bal_acc = balanced_accuracy_score(targets, preds)

            self.log("train_balanced_accuracy", bal_acc, on_epoch=True, prog_bar=True)

        # clear buffers
        self.train_preds.clear()
        self.train_targets.clear()


    def validation_step (self, batch, batch_idx):
        inputs, labels, image_paths = batch
        outputs = self(inputs)
        val_loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
    
        #IDs:
        ids = [os.path.basename(os.path.dirname(p)) for p in image_paths]
      
        #probs = torch.softmax(outputs, dim=1)
        #pos_probs = probs[:, 1] #positive class probabilities
        val_precision = precision(preds, labels, task="multiclass", num_classes=2, average="macro")
        val_recall = recall(preds, labels, task="multiclass", num_classes=2, average="macro")
        val_f1score = f1_score(preds, labels, task="multiclass", num_classes=2, average="macro")
        #val_bal_acc = balanced_accuracy(preds, labels, task="multiclass", num_classes=2)

        self.val_preds.append(preds.cpu())
        self.val_targets.append(labels.cpu())
        self.val_images.append(inputs.cpu())
        self.image_ids.extend(ids)
        #self.val_preds_probs.append(pos_probs.cpu())
        
        self.log_dict({"val_loss": val_loss, 
                       #"val_bal_acc": val_bal_acc, 
                       "val_precision": val_precision, 
                       "val_recall": val_recall, 
                       "val_f1score": val_f1score
                       }, on_epoch=True, on_step=False, logger=True)
    
        return val_loss
    
        
    def on_validation_epoch_start(self):
        self.val_preds.clear()
        self.val_targets.clear()
        self.val_images.clear()
        self.image_ids.clear()
        #self.val_preds_probs.clear()


    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            preds = torch.cat(self.val_preds).numpy()
            targets = torch.cat(self.val_targets).numpy()
            bal_acc = balanced_accuracy_score(targets, preds)

            self.log("val_balanced_accuracy", bal_acc, on_epoch=True, prog_bar=True)


    def on_fit_end(self):
        y_pred = torch.cat(self.val_preds, dim=0).numpy()
        y_true = torch.cat(self.val_targets, dim=0).numpy()
        images = torch.cat(self.val_images, dim=0).permute(0, 2, 3, 1).numpy()
        #y_probs = torch.cat(self.val_preds_probs, dim=0).numpy()
        img_ids = np.array(self.image_ids)

        class_names = ["downey_mildew", "chocolate_spot"]
        
        # roc_auc = roc_auc_score(y_true, y_probs)

        # log FP and FN

        # False Positives:
        fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]

        # False Negatives:
        fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

        fp_images = images[fp_idx]
        fn_images = images[fn_idx]

        fp_ids = img_ids[fp_idx]
        fn_ids = img_ids[fn_idx]

        print(f"fp_ids {len(fp_ids)}")
        print(f"fp_ids {fp_ids}")
        print(f"fn_ids {len(fn_ids)}")
        print(f"fn_ids {fn_ids}")
        print(f"fp_pildid {len(fp_images)}")
        print(f"fn_pildid {len(fn_images)}")

        # Limit number of logged images
        max_images = 20
        fp_images = fp_images[:max_images]
        fn_images = fn_images[:max_images]

        # Unnormalise images
        fp_images = [img * 0.5 + 0.5 for img in fp_images]
        fn_images = [img * 0.5 + 0.5 for img in fn_images]

        # Log to wandb
        self.logger.experiment.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=y_true,
                        preds=y_pred,
                        class_names=class_names
                    ),
                    "False positives": [wandb.Image(img, caption=f"ID: {id}") for img, id in zip(fp_images, fp_ids)],
                    "False negatives": [wandb.Image(img, caption=f"ID: {id}") for img, id in zip(fn_images, fn_ids)]
 
        }, commit=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        test_loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        test_acc = (preds == labels).float().mean()
        self.log_dict({"test_loss": test_loss, "test_acc": test_acc}, logger=True)
        
        return test_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)