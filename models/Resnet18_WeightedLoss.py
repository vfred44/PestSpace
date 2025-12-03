# Weighted loss model:

import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score, recall, precision, average_precision
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


class Resnet18_WeightedLoss(pl.LightningModule):
    def __init__(self, model, num_classes = 2, lr = 1e-4, class_counts=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

        # Replace final classification layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.val_preds = []
        self.val_targets = []

        # Weighted loss:
        class_counts = torch.tensor(class_counts)
        class_weights = 1.0 / class_counts #not normalized!
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        train_loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        train_acc = accuracy(preds, labels, task="multiclass", num_classes=2)
        train_precision = precision(preds, labels, task="multiclass", num_classes=2, average="macro")
        train_recall = recall(preds, labels, task="multiclass", num_classes=2, average="macro")
        train_f1score = f1_score(preds, labels, task="multiclass", num_classes=2, average="macro")
        
        self.log_dict({"train_loss": train_loss, 
                       "train_acc": train_acc, 
                       "train_precision": train_precision, 
                       "train_recall": train_recall, 
                       "train_f1score": train_f1score
                       }, on_epoch=True, on_step=False, logger=True)
                    
        return train_loss
    
    def validation_step (self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        val_acc = accuracy(preds, labels, task="multiclass", num_classes=2)
        val_precision = precision(preds, labels, task="multiclass", num_classes=2, average="macro")
        val_recall = recall(preds, labels, task="multiclass", num_classes=2, average="macro")
        val_f1score = f1_score(preds, labels, task="multiclass", num_classes=2, average="macro")

        self.val_preds.append(preds.cpu())
        self.val_targets.append(labels.cpu())
        
        self.log_dict({"val_loss": val_loss, 
                       "val_acc": val_acc, 
                       "val_precision": val_precision, 
                       "val_recall": val_recall, 
                       "val_f1score": val_f1score
                       }, on_epoch=True, on_step=False, logger=True)
    
        return val_loss
    
    def on_validation_epoch_end(self):
        y_pred = torch.cat(self.val_preds).numpy()
        y_true = torch.cat(self.val_targets).numpy()
        self.val_preds.clear()
        self.val_targets.clear()

        class_names = ["downey_mildew", "rust"]


        self.logger.experiment.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
        
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