import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score, recall, precision, average_precision, auroc
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler 
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger
import time
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from data.data import get_data_loaders

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    print("Config:", cfg)

    # Wandb
    wandb.login(key="c6296443d688c57d80b06f95f26c00000ff94a35")

    wandb_logger = WandbLogger(
    project=cfg.wandb.project,
    entity=cfg.wandb.entity,
    name=cfg.wandb.run_name,
    config={
        "batch_size": cfg.data.batch_size,
        #"learning_rate": cfg.model.lr,
        "optimizer": "Adam",
        "epochs": cfg.trainer.max_epochs
        },
        log_model=True,
        offline=False,
        reinit=True
    )

    # Data
    train_dl, val_dl, class_counts, image_paths = get_data_loaders(cfg)

    #Save best checkpoint:
    checkpoint_cb = ModelCheckpoint(
                        monitor="val_loss",
                        mode="min",
                        save_top_k=1,
                        save_last=True,
                        filename="{epoch}-{val_loss:.4f}"
                    )
    
    #Stop training when the validation loss stops improving:
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,  # stop after nr of epochs with no improvement
        mode='min'
    )

    print(f"class_counts{class_counts}")

    # Model
    model = instantiate(cfg.model,
                        class_counts=class_counts,
                        classes_to_use=cfg.data.classes_to_use
                        )
    
    # Trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=wandb_logger,
        #callbacks=checkpoint_cb
        callbacks=[checkpoint_cb, early_stop_callback]
    )

    # Train with validation every epoch
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Test
    #trainer.test(model, dataloaders=test_dl)

    wandb.finish()

if __name__ == "__main__":
    main()


