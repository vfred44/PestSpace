import torch
import os
import torch.nn as nn
import torch.nn.functional as F
#from torchmetrics.functional import accuracy, f1_score, recall, precision
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import wandb


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


class EfficientnetB0(pl.LightningModule):
    def __init__(self, model, classes_to_use=None, lr=1e-4, weight_decay=1e-3, use_focal_loss=False, use_weighted_loss=False, class_counts=None, alpha=None, gamma=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.class_names = classes_to_use
        self.num_classes = len(classes_to_use)
    

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
        self.model.classifier[1] = nn.Linear(num_features, self.num_classes)


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
        
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, logger=True)
                    
        return train_loss

    def on_train_epoch_end(self):
        if len(self.train_preds) > 0:
            preds = torch.cat(self.train_preds).numpy()
            targets = torch.cat(self.train_targets).numpy()

            train_bal_acc = balanced_accuracy_score(targets, preds)
            train_precision = precision_score(targets, preds, average="macro", zero_division=0)
            train_recall = recall_score(targets, preds, average="macro", zero_division=0)
            train_f1score = f1_score(targets, preds, average="macro", zero_division=0)
            
            #Torchmetrics:
            #train_precision = precision(targets, preds, task="multiclass", num_classes=self.num_classes, average="macro")
            #train_recall = recall(targets, preds, task="multiclass", num_classes=self.num_classes, average="macro")
            #train_f1score = f1_score(targets, preds, task="multiclass", num_classes=self.num_classes, average="macro")

            self.log_dict({"train_bal_accuracy": train_bal_acc,
                            "train_precision": train_precision, 
                            "train_recall": train_recall, 
                            "train_f1score": train_f1score
                        }, on_epoch=True, on_step=False, logger=True)

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

        self.val_preds.append(preds.cpu())
        self.val_targets.append(labels.cpu())
        self.val_images.append(inputs.cpu())
        self.image_ids.extend(ids)
        #self.val_preds_probs.append(pos_probs.cpu())
        
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, logger=True)
    
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

            val_bal_acc = balanced_accuracy_score(targets, preds)
            val_precision = precision_score(targets, preds, average="macro", zero_division=0)
            val_recall = recall_score(targets, preds, average="macro", zero_division=0)
            val_f1score = f1_score(targets, preds, average="macro", zero_division=0)

            #Torchmetrics
            #val_precision = precision(preds, labels, task="multiclass", num_classes=self.num_classes, average="macro")
            #val_recall = recall(preds, labels, task="multiclass", num_classes=self.num_classes, average="macro")
            #val_f1score = f1_score(preds, labels, task="multiclass", num_classes=self.num_classes, average="macro")

            self.log_dict({"val_bal_accuracy": val_bal_acc,
                            "val_precision": val_precision, 
                            "val_recall": val_recall, 
                            "val_f1score": val_f1score
                        }, on_epoch=True, on_step=False, logger=True)


    def on_fit_end(self):
        y_pred = torch.cat(self.val_preds, dim=0).numpy()
        y_true = torch.cat(self.val_targets, dim=0).numpy()
        images = torch.cat(self.val_images, dim=0).permute(0, 2, 3, 1).numpy()
        #y_probs = torch.cat(self.val_preds_probs, dim=0).numpy()
        img_ids = np.array(self.image_ids)

        # log FP and FN

        max_images = 20

        for c, class_name in enumerate(self.class_names):
            # False Positives for class c
            fp_idx = np.where((y_pred == c) & (y_true != c))[0]

            # False Negatives for class c
            fn_idx = np.where((y_true == c) & (y_pred != c))[0]

            fp_images = images[fp_idx]
            fn_images = images[fn_idx]

            fp_ids = img_ids[fp_idx]
            fn_ids = img_ids[fn_idx]

            print(f"fp_ids {len(fp_ids)}")
            print(f"fp_ids {fp_ids}")
            print(f"fn_ids {len(fn_ids)}")
            print(f"fn_ids {fn_ids}")
            print(f"fp_images {len(fp_images)}")
            print(f"fn_images {len(fn_images)}")

            fp_images = fp_images[:max_images]
            fn_images = fn_images[:max_images]


            # Unnormalize
            fp_images = [
                np.clip(
                    img * np.array([0.229, 0.224, 0.225]) +
                    np.array([0.485, 0.456, 0.406]),
                    0.0, 1.0
                )
                for img in fp_images
            ]

            fn_images = [
                np.clip(
                    img * np.array([0.229, 0.224, 0.225]) +
                    np.array([0.485, 0.456, 0.406]),
                    0.0, 1.0
                )
                for img in fn_images
            ]

            self.logger.experiment.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                                y_true=y_true,
                                preds=y_pred,
                                class_names=self.class_names
                            ),
                f"False positives for {class_name}": [
                    wandb.Image(img, caption=f"ID: {id}")
                    for img, id in zip(fp_images, fp_ids)
                ],
                f"False negatives for {class_name}": [
                    wandb.Image(img, caption=f"ID: {id}")
                    for img, id in zip(fn_images, fn_ids)
                ],
            }, commit=False)

        

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
