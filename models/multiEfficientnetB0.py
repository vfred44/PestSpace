import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
#from torchmetrics.functional import accuracy, f1_score, recall, precision
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import wandb

Image.MAX_IMAGE_PIXELS = None

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
        
        self.best_val_loss = float("inf")
        self.best_outputs = {
            "disease_preds": None,
            "disease_targets": None,
            "plant_preds": None,
            "plant_targets": None,
            "images": None,
            "img_ids": None
        }

        if use_focal_loss:
            self.loss_fn = FocalLoss(class_counts=class_counts, gamma=gamma, alpha=alpha)
        
        elif use_weighted_loss and class_counts is not None:
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

        #self.val_images = []
        self.image_ids = []
        self.image_paths_all = []
        self.train_image_paths = []
      
    # def forward(self, x):
    #     return self.model(x)
    
    def forward(self, x):
        features = self.model.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)

        plant_logits = self.plant_head(features)
        disease_logits = self.disease_head(features)

        return plant_logits, disease_logits

    # #For finetuning
    #
    # def on_train_epoch_start(self):
    #     if self.current_epoch == 5:
    #         print("Unfreezing backbone...")
    #         self.model.features.requires_grad_(True)

    #For grouped metrics

    def compute_grouped_metrics(self, preds, targets, paths, prefix=""):
        paths = np.array(paths)

        folder_names = np.array([
            os.path.basename(os.path.dirname(os.path.dirname(p)))
            for p in paths
        ])

        groups = {
            "PS_Fababean": np.array([
                fn.startswith("PS_Fababean") for fn in folder_names
            ]),
            "PS_Wheat": np.array([
                fn.startswith("PS_Wheat") for fn in folder_names
            ])
        }

        metrics = {}

        for group_name, mask in groups.items():
            if mask.sum() == 0:
                continue

            dp = preds[mask]
            dt = targets[mask]

            metrics.update({
                f"{prefix}_precision_{group_name}": precision_score(dt, dp, average="macro", zero_division=0),
                f"{prefix}_recall_{group_name}": recall_score(dt, dp, average="macro", zero_division=0),
                f"{prefix}_f1_{group_name}": f1_score(dt, dp, average="macro", zero_division=0),
            })

        return metrics
    
    def compute_grouped_metrics(self, preds, targets, paths, prefix=""):
        paths = np.array(paths)

        folder_names = np.array([
            os.path.basename(os.path.dirname(os.path.dirname(p)))
            for p in paths
        ])

        # 🔑 DIFFERENT GROUPS depending on phase
        if prefix.startswith("train"):
            groups = {
                "Fababean": np.array([fn.startswith("Fababean") for fn in folder_names]),
                "Wheat": np.array([fn.startswith("Wheat") for fn in folder_names]),
            }       
        else:  # validation
            groups = {
                "PS_Fababean": np.array([fn.startswith("PS_Fababean") for fn in folder_names]),
                "PS_Wheat": np.array([fn.startswith("PS_Wheat") for fn in folder_names]),
            }

        metrics = {}

        for group_name, mask in groups.items():
            if mask.sum() == 0:
                continue

            dp = preds[mask]
            dt = targets[mask]

            metrics.update({
                f"{prefix}_precision_{group_name}": precision_score(dt, dp, average="macro", zero_division=0),
                f"{prefix}_recall_{group_name}": recall_score(dt, dp, average="macro", zero_division=0),
                f"{prefix}_f1_{group_name}": f1_score(dt, dp, average="macro", zero_division=0),
            })

        return metrics



    def training_step(self, batch, batch_idx):
        inputs, plant_labels, disease_labels, image_paths = batch
        #outputs = self(inputs)
        plant_logits, disease_logits = self(inputs)
        #train_loss = self.loss_fn(outputs, labels)
        #preds = torch.argmax(outputs, dim=1)
        train_plant_loss = self.loss_fn(plant_logits, plant_labels)
        train_disease_loss = self.loss_fn(disease_logits, disease_labels)
        
        train_loss = train_plant_loss + train_disease_loss

        # predictions
        plant_preds = torch.argmax(plant_logits, dim=1)
        disease_preds = torch.argmax(disease_logits, dim=1)

        # store
        self.train_disease_preds.append(disease_preds.detach().cpu())
        self.train_disease_targets.append(disease_labels.detach().cpu())
        self.train_plant_preds.append(plant_preds.detach().cpu())
        self.train_plant_targets.append(plant_labels.detach().cpu())
        self.train_image_paths.extend(image_paths)

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
            # train_disease_bal_acc = balanced_accuracy_score(disease_targets, disease_preds)
            # train_disease_precision = precision_score(disease_targets, disease_preds, average="macro", zero_division=0)
            # train_disease_recall = recall_score(disease_targets, disease_preds, average="macro", zero_division=0)
            # train_disease_f1 = f1_score(disease_targets, disease_preds, average="macro", zero_division=0)
            
            # self.log_dict({
            #     "train_disease_bal_acc": train_disease_bal_acc,
            #     "train_disease_precision": train_disease_precision,
            #     "train_disease_recall": train_disease_recall,
            #     "train_disease_f1": train_disease_f1
            # }, on_epoch=True, on_step=False, logger=True)

            grouped_metrics = self.compute_grouped_metrics(
                disease_preds,
                disease_targets,
                self.train_image_paths,
                prefix="train_disease"
            )

            self.log_dict(grouped_metrics, on_epoch=True, on_step=False, logger=True)


        # clear buffers
        self.train_plant_preds.clear()
        self.train_plant_targets.clear()
        self.train_disease_preds.clear()
        self.train_disease_targets.clear()
        self.train_image_paths.clear()


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
        #self.val_images.append(inputs.detach().cpu())

        self.image_ids.extend(ids)
        self.image_paths_all.extend(image_paths)
        
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, logger=True)
        self.log("val_loss_plant", val_plant_loss, on_epoch=True, on_step=False, logger=True)
        self.log("val_loss_disease", val_disease_loss, on_epoch=True, on_step=False, logger=True)
    
        return val_loss
    
    
    def on_validation_epoch_start(self):
        self.val_disease_preds.clear()
        self.val_disease_targets.clear()
        self.val_plant_preds.clear()
        self.val_plant_targets.clear()

        #self.val_images.clear()
        self.image_ids.clear()
        self.image_paths_all.clear()
      

    def on_validation_epoch_end(self):

        print("\nValidation image paths:")
        for p in self.image_paths_all:
            print(p)

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
            # val_disease_bal_acc = balanced_accuracy_score(disease_targets, disease_preds)
            # val_disease_precision = precision_score(disease_targets, disease_preds, average="macro", zero_division=0)
            # val_disease_recall = recall_score(disease_targets, disease_preds, average="macro", zero_division=0)
            # val_disease_f1 = f1_score(disease_targets, disease_preds, average="macro", zero_division=0)
            
            # self.log_dict({
            #     "val_disease_bal_acc": val_disease_bal_acc,
            #     "val_disease_precision": val_disease_precision,
            #     "val_disease_recall": val_disease_recall,
            #     "val_disease_f1": val_disease_f1
            # }, on_epoch=True, on_step=False, logger=True)

            grouped_metrics = self.compute_grouped_metrics(
                disease_preds,
                disease_targets,
                self.image_paths_all,
                prefix="val_disease"
)

            self.log_dict(grouped_metrics, on_epoch=True, on_step=False, logger=True)

        #Save best epoch outputs:
        current_val_loss = self.trainer.callback_metrics["val_loss"]

        # tensor to float
        if isinstance(current_val_loss, torch.Tensor):
            current_val_loss = current_val_loss.item()

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss

            self.best_outputs["disease_preds"] = torch.cat(self.val_disease_preds, dim=0).cpu()
            self.best_outputs["disease_targets"] = torch.cat(self.val_disease_targets, dim=0).cpu()

            self.best_outputs["plant_preds"] = torch.cat(self.val_plant_preds, dim=0).cpu()
            self.best_outputs["plant_targets"] = torch.cat(self.val_plant_targets, dim=0).cpu()

            #self.best_outputs["images"] = torch.cat(self.val_images, dim=0).cpu()
            #self.best_outputs["img_ids"] = np.array(self.image_ids)

            self.best_outputs["image_paths"] = list(self.image_paths_all)
            self.best_outputs["img_ids"] = np.array(self.image_ids)


    def on_fit_end(self):
        # For disease
        disease_pred = self.best_outputs["disease_preds"].numpy()
        disease_targets = self.best_outputs["disease_targets"].numpy()

        #For plant
        plant_pred = self.best_outputs["plant_preds"].numpy()
        plant_targets = self.best_outputs["plant_targets"].numpy()

        print("Unique disease targets:", np.unique(disease_targets))
        print("Unique disease preds:", np.unique(disease_pred))

        #images = self.best_outputs["images"].permute(0, 2, 3, 1).numpy()
        def load_image(path):
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224))
            return np.array(img).astype(np.uint8)

       
        img_ids = self.best_outputs["img_ids"]

        print("img_ids:")
        print(img_ids)

        print(f"Best val_loss: {self.best_val_loss}")
        
        # log FP and FN

        max_images = 20

        def log_head(preds, targets, class_names, head_name):
            for c, class_name in enumerate(class_names):
                fp_idx = np.where((preds == c) & (targets != c))[0]
                fn_idx = np.where((targets == c) & (preds != c))[0]
              
                fp_paths = np.array(self.best_outputs["image_paths"])[fp_idx][:max_images]
                fp_ids = img_ids[fp_idx][:max_images]

                fp_images = [load_image(p) for p in fp_paths]

                fn_paths = np.array(self.best_outputs["image_paths"])[fn_idx][:max_images]
                fn_ids = img_ids[fn_idx][:max_images]

                fn_images = [load_image(p) for p in fn_paths]

                print("False positive paths:")
                print(fp_paths)

                print("False negative paths:")
                print(fn_paths)

                #fp_images = images[fp_idx][:max_images]
                #fn_images = images[fn_idx][:max_images]

                #fp_ids = img_ids[fp_idx][:max_images]
                #fn_ids = img_ids[fn_idx][:max_images]

                # # unnormalize
                # fp_images = [
                #     np.clip(img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]),0.0,1.0)
                #     for img in fp_images
                # ]
                # fn_images = [
                #     np.clip(img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]),0.0,1.0)
                #     for img in fn_images
                # ]

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
    
    # def configure_optimizers(self):
    #     return torch.optim.AdamW(
    #          self.parameters(),
    #          lr=self.hparams.lr,
    #          weight_decay=self.hparams.weight_decay
    # )
    

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
    )