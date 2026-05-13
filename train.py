import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from data.multidata import get_data_loaders

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
        "optimizer": "AdamW",
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

    #Model
    model = instantiate(cfg.model,
                        class_counts=class_counts,
                        classes_to_use=cfg.data.diseases_to_use,
                        plants_to_use=cfg.data.plants_to_use
                        )
    
    # For finetuning:
    # checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    # state_dict = checkpoint["state_dict"]

    # #remove classifier weights
    # state_dict = {
    #     k: v for k, v in state_dict.items()
    #     if not k.startswith("model.classifier")
    # }

    # model.load_state_dict(state_dict, strict=False)
   
    # model.model.features.requires_grad_(True)

    # For finetuning with plant and disease classification

    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Remove both heads
    state_dict = {
    k: v for k, v in state_dict.items()
    if not ("disease_head" in k or "plant_head" in k)
    }

    model.load_state_dict(state_dict, strict=False)
    model.model.features.requires_grad_(True)


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


