import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from models.Resnet18_focalloss import Resnet18_focalloss       
from data.data import get_data_loaders


@hydra.main(version_base=None, config_path="conf", config_name="config")
def debug(cfg: DictConfig):

    # Checkpoint
    ckpt_path = cfg.ckpt_path

    # Load model from checkpoint
    model = Resnet18_focalloss.load_from_checkpoint(ckpt_path)
    model.eval()  # Set to evaluation mode

    # Setup your DataModule for validation
    train_dl, val_dl, class_counts, image_paths = get_data_loaders(cfg)
 
    #model.image_paths = image_paths

    # Option 1: Use Lightning Trainer validate loop
    trainer = Trainer(accelerator="cpu", devices=1, logger=WandbLogger(project="PestSpace", name="debug_run"))
    val_results = trainer.validate(model=model, dataloaders=val_dl)

    print("Validation results:", val_results)

if __name__ == "__main__":
    debug()

