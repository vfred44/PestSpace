import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path

# Import your Lightning model class
from models.multiEfficientnetB0 import multiEfficientnetB0

@hydra.main(version_base=None, config_path="conf", config_name="config")
def export_onnx(cfg: DictConfig):
    """
    Exports a PyTorch Lightning checkpoint to ONNX using Hydra config.
    """
    # Load the checkpoint
    model = multiEfficientnetB0.load_from_checkpoint(
        checkpoint_path=cfg.ckpt_path,
        cfg=cfg
    )
    model.eval()

    # Dummy input
    img_size = cfg.data.transforms.val.transforms[0].size
    dummy_input = torch.randn(1, 3, img_size, img_size)

    # Output path
    output_path = Path("/Users/fredvaartnou/Downloads/model.onnx")

    # Export ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"ONNX export complete! Saved to {output_path}")

if __name__ == "__main__":
    export_onnx()