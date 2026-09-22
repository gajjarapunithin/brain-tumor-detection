"""Training loop for U-Net segmentation."""

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data.dataset import BRISCSegmentationDataset
from src.models.unet import UNet
from src.training.losses import DiceBCELoss
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def train_segmentation(config_path: str = "configs/segmentation_config.yaml", run_name: str = "segmentation") -> Path:
    """Train U-Net, validate each epoch, and return the best checkpoint path."""
    config = Config()
    with open(config_path, encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    for key, value in settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    set_seed(config.seed)
    run_dir = config.run_dir(run_name)
    logger = get_logger(run_name, run_dir / "logs")
    dataset_root = config.raw_dir / "segmentation_task"
    train_set = BRISCSegmentationDataset(dataset_root, "train", config.image_size)
    val_set = BRISCSegmentationDataset(dataset_root, "val", config.image_size)
    train_loader = DataLoader(train_set, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, config.batch_size, shuffle=False, num_workers=config.num_workers)
    model = UNet().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = DiceBCELoss()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = checkpoint_dir / "best.pt"
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Segmentation {epoch + 1}/{config.epochs}"):
            images, masks = images.to(config.device), masks.to(config.device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                validation_loss += criterion(model(images.to(config.device)), masks.to(config.device)).item()
        validation_loss /= max(len(val_loader), 1)
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f", epoch + 1, train_loss / max(len(train_loader), 1), validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({"model_state": model.state_dict(), "config": vars(config)}, best_path)
    return best_path
