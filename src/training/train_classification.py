"""Training loop for the four-class MRI classifier."""

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data.dataset import BRISCClassificationDataset
from src.models.classifier import TumorClassifier
from src.training.losses import classification_loss
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def train_classification(config_path: str = "configs/classification_config.yaml", run_name: str = "classification") -> Path:
    """Train the classifier, validate each epoch, and save the best checkpoint."""
    config = Config()
    with open(config_path, encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    for key, value in settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    set_seed(config.seed)
    run_dir = config.run_dir(run_name)
    logger = get_logger(run_name, run_dir / "logs")
    dataset_root = config.raw_dir / "classification_task"
    train_set = BRISCClassificationDataset(dataset_root, "train", config.image_size)
    val_set = BRISCClassificationDataset(dataset_root, "val", config.image_size)
    train_loader = DataLoader(train_set, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, config.batch_size, shuffle=False, num_workers=config.num_workers)
    model = TumorClassifier(config.num_classes, config.freeze_backbone).to(config.device)
    optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config.learning_rate)
    criterion = classification_loss()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_path = checkpoint_dir / "best.pt"
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Classification {epoch + 1}/{config.epochs}"):
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                predictions = model(images.to(config.device)).argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += labels.numel()
        accuracy = correct / max(total, 1)
        logger.info("epoch=%d train_loss=%.4f val_accuracy=%.4f", epoch + 1, train_loss / max(len(train_loader), 1), accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({"model_state": model.state_dict(), "config": vars(config)}, best_path)
    return best_path
