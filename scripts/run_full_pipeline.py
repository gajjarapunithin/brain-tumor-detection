"""CLI entry point for running both training stages."""

from src.training.train_classification import train_classification
from src.training.train_segmentation import train_segmentation


if __name__ == "__main__":
    segmentation_checkpoint = train_segmentation()
    classification_checkpoint = train_classification()
    print(f"Segmentation checkpoint: {segmentation_checkpoint}")
    print(f"Classification checkpoint: {classification_checkpoint}")
