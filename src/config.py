"""Central configuration for the brain tumor detection project."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    """Project paths and model/training defaults in one discoverable object."""

    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    image_size: int = 224
    num_classes: int = 4
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 25
    seed: int = 42
    num_workers: int = 2
    backbone: str = "resnet50"
    freeze_backbone: bool = True
    segmentation_threshold: float = 0.5
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def experiments_dir(self) -> Path:
        return self.root_dir / "experiments"

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    def run_dir(self, run_name: str) -> Path:
        return self.experiments_dir / run_name

    def ensure_directories(self) -> None:
        """Create runtime output directories without creating dataset files."""
        for directory in (self.processed_dir, self.splits_dir, self.experiments_dir, self.figures_dir):
            directory.mkdir(parents=True, exist_ok=True)
