"""CLI entry point for segmentation training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_segmentation import train_segmentation

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/segmentation_config.yaml")
parser.add_argument("--run-name", default="segmentation")
args = parser.parse_args()
print(train_segmentation(args.config, args.run_name))
