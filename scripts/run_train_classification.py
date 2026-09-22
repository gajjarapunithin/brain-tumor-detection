"""CLI entry point for classification training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_classification import train_classification

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/classification_config.yaml")
parser.add_argument("--run-name", default="classification")
args = parser.parse_args()
print(train_classification(args.config, args.run_name))
