"""CLI entry point for classification training."""

import argparse
import multiprocessing
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI options and launch training when explicitly invoked."""
    from src.training.train_classification import train_classification

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classification_config.yaml")
    parser.add_argument("--run-name", default="classification")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default=None)
    parser.add_argument("--resource-profile", choices=("auto", "low_memory", "balanced", "high_performance", "custom"), default=None)
    parser.add_argument("--mixed-precision", choices=("auto", "float32", "float16", "bfloat16"), default=None)
    args = parser.parse_args(argv)
    print(train_classification(args.config, args.run_name, args.device, args.resource_profile, args.mixed_precision))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
