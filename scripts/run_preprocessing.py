"""CLI entry point for preprocessing checks and directory setup."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare runtime directories for BRISC2025.")
    parser.add_argument("--root", default=None, help="Optional project root override.")
    args = parser.parse_args()
    config = Config() if args.root is None else Config(root_dir=args.root)
    config.ensure_directories()
    print(f"Prepared directories under {config.data_dir}")


if __name__ == "__main__":
    main()
