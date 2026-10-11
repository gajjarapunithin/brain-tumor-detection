"""Print the hardware capabilities visible to this PyTorch installation."""

import argparse
import json

from src.utils.hardware import hardware_report, resolve_resources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default="auto")
    parser.add_argument("--resource-profile", choices=("auto", "low_memory", "balanced", "high_performance", "custom"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    report = hardware_report(args.device)
    report["example_resources"] = resolve_resources(
        report, profile=args.resource_profile, configured_batch_size=args.batch_size,
        configured_workers=args.num_workers,
    ).__dict__
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
