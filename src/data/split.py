"""Reproducible train/validation/test splitting helpers."""

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


def create_split_manifest(records: Sequence[dict], output_path: Path | str, seed: int = 42,
                          validation_fraction: float = 0.15, test_fraction: float = 0.15) -> pd.DataFrame:
    """Create a stratified manifest with train, validation, and test assignments."""
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Cannot split an empty record collection.")
    labels = frame["label"] if "label" in frame.columns else None
    train, temporary = train_test_split(frame, test_size=validation_fraction + test_fraction,
                                        random_state=seed, stratify=labels)
    temporary_labels = temporary["label"] if labels is not None else None
    relative_test = test_fraction / (validation_fraction + test_fraction)
    validation, test = train_test_split(temporary, test_size=relative_test,
                                        random_state=seed, stratify=temporary_labels)
    train = train.assign(split="train")
    validation = validation.assign(split="val")
    test = test.assign(split="test")
    result = pd.concat([train, validation, test], ignore_index=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
