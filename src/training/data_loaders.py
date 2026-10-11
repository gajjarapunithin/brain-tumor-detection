"""Shared DataLoader construction for training and validation datasets."""

import torch
from torch.utils.data import DataLoader, Dataset, Subset


def build_train_validation_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Construct loaders while respecting the configured worker count."""
    if num_workers < 0:
        raise ValueError("num_workers must be zero or greater")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, validation_loader


def make_seeded_train_validation_subsets(
    training_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    validation_fraction: float = 0.15,
    seed: int = 42,
    group_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[Subset, Subset]:
    """Split a cleaned population reproducibly without separating duplicate groups."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if len(training_dataset) != len(validation_dataset):
        raise ValueError("Training and validation views must have the same cleaned population")
    train_samples = getattr(training_dataset, "samples", None)
    validation_samples = getattr(validation_dataset, "samples", None)
    if train_samples is not None and validation_samples is not None:
        if [(path, label) for path, label in train_samples] != [
            (path, label) for path, label in validation_samples
        ]:
            raise ValueError("Training and validation views do not contain the same samples in the same order")
    train_group_ids = getattr(training_dataset, "duplicate_group_ids", None)
    validation_group_ids = getattr(validation_dataset, "duplicate_group_ids", None)
    if train_group_ids is not None and validation_group_ids is not None:
        if tuple(train_group_ids) != tuple(validation_group_ids):
            raise ValueError("Training and validation views disagree on duplicate-content group assignments")
        if group_ids is None:
            group_ids = train_group_ids
    if group_ids is None:
        group_ids = tuple(f"sample:{index}" for index in range(len(training_dataset)))
    if len(group_ids) != len(training_dataset):
        raise ValueError("group_ids must have one entry per cleaned training sample")
    group_members: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        group_members.setdefault(str(group_id), []).append(index)
    validation_count = max(1, int(len(training_dataset) * validation_fraction))
    generator = torch.Generator().manual_seed(seed)
    group_names = list(group_members)
    order = torch.randperm(len(group_names), generator=generator).tolist()
    validation_indices: list[int] = []
    training_indices: list[int] = []
    for group_index in order:
        members = group_members[group_names[group_index]]
        if len(validation_indices) < validation_count and len(training_indices) + len(members) < len(training_dataset):
            validation_indices.extend(members)
        else:
            training_indices.extend(members)
    if not training_indices or not validation_indices:
        raise ValueError("Unable to form nonempty train and validation subsets from duplicate groups")
    if set(group_ids[index] for index in training_indices) & set(group_ids[index] for index in validation_indices):
        raise RuntimeError("Duplicate content group crossed the training/validation split")
    return (
        Subset(training_dataset, training_indices),
        Subset(validation_dataset, validation_indices),
    )
