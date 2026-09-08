import random
from math import ceil
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.config import NUM_CLASSES


def set_seed(seed: int) -> None:
    """Zorgt dat experimenten reproduceerbaar zijn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_targets(dataset: Any) -> np.ndarray:
    """Haalt de labels uit de dataset."""

    # train/validatie split is een Subset.
    if isinstance(dataset, Subset):
        return getattr(dataset.dataset, "y")[dataset.indices].cpu().numpy()

    return dataset.y.cpu().numpy()


def split_train_validation(
    dataset: Any,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Maakt een gestratificeerde train- en validatieset."""

    indices = np.arange(len(dataset))
    labels = get_targets(dataset)

    train_idx, valid_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels,
    )

    return Subset(dataset, train_idx), Subset(dataset, valid_idx)


def calculate_class_weights(
    dataset: Any,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Geeft zeldzame klassen een hoger gewicht."""

    labels = torch.tensor(get_targets(dataset))
    counts = torch.bincount(labels, minlength=num_classes).float()

    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(
            f"Kan geen class weights berekenen; ontbrekende klassen: {missing}."
        )

    return counts.sum() / (num_classes * counts)


class FullDataStreamer:
    """Herhaalbare datastream die ook de laatste batch teruggeeft."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return ceil(len(self.dataset) / self.batch_size)

    def stream(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=False,
            )

            yield from loader