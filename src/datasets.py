from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.config import ECG_FEATURES


class HeartDataset1D(Dataset[tuple[Tensor, Tensor]]):
    """Dataset voor het 1D CNN-model."""

    def __init__(self, path: str | Path, target: str) -> None:
        df = pd.read_parquet(path)
        features = df.drop(target, axis=1)

        if features.shape[1] != ECG_FEATURES:
            raise ValueError(
                f"HeartDataset1D verwacht {ECG_FEATURES} ECG-meetpunten, "
                f"maar kreeg {features.shape[1]}."
            )

        # De originele 187 ECG-meetpunten gebruiken.
        self.x = torch.tensor(
            features.values,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            df[target].values,
            dtype=torch.int64,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # Eén kanaal toevoegen voor Conv1d.
        return self.x[idx].unsqueeze(1), self.y[idx]

    def __repr__(self) -> str:
        return f"HeartDataset1D ({len(self)})"


class HeartDataset2D(Dataset[tuple[Tensor, Tensor]]):
    """Dataset voor de 2D CNN-modellen."""

    def __init__(
        self,
        path: str | Path,
        target: str,
        shape: tuple[int, int] = (16, 12),
    ) -> None:
        df = pd.read_parquet(path)
        features = df.drop(target, axis=1)
        expected_features = ECG_FEATURES

        if features.shape[1] != expected_features:
            raise ValueError(
                f"HeartDataset2D met shape {shape} verwacht "
                f"{expected_features} ECG-meetpunten, "
                f"maar kreeg {features.shape[1]}."
            )

        x = torch.tensor(
            features.values,
            dtype=torch.float32,
        )

        # Pad naar het aantal waarden dat de matrixvorm nodig heeft.
        padded_features = shape[0] * shape[1]
        x = torch.nn.functional.pad(x, (0, padded_features - x.size(1)))

        # Omvormen naar een 16 x 12 matrix.
        self.x = x.reshape(-1, 1, *shape)

        self.y = torch.tensor(
            df[target].values,
            dtype=torch.int64,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]

    def __repr__(self) -> str:
        return f"HeartDataset2D ({len(self)})"