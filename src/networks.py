import torch
from torch import nn

from src.config import ECG_FEATURES, MATRIX_SHAPE, NUM_CLASSES


class CNN1D(nn.Module):
    """Eenvoudig 1D CNN-model."""

    def __init__(
        self,
        features: int = 1,
        num_classes: int = NUM_CLASSES,
        kernel_size: int = 5,
        filters: int = 16,
        input_size: int = ECG_FEATURES,
    ) -> None:
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv1d(
                features,
                filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Pooling halveert het aantal meetpunten.
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * (input_size // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Van (batch, 187, 1) naar (batch, 1, 187).
        x = x.permute(0, 2, 1)

        x = self.convolutions(x)
        return self.dense(x)


class CNN2D(nn.Module):
    """Eenvoudig 2D CNN-model."""

    def __init__(
        self,
        features: int = 1,
        num_classes: int = NUM_CLASSES,
        kernel_size: int = 3,
        filters: int = 16,
        matrixshape: tuple[int, int] = MATRIX_SHAPE,
    ) -> None:
        super().__init__()

        # Convolutie over de 16 x 12 ECG-matrix.
        self.convolutions = nn.Sequential(
            nn.Conv2d(
                features,
                filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Pooling halveert hoogte en breedte.
        flatten_size = (
            filters
            * (matrixshape[0] // 2)
            * (matrixshape[1] // 2)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        return self.dense(x)


class ParallelCNN2D(nn.Module):
    """2D CNN met twee parallelle kernelgroottes."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        filters: int = 16,
        matrixshape: tuple[int, int] = MATRIX_SHAPE,
        large_kernel: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Kleine kernel zoekt lokale patronen.
        self.small_route = self._conv_block(filters, 3)

        # Grote kernel kijkt naar een groter deel van het ECG.
        self.large_route = self._conv_block(filters, large_kernel)

        # Beide routes worden samengevoegd, dus 2 keer zoveel features.
        flatten_size = (
            2
            * filters
            * (matrixshape[0] // 2)
            * (matrixshape[1] // 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flatten_size, num_classes),
        )

    @staticmethod
    def _conv_block(filters: int, kernel_size: int) -> nn.Sequential:
        """Maakt één convolutioneel blok."""
        return nn.Sequential(
            nn.Conv2d(
                1,
                filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Beide routes verwerken dezelfde invoer.
        small = self.small_route(x)
        large = self.large_route(x)

        # Uitkomsten samenvoegen over de filters.
        x = torch.cat([small, large], dim=1)

        return self.classifier(x)