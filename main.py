import argparse
from functools import partial
from argparse import Namespace
from collections.abc import Callable

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from src.config import (
    DEFAULT_SEED,
    ECG_FEATURES,
    MATRIX_SHAPE,
    NUM_CLASSES,
    RESULTS_FILE,
    TARGET_COLUMN,
    TEST_FILE,
    TRAIN_FILE,
    TUNING_RESULTS_FILE,
    VALIDATION_FRACTION,
)
from src.data_utils import (
    FullDataStreamer,
    calculate_class_weights,
    split_train_validation,
)
from src.datasets import HeartDataset1D, HeartDataset2D
from src.helpers import run_experiment
from src.networks import CNN1D, CNN2D, ParallelCNN2D
from src.tuning import run_ray_tuning


def build_streamer(dataset: Dataset, batch_size: int) -> FullDataStreamer:
    """Maakt een datastreamer voor training, validatie of test."""
    return FullDataStreamer(dataset, batch_size=batch_size)


def load_data(
    use_2d: bool,
    seed: int,
) -> tuple[Subset, Subset, Dataset]:
    """Laadt de data en splitst de trainingsdata in train en validatie."""

    if use_2d:
        train_data: Dataset = HeartDataset2D(
            TRAIN_FILE,
            target=TARGET_COLUMN,
            shape=MATRIX_SHAPE,
        )
        test_data: Dataset = HeartDataset2D(
            TEST_FILE,
            target=TARGET_COLUMN,
            shape=MATRIX_SHAPE,
        )
    else:
        train_data = HeartDataset1D(
            TRAIN_FILE,
            target=TARGET_COLUMN,
        )
        test_data = HeartDataset1D(
            TEST_FILE,
            target=TARGET_COLUMN,
        )

    train, valid = split_train_validation(
        train_data,
        validation_fraction=VALIDATION_FRACTION,
        seed=seed,
    )

    return train, valid, test_data


def run_standard_experiment(args: Namespace) -> None:
    """Voert één van de standaard experimenten uit."""

    # Alleen de 1D-experimenten gebruiken de 1D-dataset.
    use_2d = args.experiment not in {"baseline_1d", "weighted_1d"}

    train, valid, test = load_data(
        use_2d=use_2d,
        seed=args.seed,
    )

    # Gewichten om rekening te houden met ongelijke klassen.
    weights = calculate_class_weights(train)

    # Bepalen of class weights gebruikt moeten worden.
    weighted = args.experiment in {
        "weighted_1d",
        "weighted_2d",
        "parallel",
    }

    parallel = args.experiment == "parallel"

    # Kies het juiste model.
    model_function: Callable[[], nn.Module]

    if args.experiment in {"baseline_1d", "weighted_1d"}:
        model_function = partial(
            CNN1D,
            num_classes=NUM_CLASSES,
            filters=args.filters,
            input_size=ECG_FEATURES,
        )

    elif args.experiment in {"baseline_2d", "weighted_2d"}:
        model_function = partial(
            CNN2D,
            num_classes=NUM_CLASSES,
            filters=args.filters,
            matrixshape=MATRIX_SHAPE,
        )

    elif args.experiment == "parallel":
        model_function = partial(
            ParallelCNN2D,
            num_classes=NUM_CLASSES,
            filters=args.filters,
            matrixshape=MATRIX_SHAPE,
            large_kernel=args.large_kernel,
            dropout=args.dropout,
        )

    else:
        raise ValueError(f"Onbekend experiment: {args.experiment}")

    # Experiment uitvoeren.
    run_experiment(
        model_function=model_function,
        experiment_name=args.experiment,
        trainstreamer=build_streamer(train, args.batch_size),
        validstreamer=build_streamer(valid, args.batch_size),
        teststreamer=build_streamer(test, args.batch_size),
        loss_fn=torch.nn.CrossEntropyLoss(
            weight=weights if weighted else None
        ),
        class_weights=weighted,
        parallel=parallel,
        epochs=args.epochs,
        evaluate_on_test=args.test_evaluation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        filename=RESULTS_FILE,
        seed=args.seed,
    )


def run_tuning(args: Namespace) -> None:
    """Zoekt met Ray Tune naar goede hyperparameters."""

    train, valid, _ = load_data(
        use_2d=True,
        seed=args.seed,
    )

    weights = calculate_class_weights(train)

    results = run_ray_tuning(
        train_dataset=train,
        valid_dataset=valid,
        class_weights=weights,
        num_samples=args.num_samples,
        max_epochs=args.epochs,
    )

    # Alleen de belangrijkste resultaten tonen.
    columns = [
        "macro_recall",
        "config/learning_rate",
        "config/filters",
        "config/batch_size",
        "config/weight_decay",
        "config/dropout",
        "config/large_kernel",
    ]

    print(
        results[columns]
        .head(5)
        .to_string(index=False)
    )


def run_final(args: Namespace) -> None:
    """Train het uiteindelijke model met de beste instellingen uit Ray Tune."""

    tuning_file = TUNING_RESULTS_FILE

    if not tuning_file.exists():
        raise FileNotFoundError(
            "Voer eerst --experiment tune uit. "
            "results/ray_tuning_results.csv ontbreekt."
        )

    # Resultaten van Ray Tune inlezen.
    tuning = pd.read_csv(tuning_file)

    # Rij met de hoogste macro recall kiezen.
    best = (
        tuning
        .sort_values("macro_recall", ascending=False)
        .iloc[0]
    )

    train, valid, test = load_data(
        use_2d=True,
        seed=args.seed,
    )

    weights = calculate_class_weights(train)

    # Beste batch size uit de tuning gebruiken.
    batch_size = int(best["config/batch_size"])

    run_experiment(
        model_function=lambda: ParallelCNN2D(
            num_classes=NUM_CLASSES,
            filters=int(best["config/filters"]),
            matrixshape=MATRIX_SHAPE,
            large_kernel=int(best["config/large_kernel"]),
            dropout=float(best["config/dropout"]),
        ),
        experiment_name="parallel_hypertuned_final",
        trainstreamer=build_streamer(train, batch_size),
        validstreamer=build_streamer(valid, batch_size),
        teststreamer=build_streamer(test, batch_size),
        loss_fn=torch.nn.CrossEntropyLoss(weight=weights),
        class_weights=True,
        parallel=True,
        epochs=args.epochs,
        evaluate_on_test=True,
        learning_rate=float(best["config/learning_rate"]),
        weight_decay=float(best["config/weight_decay"]),
        filename=RESULTS_FILE,
        seed=args.seed,
    )


def parse_args() -> Namespace:
    """Leest de instellingen uit de command line."""

    parser = argparse.ArgumentParser(
        description="ECG CNN experiment runner"
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "baseline_1d",
            "baseline_2d",
            "weighted_1d",
            "weighted_2d",
            "parallel",
            "tune",
            "final",
        ],
    )

    # Algemene instellingen.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument(
        "--large-kernel",
        type=int,
        choices=[5, 7],
        default=5,
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--test-evaluation",
        action="store_true",
        help="Evaluate a standard experiment once on the test set.",
    )

    return parser.parse_args()


def main() -> None:
    """Start het gekozen experiment."""

    args = parse_args()

    if args.experiment == "tune":
        run_tuning(args)

    elif args.experiment == "final":
        run_final(args)

    else:
        run_standard_experiment(args)


if __name__ == "__main__":
    main()