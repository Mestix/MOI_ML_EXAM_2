from __future__ import annotations

from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ray import tune
from ray.tune import TuneConfig, Tuner
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import Dataset

from src.config import MATRIX_SHAPE, NUM_CLASSES, TUNING_RESULTS_FILE
from src.data_utils import FullDataStreamer
from src.networks import ParallelCNN2D


def run_ray_tuning(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    class_weights: torch.Tensor,
    num_samples: int = 20,
    max_epochs: int = 10,
    output_dir: str = "ray_results",
) -> pd.DataFrame:
    """Optimaliseer de parallelle 2D-CNN op basis van macro-recall op de validatieset."""

    def train_tune_model(config: dict[str, Any]) -> None:
        # Maak datastreamers voor de trainings- en validatieset.
        trainstreamer = FullDataStreamer(
            train_dataset,
            batch_size=config["batch_size"],
        )
        validstreamer = FullDataStreamer(
            valid_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
        )

        # Bouw het parallelle 2D-CNN-model op basis van de gekozen configuratie.
        model = ParallelCNN2D(
            num_classes=NUM_CLASSES,
            filters=config["filters"],
            matrixshape=MATRIX_SHAPE,
            large_kernel=config["large_kernel"],
            dropout=config["dropout"],
        )

        # Gebruik class weights zodat zeldzame klassen zwaarder meetellen in de loss.
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Maak de optimizer met de door Ray Tune gekozen hyperparameters.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # Train en evalueer het model gedurende met aantal epochs.
        for epoch in range(max_epochs):
            model.train()

            train_data = trainstreamer.stream()
            for _ in range(len(trainstreamer)):
                x, y = next(train_data)

                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()

            # Evalueer de configuratie op de validatieset.
            # De testset wordt hier bewust niet gebruikt voor modelselectie.
            model.eval()
            y_true: list[int] = []
            y_pred: list[int] = []

            valid_data = validstreamer.stream()

            with torch.no_grad():
                for _ in range(len(validstreamer)):
                    x, y = next(valid_data)

                    predictions = model(x).argmax(dim=1)

                    y_true.extend(y.detach().cpu().tolist())
                    y_pred.extend(predictions.detach().cpu().tolist())

            # Bereken de recall per klasse.
            recalls = recall_score(
                y_true,
                y_pred,
                labels=list(range(NUM_CLASSES)),
                average=None,
                zero_division=0,
            )

            # Rapporteer de resultaten aan Ray Tune.
            # Macro-recall is de belangrijkste selectiemaat.
            tune.report(
                {
                    "epoch": epoch + 1,
                    "accuracy": accuracy_score(y_true, y_pred),
                    "macro_recall": float(recalls.mean()),
                    "recall_N": float(recalls[0]),
                    "recall_S": float(recalls[1]),
                    "recall_V": float(recalls[2]),
                    "recall_F": float(recalls[3]),
                    "recall_Q": float(recalls[4]),
                }
            )

    # Zoekruimte voor de hyperparameters.
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "filters": tune.choice([8, 16, 32, 64]),
        "batch_size": tune.choice([16, 32, 64]),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "dropout": tune.uniform(0.0, 0.4),

        # De eerste parallelle route gebruikt vast een 3x3-kernel.
        # Alleen de kernelgrootte van de tweede route wordt geoptimaliseerd.
        "large_kernel": tune.choice([5, 7]),
    }

    # ASHA stopt zwakke configuraties vroegtijdig,
    # zodat de beschikbare rekentijd beter wordt gebruikt.
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=max_epochs,
        grace_period=min(3, max_epochs),
        reduction_factor=2,
    )

    # Stel Ray Tune in.
    tuner = Tuner(
        tune.with_resources(
            train_tune_model,
            resources={"cpu": 1},
        ),
        param_space=search_space,
        tune_config=TuneConfig(
            metric="macro_recall",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=1,
        ),
        run_config=tune.RunConfig(
            name="parallel_2d_hypertuning",
            storage_path=str(Path(output_dir).resolve()),
        ),
    )

    # Start de hyperparameteroptimalisatie.
    results = tuner.fit()

    # Zet alle resultaten om naar een DataFrame en sorteer op macro-recall.
    result_df = results.get_dataframe().sort_values(
        "macro_recall",
        ascending=False,
    )

    # Sla de tuningresultaten op zodat ze later bekeken kunnen worden.
    Path("results").mkdir(parents=True, exist_ok=True)
    result_df.to_csv(TUNING_RESULTS_FILE, index=False)
    save_tuning_outputs(result_df)

    return result_df


def save_tuning_outputs(result_df: pd.DataFrame) -> None:
    """Slaat een samenvatting en figuur van de validatieresultaten op."""

    best = result_df.iloc[0]
    summary = pd.DataFrame([{
        "Trial": best["trial_id"],
        "Epoch": best["epoch"],
        "Validation Accuracy": best["accuracy"],
        "Validation Macro Recall": best["macro_recall"],
        "Recall N": best["recall_N"],
        "Recall S": best["recall_S"],
        "Recall V": best["recall_V"],
        "Recall F": best["recall_F"],
        "Recall Q": best["recall_Q"],
        "Learning Rate": best["config/learning_rate"],
        "Filters": best["config/filters"],
        "Batch Size": best["config/batch_size"],
        "Weight Decay": best["config/weight_decay"],
        "Dropout": best["config/dropout"],
        "Large Kernel": best["config/large_kernel"],
    }])
    Path("results").mkdir(parents=True, exist_ok=True)
    summary.to_csv("results/ray_tuning_summary.csv", index=False)

    figure_data = result_df.sort_values("macro_recall")
    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.barh(figure_data["trial_id"], figure_data["macro_recall"])
    plt.xlabel("Validatie macro-recall")
    plt.ylabel("Trial")
    plt.title("Ray Tune: macro-recall per configuratie")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig("figures/ray_tuning_top_trials.png", dpi=300)
    plt.close()