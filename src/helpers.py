import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mltrainer import ReportTypes, Trainer, TrainerSettings
from mltrainer.metrics import Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from torch import nn

from src.config import CLASS_LABELS
from src.data_utils import FullDataStreamer, set_seed


def run_experiment(
    model_function: Callable[[], nn.Module],
    experiment_name: str,
    trainstreamer: FullDataStreamer,
    validstreamer: FullDataStreamer,
    teststreamer: FullDataStreamer,
    loss_fn: Callable[..., torch.Tensor],
    class_weights: bool = False,
    parallel: bool = False,
    epochs: int = 15,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    filename: str | Path = "results/results.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Train en test één experiment één keer."""

    run = 1
    set_seed(seed)
    print(f"\n{experiment_name} - run {run} (seed={seed})")

    model = model_function()
    settings = TrainerSettings(
        epochs=epochs,
        metrics=[Accuracy()],
        logdir=f"logs/{safe_name(experiment_name)}/run_{run}",
        train_steps=len(trainstreamer),
        valid_steps=len(validstreamer),
        reporttypes=[ReportTypes.TENSORBOARD],
        optimizer_kwargs={
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        earlystop_kwargs={
            "save": False,
            "verbose": True,
            "patience": 3,
        },
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam,
        traindataloader=trainstreamer.stream(),
        validdataloader=validstreamer.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    start = time.perf_counter()
    trainer.loop()
    training_time = time.perf_counter() - start

    y_true, y_pred = predict(model, teststreamer)
    recalls = recall_score(
        y_true,
        y_pred,
        labels=range(5),
        average=None,
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    macro_recall = recalls.mean()

    result = {
        "Experiment": experiment_name,
        "Run": run,
        "Seed": seed,
        "Class Weights": class_weights,
        "Parallel": parallel,
        "Epochs": epochs,
        "Learning Rate": learning_rate,
        "Weight Decay": weight_decay,
        "Accuracy": accuracy,
        "Macro Recall": macro_recall,
        "Training Time Seconds": training_time,
    }
    for label, recall in zip(CLASS_LABELS, recalls):
        result[f"Recall_{label}"] = recall

    save_result(result, filename)
    plot_confusion_matrix(
        [confusion_matrix(
            y_true,
            y_pred,
            labels=range(5),
            normalize="true",
        )],
        experiment_name,
    )

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro-recall: {macro_recall:.3f}")
    print(f"Recall: {recalls.round(3)}")
    print(f"Trainingstijd: {training_time:.1f} seconden")

    summary = save_summary([result], experiment_name, Path(filename).parent)
    print("\nResultaat:")
    print(summary.to_string(index=False))
    return pd.DataFrame([result])


def predict(
    model: nn.Module,
    datastreamer: FullDataStreamer,
) -> tuple[list[int], list[int]]:
    """Maakt voorspellingen op de testdata."""

    model.eval()
    y_true = []
    y_pred = []

    stream = datastreamer.stream()

    with torch.no_grad():
        for _ in range(len(datastreamer)):
            x, y = next(stream)

            voorspelling = model(x).argmax(dim=1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(voorspelling.cpu().tolist())

    return y_true, y_pred


def save_result(result: dict[str, Any], filename: str | Path) -> None:
    """Slaat een run op in het resultatenbestand."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    nieuwe_rij = pd.DataFrame([result])

    if filename.exists() and filename.stat().st_size > 0:
        df = pd.read_csv(filename)

        # Oude versie van dezelfde run verwijderen.
        df = df[
            ~(
                (df["Experiment"] == result["Experiment"])
                & (df["Run"] == result["Run"])
            )
        ]

        nieuwe_rij = pd.concat([df, nieuwe_rij], ignore_index=True)

    nieuwe_rij.to_csv(filename, index=False)


def save_summary(
    resultaten: list[dict[str, Any]],
    experiment_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Slaat de resultaten van één experiment op."""

    df = pd.DataFrame(resultaten)

    kolommen = [
        "Accuracy",
        "Macro Recall",
        *[f"Recall_{label}" for label in CLASS_LABELS],
        "Training Time Seconds",
    ]

    summary = {
        "Experiment": experiment_name,
        "Runs": len(df),
    }

    # De samenvatting bevat de ene uitgevoerde meting.
    for kolom in kolommen:
        summary[f"{kolom} Mean"] = df[kolom].mean()

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([summary])

    summary.to_csv(
        output_dir / f"{safe_name(experiment_name)}_summary.csv",
        index=False,
    )

    return summary


def plot_confusion_matrix(
    confusion_matrices: list[np.ndarray],
    experiment_name: str,
) -> None:
    """Maakt de confusion matrix van het experiment."""

    mean_cm = np.mean(confusion_matrices, axis=0)

    Path("figures").mkdir(exist_ok=True)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        mean_cm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )

    plt.xlabel("Voorspelde klasse")
    plt.ylabel("Werkelijke klasse")
    plt.title(f"Gemiddelde confusion matrix – {experiment_name}")
    plt.tight_layout()

    plt.savefig(
        f"figures/{safe_name(experiment_name)}_mean_confusion_matrix.png",
        dpi=300,
    )

    plt.close()


def safe_name(naam: str) -> str:
    """Maakt een naam geschikt voor een bestand of map."""
    return naam.lower().replace(" ", "_").replace("-", "_")