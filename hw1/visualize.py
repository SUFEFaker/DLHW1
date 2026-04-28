from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path

import numpy as np


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def plot_training_history(history: list[dict], output_path: str | Path) -> bool:
    plt = _load_pyplot()
    if plt is None or not history:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Accuracy")
    axes[1].plot(epochs, val_acc, label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_weight_grid(
    weight_matrix: np.ndarray,
    input_shape: tuple[int, int, int],
    output_path: str | Path,
    max_filters: int = 25,
) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_filters = min(max_filters, weight_matrix.shape[1])
    cols = int(ceil(sqrt(num_filters)))
    rows = int(ceil(num_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(rows, cols)

    for index in range(rows * cols):
        axis = axes.flat[index]
        axis.axis("off")
        if index >= num_filters:
            continue
        image = weight_matrix[:, index].reshape(input_shape)
        image = image - image.min()
        max_value = image.max()
        if max_value > 0:
            image = image / max_value
        axis.imshow(image)
        axis.set_title(f"Neuron {index}", fontsize=9)

    fig.suptitle("First Layer Weight Visualization")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_image_grid(
    images: list[np.ndarray], titles: list[str], output_path: str | Path, cols: int = 4
) -> bool:
    plt = _load_pyplot()
    if plt is None or not images:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = int(ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(rows, cols)

    for index in range(rows * cols):
        axis = axes.flat[index]
        axis.axis("off")
        if index >= len(images):
            continue
        axis.imshow(images[index])
        axis.set_title(titles[index], fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_confusion_heatmap(
    matrix: np.ndarray, class_names: list[str], output_path: str | Path
) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True
