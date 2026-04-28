from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .data import EuroSATDataModule
from .engine import run_epoch
from .metrics import confusion_matrix, save_confusion_matrix_csv
from .nn import ThreeLayerMLP
from .optim import ExponentialLRScheduler, SGD
from .utils import ensure_dir, load_state_dict, save_json, save_state_dict
from .visualize import plot_confusion_heatmap, plot_training_history


@dataclass
class TrainingConfig:
    hidden_dims: tuple[int, int]
    activation: str = "relu"
    learning_rate: float = 0.05
    lr_decay: float = 0.95
    min_lr: float = 1e-4
    epochs: int = 20
    batch_size: int = 64
    weight_decay: float = 0.0
    seed: int = 42

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        return payload


def train_experiment(
    datamodule: EuroSATDataModule,
    config: TrainingConfig,
    run_dir: str | Path,
    save_plots: bool = True,
    evaluate_test: bool = True,
    log_interval: int = 50,
) -> dict:
    run_dir = ensure_dir(run_dir)
    datamodule.batch_size = config.batch_size

    model = ThreeLayerMLP(
        input_dim=datamodule.input_dim,
        hidden_dims=config.hidden_dims,
        num_classes=datamodule.num_classes,
        activation=config.activation,
    )
    optimizer = SGD(model.parameters(), lr=config.learning_rate)
    scheduler = ExponentialLRScheduler(
        optimizer,
        decay=config.lr_decay,
        min_lr=config.min_lr,
    )

    split_payload = {
        "class_names": datamodule.class_names,
        "splits": datamodule.serialize_splits(),
    }
    # 保存数据划分，保证之后 evaluate/analyze 使用同一批验证集和测试集。
    save_json(run_dir / "splits.json", split_payload)

    history: list[dict] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    metadata_path = run_dir / "best_model.json"
    checkpoint_path = run_dir / "best_model.npz"

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch:02d}/{config.epochs:02d} started", flush=True)
        train_metrics = run_epoch(
            model,
            datamodule,
            split="train",
            optimizer=optimizer,
            weight_decay=config.weight_decay,
            epoch=epoch,
            log_interval=log_interval,
        )
        val_metrics = run_epoch(
            model,
            datamodule,
            split="val",
            optimizer=None,
            weight_decay=0.0,
            epoch=epoch,
        )

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)

        if val_metrics["accuracy"] > best_val_accuracy:
            # 按验证集准确率选择最优模型，而不是直接使用最后一个 epoch。
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            save_state_dict(checkpoint_path, model.state_dict())
            save_json(
                metadata_path,
                {
                    "config": config.to_dict(),
                    "data_root": str(datamodule.data_root),
                    "class_names": datamodule.class_names,
                    "input_shape": list(datamodule.input_shape),
                    "input_dim": datamodule.input_dim,
                    "mean": datamodule.mean.tolist(),
                    "std": datamodule.std.tolist(),
                    "split_ratios": list(datamodule.split_ratios),
                    "split_sizes": datamodule.split_sizes(),
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "checkpoint_file": checkpoint_path.name,
                    "splits_file": "splits.json",
                },
            )

        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} "
            f"lr {optimizer.lr:.6f} "
            f"train_loss {train_metrics['loss']:.4f} "
            f"train_acc {train_metrics['accuracy']:.4f} "
            f"val_loss {val_metrics['loss']:.4f} "
            f"val_acc {val_metrics['accuracy']:.4f}",
            flush=True,
        )
        scheduler.step()

    save_json(run_dir / "history.json", history)
    plot_created = plot_training_history(history, run_dir / "training_curves.png") if save_plots else False

    best_state = load_state_dict(checkpoint_path)
    model.load_state_dict(best_state)

    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "plot_created": plot_created,
    }

    if evaluate_test:
        # 训练结束后载入最佳权重，在独立测试集上做最终评估。
        test_metrics = run_epoch(
            model,
            datamodule,
            split="test",
            optimizer=None,
            weight_decay=0.0,
            epoch=0,
            collect_outputs=True,
        )
        matrix = confusion_matrix(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            datamodule.num_classes,
        )
        save_confusion_matrix_csv(run_dir / "test_confusion_matrix.csv", matrix, datamodule.class_names)
        plot_confusion_heatmap(
            matrix,
            datamodule.class_names,
            run_dir / "test_confusion_matrix.png",
        )
        save_json(
            run_dir / "test_metrics.json",
            {
                "loss": test_metrics["loss"],
                "accuracy": test_metrics["accuracy"],
                "num_examples": test_metrics["num_examples"],
            },
        )
        summary["test_accuracy"] = test_metrics["accuracy"]
        summary["test_loss"] = test_metrics["loss"]

    save_json(run_dir / "summary.json", summary)
    return summary
