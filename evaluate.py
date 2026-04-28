from __future__ import annotations

import argparse
from pathlib import Path

from hw1.data import EuroSATDataModule
from hw1.engine import run_epoch
from hw1.metrics import confusion_matrix, save_confusion_matrix_csv
from hw1.nn import ThreeLayerMLP
from hw1.utils import load_json, load_state_dict, resolve_metadata_path
from hw1.visualize import plot_confusion_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained NumPy MLP on EuroSAT.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_path = resolve_metadata_path(args.checkpoint, args.metadata)
    metadata = load_json(metadata_path)
    split_payload = load_json(metadata_path.parent / metadata["splits_file"])

    data_root = args.data_root or Path(metadata["data_root"])
    batch_size = args.batch_size or int(metadata["config"]["batch_size"])
    output_dir = args.output_dir or args.checkpoint.parent

    datamodule = EuroSATDataModule(
        data_root=data_root,
        batch_size=batch_size,
        image_size=tuple(metadata["input_shape"][:2]),
        seed=int(metadata["config"]["seed"]),
        split_ratios=tuple(metadata["split_ratios"]),
        split_definitions=split_payload["splits"],
        class_names=split_payload["class_names"],
        mean=metadata["mean"],
        std=metadata["std"],
    )
    datamodule.prepare()

    model = ThreeLayerMLP(
        input_dim=int(metadata["input_dim"]),
        hidden_dims=tuple(metadata["config"]["hidden_dims"]),
        num_classes=len(metadata["class_names"]),
        activation=metadata["config"]["activation"],
    )
    model.load_state_dict(load_state_dict(args.checkpoint))

    metrics = run_epoch(
        model,
        datamodule,
        split=args.split,
        optimizer=None,
        weight_decay=0.0,
        collect_outputs=True,
    )

    matrix = confusion_matrix(metrics["y_true"], metrics["y_pred"], datamodule.num_classes)
    save_confusion_matrix_csv(
        output_dir / f"{args.split}_confusion_matrix.csv",
        matrix,
        datamodule.class_names,
    )
    plot_confusion_heatmap(
        matrix,
        datamodule.class_names,
        output_dir / f"{args.split}_confusion_matrix.png",
    )

    print(f"Split: {args.split}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    for row in matrix.tolist():
        print(row)


if __name__ == "__main__":
    main()
