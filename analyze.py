from __future__ import annotations

import argparse
from pathlib import Path

from hw1.data import EuroSATDataModule
from hw1.engine import run_epoch
from hw1.nn import ThreeLayerMLP
from hw1.utils import load_json, load_state_dict, resolve_metadata_path, save_json
from hw1.visualize import plot_image_grid, plot_weight_grid


def build_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualization helpers for the EuroSAT homework.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    return parser


def load_experiment(args: argparse.Namespace):
    metadata_path = resolve_metadata_path(args.checkpoint, args.metadata)
    metadata = load_json(metadata_path)
    split_payload = load_json(metadata_path.parent / metadata["splits_file"])

    data_root = args.data_root or Path(metadata["data_root"])
    datamodule = EuroSATDataModule(
        data_root=data_root,
        batch_size=int(metadata["config"]["batch_size"]),
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
    return metadata, datamodule, model


def weights_command(args: argparse.Namespace) -> None:
    _, datamodule, model = load_experiment(args)
    output_path = args.output or args.checkpoint.parent / "first_layer_weights.png"
    created = plot_weight_grid(
        model.linear1.weight.data,
        datamodule.input_shape,
        output_path,
        max_filters=args.max_filters,
    )
    if not created:
        raise RuntimeError("matplotlib is required for visualization. Install it with pip.")
    print(f"Saved weight visualization to: {output_path}")


def errors_command(args: argparse.Namespace) -> None:
    _, datamodule, model = load_experiment(args)
    output_path = args.output or args.checkpoint.parent / "misclassified_examples.png"
    json_path = args.json_output or args.checkpoint.parent / "misclassified_examples.json"

    results = run_epoch(
        model,
        datamodule,
        split=args.split,
        optimizer=None,
        weight_decay=0.0,
        collect_outputs=True,
    )

    mistakes = []
    for relative_path, true_label, pred_label in zip(
        results["paths"],
        results["y_true"].tolist(),
        results["y_pred"].tolist(),
    ):
        if true_label != pred_label:
            mistakes.append(
                {
                    "path": relative_path,
                    "true_label": datamodule.class_names[true_label],
                    "pred_label": datamodule.class_names[pred_label],
                }
            )

    mistakes = mistakes[: args.num_samples]
    save_json(json_path, mistakes)

    images = [datamodule.load_display_image(item["path"]) for item in mistakes]
    titles = [
        f"T: {item['true_label']}\nP: {item['pred_label']}\n{Path(item['path']).name}"
        for item in mistakes
    ]
    created = plot_image_grid(images, titles, output_path, cols=args.cols)
    if not created:
        raise RuntimeError("matplotlib is required for visualization. Install it with pip.")
    print(f"Saved misclassified example grid to: {output_path}")
    print(f"Saved misclassified metadata to: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = build_common_parser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    weights_parser = subparsers.add_parser("weights", help="Visualize first-layer weights.")
    weights_parser.add_argument("--output", type=Path, default=None)
    weights_parser.add_argument("--max-filters", type=int, default=25)
    weights_parser.set_defaults(func=weights_command)

    errors_parser = subparsers.add_parser("errors", help="Visualize misclassified examples.")
    errors_parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    errors_parser.add_argument("--num-samples", type=int, default=16)
    errors_parser.add_argument("--cols", type=int, default=4)
    errors_parser.add_argument("--output", type=Path, default=None)
    errors_parser.add_argument("--json-output", type=Path, default=None)
    errors_parser.set_defaults(func=errors_command)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
