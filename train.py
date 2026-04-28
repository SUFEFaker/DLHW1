from __future__ import annotations

import argparse
from pathlib import Path

from hw1.data import EuroSATDataModule
from hw1.experiments import TrainingConfig, train_experiment
from hw1.utils import ensure_dir, set_seed, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy-based three-layer MLP on EuroSAT.")
    parser.add_argument("--data-root", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", nargs=2, type=int, default=[64, 64])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print training progress every N batches. Use 0 to disable batch progress.",
    )
    return parser.parse_args()


def normalize_hidden_dims(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("--hidden-dims must provide one or two integers.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    hidden_dims = normalize_hidden_dims(args.hidden_dims)
    run_name = args.run_name or f"mlp_{timestamp()}"
    run_dir = ensure_dir(args.output_dir / run_name)

    datamodule = EuroSATDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        seed=args.seed,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )
    datamodule.prepare()

    config = TrainingConfig(
        hidden_dims=hidden_dims,
        activation=args.activation,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        min_lr=args.min_lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    print("Dataset split sizes:", datamodule.split_sizes())
    print("Classes:", datamodule.class_names)
    print("Training config:", config.to_dict())

    summary = train_experiment(
        datamodule=datamodule,
        config=config,
        run_dir=run_dir,
        save_plots=True,
        evaluate_test=True,
        log_interval=args.log_interval,
    )

    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
    if "test_accuracy" in summary:
        print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"Checkpoint saved to: {summary['checkpoint_path']}")
    print(f"Run directory: {summary['run_dir']}")


if __name__ == "__main__":
    main()
