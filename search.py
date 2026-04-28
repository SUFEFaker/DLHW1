from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from hw1.data import EuroSATDataModule
from hw1.experiments import TrainingConfig, train_experiment
from hw1.utils import ensure_dir, save_json, set_seed, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid/random search for EuroSAT MLP hyperparameters.")
    parser.add_argument("--data-root", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/search"))
    parser.add_argument("--search-name", type=str, default=None)
    parser.add_argument("--search-mode", type=str, default="grid", choices=["grid", "random"])
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rates", nargs="+", type=float, required=True)
    parser.add_argument("--weight-decays", nargs="+", type=float, required=True)
    parser.add_argument(
        "--hidden-dim-options",
        nargs="+",
        type=str,
        required=True,
        help='Examples: "256,128" "512,256" "128"',
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        type=str,
        default=["relu", "tanh"],
        choices=["relu", "sigmoid", "tanh"],
    )
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", nargs=2, type=int, default=[64, 64])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def parse_hidden_dims(text: str) -> tuple[int, int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError(f"Invalid hidden dim specification: {text}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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

    hidden_dim_options = [parse_hidden_dims(item) for item in args.hidden_dim_options]
    # 先生成完整超参数笛卡尔积，grid 模式全跑，random 模式从中抽样。
    combinations = list(
        itertools.product(
            args.learning_rates,
            hidden_dim_options,
            args.weight_decays,
            args.activations,
        )
    )

    if args.search_mode == "random":
        rng = np.random.default_rng(args.seed)
        choose = min(args.num_trials, len(combinations))
        # 不放回抽样，避免同一组超参数重复训练。
        selected_indices = rng.choice(len(combinations), size=choose, replace=False)
        combinations = [combinations[int(index)] for index in selected_indices]

    search_name = args.search_name or f"search_{timestamp()}"
    search_dir = ensure_dir(args.output_dir / search_name)
    results: list[dict] = []

    for trial_index, (lr, hidden_dims, weight_decay, activation) in enumerate(combinations, start=1):
        trial_name = (
            f"trial_{trial_index:03d}_lr{lr:g}_hd{hidden_dims[0]}x{hidden_dims[1]}_"
            f"wd{weight_decay:g}_{activation}"
        )
        print(f"[{trial_index}/{len(combinations)}] {trial_name}")

        config = TrainingConfig(
            hidden_dims=hidden_dims,
            activation=activation,
            learning_rate=lr,
            lr_decay=args.lr_decay,
            min_lr=args.min_lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=weight_decay,
            seed=args.seed,
        )
        summary = train_experiment(
            datamodule=datamodule,
            config=config,
            run_dir=search_dir / trial_name,
            save_plots=False,
            evaluate_test=False,
        )
        result = {
            "trial_name": trial_name,
            "learning_rate": lr,
            "hidden_dims": list(hidden_dims),
            "weight_decay": weight_decay,
            "activation": activation,
            "best_epoch": summary["best_epoch"],
            "best_val_accuracy": summary["best_val_accuracy"],
            "run_dir": summary["run_dir"],
        }
        results.append(result)
        # 每个 trial 结束就落盘，长时间搜索中断后也能保留已有结果。
        save_json(search_dir / "search_results.json", results)

    best_result = max(results, key=lambda item: item["best_val_accuracy"])
    save_json(search_dir / "best_trial.json", best_result)

    print("Best configuration:")
    print(best_result)


if __name__ == "__main__":
    main()
