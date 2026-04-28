from __future__ import annotations

import numpy as np

from .autograd import Tensor
from .data import EuroSATDataModule
from .losses import l2_regularization, softmax_cross_entropy
from .metrics import accuracy_score
from .nn import Module
from .optim import SGD


def run_epoch(
    model: Module,
    datamodule: EuroSATDataModule,
    split: str,
    optimizer: SGD | None = None,
    weight_decay: float = 0.0,
    epoch: int = 0,
    collect_outputs: bool = False,
    log_interval: int = 0,
) -> dict:
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    weight_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    ]

    total_loss = 0.0
    total_examples = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_paths: list[str] = []

    batch_iterator = datamodule.iter_batches(
        split,
        shuffle=is_training,
        with_paths=collect_outputs,
        epoch=epoch,
    )

    for batch_index, batch in enumerate(batch_iterator, start=1):
        if collect_outputs:
            features, targets, paths = batch
            all_paths.extend(paths)
        else:
            features, targets = batch

        inputs = Tensor(features, requires_grad=False)
        logits = model(inputs)
        ce_loss = softmax_cross_entropy(logits, targets)
        objective = ce_loss

        if is_training and weight_decay > 0.0:
            # L2 正则只加到训练目标里，验证和测试只统计交叉熵。
            objective = objective + l2_regularization(weight_parameters, weight_decay)

        if is_training:
            # 清空旧梯度 -> 反向传播 -> SGD 更新参数。
            model.zero_grad()
            objective.backward()
            optimizer.step()

        predictions = logits.data.argmax(axis=1)
        batch_size = targets.shape[0]
        total_examples += batch_size
        total_loss += ce_loss.item() * batch_size
        all_targets.append(targets)
        all_predictions.append(predictions)

        if log_interval > 0 and (batch_index % log_interval == 0):
            # 长时间训练时输出 running 指标，便于判断程序不是卡住。
            running_loss = total_loss / max(total_examples, 1)
            running_accuracy = accuracy_score(
                np.concatenate(all_targets),
                np.concatenate(all_predictions),
            )
            print(
                f"Epoch {epoch:02d} {split:<5} "
                f"batch {batch_index:04d} "
                f"examples {total_examples:05d} "
                f"loss {running_loss:.4f} "
                f"acc {running_accuracy:.4f}",
                flush=True,
            )

    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    y_pred = (
        np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)
    )
    result = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy_score(y_true, y_pred) if total_examples else 0.0,
        "num_examples": total_examples,
    }
    if collect_outputs:
        result["y_true"] = y_true
        result["y_pred"] = y_pred
        result["paths"] = all_paths
    return result
