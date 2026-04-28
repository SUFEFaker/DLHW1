from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .autograd import Tensor


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    if logits.data.ndim != 2:
        raise ValueError("softmax_cross_entropy expects logits with shape [batch, classes].")

    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 1 or targets.shape[0] != logits.data.shape[0]:
        raise ValueError("targets must be a 1D array with the same batch size as logits.")

    # 先减去每行最大值，避免 exp(logits) 数值溢出。
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    batch_indices = np.arange(targets.shape[0])
    loss_value = -np.log(probs[batch_indices, targets] + 1e-12).mean()

    out = Tensor(loss_value, requires_grad=logits.requires_grad, _children=(logits,), _op="ce")

    def _backward() -> None:
        if out.grad is None or not logits.requires_grad:
            return
        # softmax + cross entropy 的合并梯度：p - one_hot(y)。
        grad = probs.copy()
        grad[batch_indices, targets] -= 1.0
        grad /= targets.shape[0]
        logits._accumulate_grad(grad * out.grad)

    out._backward = _backward
    return out


def l2_regularization(parameters: Iterable[Tensor], strength: float) -> Tensor:
    # 只对权重矩阵做 L2 penalty，bias 在调用处不会传进来。
    penalty: Tensor | None = None
    for param in parameters:
        term = 0.5 * strength * param.square().sum()
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        return Tensor(0.0, requires_grad=False)
    return penalty
