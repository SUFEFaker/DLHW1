from __future__ import annotations

from collections.abc import Iterable

from .autograd import Parameter


class SGD:
    def __init__(self, parameters: Iterable[Parameter], lr: float) -> None:
        self.parameters = list(parameters)
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            # SGD: theta <- theta - lr * grad。
            parameter.data -= self.lr * parameter.grad


class ExponentialLRScheduler:
    def __init__(self, optimizer: SGD, decay: float = 0.95, min_lr: float = 1e-5) -> None:
        self.optimizer = optimizer
        self.decay = float(decay)
        self.min_lr = float(min_lr)

    def step(self) -> float:
        # 指数衰减学习率，并用 min_lr 防止学习率过早降到 0 附近。
        updated = self.optimizer.lr * self.decay
        self.optimizer.lr = max(updated, self.min_lr)
        return self.optimizer.lr
