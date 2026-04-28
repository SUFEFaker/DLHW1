from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .autograd import Parameter, Tensor


class Module:
    def __init__(self) -> None:
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        # 递归收集子模块参数，方便 optimizer 和 checkpoint 统一处理。
        for name, value in self.__dict__.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Parameter):
                yield full_name, value
            elif isinstance(value, Module):
                yield from value.named_parameters(full_name)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    item_name = f"{full_name}.{index}"
                    if isinstance(item, Parameter):
                        yield item_name, item
                    elif isinstance(item, Module):
                        yield from item.named_parameters(item_name)

    def parameters(self) -> list[Parameter]:
        return [param for _, param in self.named_parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: parameter.data.copy() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        current = dict(self.named_parameters())
        missing = sorted(set(current) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(current))
        if missing or unexpected:
            message = []
            if missing:
                message.append(f"missing keys: {missing}")
            if unexpected:
                message.append(f"unexpected keys: {unexpected}")
            raise KeyError("; ".join(message))
        for name, parameter in current.items():
            parameter.data = np.asarray(state_dict[name], dtype=np.float32).copy()


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # Xavier/Glorot 初始化，避免深层全连接网络初始激活过大或过小。
        limit = np.sqrt(6.0 / (in_features + out_features))
        weight = np.random.uniform(-limit, limit, size=(in_features, out_features)).astype(
            np.float32
        )
        bias = np.zeros((1, out_features), dtype=np.float32)
        self.weight = Parameter(weight, name="weight")
        self.bias = Parameter(bias, name="bias")

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


class ThreeLayerMLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] | list[int],
        num_classes: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0])
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain one or two integers.")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.activation = activation.lower()

        if self.activation not in {"relu", "sigmoid", "tanh"}:
            raise ValueError("activation must be one of: relu, sigmoid, tanh.")

        self.linear1 = Linear(input_dim, hidden_dims[0])
        self.linear2 = Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = Linear(hidden_dims[1], num_classes)

    def _apply_activation(self, x: Tensor) -> Tensor:
        if self.activation == "relu":
            return x.relu()
        if self.activation == "sigmoid":
            return x.sigmoid()
        return x.tanh()

    def forward(self, x: Tensor) -> Tensor:
        # 三个线性层：前两层接激活函数，最后一层输出 logits。
        x = self._apply_activation(self.linear1(x))
        x = self._apply_activation(self.linear2(x))
        return self.linear3(x)
