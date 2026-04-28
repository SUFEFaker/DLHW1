from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _to_float32_array(data: Any) -> np.ndarray:
    return np.asarray(data, dtype=np.float32)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    # 反向传播遇到 NumPy broadcasting 时，需要把梯度压回原张量形状。
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
        name: str | None = None,
    ) -> None:
        self.data = _to_float32_array(data)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.name = name

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data!r}, requires_grad={self.requires_grad}, "
            f"name={self.name!r})"
        )

    @staticmethod
    def ensure_tensor(value: Any) -> "Tensor":
        if isinstance(value, Tensor):
            return value
        return Tensor(value, requires_grad=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def zero_grad(self) -> None:
        self.grad = None

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        # 同一个参数可能被多个计算路径使用，梯度需要累加而不是覆盖。
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = grad.astype(np.float32, copy=True)
        else:
            self.grad += grad.astype(np.float32, copy=False)

    def __add__(self, other: Any) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # 加法两侧梯度都是 1，但要处理 broadcasting 后的维度还原。
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: Any) -> "Tensor":
        return self + (-Tensor.ensure_tensor(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return Tensor.ensure_tensor(other) - self

    def __mul__(self, other: Any) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad * other.data
                self._accumulate_grad(_unbroadcast(grad_self, self.data.shape))
            if other.requires_grad:
                grad_other = out.grad * self.data
                other._accumulate_grad(_unbroadcast(grad_other, other.data.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __truediv__(self, other: Any) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="div",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad / other.data
                self._accumulate_grad(_unbroadcast(grad_self, self.data.shape))
            if other.requires_grad:
                grad_other = -out.grad * self.data / (other.data ** 2)
                other._accumulate_grad(_unbroadcast(grad_other, other.data.shape))

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Any) -> "Tensor":
        return Tensor.ensure_tensor(other) / self

    def matmul(self, other: Any) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # 矩阵乘法反向传播：dA = dY @ B.T, dB = A.T @ dY。
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other: Any) -> "Tensor":
        return self.matmul(other)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad, self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                normalized_axes = tuple(
                    ax if ax >= 0 else self.data.ndim + ax for ax in axes
                )
                if not keepdims:
                    for ax in sorted(normalized_axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.data.shape)
            self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            count = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            count = 1
            for ax in axes:
                count *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) / float(count)

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def square(self) -> "Tensor":
        return self * self

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * (self.data > 0.0))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(
            sig,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * sig * (1.0 - sig))

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        tanh_value = np.tanh(self.data)
        out = Tensor(
            tanh_value,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * (1.0 - tanh_value ** 2))

        out._backward = _backward
        return out

    def backward(self, grad: Any | None = None) -> None:
        if not self.requires_grad:
            return
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Gradient must be specified for non-scalar tensors.")
            grad_array = np.ones_like(self.data, dtype=np.float32)
        else:
            grad_array = _to_float32_array(grad)

        # 先对计算图做拓扑排序，再按反向拓扑顺序依次调用局部 backward。
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = grad_array
        for node in reversed(topo):
            node._backward()


class Parameter(Tensor):
    def __init__(self, data: Any, name: str | None = None) -> None:
        super().__init__(data, requires_grad=True, name=name)
