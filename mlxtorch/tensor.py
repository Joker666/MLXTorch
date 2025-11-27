"""Core Tensor class with lightweight autograd support built on MLX."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Set, Tuple, Union

import mlx.core as mx

Arrayable = Union[mx.array, float, int, bool, list[Any], tuple[Any, ...]]
TensorLike = Union[Arrayable, "Tensor"]


def _ensure_array(data: TensorLike) -> mx.array:
    """Convert input (tensor or array-like) to an MLX array."""
    if isinstance(data, Tensor):
        return data.data
    return mx.array(data)


def _match_shape(grad: mx.array, shape: Tuple[int, ...]) -> mx.array:
    """Sum gradient to match a broadcasted shape."""
    if grad.shape == shape:
        return grad

    # Remove leading broadcasted dimensions
    while len(shape) < grad.ndim:
        grad = mx.sum(grad, axis=0)

    # Collapse axes that were broadcasted with dimension 1
    for axis, dim in enumerate(shape):
        if dim == 1:
            grad = mx.sum(grad, axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """Tensor wrapper around `mlx.core.array` that supports autograd."""

    def __init__(
        self,
        data: TensorLike,
        requires_grad: bool = False,
        _backward: Optional[Callable[[], None]] = None,
        _prev: Optional[Iterable["Tensor"]] = None,
    ) -> None:
        self.data: mx.array = _ensure_array(data)
        self.grad: Optional[mx.array] = None
        self.requires_grad = requires_grad
        self._backward: Callable[[], None] = _backward or (lambda: None)
        self._prev: Set["Tensor"] = set(_prev) if _prev is not None else set()

    def __repr__(self) -> str:  # pragma: no cover - string repr is trivial
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def _add_grad(self, grad: mx.array) -> None:
        """Accumulate gradient."""
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

    # ---- arithmetic operations ----
    def __add__(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other_tensor.data,
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _prev=(self, other_tensor),
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._add_grad(_match_shape(out.grad, self.data.shape))
            if other_tensor.requires_grad:
                other_tensor._add_grad(_match_shape(out.grad, other_tensor.data.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        return self + other

    def __sub__(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        return self + (-1 * other)

    def __rsub__(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        return (-1 * self) + other

    def __mul__(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other_tensor.data,
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _prev=(self, other_tensor),
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad * other_tensor.data
                self._add_grad(_match_shape(grad_self, self.data.shape))
            if other_tensor.requires_grad:
                grad_other = out.grad * self.data
                other_tensor._add_grad(_match_shape(grad_other, other_tensor.data.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        return self * other

    def __truediv__(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other_tensor.data,
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _prev=(self, other_tensor),
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad / other_tensor.data
                self._add_grad(_match_shape(grad_self, self.data.shape))
            if other_tensor.requires_grad:
                grad_other = -out.grad * self.data / (other_tensor.data**2)
                other_tensor._add_grad(_match_shape(grad_other, other_tensor.data.shape))

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        return Tensor(other) / self

    def __pow__(self, power: Union[int, float]) -> "Tensor":
        out = Tensor(
            self.data**power,
            requires_grad=self.requires_grad,
            _prev=(self,),
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = out.grad * (power * (self.data ** (power - 1)))
            self._add_grad(_match_shape(grad_self, self.data.shape))

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1

    # ---- matrix operations ----
    def matmul(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            mx.matmul(self.data, other_tensor.data),
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _prev=(self, other_tensor),
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = mx.matmul(out.grad, mx.transpose(other_tensor.data))
                self._add_grad(grad_self)
            if other_tensor.requires_grad:
                grad_other = mx.matmul(mx.transpose(self.data), out.grad)
                other_tensor._add_grad(grad_other)

        out._backward = _backward
        return out

    def __matmul__(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        return self.matmul(other)

    def __rmatmul__(self, other: Union["Tensor", Arrayable]) -> "Tensor":
        return Tensor(other).matmul(self)

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> "Tensor":
        """Transpose tensor."""
        out = Tensor(
            mx.transpose(self.data, axes=axes),
            requires_grad=self.requires_grad,
            _prev=(self,),
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            inv_axes: Optional[Tuple[int, ...]]
            if axes is None:
                inv_axes = None
            else:
                inv = [0] * len(axes)
                for i, axis in enumerate(axes):
                    inv[axis] = i
                inv_axes = tuple(inv)
            self._add_grad(mx.transpose(out.grad, axes=inv_axes))

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        """Shorthand for transpose with axes reversed."""
        return self.transpose()

    # ---- reduction operations ----
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        """Sum elements over axis."""
        out = Tensor(
            mx.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _prev=(self,),
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_out = out.grad
            if axis is None:
                grad_self = mx.broadcast_to(grad_out, self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                if not keepdims:
                    for ax in sorted(axes):
                        grad_out = mx.expand_dims(grad_out, ax)
                grad_self = mx.broadcast_to(grad_out, self.data.shape)
            self._add_grad(grad_self)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        """Mean over axis."""
        out = Tensor(
            mx.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _prev=(self,),
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_out = out.grad
            if axis is None:
                count = self.data.size
                grad_self = mx.broadcast_to(grad_out / count, self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                count = 1
                for ax in axes:
                    count *= self.data.shape[ax]
                if not keepdims:
                    for ax in sorted(axes):
                        grad_out = mx.expand_dims(grad_out, ax)
                grad_self = mx.broadcast_to(grad_out / count, self.data.shape)
            self._add_grad(grad_self)

        out._backward = _backward
        return out

    # ---- autograd ----
    def backward(self, grad: Optional[mx.array] = None) -> None:
        """Run backpropagation from this tensor."""
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be specified for non-scalar outputs")
            grad = mx.ones_like(self.data)
        self.grad = grad

        topo: list[Tensor] = []
        visited: Set[Tensor] = set()

        def build(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        """Clear stored gradients."""
        self.grad = None


def tensor(data: Arrayable, requires_grad: bool = False) -> Tensor:
    """Create a Tensor from input data."""
    return Tensor(data, requires_grad=requires_grad)


def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """Create a zero tensor."""
    return Tensor(mx.zeros(shape), requires_grad=requires_grad)


def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """Create an all-ones tensor."""
    return Tensor(mx.ones(shape), requires_grad=requires_grad)


def randn(shape: Tuple[int, ...], requires_grad: bool = False, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Create a tensor with samples from a normal distribution."""
    data = mx.random.normal(shape=shape, loc=mean, scale=std)
    return Tensor(data, requires_grad=requires_grad)
