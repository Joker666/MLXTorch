import pytest
import mlx.core as mx

from mlxtorch.tensor import tensor


def test_backward_requires_grad_for_non_scalar_output():
    x = tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        x.backward()


def test_backward_accepts_external_grad():
    x = tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0
    upstream = mx.array([1.0, 1.5])
    y.backward(upstream)

    assert x.grad is not None
    assert mx.allclose(x.grad, upstream * 2.0)


def test_gradient_accumulates_across_multiple_calls():
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * 2.0).sum()
    y.backward()
    y.backward()

    expected = mx.array([4.0, 4.0, 4.0])
    assert x.grad is not None
    assert mx.allclose(x.grad, expected)


def test_broadcasted_gradient_reduction():
    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = tensor([1.0, -1.0, 0.5], requires_grad=True)
    z = (x + y).sum()
    z.backward()

    assert x.grad is not None
    assert y.grad is not None
    assert mx.allclose(x.grad, mx.ones_like(x.data))
    assert mx.allclose(y.grad, mx.array([2.0, 2.0, 2.0]))


def test_branching_graph_accumulates_from_multiple_paths():
    x = tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    z = y + x
    out = z.sum()
    out.backward()

    assert x.grad is not None
    assert mx.allclose(x.grad, mx.array([3.0, 5.0]))


def test_scalar_multiplier_broadcasts_gradients():
    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    scale = tensor(2.0, requires_grad=True)
    out = (x * scale).sum()
    out.backward()

    assert x.grad is not None
    assert mx.allclose(x.grad, mx.ones_like(x.data) * 2.0)
    assert scale.grad is not None
    assert mx.allclose(scale.grad, mx.array(10.0))


def test_shared_subgraph_propagates_through_all_paths():
    a = tensor([1.0, 2.0], requires_grad=True)
    b = a * 2.0
    c = a + b
    d = b * c  # 6 * a^2
    loss = d.sum()
    loss.backward()

    expected = mx.array([12.0, 24.0])
    assert a.grad is not None
    assert mx.allclose(a.grad, expected)
    assert b.grad is not None and c.grad is not None  # intermediate grads exist


def test_no_grad_leaf_remains_none():
    x = tensor([1.0, 2.0], requires_grad=True)
    y = tensor([3.0, 4.0], requires_grad=False)
    out = (x * y).sum()
    out.backward()

    assert x.grad is not None
    assert mx.allclose(x.grad, y.data)
    assert y.grad is None
