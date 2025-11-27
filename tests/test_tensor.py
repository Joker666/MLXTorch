import mlx.core as mx

from mlxtorch.tensor import Tensor, ones, randn, tensor, zeros


def test_add_and_backward():
    a = tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a + b
    out = c.sum()
    out.backward()

    assert mx.allclose(c.data, mx.array([5.0, 7.0, 9.0]))
    assert mx.allclose(a.grad, mx.ones_like(a.data))
    assert mx.allclose(b.grad, mx.ones_like(b.data))


def test_subtraction_with_python_list_and_grad():
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x - [0.5, 1.0, 1.5]
    total = y.sum()
    total.backward()

    assert mx.allclose(y.data, mx.array([0.5, 1.0, 1.5]))
    assert mx.allclose(x.grad, mx.ones_like(x.data))


def test_mul_div_backward():
    x = tensor([2.0, 4.0], requires_grad=True)
    y = tensor([3.0, 1.5], requires_grad=True)
    z = (x * y) / 2.0
    z_sum = z.sum()
    z_sum.backward()

    assert mx.allclose(x.grad, y.data / 2.0)
    assert mx.allclose(y.grad, x.data / 2.0)


def test_pow_backward():
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    assert mx.allclose(x.grad, mx.array([2.0, 4.0, 6.0]))


def test_matmul_backward():
    a = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = tensor([[5.0], [6.0]], requires_grad=True)
    c = a.matmul(b)
    loss = c.sum()
    loss.backward()

    assert c.data.shape == (2, 1)
    assert mx.allclose(a.grad, mx.array([[5.0, 6.0], [5.0, 6.0]]))
    assert mx.allclose(b.grad, mx.array([[4.0], [6.0]]))


def test_transpose_backward():
    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.transpose()
    loss = y.sum()
    loss.backward()

    assert y.data.shape == (2, 2)
    assert mx.allclose(x.grad, mx.ones_like(x.data))


def test_sum_and_mean_backward():
    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    s = x.sum()
    m = x.mean()

    s.backward()
    assert mx.allclose(x.grad, mx.ones_like(x.data))

    x.zero_grad()
    m.backward()
    assert mx.allclose(x.grad, mx.ones_like(x.data) * 0.25)

    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    s_axis = x.sum(axis=0)
    upstream = mx.ones_like(s_axis.data)
    s_axis.backward(upstream)
    assert mx.allclose(x.grad, mx.ones_like(x.data))

    x.zero_grad()
    m_axis = x.mean(axis=1)
    m_axis.backward(mx.array([1.0, 2.0]))
    expected = mx.array([[1.0 / 3.0] * 3, [2.0 / 3.0] * 3])
    assert mx.allclose(x.grad, expected)


def test_requires_grad_propagation():
    a = tensor([1.0, 2.0], requires_grad=True)
    b = tensor([3.0, 4.0], requires_grad=False)
    c = a + b
    d = b * 2.0
    assert c.requires_grad
    assert not d.requires_grad


def test_zero_grad_resets():
    x = tensor([1.0, -1.0], requires_grad=True)
    y = x.sum()
    y.backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None


def test_convenience_creators():
    z = zeros((2, 2))
    o = ones((2, 2))
    r = randn((2, 2))
    assert isinstance(z, Tensor) and isinstance(o, Tensor) and isinstance(r, Tensor)
    assert mx.allclose(z.data, mx.zeros((2, 2)))
    assert mx.allclose(o.data, mx.ones((2, 2)))
