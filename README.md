# MLXTorch

MLXTorch is a learning-focused deep learning mini-framework that mirrors the spirit of PyTorch while running on Apple Silicon through [MLX](https://github.com/apple/mlx). It is intentionally small so you can read every file, understand how tensors, autograd, layers, optimizers, and training loops work together, and extend it with your own experiments.

## Why Another Torch?

- **Education first** – each subsystem is tiny and self-contained, making it ideal for studying modern DL internals.
- **Apple Silicon native** – powered by MLX arrays, so it runs efficiently on M-series GPUs/NPUs without extra dependencies.
- **PyTorch-like ergonomics** – familiar API surface (`Tensor`, `nn.Module`, `optim`, losses) with only the essentials implemented.

## Project Scope

- `mlxtorch.tensor.Tensor`: tape-based autograd over `mlx.core.array`.
- Elementwise, matrix, and reduction ops with gradient tracking.
- Basic neural network layers (`Linear`, `ReLU`, `Sigmoid`, `Tanh`) built on a lightweight `Module` base.
- Optimizers (`SGD`, `Adam`) that update tensors in place.
- Standard losses (MSE, CrossEntropy) with logits + label support.
- Minimal dataset/dataloader utilities plus a tiny training script that stitches everything together.
- Comprehensive unit tests for every subsystem.

> **Status:** Tensor + autograd foundations are in progress; higher-level modules will land next. The README serves as a north star even before the full implementation is done.

## Installation

```bash
git clone git@github.com:Joker666/MLXTorch.git
cd MLXTorch
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependencies are intentionally light: Python ≥3.10 and MLX ≥0.20.0.

## Quick Start

```python
import mlx.core as mx
from mlxtorch import Tensor

x = Tensor(mx.array([1.0, 2.0, 3.0]), requires_grad=True)
y = (x * 2).sum()
y.backward()

print(x.grad)  # -> mx.array([2., 2., 2.])
```

Once `mlxtorch.nn`, `mlxtorch.optim`, and `mlxtorch.losses` are finalized, you will be able to compose models and training loops in a PyTorch-like style:

```python
from mlxtorch import tensor
from mlxtorch.nn import Linear, ReLU, Module
from mlxtorch.optim import SGD

class TinyNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 4)
        self.fc2 = Linear(4, 1)

    def forward(self, x):
        return self.fc2(ReLU()(self.fc1(x)))

model = TinyNet()
optim = SGD(model.parameters(), lr=0.1)
```

## Repository Layout

```
mlxtorch/
├── tensor.py           # Tensor + autograd core
├── nn/                 # Module base class + layers (planned)
├── optim/              # SGD, Adam (planned)
├── losses.py           # Loss functions (planned)
├── data/               # Dataset + DataLoader utilities (planned)
├── train.py            # Example training loop (planned)
tests/
├── test_tensor.py
├── test_autograd.py    # planned
└── ...                 # more subsystem tests
```

## Roadmap

1. Finalize tensor ops and autograd parity with the original PyTorch-inspired API.
2. Flesh out `mlxtorch.nn` layers and module parameter management.
3. Implement optimizers, losses, and datasets.
4. Deliver a polished end-to-end training demo showcasing gradient descent on a toy dataset.
5. Add grad-check utilities and additional documentation/blog-style walkthroughes.

## Contributing

Contributions focused on clarity, tests, and documentation are especially welcome. Please:

1. Open an issue describing the change.
2. Keep PRs small and well-tested (`pytest` must pass).
3. Add docstrings and comments where the code isn’t self-explanatory.

## License

MIT License – see `LICENSE`. Feel free to learn from or build upon MLXTorch for your own projects.

---

*MLXTorch: learn, tweak, and build deep learning tools the MLX-native way.*

