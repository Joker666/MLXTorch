**Project**: MLXTorch
**Goal**: Implement a minimal deep learning framework inspired by PyTorch. It must run on Apple Silicon using the MLX framework. It should support tensor operations, autograd, simple neural network layers, optimizers, loss functions, basic datasets, and a small training loop.

## Shared Conventions

* All code must use Python with MLX.
* All tensors are `mlx.core.array` objects.
* Autograd must follow a tape style.
* API surface should look similar to PyTorch, but smaller and simpler.
* Keep each file small and readable.
* Add docstrings to all public functions and classes.
* Use type hints.
* Use unit tests for each component.
* Avoid unnecessary abstractions.

---

# Agent 1 Tensor Architect

**Responsibility**
Design and implement the foundational Tensor class. It wraps `mlx.core.array`, tracks gradients, and stores references to backward functions.

**Tasks**

1. Create `tensor.py`.
2. Implement `class Tensor` with fields: `data`, `grad`, `requires_grad`, `_backward`, `_prev`.
3. Implement elementwise ops: add, sub, mul, div, pow.
4. Implement matrix ops: matmul, transpose.
5. Implement reduction ops: sum, mean.
6. Connect ops to autograd using function closures.
7. Implement `zero_grad()`.
8. Add convenience constructors: `tensor`, `zeros`, `ones`, `randn`.
9. Add unit tests for all ops.

**Deliverables**

* `tensor.py`
* `tests/test_tensor.py`

---

# Agent 2 Autograd Weaver

**Responsibility**
Build the autograd engine for Tensor. Execute backward passes, handle topological sorting, and provide gradient propagation support.

**Tasks**

1. Implement a topological sort for the computation graph.
2. Add a `backward()` method on Tensor that accumulates gradients.
3. Support broadcasting gradients.
4. Ensure gradient accumulation is correct.
5. Add a simple check to prevent backward on non scalar outputs without specifying grad.
6. Add unit tests for autograd correctness.

**Deliverables**

* Updates to `tensor.py`
* `tests/test_autograd.py`

---

# Agent 3 NN Layer Builder

**Responsibility**
Implement a tiny neural network module system inspired by PyTorch's `nn`.

**Tasks**

1. Create `nn/module.py` with a base `Module` class.
2. Implement `parameters()` and `zero_grad()`.
3. Create common layers in `nn/layers.py`:

   * Linear
   * ReLU
   * Sigmoid
   * Tanh
4. Ensure layers use MLXTorch tensors internally.
5. Add unit tests.

**Deliverables**

* `nn/` folder with `module.py` and `layers.py`
* `tests/test_layers.py`

---

# Agent 4 Optimizer Smith

**Responsibility**
Implement a small set of optimizers similar to PyTorch.

**Tasks**

1. Create `optim/sgd.py` with basic SGD and momentum.
2. Create `optim/adam.py` with minimal Adam implementation.
3. Optimizers accept a list of parameters and update them in place.
4. Add learning rate, weight decay, and beta configs.
5. Add unit tests.

**Deliverables**

* `optim/` folder with optimizer implementations
* `tests/test_optim.py`

---

# Agent 5 Loss Alchemist

**Responsibility**
Implement a small standard set of loss functions.

**Tasks**

1. Create `losses.py`.
2. Implement MSELoss.
3. Implement CrossEntropyLoss using logits and integer class labels.
4. Add unit tests.

**Deliverables**

* `losses.py`
* `tests/test_losses.py`

---

# Agent 6 Data Artisan

**Responsibility**
Implement a simple dataset and dataloader system.

**Tasks**

1. Create `data/dataset.py` with a `Dataset` base class.
2. Create `data/dataloader.py` with a basic iterator.
3. Add shuffle and batch size.
4. Work with numpy arrays and produce MLXTorch tensors.
5. Add a toy dataset for testing.

**Deliverables**

* `data/` folder
* `tests/test_data.py`

---

# Agent 7 Train Loop Conductor

**Responsibility**
Assemble a small training loop and example script to demonstrate MLXTorch.

**Tasks**

1. Create `train.py`.
2. Write a minimal training workflow:

   * Build model
   * Load dataset
   * Run forward
   * Compute loss
   * Backward
   * Step optimizer
3. Add an example model such as a tiny MLP for MNIST or a synthetic dataset.
4. Add comments so new users can follow the logic.

**Deliverables**

* `train.py`

---

# Agent 8 Packaging Scribe

**Responsibility**
Prepare the project so it can be installed and used like a micro library.

**Tasks**

1. Add `__init__.py` files.
2. Write `setup.py` or pyproject.
3. Include a README.
4. Include installation instructions and MLXTorch philosophy.
5. Add version string.
6. Add Makefile for tests.

**Deliverables**

* Packaging files
* README
* Version tag

---

# Agent 9 QA Sentinel

**Responsibility**
Verify end to end correctness.

**Tasks**

1. Run full test suite.
2. Verify gradients against numerical gradients for a few ops.
3. Verify training loop reduces loss.
4. Report issues and request fixes.

**Deliverables**

* QA report
* Issues list
