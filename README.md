# Neural Networks: Zero to Hero — Personal Implementations

My implementations following Andrej Karpathy's "Neural Networks: Zero to Hero" series.
Built from scratch, without copy-pasting, focusing on deep understanding over speed.

---

## 1. Micrograd — `micrograd_selfcoding.py`

A tiny scalar-valued autograd engine built from scratch.

- **Engine**: `Value` class with forward and backward pass
- **Layers**: `Neuron`, `Layer`, `MLP` classes
- **Backpropagation**: Manual chain rule implementation
- **Activation**: Tanh

This replicates the core mechanism behind PyTorch's autograd in ~100 lines.

---

## 2. Makemore — `main.ipynb`

A character-level bigram language model that generates human-like names.

- **Statistical Model**: Counts bigram frequencies across 32,000 names,
  normalizes to probabilities, and samples new names.
- **Neural Network Model**: Single-layer neural network trained with
  gradient descent to learn the same bigram distribution.
- **Loss Function**: Negative Log Likelihood (NLL) used to evaluate
  and minimize prediction error.
- **Techniques**: One-hot encoding, softmax, Laplace smoothing,
  backpropagation via PyTorch autograd.

**Key result:** Both models converge to the same NLL (~2.45), proving that a neural
network can learn what a counting table knows — purely through gradient descent.

### Dataset
`names.txt` — 32,033 human first names from [ssa.gov](https://www.ssa.gov/oact/babynames/)

---

## Based on

Andrej Karpathy — [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
