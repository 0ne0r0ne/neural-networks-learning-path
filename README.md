# Makemore — Character-Level Language Model

Implementation of a character-level bigram language model built from scratch,
following Andrej Karpathy's "Neural Networks: Zero to Hero" series.

## What's inside

- **Statistical Model**: Counts bigram frequencies across 32,000 names,
  normalizes to probabilities, and samples new names.
- **Neural Network Model**: Single-layer neural network trained with
  gradient descent to learn the same bigram distribution.
- **Loss Function**: Negative Log Likelihood (NLL) used to evaluate
  and minimize prediction error.
- **Techniques**: One-hot encoding, softmax, Laplace smoothing,
  backpropagation via PyTorch autograd.

## Key result

Both models converge to the same NLL (~2.45), proving that a neural
network can learn what a counting table knows — purely through gradient descent.

## Dataset

`names.txt` — 32,033 human first names from [ssa.gov](https://www.ssa.gov/oact/babynames/)

## Based on

Andrej Karpathy — [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
