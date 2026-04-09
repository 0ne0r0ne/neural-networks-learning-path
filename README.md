# simple-autograd

A scalar-valued autograd engine and a basic neural network library.

I already studied the calculus and ML theory behind these concepts, so I built this to bridge the gap between the math and the actual Python implementation. This repo shows how backpropagation and the chain rule work in code without using heavy frameworks.

## Quick Look
- **engine.py**: The core `Value` class that handles the math and calculates gradients.
- **nn.py**: A simple MLP (Multi-Layer Perceptron) built on top of the engine.

---
*Inspired by Andrej Karpathy's "Neural Networks: Zero to Hero" series (https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).*
