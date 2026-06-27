"""Small neural-network building blocks built on the autograd engine."""
from __future__ import annotations

import numpy as np

from src.autograd import Tensor


class MLP:
    """Multilayer perceptron with tanh hidden activations and a linear head."""

    def __init__(self, sizes: list[int], seed: int = 0, out_scale: float = 1.0):
        rng = np.random.default_rng(seed)
        self.layers: list[tuple[Tensor, Tensor]] = []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])
            W = Tensor(rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            b = Tensor(np.zeros((1, sizes[i + 1])))
            self.layers.append((W, b))
        # shrink the output layer so initial dynamics are gentle
        Wo, bo = self.layers[-1]
        Wo.data *= out_scale

    def __call__(self, x: Tensor) -> Tensor:
        h = x
        for k, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if k < len(self.layers) - 1:
                h = h.tanh()
        return h

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer]


class Adam:
    """Adam optimizer over a list of Tensors."""

    def __init__(self, params: list[Tensor], lr: float = 1e-2,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
