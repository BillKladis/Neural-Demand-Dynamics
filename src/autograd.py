"""
A minimal reverse-mode automatic differentiation engine (NumPy only).

Just enough to backpropagate through an unrolled ODE solver and a small MLP.
Tensors form a computation graph; `backward()` performs reverse-mode autodiff
over a topological order. Broadcasting is handled by summing gradients back to
each operand's shape. Correctness is verified by finite-difference gradient
checks in `tests/` and at the bottom of this module.
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` so its shape matches `shape` (reverse of NumPy broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev")

    def __init__(self, data, _children=(), requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)

    # ----- elementwise / linear ops -----
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad = other.grad + _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad = other.grad + _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        return out

    # ----- nonlinearities -----
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def softplus(self):
        # numerically stable softplus and its sigmoid derivative
        x = self.data
        sp = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        out = Tensor(sp, (self,))
        sig = 1.0 / (1.0 + np.exp(-x))

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + sig * out.grad

        out._backward = _backward
        return out

    # ----- reductions -----
    def sum(self):
        out = Tensor(self.data.sum(), (self,))

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(), (self,))
        n = self.data.size

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + np.ones_like(self.data) * (out.grad / n)

        out._backward = _backward
        return out

    # ----- sugar -----
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def backward(self):
        # Iterative post-order DFS so deep graphs (long ODE unrolls) don't
        # blow Python's recursion limit.
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            v, processed = stack.pop()
            if processed:
                topo.append(v)
                continue
            if id(v) in visited:
                continue
            visited.add(id(v))
            stack.append((v, True))
            for child in v._prev:
                if id(child) not in visited:
                    stack.append((child, False))

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def concat1(a: "Tensor", b: "Tensor") -> "Tensor":
    """Concatenate two [B, .] tensors along axis 1."""
    out = Tensor(np.concatenate([a.data, b.data], axis=1), (a, b))
    split = a.data.shape[1]

    def _backward():
        if a.requires_grad:
            a.grad = a.grad + out.grad[:, :split]
        if b.requires_grad:
            b.grad = b.grad + out.grad[:, split:]

    out._backward = _backward
    return out


def gradient_check(fn, params, eps: float = 1e-6, tol: float = 1e-4) -> float:
    """Compare autodiff gradients to central finite differences. Returns max rel error."""
    loss = fn()
    for p in params:
        p.zero_grad()
    loss.backward()
    analytic = [p.grad.copy() for p in params]

    max_rel = 0.0
    for pi, p in enumerate(params):
        flat = p.data.ravel()
        for idx in range(flat.size):
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = fn().data
            flat[idx] = orig - eps
            lm = fn().data
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[pi].ravel()[idx]
            denom = max(1e-8, abs(num) + abs(ana))
            max_rel = max(max_rel, abs(num - ana) / denom)
    return max_rel
