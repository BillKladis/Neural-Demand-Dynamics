"""
Two neural ordinary-differential-equation models, trained identically.

Both learn dD/dt from observed trajectories by backpropagating through an
unrolled RK4 solver (discretize-then-optimize). They differ only in the
structural prior placed on the right-hand side:

  UDEModel  (grey-box):  dD/dt = alpha * (f_theta(p) - D)
      The first-order relaxation structure is known; the neural network learns
      only the price-equilibrium curve f_theta(p), and alpha is a scalar param.

  BlackBoxODE (black-box):  dD/dt = g_theta(D, p)
      The entire right-hand side is an unconstrained neural network.

The shared experiment shows the structural prior is what enables
generalization to counterfactual price schedules.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.autograd import Tensor, concat1
from src.nn import MLP


@dataclass
class Norm:
    p_mean: float
    p_std: float
    d_mean: float
    d_std: float

    @classmethod
    def from_data(cls, prices: np.ndarray, demands: np.ndarray) -> "Norm":
        return cls(float(prices.mean()), float(prices.std()),
                   float(demands.mean()), float(demands.std()))


def _rk4(rhs, D: Tensor, dt: float) -> Tensor:
    k1 = rhs(D)
    k2 = rhs(D + 0.5 * dt * k1)
    k3 = rhs(D + 0.5 * dt * k2)
    k4 = rhs(D + dt * k3)
    return D + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)


class UDEModel:
    """Grey-box: known relaxation structure, learned equilibrium curve."""

    short = "ude"
    name = "Universal DE (grey-box)"

    def __init__(self, norm: Norm, hidden: int = 16, seed: int = 0):
        self.norm = norm
        # f_theta maps normalized price -> demand (linear head, modest init)
        self.f = MLP([1, hidden, hidden, 1], seed=seed, out_scale=0.1)
        self.raw_alpha = Tensor(np.array([[-1.0]]))  # softplus(-1) ~ 0.31

    def parameters(self):
        return self.f.parameters() + [self.raw_alpha]

    def equilibrium(self, p: Tensor) -> Tensor:
        p_n = (p + (-self.norm.p_mean)) * (1.0 / self.norm.p_std)
        return self.f(p_n) * self.norm.d_std + self.norm.d_mean

    def rollout(self, price: np.ndarray, D0: np.ndarray, dt: float = 1.0) -> Tensor:
        """Integrate from D0 over the price schedule. Returns [B, T] predictions."""
        B, T = price.shape
        D = Tensor(D0.reshape(B, 1), requires_grad=False)
        alpha = self.raw_alpha.softplus()
        preds = []
        for t in range(T):
            p_t = Tensor(price[:, t].reshape(B, 1), requires_grad=False)
            f_eq = self.equilibrium(p_t)  # depends on price only

            def rhs(D_, f_eq=f_eq, alpha=alpha):
                return alpha * (f_eq - D_)

            preds.append(D)
            D = _rk4(rhs, D, dt)
        return _stack_time(preds)

    @property
    def alpha_value(self) -> float:
        return float(self.raw_alpha.softplus().data.ravel()[0])


class BlackBoxODE:
    """Black-box: the whole right-hand side is a neural network of (D, p)."""

    short = "blackbox"
    name = "Black-box neural ODE"

    def __init__(self, norm: Norm, hidden: int = 24, seed: int = 0):
        self.norm = norm
        self.g = MLP([2, hidden, hidden, 1], seed=seed, out_scale=0.1)

    def parameters(self):
        return self.g.parameters()

    def rollout(self, price: np.ndarray, D0: np.ndarray, dt: float = 1.0) -> Tensor:
        B, T = price.shape
        D = Tensor(D0.reshape(B, 1), requires_grad=False)
        preds = []
        for t in range(T):
            p_n = Tensor(((price[:, t].reshape(B, 1) - self.norm.p_mean)
                          / self.norm.p_std), requires_grad=False)

            def rhs(D_, p_n=p_n):
                D_n = (D_ + (-self.norm.d_mean)) * (1.0 / self.norm.d_std)
                r = self.g(concat1(D_n, p_n))
                return r * self.norm.d_std   # back to demand-rate units

            preds.append(D)
            D = _rk4(rhs, D, dt)
        return _stack_time(preds)


def _stack_time(preds: list[Tensor]) -> Tensor:
    """Stack a list of [B,1] tensors into one [B,T] tensor (autograd-aware)."""
    B = preds[0].data.shape[0]
    T = len(preds)
    out = Tensor(np.concatenate([p.data for p in preds], axis=1), tuple(preds))

    def _backward():
        for t, p in enumerate(preds):
            if p.requires_grad:
                p.grad = p.grad + out.grad[:, t:t + 1]

    out._backward = _backward
    return out


def mse_loss(pred: Tensor, target: np.ndarray) -> Tensor:
    diff = pred - Tensor(target, requires_grad=False)
    return (diff * diff).mean()
