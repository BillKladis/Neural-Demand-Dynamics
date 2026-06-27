"""Evaluation metrics for the demand-dynamics models."""
from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def predict(model, prices: np.ndarray, D0: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Run a forward rollout and return a plain NumPy [B, T] array."""
    return model.rollout(prices, D0, dt=dt).data


def curve_recovery_error(model, par, price_grid: np.ndarray) -> float:
    """Relative L2 error between learned and true equilibrium curve Q*(p)."""
    from src.dynamics import equilibrium
    true = equilibrium(price_grid, par)
    learned = model.equilibrium(_as_col(price_grid)).data.ravel()
    return float(np.linalg.norm(learned - true) / np.linalg.norm(true))


def _as_col(x: np.ndarray):
    from src.autograd import Tensor
    return Tensor(np.asarray(x, dtype=float).reshape(-1, 1), requires_grad=False)
