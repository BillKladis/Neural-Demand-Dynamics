"""Training loop shared by both neural-ODE models."""
from __future__ import annotations

import numpy as np

from src.models import mse_loss
from src.nn import Adam


def train(model, prices: np.ndarray, demands: np.ndarray, epochs: int = 1500,
          lr: float = 1e-2, dt: float = 1.0, verbose: bool = True,
          log_every: int = 250) -> list[float]:
    """Fit a model to observed trajectories by trajectory matching.

    Returns the per-epoch training-loss history.
    """
    opt = Adam(model.parameters(), lr=lr)
    D0 = demands[:, 0].copy()
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        pred = model.rollout(prices, D0, dt=dt)
        loss = mse_loss(pred, demands)
        loss.backward()
        opt.step()
        history.append(float(loss.data))
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            extra = f" alpha={model.alpha_value:.3f}" if hasattr(model, "alpha_value") else ""
            print(f"  [{model.short:8s}] epoch {ep:4d}  loss={loss.data:.4f}{extra}")
    return history
