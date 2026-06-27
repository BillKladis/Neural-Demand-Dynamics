"""
Ground-truth demand dynamics and synthetic data generation.

The "true" world is a first-order delay system: demand relaxes toward a
price-dependent equilibrium with inertia (habit formation, word of mouth):

    dD/dt = alpha * (Q*(p(t)) - D(t))

where the equilibrium demand curve Q*(p) is a saturating sigmoid in price:

    Q*(p) = D_max * sigmoid(-k * (p - p0))

Two things are unknown to the learner: the inertia alpha and the nonlinear
curve Q*. Training schedules vary price *slowly*, so demand stays near
equilibrium. The counterfactual test schedule contains an abrupt promotion,
driving demand far from equilibrium - a region of state space the slowly
varying training data never visits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrueParams:
    alpha: float = 0.25      # inertia / relaxation rate
    D_max: float = 100.0     # market saturation
    k: float = 0.65          # price sensitivity (curve steepness)
    p0: float = 11.0         # price at which equilibrium demand is half of D_max
    noise: float = 1.5       # observation noise (std, in demand units)


def equilibrium(p: np.ndarray, par: TrueParams) -> np.ndarray:
    """Q*(p): the true equilibrium demand curve."""
    return par.D_max / (1.0 + np.exp(par.k * (p - par.p0)))


def _rk4_step(D, p, dt, par):
    def f(D_):
        return par.alpha * (equilibrium(np.array(p), par) - D_)
    k1 = f(D)
    k2 = f(D + 0.5 * dt * k1)
    k3 = f(D + 0.5 * dt * k2)
    k4 = f(D + dt * k3)
    return D + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(price_schedule: np.ndarray, par: TrueParams, D0: float,
             dt: float = 1.0, substeps: int = 4) -> np.ndarray:
    """Integrate the true ODE under a piecewise-constant price schedule (ZOH).

    Uses finer substeps internally so the data is not trivially reproducible by
    a single coarse RK4 step (a realistic discretization gap for the learner).
    """
    T = len(price_schedule)
    D = np.full((), float(D0))
    out = np.empty(T)
    h = dt / substeps
    for t in range(T):
        out[t] = D
        for _ in range(substeps):
            D = _rk4_step(D, price_schedule[t], h, par)
    return out


def _slow_price_walk(rng, T, lo, hi, hold=6):
    """Piecewise-constant price that changes gently every `hold` steps."""
    p = np.empty(T)
    level = rng.uniform(lo, hi)
    for t in range(T):
        if t % hold == 0 and t > 0:
            level = np.clip(level + rng.normal(0, 0.7), lo, hi)
        p[t] = level
    return p


def make_training_set(par: TrueParams, n_traj: int = 12, T: int = 60,
                      price_lo: float = 9.0, price_hi: float = 13.0,
                      seed: int = 0):
    """Slowly varying price schedules -> demand stays near equilibrium."""
    rng = np.random.default_rng(seed)
    prices, demands = [], []
    for _ in range(n_traj):
        p = _slow_price_walk(rng, T, price_lo, price_hi)
        D0 = equilibrium(np.array(p[0]), par) + rng.normal(0, 2.0)
        clean = simulate(p, par, D0)
        noisy = clean + rng.normal(0, par.noise, size=T)
        prices.append(p)
        demands.append(noisy)
    return np.array(prices), np.array(demands)


def counterfactual_schedules(T: int = 60) -> dict[str, np.ndarray]:
    """Price schedules with shapes never seen in the slow training data."""
    # 1) Realistic abrupt promotion: high, deep promo window, partial recovery.
    promo = np.full(T, 12.5)
    promo[20:35] = 9.2
    promo[35:] = 11.5

    # 2) Rapid repricing stress test: steady, then aggressive square-wave
    #    repricing (demand never settles -> large disequilibrium), then settle.
    rapid = np.full(T, 12.0)
    for t in range(15, 45):
        rapid[t] = 13.0 if ((t - 15) // 3) % 2 == 0 else 9.0
    rapid[45:] = 10.5

    # 3) Gradual triangular ramp (slow -> demand tracks equilibrium closely).
    ramp = np.concatenate([np.linspace(13, 9, T // 2), np.linspace(9, 13, T - T // 2)])

    return {"Promotion": promo, "Rapid repricing": rapid, "Gradual ramp": ramp}


def realize(p: np.ndarray, par: TrueParams, seed: int = 99):
    """Integrate a price schedule and return (price, noisy_demand, clean_demand)."""
    rng = np.random.default_rng(seed)
    D0 = equilibrium(np.array(p[0]), par)
    clean = simulate(p, par, D0)
    noisy = clean + rng.normal(0, par.noise, size=len(p))
    return p, noisy, clean
