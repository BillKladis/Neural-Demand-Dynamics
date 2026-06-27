"""Save / load trained model parameters (NumPy npz)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models import BlackBoxODE, Norm, UDEModel


def save_models(ude: UDEModel, bb: BlackBoxODE, norm: Norm,
                path: str = "data/models.npz") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "norm": np.array([norm.p_mean, norm.p_std, norm.d_mean, norm.d_std]),
        "ude_hidden": np.array([ude.f.layers[0][0].data.shape[1]]),
        "bb_hidden": np.array([bb.g.layers[0][0].data.shape[1]]),
    }
    for i, p in enumerate(ude.parameters()):
        blob[f"ude_{i}"] = p.data
    for i, p in enumerate(bb.parameters()):
        blob[f"bb_{i}"] = p.data
    np.savez(path, **blob)


def load_models(path: str = "data/models.npz"):
    d = np.load(path)
    pm, ps, dm, ds = d["norm"]
    norm = Norm(float(pm), float(ps), float(dm), float(ds))
    ude = UDEModel(norm, hidden=int(d["ude_hidden"][0]))
    bb = BlackBoxODE(norm, hidden=int(d["bb_hidden"][0]))
    for i, p in enumerate(ude.parameters()):
        p.data = d[f"ude_{i}"]
    for i, p in enumerate(bb.parameters()):
        p.data = d[f"bb_{i}"]
    return ude, bb, norm
