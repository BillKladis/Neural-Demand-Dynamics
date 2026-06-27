"""
Run the full Neural Demand Dynamics study and produce all figures + metrics.

Pipeline:
  1. Generate slowly varying training trajectories from the true ODE.
  2. Train both models (grey-box UDE and black-box neural ODE) over several seeds.
  3. Evaluate on held-out counterfactual price schedules (vs the clean truth).
  4. Recover the equilibrium curve Q*(p) and the inertia alpha.
  5. Save figures to figures/ and metrics to figures/metrics.json.

Single command:  python experiments.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import viz
from src.dynamics import (TrueParams, counterfactual_schedules, equilibrium,
                          make_training_set, realize)
from src.evaluate import curve_recovery_error, predict, r2, rmse
from src.models import BlackBoxODE, Norm, UDEModel
from src.train import train
import matplotlib.pyplot as plt

FIG = Path("figures")
SEEDS = [1, 2, 3]
EPOCHS = 1500
LR = 1e-2


def main() -> None:
    FIG.mkdir(exist_ok=True)
    par = TrueParams()
    prices, demands = make_training_set(par, n_traj=12, T=60, seed=0)
    norm = Norm.from_data(prices, demands)
    schedules = counterfactual_schedules(T=60)
    realized = {name: realize(p, par, seed=99) for name, p in schedules.items()}
    grid = np.linspace(prices.min(), prices.max(), 60)

    # ---- train over seeds, collect metrics ----
    per_seed = {"ude": [], "blackbox": []}
    keep = {}  # seed -> (ude, bb, histories) for the representative seed
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        ude = UDEModel(norm, seed=seed)
        h_ude = train(ude, prices, demands, epochs=EPOCHS, lr=LR, verbose=False)
        bb = BlackBoxODE(norm, seed=seed)
        h_bb = train(bb, prices, demands, epochs=EPOCHS, lr=LR, verbose=False)

        row_u = {"train_rmse": rmse(predict(ude, prices, demands[:, 0].copy()), demands),
                 "alpha": ude.alpha_value,
                 "curve_relL2": curve_recovery_error(ude, par, grid)}
        row_b = {"train_rmse": rmse(predict(bb, prices, demands[:, 0].copy()), demands)}
        for name, (p, _, clean) in realized.items():
            P1, D0 = p.reshape(1, -1), np.array([clean[0]])
            row_u[name] = rmse(predict(ude, P1, D0).ravel(), clean)
            row_b[name] = rmse(predict(bb, P1, D0).ravel(), clean)
        per_seed["ude"].append(row_u)
        per_seed["blackbox"].append(row_b)
        print(f"  ude  train={row_u['train_rmse']:.2f} alpha={row_u['alpha']:.3f} "
              f"curve={row_u['curve_relL2']:.3f}")
        print(f"  bb   train={row_b['train_rmse']:.2f}")
        for name in realized:
            print(f"    CF {name:16s} ude={row_u[name]:.2f}  bb={row_b[name]:.2f}")
        if seed == SEEDS[0]:
            keep = dict(ude=ude, bb=bb, h_ude=h_ude, h_bb=h_bb)
            from src.persist import save_models
            save_models(ude, bb, norm)
            print("  saved representative models -> data/models.npz")

    metrics = _aggregate(per_seed, realized, par)
    (FIG / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== aggregated (mean +/- std over seeds) ===")
    print(json.dumps(metrics, indent=2))

    # ---- figures (representative seed) ----
    _fig_dynamics(par, prices, demands)
    _fig_loss(keep["h_ude"], keep["h_bb"])
    _fig_counterfactual(keep["ude"], keep["bb"], realized["Rapid repricing"], par)
    _fig_curve(keep["ude"], par, grid, norm)
    _fig_summary(metrics, realized)
    print(f"\nSaved figures to {FIG}/")


def _aggregate(per_seed, realized, par):
    def ms(vals):
        a = np.array(vals, float)
        return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4)}

    out = {"true_alpha": par.alpha, "seeds": len(per_seed["ude"]), "ude": {}, "blackbox": {}}
    out["ude"]["train_rmse"] = ms([r["train_rmse"] for r in per_seed["ude"]])
    out["ude"]["alpha"] = ms([r["alpha"] for r in per_seed["ude"]])
    out["ude"]["curve_relL2"] = ms([r["curve_relL2"] for r in per_seed["ude"]])
    out["blackbox"]["train_rmse"] = ms([r["train_rmse"] for r in per_seed["blackbox"]])
    for name in realized:
        out["ude"][name] = ms([r[name] for r in per_seed["ude"]])
        out["blackbox"][name] = ms([r[name] for r in per_seed["blackbox"]])
    return out


# --------------------------------------------------------------------------- figures
def _fig_dynamics(par, prices, demands):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.6]})
    for i in range(min(6, len(prices))):
        ax1.plot(prices[i], color=viz.PALETTE["price"], alpha=0.5)
        ax2.plot(demands[i], color=viz.PALETTE["ude"], alpha=0.5)
    ax1.set_ylabel("Price")
    ax1.set_title("Training data: slowly varying prices, demand near equilibrium")
    ax2.set_ylabel("Demand")
    ax2.set_xlabel("Time step")
    fig.tight_layout()
    fig.savefig(FIG / "01_training_dynamics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_loss(h_ude, h_bb):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(h_ude, color=viz.PALETTE["ude"], label="UDE (grey-box)")
    ax.semilogy(h_bb, color=viz.PALETTE["blackbox"], label="Black-box neural ODE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE (log scale)")
    ax.set_title("Training convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_training_loss.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_counterfactual(ude, bb, realized, par):
    p, noisy, clean = realized
    P1, D0 = p.reshape(1, -1), np.array([clean[0]])
    u = predict(ude, P1, D0).ravel()
    b = predict(bb, P1, D0).ravel()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2]})
    ax1.plot(p, color=viz.PALETTE["price"])
    ax1.set_ylabel("Price")
    ax1.set_title("Counterfactual: rapid repricing never seen in training")
    ax2.plot(clean, color=viz.PALETTE["truth"], label="True demand", linewidth=2.4)
    ax2.plot(u, color=viz.PALETTE["ude"], ls="--", label=f"UDE (RMSE {rmse(u,clean):.2f})")
    ax2.plot(b, color=viz.PALETTE["blackbox"], ls="--",
             label=f"Black-box (RMSE {rmse(b,clean):.2f})")
    ax2.set_ylabel("Demand")
    ax2.set_xlabel("Time step")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIG / "03_counterfactual_forecast.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_curve(ude, par, grid, norm):
    true = equilibrium(grid, par)
    learned = ude.equilibrium(_col(grid)).data.ravel()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, true, color=viz.PALETTE["truth"], linewidth=2.4, label="True Q*(p)")
    ax.plot(grid, learned, color=viz.PALETTE["ude"], ls="--",
            label="UDE recovered f_theta(p)")
    ax.axvspan(norm.p_mean - norm.p_std, norm.p_mean + norm.p_std,
               color=viz.PALETTE["muted"], alpha=0.18, label="+/-1 sd training prices")
    ax.set_xlabel("Price p")
    ax.set_ylabel("Equilibrium demand")
    ax.set_title("Recovered price-equilibrium curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "04_recovered_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_summary(metrics, realized):
    names = list(realized.keys())
    ude_vals = [metrics["ude"][n]["mean"] for n in names]
    ude_err = [metrics["ude"][n]["std"] for n in names]
    bb_vals = [metrics["blackbox"][n]["mean"] for n in names]
    bb_err = [metrics["blackbox"][n]["std"] for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w / 2, ude_vals, w, yerr=ude_err, capsize=4,
           color=viz.PALETTE["ude"], label="UDE (grey-box)")
    ax.bar(x + w / 2, bb_vals, w, yerr=bb_err, capsize=4,
           color=viz.PALETTE["blackbox"], label="Black-box neural ODE")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Counterfactual RMSE (vs true demand)")
    ax.set_title(f"Counterfactual generalization ({metrics['seeds']} seeds, mean +/- std)")
    ax.legend()
    for xi, v in zip(x - w / 2, ude_vals):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for xi, v in zip(x + w / 2, bb_vals):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "05_generalization_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _col(x):
    from src.autograd import Tensor
    return Tensor(np.asarray(x, float).reshape(-1, 1), requires_grad=False)


if __name__ == "__main__":
    main()
