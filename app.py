"""
Pricing Strategy Simulator - forecast demand and revenue for a price plan you
have never run, using the trained grey-box Universal Differential Equation.

Run: streamlit run app.py   (after: python experiments.py, to train models)
"""
import numpy as np
import streamlit as st

st.set_page_config(page_title="Neural Demand Dynamics", layout="wide")

from pathlib import Path

import matplotlib.pyplot as plt

from src import viz
from src.dynamics import TrueParams, simulate, equilibrium, counterfactual_schedules
from src.evaluate import predict, rmse
from src.persist import load_models

PAR = TrueParams()


@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    if not Path("data/models.npz").exists():
        return None
    return load_models()


models = get_models()
if models is None:
    st.error("No trained models found. Run `python experiments.py` first to train and save them.")
    st.stop()
ude, bb, norm = models

st.title("Neural Demand Dynamics")
st.caption("Counterfactual price-response forecasting with a grey-box Universal "
           "Differential Equation. Simulate the demand and revenue impact of a "
           "pricing strategy the model never saw in training.")

# ----- Sidebar: build a price plan -----
with st.sidebar:
    st.subheader("Pricing strategy")
    _opts = ["Promotion", "Rapid repricing", "Gradual ramp", "Custom"]
    _qp = st.query_params.get("scenario", "Promotion")
    _idx = _opts.index(_qp) if _qp in _opts else 0
    scenario = st.selectbox(
        "Scenario", _opts, index=_idx,
        help="Preset counterfactual schedules, or build your own.",
    )
    T = 60
    presets = counterfactual_schedules(T=T)
    if scenario == "Custom":
        base = st.slider("Base price", 9.0, 13.0, 12.0, 0.1)
        promo_price = st.slider("Promo price", 8.5, 13.0, 9.5, 0.1)
        start = st.slider("Promo start (step)", 0, T - 5, 20)
        length = st.slider("Promo length (steps)", 1, 30, 12)
        price = np.full(T, base)
        price[start:start + length] = promo_price
    else:
        price = presets[scenario].copy()

    st.divider()
    st.subheader("What the model learned")
    st.metric("Recovered inertia alpha", f"{ude.alpha_value:.3f}", help="True value: 0.25")
    show_bb = st.checkbox("Compare black-box neural ODE", value=True)
    show_truth = st.checkbox("Show ground-truth simulator", value=True)

# ----- Forecast -----
D0 = float(equilibrium(np.array(price[0]), PAR))
clean = simulate(price, PAR, D0)
ude_pred = predict(ude, price.reshape(1, -1), np.array([D0])).ravel()
bb_pred = predict(bb, price.reshape(1, -1), np.array([D0])).ravel()

ude_rev = float(np.sum(ude_pred * price))
true_rev = float(np.sum(clean * price))
bb_rev = float(np.sum(bb_pred * price))

# ----- KPIs -----
c = st.columns(4)
c[0].metric("Forecast revenue (UDE)", f"${ude_rev:,.0f}")
c[1].metric("Peak demand (UDE)", f"{ude_pred.max():.0f}")
c[2].metric("Avg price", f"${price.mean():.2f}")
if show_truth:
    err_pct = abs(ude_rev - true_rev) / true_rev * 100
    c[3].metric("Revenue error vs truth", f"{err_pct:.1f}%")

# ----- Plot -----
viz.apply_theme()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                               gridspec_kw={"height_ratios": [1, 2]})
ax1.plot(price, color=viz.PALETTE["price"])
ax1.set_ylabel("Price")
ax1.set_title(f"Price plan: {scenario}")
if show_truth:
    ax2.plot(clean, color=viz.PALETTE["truth"], linewidth=2.4, label="True demand")
ax2.plot(ude_pred, color=viz.PALETTE["ude"], ls="--",
         label="UDE forecast" + (f" (RMSE {rmse(ude_pred, clean):.2f})" if show_truth else ""))
if show_bb:
    ax2.plot(bb_pred, color=viz.PALETTE["blackbox"], ls="--",
             label="Black-box forecast" + (f" (RMSE {rmse(bb_pred, clean):.2f})" if show_truth else ""))
ax2.set_ylabel("Demand")
ax2.set_xlabel("Time step")
ax2.legend()
fig.tight_layout()
st.pyplot(fig)

with st.expander("How this works", expanded=False):
    st.markdown(
        "- The model is a **Universal Differential Equation**: a known relaxation "
        "law `dD/dt = alpha * (f(p) - D)` with a neural network for the unknown "
        "price-equilibrium curve `f(p)`.\n"
        "- It was trained only on **slowly varying** historical prices, yet it "
        "forecasts demand under **new** pricing strategies because it learned the "
        "governing dynamics, not surface correlations.\n"
        "- The black-box neural ODE fits the same training data but breaks under "
        "rapid repricing - it never saw demand far from equilibrium."
    )

# ----- Recovered curve -----
with st.expander("Recovered price-equilibrium curve", expanded=False):
    grid = np.linspace(9, 13, 60)
    from src.autograd import Tensor
    learned = ude.equilibrium(Tensor(grid.reshape(-1, 1), requires_grad=False)).data.ravel()
    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, equilibrium(grid, PAR), color=viz.PALETTE["truth"], lw=2.2, label="True Q*(p)")
    ax.plot(grid, learned, color=viz.PALETTE["ude"], ls="--", label="UDE recovered")
    ax.set_xlabel("Price"); ax.set_ylabel("Equilibrium demand"); ax.legend()
    st.pyplot(fig2)
