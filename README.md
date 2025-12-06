# Multi-Regime CFD Surrogate Modeling Using Mixture-of-Experts (MoE)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17843437.svg)](https://doi.org/10.5281/zenodo.17843437)

This repository contains the code and experiments for the paper:

> **“Multi-Regime CFD Surrogate Modeling Using Mixture-of-Experts (MoE):  
> A Unified Framework for Complex Aerodynamic Regimes”**  
> Apichet Janya, Cheetah Research Lab  
> DOI: https://doi.org/10.5281/zenodo.17843437

The goal of this project is to build a **multi-regime surrogate model** for aerodynamic CFD using a **Mixture-of-Experts (MoE)** architecture that:

- Handles **laminar → transitional → turbulent / separated / stalled** regimes,
- Eliminates **error spikes near stall** seen in monolithic surrogates,
- Provides **physically interpretable** gating (experts specializing in attached / transitional / stall regimes),
- Supports **active learning** via expert-disagreement uncertainty.

---

## 🔍 Key Ideas

- **Multi-regime physics**  
  Aerodynamic flows are not globally smooth. They contain distinct regimes (attached, transitional, stall) that are difficult to approximate with a single monolithic model.

- **Mixture-of-Experts surrogate**  
  A gating network learns soft assignments \( g_i(x) \) over expert networks \( f_i(x) \), producing:
  \[
  \hat{y}(x) = \sum_{i=1}^{E} g_i(x)\,f_i(x),
  \]
  allowing each expert to specialize in a coherent flow regime.

- **Synthetic CFD dataset**  
  A controlled generator emulates nonlinear aerodynamic responses as a function of:
  - camber, thickness,
  - Reynolds number,
  - angle of attack (AoA),
  - turbulence-model proxies,
  - mesh-fidelity / noise factors.

- **Interpretability**  
  Gating maps show:
  - **Expert 0** → low AoA / attached flow,  
  - **Expert 1** → transitional regime,  
  - **Expert 2** → high AoA / stall region.

- **Active learning via expert disagreement**  
  Uncertainty is defined as:
  \[
  \sigma^2(x) = \sum_i g_i(x)\left(f_i(x)-\hat{y}(x)\right)^2,
  \]
  which naturally highlights regime boundaries and sparsely sampled regions.

---

## 📁 Repository Structure (suggested)

Your layout may differ, but a typical structure for this project is:

```text
.
├─ notebooks/
│  └─ moe_cfd_experiments.ipynb      # End-to-end experiments (training + plots)
├─ src/
│  ├─ data/
│  │  └─ synthetic_cfd_dataset.py    # Synthetic CFD generator
│  ├─ models/
│  │  ├─ moe_surrogate.py            # MoE surrogate (PyTorch)
│  │  └─ baselines.py                # MLP / RF / SVR / GPR baselines
│  └─ utils/
│     └─ plotting.py                 # Error plots, gating visualizations
├─ scripts/
│  ├─ train_baselines.py             # Train monolithic surrogates
│  ├─ train_moe_pytorch.py           # Train MoE (PyTorch)
│  └─ make_figures.py                # Reproduce paper figures
├─ requirements.txt
└─ README.md
