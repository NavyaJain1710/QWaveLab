# ⚛️ QWaveLab: Time Evolution of a Quantum Wave Packet

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/test_physics.py)

> **Problem 7:** Numerically solve the time-dependent Schrödinger equation for a Gaussian wave packet in a one-dimensional potential. Investigate its propagation, spreading, reflection, and transmission when it encounters a potential barrier. Visualize the time evolution of the probability density and discuss the role of quantum interference.

---

## 🌟 Overview

**QWaveLab** is an interactive quantum physics laboratory built with Streamlit, NumPy, and SciPy. It provides a real-time numerical simulation of the 1-D Time-Dependent Schrödinger Equation (TDSE) using a strictly unitary **Crank–Nicolson** finite-difference solver.

Every plot, percentage, and curve is computed live from the discrete wave function—no hard-coded curves or canned data.

---

## 🔬 Physics & Mathematical Formulation

### 1. Time-Dependent Schrödinger Equation (TDSE)
$$i\hbar \frac{\partial \psi(x,t)}{\partial t} = \hat{H}\psi(x,t) = \left[ -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x) \right]\psi(x,t)$$

### 2. Initial Gaussian Wave Packet
$$\psi(x, 0) = \left(2\pi\sigma^2\right)^{-1/4} \exp\left(-\frac{(x - x_0)^2}{4\sigma^2}\right) \exp(i k_0 x)$$
* **Position Uncertainty:** $\Delta x = \sigma$
* **Momentum Uncertainty:** $\Delta p = \frac{\hbar}{2\sigma}$
* **Minimum Uncertainty State:** $\Delta x \cdot \Delta p = \frac{\hbar}{2}$
* **Group Velocity:** $v_g = \frac{\hbar k_0}{m}$

### 3. Crank–Nicolson Unitary Time-Stepping
$$\left(I + \frac{i\Delta t}{2\hbar}H\right)\psi^{n+1} = \left(I - \frac{i\Delta t}{2\hbar}H\right)\psi^n$$
* **Unconditionally Stable & Unitary:** Preserves probability $\int |\psi|^2 dx = 1$ to floating-point precision ($\sim 10^{-13}$).
* **Sparse LU Factorization:** Pre-factorizes $(I + \frac{i\Delta t}{2\hbar}H)$ once via `scipy.sparse.linalg.factorized` for high performance.

### 4. Dynamic Reflection ($R$), Transmission ($T$), and Barrier Penetration ($B$)
Partitioning the spatial grid around barrier $[a, b]$:
$$R(t) = \int_{-\infty}^{a} |\psi(x,t)|^2 dx, \quad B(t) = \int_{a}^{b} |\psi(x,t)|^2 dx, \quad T(t) = \int_{b}^{\infty} |\psi(x,t)|^2 dx$$
$$\text{At all times: } R(t) + B(t) + T(t) = 1.0000000000$$

### 5. Quantum Interference Isolation
Using spatial Fourier transforms, $\psi(x,t)$ is decomposed into right-moving ($p > 0$) and left-moving ($p < 0$) components:
$$\psi(x,t) = \psi_R(x,t) + \psi_L(x,t)$$
$$|\psi(x,t)|^2 = \underbrace{|\psi_R|^2}_{\text{Incident/Transmitted}} + \underbrace{|\psi_L|^2}_{\text{Reflected}} + \underbrace{2\,\text{Re}\left(\psi_R^* \psi_L\right)}_{\text{Interference Fringes}}$$
* **Key Insight:** $\int 2\,\text{Re}\left(\psi_R^* \psi_L\right) dx \approx 0$ — interference redistributes probability in space without creating or destroying probability.

---

## 🖥️ Application Modules & Pages

1. **🏠 Home – Quantum Laboratory:** Project overview, governing equations, live preview, and physical parameters.
2. **🌊 Wave Packet Simulator:** Full interactive simulation with real-time Play/Pause/Scrub controls, phase visualization ($\text{Re}(\psi), \text{Im}(\psi)$), space-time heatmaps, and energy tracking.
3. **🧪 Potential Laboratory:** Investigate five distinct potential profiles:
   * Free particle (dispersive spreading)
   * Rectangular barrier (tunnelling vs over-barrier reflection)
   * Gaussian barrier (adiabatic reflectionless scattering)
   * Double barrier (Fabry–Pérot resonance)
   * Finite attractive well (quantum reflection from wells)
4. **🚇 Quantum Tunnelling Explorer:** Compare numerical wave packet $T$ with plane-wave Transfer Matrix, exact textbook formulas, and WKB approximation ($\exp(-2\kappa a)$).
5. **🎯 Resonant Tunnelling Lab:** Numerical search for quasi-bound state transmission peaks ($T \to 1$), lifetime calculation $\tau = \hbar/\Gamma$, and standing wave resonance.
6. **〰️ Interference & Momentum Analysis:** Real-time spatial interference fringes ($\Delta x = \pi/k_0$), FFT momentum spectrum, and Heisenberg uncertainty product $\Delta x \cdot \Delta p \ge \hbar/2$.
7. **⚖️ Classical vs Quantum:** Direct side-by-side comparison of Schrödinger wave evolution against a 40,000-particle Newtonian ensemble.
8. **📊 Experiment Analysis:** Automated parametric sweeps:
   * **Experiment A:** Transmission & Reflection vs Barrier Height ($V_0$)
   * **Experiment B:** Exponential Tunnelling Decay vs Barrier Width ($a$)
   * **Experiment C:** Transmission vs Carrier Momentum ($k_0$)
9. **✅ Numerical Validation:** Formal convergence benchmarks proving $O(\Delta x^2)$ and $O(\Delta t^2)$ order of accuracy against the exact analytical free-spreading solution.

---

## 🚀 Quickstart & Local Installation

### Prerequisites
* Python 3.10, 3.11, or 3.12
* `pip` package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/QWaveLab.git
cd QWaveLab

# Install required dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
The application will open automatically at `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Fork or push this repository to your GitHub account.
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New App"** and select:
   * **Repository:** `<your-username>/QWaveLab`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Click **Deploy!**

---

## 🧪 Testing & Verification

Run the automated test suite covering physics conservation, transfer matrix benchmarking, and convergence order:
```bash
python -m pytest -v tests/test_physics.py
```

### Verified Benchmark Criteria
* **Unitarity:** $\max |\|\psi(t)\|^2 - 1| < 10^{-10}$
* **Energy Drift:** $\Delta \langle H \rangle / \langle H \rangle < 10^{-10}$
* **Minimum Uncertainty Initial State:** $\Delta x \cdot \Delta p = 0.500000 \pm 10^{-6}$
* **Convergence Order:** $> 1.8$ (nominal 2nd-order Crank–Nicolson scheme)

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── config.toml         # Custom dark-mode UI theme configuration
├── tests/
│   └── test_physics.py     # Automated pytest test cases
├── analysis.py             # Transfer matrix, momentum FFT, classical ensemble
├── app.py                  # Main entry point & page routing
├── experiments.py          # Automated parameter sweeps (Exp A/B/C)
├── pages.py                # UI layouts and interactive controllers
├── physics_engine.py       # Crank-Nicolson TDSE solver & Hamiltonian builder
├── potentials.py           # Potential energy profiles V(x)
├── report.py               # PDF and CSV report export generator
├── requirements.txt        # Python package dependencies
├── ui.py                   # UI styling, metric cards, animation player
├── visualizations.py       # Plotly (web) and Matplotlib (export) figures
└── README.md               # Project documentation
```

---

## 📜 Academic Attribution & License

Developed for **Problem 7: Time Evolution of a Quantum Wave Packet**.  
Distributed under the [MIT License](LICENSE).
