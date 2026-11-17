# ⚛ QWaveLab – Interactive Quantum Wave Packet Simulation Laboratory

A Streamlit application that solves the 1-D time-dependent Schrödinger equation
`iħ ∂ψ/∂t = [-ħ²/2m ∂²/∂x² + V(x)]ψ` with a **Crank–Nicolson finite-difference solver**
(sparse matrices, one LU factorisation reused for every step) and lets you explore wave-packet
propagation, spreading, reflection, transmission, tunnelling, interference and resonant tunnelling.

Every number and curve is computed by the solver (or by the transfer-matrix / FFT analysis built on it);
nothing is hard-coded. The solver in `physics_engine.py` reuses the functions from the Colab notebook
(`gaussian_wave_packet`, Hamiltonian construction, `simulate_wave_packet`).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository (`git init && git add . && git commit -m "QWaveLab" && git push`).
2. On https://share.streamlit.io choose the repo, branch and **`app.py`** as the main file. Dependencies come from `requirements.txt`.

## Files
| File | Role |
|---|---|
| `app.py` | Entry point, navigation sidebar, routing |
| `pages.py` | The 9 pages (UI only) |
| `ui.py` | CSS theme, metric cards, animation player (Play/Pause/Reset/Step/slider/speed), export panel |
| `physics_engine.py` | Grid, Gaussian packet, Hamiltonian, Crank–Nicolson evolution, observables, input validation |
| `potentials.py` | Free, rectangular, Gaussian, double barrier, finite well + parameter specs |
| `analysis.py` | Transfer-matrix T(E), resonance finder, momentum/interference decomposition, classical ensemble, validation/convergence |
| `experiments.py` | Automatic domain/duration, scattering runs, experiments A/B/C, interpretation text |
| `visualizations.py` | Plotly figures (UI) and Matplotlib figures (PNG/PDF export) |
| `report.py` | CSV, Markdown summary, PNG/ZIP and PDF report generation |

## Pages
1. Home · 2. Wave Packet Simulator · 3. Potential Laboratory · 4. Quantum Tunnelling Explorer · 5. Resonant Tunnelling Lab ·
6. Interference & Momentum · 7. Classical vs Quantum · 8. Experiment Analysis (A/B/C + side-by-side compare) · 9. Numerical Validation

## Method notes
* Units: ħ = 1, m = 1 by default (dimensionless). Σ|ψ|²Δx = 1 is enforced at t = 0.
* `(I + iΔtH/2ħ)ψⁿ⁺¹ = (I − iΔtH/2ħ)ψⁿ` is unitary: norm and ⟨H⟩ are conserved to ~1e-13.
* R, T, B = probability left of / right of / inside the potential region (partition of the grid, so R+T+B = norm).
  They are final reflection/transmission coefficients only after the collision (the app flags incomplete collisions).
* Barrier edges are placed on the grid by cell-overlap weighting, which preserves ∫V dx for any Δx.
* T(E) (page 5) uses a transfer matrix on the same discretised potential; resonances are found with `scipy.signal.find_peaks`
  and refined numerically. Wave-packet runs can be overlaid as an independent check.
* Measured convergence (free packet vs exact solution): order ≈ 2 in Δx and ≈ 2 in Δt.

## Limitations
One dimension, single particle, time-independent V; hard (Dirichlet) walls reflect anything reaching them (the app warns);
lattice dispersion needs ≳ 10 grid points per wavelength; a packet's T is an energy-average of the plane-wave T(E).
The Streamlit Community Cloud free tier has limited memory/CPU – keep Δx ≥ 0.025 and moderate durations.
