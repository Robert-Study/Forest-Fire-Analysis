# Forest Fire Simulation & Analysis

An optimised Python implementation of a stochastic forest-fire cellular automaton, developed to investigate self-organised criticality, oscillatory population dynamics and scaling behaviour.

The project began as assessed university coursework and has since been substantially extended into a broader simulation and analysis toolkit, including parameter sweeps, statistical fitting, fire-size analysis and interactive visualisation.

Large-scale runs have processed more than **15 trillion cell updates**.

> **Academic result:** **76% — First-Class mark.** The public repository has subsequently been extended beyond the assessed submission.

## Highlights

- Optimised stochastic cellular-automaton simulation in Python
- 15+ trillion cell updates in large-scale runs
- Self-organised criticality and equilibrium analysis
- Decaying-oscillation fitting and parameter exploration
- Pure and truncated power-law modelling
- Cluster geometry and fractal-boundary analysis
- Automated parameter sweeps and timescale estimation
- Interactive Streamlit visualisation

### 🔗 Interactive Demo
[Launch the Streamlit simulation](https://forest-fire-analysis.streamlit.app/)

### 📄 Technical Report
[View the original technical report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDim1tykc0nTKl1fFVf2T52AaR-wLgcwbBSEvPdq3rfR-U?e=T9fe1s)

## Model Overview

The forest-fire model is a stochastic cellular automaton defined on a two-dimensional grid:

- Empty cells grow trees with probability **p**
- Trees ignite spontaneously with probability **f**
- Fire spreads to neighbouring trees
- Burning cells become empty

Despite these simple local rules, the model exhibits complex collective behaviour including decaying population oscillations, equilibrium regimes, heavy-tailed fire-size distributions and fractal cluster boundaries.

The project investigates how these behaviours vary across the model's parameter space and how characteristic quantities scale with **p** and **f**.

## Analysis

The repository includes tools for:

- Population time-series analysis
- Decaying-sine fitting
- Equilibrium tree-coverage estimation
- Oscillation-frequency and decay-rate extraction
- Structured parameter sweeps
- Fire-size distribution analysis
- Pure and truncated power-law fitting
- Cluster area--perimeter scaling
- Simulation visualisation and GIF generation

## Repository Structure

```text
Forest-Fire-Analysis/
├── code/                          # Core simulation and analysis code
│
│   ├── core_simulations/          # Cellular-automaton implementations
│   │   ├── fast_simulation.py     # Optimised default simulation
│   │   ├── id_simulation.py       # Fire-ID tracking for size/lifetime statistics
│   │   └── simulation_32bit.py    # Higher-precision model for cluster analysis
│   │
│   ├── analysis/                  # Statistical analysis and fitting
│   │   ├── decaying_sine_fit.py
│   │   ├── parameter_sweep.py
│   │   ├── power_law_fit.py
│   │   ├── truncated_power_law_fit.py
│   │   ├── cluster_area_perimeter.py
│   │   ├── equilibrium_coverage.py
│   │   ├── angular_frequency.py
│   │   ├── exponential_decay.py
│   │   └── timescales.py
│   │
│   └── display/                   # Plotting and visualisation
│       ├── simulation_gif.py
│       ├── population_time.py
│       ├── decaying_sine.py
│       ├── parameter_sweep.py
│       ├── equilibrium_coverage_growth.py
│       ├── equilibrium_coverage_lightning.py
│       ├── angular_frequency_growth.py
│       ├── exponential_decay_lightning.py
│       ├── power_law.py
│       ├── truncated_power_law.py
│       └── cluster_area_perimeter.py
│
├── outputs/                       # Example and generated results
│   ├── gifs/
│   ├── figures/
│   ├── sweeps/
│   └── tables/
│
├── requirements.txt
├── streamlit_simulation.py
├── README.md
└── .gitignore
