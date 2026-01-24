# Forest Fire Model – Large-Scale Simulation & Analysis

A high-performance, research-grade implementation of a stochastic forest-fire model, including advanced statistical analysis, with visualization tools.
This project was developed to study **self-organized criticality**, **oscillatory dynamics**, and **scaling laws** in spatially extended systems using large grid simulations.

See interactive demo (hosted on streamlit): https://forest-fire-analysis.streamlit.app/
See code report: https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDim1tykc0nTKl1fFVf2T52AaR-wLgcwbBSEvPdq3rfR-U?e=T9fe1s

This code provides:

- Various optimized simulation engines  
- Automatic application of time-scales
- Population plots fitted to a decaying-sine model
- Decaying-sine parameters exploration as power-laws
- Power-law and truncated power-law fitting of fire sizes
- Cluster geometry & fractal boundary analysis  

## Model overview

The forest-fire model is a cellular automaton defined on a 2D grid:

- Empty cells may grow trees with probability **p**
- Trees ignite spontaneously with probability **f**
- Fire spreads to neighboring trees
- Burning cells burn out and become empty

Despite its simple rules, the system exhibits:

- Decaying oscillations in forest density  
- Self-organized critical behavior  
- Power-laws for equilibrium populations, decaying rate, fire size distributions (and more.)
- Fractal cluster boundaries  


## Repository structure

```text
forest-fire-model/
├── code/                         # All source code (library-style, no auto execution)
│
│ ├── core_simulations/           # Core cellular automaton implementations
│ │ ├── fast_simulation.py        # Default model (fastest, uint16 probabilities)
│ │ ├── id_simulation.py          # Model with fire-ID tracking for fire size/lifetime statistics
│ │ └── simulation_32bit.py       # High-precision uint32 probability model (used for cluster analysis)
│ │
│ ├── analysis/                   # Data analysis, fitting, and parameter exploration
│ │ ├── decaying_sine_fit.py      # Fit decaying-sine model to population time series
│ │ ├── parameter_sweep.py        # Generate geometric p,f values and run structured sweeps
│ │ ├── power_law_fit.py          # Fire size distribution power-law fitting utilities
│ │ ├── truncated_power_law_fit.py# Alternative truncated power-law model
│ │ ├── cluster_area_perimeter.py # Cluster area–perimeter extraction and scaling analysis
│ │ ├── equilibrium_coverage.py  # Equilibrium tree coverage metrics from fitted models
│ │ ├── angular_frequency.py     # Angular frequency extraction from decaying-sine fits
│ │ ├── exponential_decay.py     # Decay-rate extraction and log–log fitting tools
│ │ └── simulation_time_scales/
│ │     └── timescales.py         # Oscillatory regime end, critical regime start, and run-length policy
│ │
│ ├── display/                    # Plotting and visualisation utilities
│ │ ├── simulation_gif.py         # Render simulation frames to animated GIFs
│ │ ├── population_time.py        # Tree/burn fraction vs time plots
│ │ ├── decaying_sine.py          # Plot decaying-sine fit over time series
│ │ ├── parameter_sweep.py        # Visualisation of p,f parameter grids
│ │ ├── equilibrium_coverage_growth.py    # Equilibrium coverage vs p plots
│ │ ├── equilibrium_coverage_lightning.py # Equilibrium coverage vs f plots
│ │ ├── angular_frequency_growth.py        # Angular frequency vs p plots
│ │ ├── exponential_decay_lightning.py    # Decay rate vs f plots
│ │ ├── power_law.py              # Fire-size distributions with fitted power-law
│ │ ├── truncated_power_law.py    # Fire-size distributions with truncated power-law
│ │ └── cluster_area_perimeter.py # Cluster area–perimeter log–log plots and fitted scaling
│
├── outputs/                      # Example outputs and user-generated results
│ ├── gifs/                       # Animated simulations
│ ├── figures/                    # Saved plots
│ ├── sweeps/                     # Parameter sweep outputs
│ └── tables/                     # CSV result tables
│
├── requirements.txt              # Python dependencies
├── README.md
└── .gitignore
```



## Project history

This project originally began as a university physics coursework assignment investigating self-organized criticality in forest-fire models, for which it received a grade of 76%.
Since then, the codebase has been rewritten and expanded on, with major improvements including:

- Multiple optimized simulation engines
- Automatic time-scale detection
- Robust parameter sweep infrastructure
- Advanced statistical fitting (decaying-sine, power laws, truncated power laws)
- Cluster geometry and fractal boundary analysis
- Modular design suitable for reuse and further research

The project is now maintained as a general research and experimentation toolkit rather than a single coursework submission.
