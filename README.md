# Forest Fire Simulation

A stochastic forest-fire cellular automaton exploring **population oscillations, fire-size distributions and cluster geometry**. I extended the assessed coursework with larger simulations, parameter sweeps, fire tracking and an interactive application.

**74% — First-Class mark**

[Interactive simulation](https://forest-fire-analysis.streamlit.app/) · [Project report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDim1tykc0nTKl1fFVf2T52AaR-wLgcwbBSEvPdq3rfR-U?e=Kbm6k0)

## Large-scale population dynamics

**16,000 × 16,000 cells · 8,000 steps · 2.048 trillion cell-time updates**  
**Growth probability p = 0.0153 · Lightning probability f = 0.000153**

<p align="center">
  <img width="1231" alt="Original large-scale forest-fire result: tree coverage and burning population over time" src="https://github.com/user-attachments/assets/7e38b3ea-7841-48c1-9fb0-33c6b6b96c5b" />
</p>

*Original project figure for the large-scale run. Tree growth and large fire outbreaks produce damped population oscillations before the system approaches equilibrium.*

When growth dominates over lightning, connected forests build up before being disrupted by large fires. Repeated growth and destruction produce the oscillations above. I fitted a decaying sinusoid to estimate equilibrium coverage, oscillation frequency and decay rate, then studied how those quantities changed with growth and ignition probabilities.

## The model

Each cell is empty, a tree or burning. Updates are synchronous: burning cells become empty, fire spreads to neighbouring trees, lightning ignites other trees with probability f, and previously empty cells grow trees with probability p.

**The grid wraps in both directions.** Fire crossing the top edge continues at the bottom; fire crossing the left edge continues at the right. Four-neighbour connectivity and periodic boundaries apply to the simulation and tree-cluster analysis.

The core uses NumPy array operations, a compact state grid and integer probability draws. The default uses 32-bit probability precision. Output can be written incrementally instead of retaining every frame in memory.

## What I investigated

- **Population dynamics:** decaying-sine fits to tree coverage and the timing of the burning population.
- **Parameter dependence:** sweeps over growth and lightning probabilities to examine equilibrium coverage, oscillation frequency and decay.
- **Fire-size distributions:** tracking ignition lineages and comparing pure and truncated power-law fits.
- **Cluster geometry:** measuring surviving tree-cluster area and exposed perimeter.

The large runs required attention to temporary arrays, probability resolution and output costs. A single 16,000² × 8,000 run contains **2.048 trillion cell-time updates**; this counts cells multiplied by timesteps.

## Fire tracking and interpretation

Each lightning ignition starts an identified lineage. Spread takes precedence over a new ignition at the same cell, and a collision assigns the shared cell to the oldest neighbouring lineage. Events are counted over their complete lifetime; fires still active when the run ends are excluded from completed-event statistics.

Event size is the number of burned cells over a lifetime, so it can exceed the grid's linear width. The app retains those large events.

The project investigates self-organised criticality using population dynamics, fire-size fits and spatial measurements. The fits are exploratory; the report and [methods notes](docs/methods.md) discuss sampling, scaling assumptions and the distinction between historical results and subsequent code corrections.

## Code and core tests

- `forest_fire/core_simulations/` — population updates and event tracking.
- `forest_fire/analysis/` — fitting, sweeps, sampling and cluster measurements.
- `forest_fire/display/` — plotting.
- `Streamlit_Simulation.py` — interactive controls and visualisation.

Core checks cover periodic boundaries, synchronous updates, probability precision and fire-event accounting.

```bash
python -m pip install -r requirements.txt
python -m streamlit run Streamlit_Simulation.py
python -m unittest discover -s tests -v
```

[Core tests](tests/test_core.py) · [Simulation tests](tests/test_simulation.py) · [GitHub Actions](https://github.com/Robert-Study/Forest-Fire-Analysis/actions)

**Methods:** Python, NumPy, SciPy, Matplotlib, Streamlit, stochastic simulation, numerical optimisation and statistical fitting.
