# Forest Fire Simulation & Analysis

An optimised Python implementation of a stochastic forest-fire cellular automaton, developed to investigate **self-organised criticality, oscillatory population dynamics and fire-size scaling**.

The project began as assessed university coursework and has since been substantially extended with larger simulations, parameter sweeps, additional statistical analysis, cluster geometry and an interactive Streamlit demonstration.

> **Academic result: 74% — First-Class mark**  
> The public repository has subsequently been extended beyond the original assessed submission.

🔗 **[Interactive Streamlit Demo](https://forest-fire-analysis.streamlit.app/)**  
📄 **[Technical Report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDim1tykc0nTKl1fFVf2T52AaR-wLgcwbBSEvPdq3rfR-U?e=Kbm6k0)**

---

## Example Result

### Rainforest — Good Growth Conditions and Average Storm Activity

**Simulation parameters:** `p = 0.0153` · `f = 0.000153` · `Grid size = 16000`

<p align="center">
  <img
    width="1231"
    alt="Tree coverage and burning population through time"
    src="https://github.com/user-attachments/assets/7e38b3ea-7841-48c1-9fb0-33c6b6b96c5b"
  />
</p>

<p align="center">
  <em>Figure 1. Tree coverage and burning population through time for a high-growth forest-fire simulation. Large fire events produce pronounced damped oscillations before the system approaches equilibrium.</em>
</p>

When tree growth dominates over lightning (`f:p = 1:100`), dense and highly connected forests develop before being disrupted by large fire outbreaks. Repeated cycles of growth and destruction produce the damped oscillatory behaviour shown above.

The oscillations are modelled using a decaying-sine function to quantify the **equilibrium tree coverage, oscillation frequency and exponential decay rate**, providing a way to characterise the system's progression towards its self-organised regime.

---

## Project Highlights

- Large-scale stochastic cellular-automaton simulation written in Python
- Simulations performed on grids up to **16,000 × 16,000 cells**
- Individual large-scale runs exceeding **2 trillion cell updates**
- Decaying-sine analysis of tree-population dynamics
- Self-organised criticality and equilibrium analysis
- Pure and truncated power-law modelling of fire-size distributions
- Automated parameter sweeps across growth and lightning probabilities
- Cluster area–perimeter scaling and fractal-boundary analysis
- Interactive visualisation with Streamlit

---

## Model

The forest-fire model operates on a two-dimensional grid in which each cell is either:

- **Empty**
- **Occupied by a tree**
- **Burning**

At each timestep:

- Empty cells may grow a tree with probability **p**
- Trees may ignite spontaneously due to lightning with probability **f**
- Fire spreads to neighbouring trees
- Burning cells become empty after one frame

The simulation uses **periodic boundary conditions**, allowing fire to wrap across the edges of the grid. This keeps boundary cells statistically equivalent to cells within the interior.

Despite these simple local rules, the system produces complex large-scale behaviour including damped population oscillations, equilibrium states, heavy-tailed fire-size distributions and spatial scaling relationships.

---

## Analysis

### Population Dynamics

Tree coverage and burning fraction are tracked through time.

High-growth conditions produce pronounced oscillations as large connected forests repeatedly grow and burn. These oscillations gradually decay as the system approaches equilibrium.

The population behaviour is modelled using a decaying sine of the form:

```text
y(t) = C + A exp(-Dt) sin(ωt + φ)
```

where the fitted parameters describe:

- **C** — equilibrium tree coverage
- **A** — oscillation amplitude
- **D** — exponential decay rate
- **ω** — angular frequency
- **φ** — phase offset

The tree and burning populations were also observed to exhibit an approximately **π/2 phase lag**, reflecting the delay between forest build-up and subsequent fire outbreaks.

### Parameter Scaling

Several model observables were investigated as functions of the growth probability **p** and lightning probability **f**.

Log-log analysis was used to examine power-law relationships involving:

- Equilibrium tree coverage
- Growth probability
- Lightning probability
- Oscillation decay rate

The decay-rate relationship was also used to define characteristic timescales separating the early oscillatory regime from the later equilibrium region.

### Fire-Size Distributions

Individual fires are assigned unique identifiers at ignition, allowing their complete evolution to be tracked.

The total number of cells burned during each event is then used as the fire size.

Near criticality, the fire-size distribution approximately follows a power law:

```text
N(s) ∝ s^(-α)
```

When lightning becomes more significant relative to tree growth, very large fires become increasingly suppressed. This behaviour was investigated using a truncated power law:

```text
N(s) = A s^(-α) exp(-s/s_c)
```

where `s_c` defines the characteristic scale beyond which the largest events become increasingly suppressed.

### Cluster Geometry

The geometry of surviving tree clusters was also investigated following fire events.

For each cluster:

- **Area** was measured from the number of tree cells
- **Perimeter** was measured from exposed cluster edges

The relationship between cluster area and perimeter was analysed using a power law, providing a route towards characterising the fractal structure of the forest near criticality.

---

## Performance & Optimisation

Large forest-fire simulations are computationally and memory intensive, so substantial optimisation was required.

The implementation includes:

- NumPy-based array operations rather than cell-by-cell Python loops
- Boolean masks for simulation-state calculations
- Efficient neighbour propagation using `np.roll`
- Compact probability representations
- Incremental output to CSV files
- Incremental GIF generation rather than retaining all frames in RAM
- Reusable parameter-sweep infrastructure
- Weighted fitting using `scipy.curve_fit`

Large runs reached:

| Property | Scale |
| --- | ---: |
| Grid width | **16,000 cells** |
| Cells per frame | **256 million** |
| Run length | **up to 8,000 frames** |
| Cell updates | **2.048 trillion per run** |

Memory overhead in the optimised implementation was reduced to approximately **9.5 bytes per cell**.

---

## Parameter Sweeps

A reusable parameter-sweep framework was developed to explore combinations of **p** and **f** systematically.

Typical sweeps use a **5 × 5 grid** of geometrically spaced parameter combinations covering approximately a **100× range**.

This makes it possible to investigate scaling relationships across model regimes without relying on isolated individual simulations.

---

## Repository Structure

```text
Forest-Fire-Analysis/
├── code/
│   ├── core_simulations/        # Cellular-automaton simulation engines
│   ├── analysis/                # Statistical analysis and fitting
│   └── display/                 # Plotting and visualisation
│
├── outputs/
│   ├── figures/
│   ├── gifs/
│   ├── sweeps/
│   └── tables/
│
├── requirements.txt
├── streamlit_simulation.py
├── README.md
└── .gitignore
```

---

## Project Development

The original assessed project investigated self-organised criticality in a stochastic forest-fire model and received a mark of **74%**.

Following submission, the project was substantially extended beyond the original coursework, including:

- Larger simulation engines
- Improved memory handling
- Automated parameter sweeps
- Additional decaying-sine analysis
- Characteristic timescale estimation
- Fire identification and size tracking
- Truncated power-law modelling
- Cluster geometry analysis
- Interactive Streamlit visualisation

The repository therefore represents an **extended version of the project**, rather than the assessed coursework submission alone.

---

## Future Work

Potential extensions include:

- Radius-of-gyration analysis of tree clusters
- Estimation of cluster fractal dimension
- CCDF-based analysis of fire-size distributions
- More rigorous comparison of heavy-tail models
- Larger parameter sweeps across **p** and **f**
- Further optimisation of very large simulations

---

## Technologies

`Python` · `NumPy` · `SciPy` · `Matplotlib` · `Streamlit` · `tqdm`

**Methods:** numerical simulation · statistical fitting · parameter sweeps · power-law analysis · data visualisation · optimisation

