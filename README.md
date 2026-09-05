# Forest Fire Simulation & Analysis

An optimised Python implementation of a stochastic forest-fire cellular automaton, developed to investigate **self-organised criticality**, **oscillatory population dynamics** and **scaling behaviour**.

The project began as assessed university coursework and has since been substantially extended into a broader simulation and analysis toolkit, including parameter sweeps, statistical fitting, fire-size analysis, cluster geometry and interactive visualisation.

> **Academic result:** **74% — First-Class mark.**  
> The public repository has subsequently been extended beyond the original assessed submission.

🔗 **[Interactive Streamlit Demo](https://forest-fire-analysis.streamlit.app/)**  
📄 **[View the Technical Report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDim1tykc0nTKl1fFVf2T52AaR-wLgcwbBSEvPdq3rfR-U?e=Kbm6k0)**

---

## Example Result

### Rainforest — Good Growth Conditions and Average Storm Activity

**Parameters:** `p = 0.0153` · `f = 0.000153` · `Grid size = 16000`

![Tree coverage and burning population through time]
<img width="1231" height="457" alt="Population_Plot" src="https://github.com/user-attachments/assets/7e38b3ea-7841-48c1-9fb0-33c6b6b96c5b" />
*Figure 1. Tree coverage and burning population through time for a high-growth forest-fire simulation. The system exhibits pronounced damped oscillations before approaching an equilibrium state.*

When tree growth dominates over lightning, dense and highly connected forests form before being disrupted by large fire events. Repeated growth and destruction produces the oscillatory behaviour above, with the amplitude gradually decreasing as the system approaches equilibrium.

The oscillatory population is modelled using a decaying sine function,

\[
y(t) = C + Ae^{-Dt}\sin(\omega t + \phi),
\]

allowing the equilibrium tree coverage, oscillation frequency and exponential decay rate to be quantified.

---

## Highlights

- Large-scale stochastic cellular-automaton simulation written in Python
- Simulations performed on grids up to **16000 × 16000 cells**
- Individual large-scale runs exceeding **2 trillion cell updates**
- Self-organised criticality and equilibrium analysis
- Decaying-sine modelling of population dynamics
- Power-law relationships between model parameters and observables
- Pure and truncated power-law modelling of fire-size distributions
- Automated parameter sweeps across growth and lightning probabilities
- Cluster area–perimeter scaling and fractal-boundary analysis
- Interactive Streamlit visualisation

---

## Model Overview

The forest-fire model is a stochastic cellular automaton defined on a two-dimensional grid.

Each cell can be in one of three states:

- **Empty**
- **Tree**
- **Burning**

At each timestep:

- Empty cells may grow trees with probability **p**
- Trees may ignite spontaneously due to lightning with probability **f**
- Fire propagates to neighbouring trees
- Burning cells burn for one frame before becoming empty

The simulation uses **periodic boundary conditions**, allowing fire propagation to wrap across the edges of the grid. This prevents boundary cells from behaving differently from cells in the interior.

Despite these simple local rules, the system produces complex large-scale behaviour including oscillations, equilibrium states, heavy-tailed fire-size distributions and spatial scaling relationships.

---

## Population Dynamics

In high-growth regimes, the forest repeatedly builds towards high tree coverage before large fires rapidly destroy connected regions.

This produces damped oscillations in both the tree and burning populations.

The fitted decaying-sine model provides estimates of:

- Equilibrium tree coverage, **C**
- Initial oscillation amplitude, **A**
- Exponential decay coefficient, **D**
- Angular frequency, **ω**
- Phase offset, **φ**

The tree and burning populations were also observed to exhibit an approximately **π/2 phase difference**, reflecting the delay between forest growth and subsequent large fire outbreaks.

---

## Power-Law Scaling

Several properties of the model were investigated using power-law relationships.

The equilibrium tree population was found to vary systematically with both the growth probability **p** and lightning probability **f**.

The exponential decay rate of the oscillations was also found to scale with lightning probability, allowing characteristic timescales to be estimated for the transition from the initial oscillatory regime towards equilibrium.

These timescales were subsequently used to separate fire-size measurements into different dynamical regions.

---

## Fire-Size Distributions

Individual fires were assigned unique identifiers at ignition so their propagation could be tracked over time.

The total number of cells burned during each event was then used as the fire size.

Near the critical regime, the distribution of fire sizes approximately follows a power law:

\[
N(s) \propto s^{-\alpha}.
\]

When lightning becomes non-negligible relative to tree growth, very large fires become increasingly suppressed by interactions between separate outbreaks.

This behaviour was investigated using a truncated power law,

\[
N(s) = As^{-\alpha}e^{-s/s_c},
\]

where \(s_c\) represents the characteristic scale beyond which large fires are increasingly suppressed.

---

## Cluster Geometry

The spatial structure of surviving tree clusters was also investigated following fire events.

For each cluster:

- **Area** was measured as the number of occupied tree cells
- **Perimeter** was measured from the number of exposed cluster edges

The relationship between cluster perimeter and area was investigated using a power law,

\[
P \propto A^m,
\]

providing a route towards characterising the fractal geometry of forest structures near criticality.

---

## Performance & Optimisation

Large simulations are computationally and memory intensive, so significant optimisation was required.

Key improvements included:

- NumPy-based array operations rather than cell-by-cell Python loops
- Boolean masks for simulation state calculations
- Efficient random-number representations for probabilistic updates
- `np.roll` operations for neighbour propagation
- Incremental saving of simulation data rather than retaining complete runs in RAM
- Direct appending of results to CSV files
- Incremental GIF frame generation
- Reusable structured parameter sweeps
- Weighted least-squares fitting using `scipy.curve_fit`

Large-scale simulations reached:

- **Grid size:** `L = 16000`
- **Cells per frame:** `256,000,000`
- **Run length:** up to `8000` frames
- **Cell updates per individual run:** over **2 trillion**

Memory overhead for the optimised simulation was reduced to approximately **9.5 bytes per cell**.

---

## Parameter Sweeps

A reusable parameter-sweep framework was developed to explore combinations of growth probability **p** and lightning probability **f**.

A typical sweep uses a **5 × 5 parameter grid**, covering a geometrically spaced range of values spanning approximately two orders of magnitude.

This allows relationships between model parameters and emergent properties to be studied systematically rather than through isolated simulations.

---

## Project History

The project originally began as a university physics coursework assignment investigating self-organised criticality in forest-fire models and received a mark of **74%**.

Following the assessed submission, the project was substantially extended beyond the original coursework.

Further development included:

- Larger and more efficient simulation engines
- Automated parameter sweeps
- Decaying-sine analysis
- Characteristic timescale estimation
- Fire identification and size tracking
- Pure and truncated power-law modelling
- Cluster geometry analysis
- Improved visualisation tools
- Interactive Streamlit demonstration

The repository is now maintained as an independent simulation and analysis project rather than a direct copy of the original assessed submission.

---

## Future Work

Potential extensions include:

- Radius-of-gyration analysis of tree clusters
- Estimation of cluster fractal dimension across dynamical regimes
- CCDF-based analysis of fire-size distributions
- More rigorous statistical comparison of heavy-tail models
- Larger parameter sweeps across **p** and **f**
- Further optimisation of large-grid simulations
- Comparison of equilibrium tree density estimated from population and fire-size statistics

---

## Technologies

`Python` · `NumPy` · `SciPy` · `Matplotlib` · `Streamlit` · `tqdm`

**Methods:** numerical simulation · statistical fitting · parameter sweeps · power-law analysis · scientific visualisation · optimisation
