# Forest Fire Simulation: methods and interpretation

## What counts as a fire?

Every lightning ignition starts a labelled lineage. If an existing fire reaches a tree during the same synchronous update, spread takes precedence over assigning a new lightning ID. If several lineages reach the tree, the oldest neighbouring ID receives that cell. Lineages are not merged into one connected outbreak.

Event size is the sum of burning cells assigned to a lineage over its lifetime. Events are grouped by ignition time and included only after they finish. Events still burning at the final frame are right-censored and omitted. Their omission avoids counting partial lifetimes as complete small fires, but finite observation windows can still underrepresent long-lived events. Longer runs and sensitivity to the observation window remain necessary.

No upper cutoff based on the linear grid width is imposed. In addition, regrowth means a sufficiently long-lived lineage could revisit a site: event size and the number of unique burned sites are different quantities.

## Sampling window

The legacy sampling rule uses `D(f) = A f^B`, with fitted values `A = 0.061399940272506136` and `B = 0.3081816633412444`. Its thresholds correspond to the fitted oscillation envelope falling to 10% and 0.1% of its initial amplitude.

These constants came from an earlier sweep. They are empirical, depend on the model regime, and are not universal forest-fire constants. The function named `critical_regime_start` is retained for compatibility, but its output is a transient-sampling heuristic. It does not test criticality.

For a new study, inspect population traces and residuals, vary the burn-in interval, and compare independent seeds and lattice sizes before interpreting a scaling exponent.

## Probability precision and performance

For a b-bit integer draw, the effective probability is `floor(p × 2^b) / 2^b`. The default core engine uses 32 bits; its resolution is approximately 2.33 × 10⁻¹⁰. The optional 16-bit representation has resolution approximately 1.53 × 10⁻⁵. Positive probabilities below the selected resolution are rejected.

The demo records effective probabilities and environment versions in its JSON output. A fixed seed reproduces a trajectory for the same implementation, NumPy random generator and precision. Changing the bit depth changes the random draws and trajectory.

The historical report discusses grids up to L = 16,000 and an estimated working-memory overhead of about 9.5 bytes per cell. That estimate is not a measured peak resident-set size for the current implementation. NumPy Boolean masks use one byte per cell, and temporary arrays contribute to peak memory. Benchmark results should include grid size, steps, precision, outputs, environment and timing scope.

## Statistical fits

Pure and truncated power-law fits use least squares on observed counts. Count uncertainty is not constant, adjacent observations can be correlated, and the choice of fit range matters. An R² value or a covariance error is not a goodness-of-fit test for the underlying probability distribution and does not establish that a power law is preferred to alternatives.

![Historical fire-size fit from the assessed report](../assets/fire-size-power-law.jpg)

*Historical report figure. It illustrates the original analysis; the large-run data have not been rerun after the event-accounting corrections.*

Useful further work includes maximum-likelihood fitting of discrete event sizes, cutoff selection, goodness-of-fit checks, comparison with other heavy-tail families and uncertainty estimation across independent runs. Those procedures are not claimed to be implemented here.

## Cluster geometry

Tree clusters use four-neighbour connectivity with periodic boundaries. Area is the number of tree sites; perimeter counts edges bordering any non-tree state.

![Historical area–perimeter fit from the assessed report](../assets/cluster-area-perimeter.jpg)

*A limited-range area–perimeter fit from the report. The plot is not an independent measurement of a universal fractal dimension.*

If cluster area scales as `R^D` and perimeter as `R^Dp`, an area–perimeter slope m gives `m = Dp/D`. The often-used conversion `Dp = 2m` additionally assumes compact area scaling with D = 2. Radius-of-gyration measurements and a broader scaling range would be needed to check that assumption.

The interactive app's perimeter analysis concerns the union of sites burned by an event. It is a different object from a surviving tree cluster and should not be interpreted as the same geometric measurement.

[Original Forest Fire Simulation report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDim1tykc0nTKl1fFVf2T52AaR-wLgcwbBSEvPdq3rfR-U?e=Kbm6k0)
