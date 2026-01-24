import numpy as np

# Global parameters (from the original notebook)
seed = 0  # reproducible report results used seed=0 in the notebook (no seeding is performed on import here)

fps = 60  # default fps for GIF output (hardware maximum)

EMPTY, TREE, FIRE = 0, 1, 2  # cell state constants

# Colour map: cmap[state] -> RGB uint8
cmap = np.array(
    [
        [0, 0, 0],       # EMPTY (black)
        [0, 180, 0],     # TREE  (green)
        [255, 0, 0],     # FIRE  (red)
    ],
    dtype=np.uint8,
)

# Exponential decay model to reach criticality (used in several analysis sections)
A = 0.061399940272506136
B = 0.3081816633412444

# Convenience aliases (optional)
CMAP = cmap
DEFAULT_FPS = fps
