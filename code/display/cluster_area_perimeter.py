import numpy as np
import matplotlib.pyplot as plt

from code.analysis.cluster_area_perimeter import model, fit_area_perimeter


def plot_cluster_area_perimeter(areas, perimeters, L, p, f):
    """Log-log plot of perimeter vs area with fitted slope m and boundary fractal dimension D_p=2m."""
    areas = np.asarray(areas, float)
    perimeters = np.asarray(perimeters, float)
    C, m, C_err, m_err, r2 = fit_area_perimeter(areas, perimeters)

    mask = (areas > 0) & (perimeters > 0)
    A_fit = areas[mask]
    P_fit = perimeters[mask]

    plt.figure(figsize=(12,5))
    plt.loglog(A_fit, P_fit, 'o', alpha=0.5, label="Clusters")
    A_ref = np.linspace(A_fit.min(), A_fit.max(), 500)
    plt.loglog(A_ref, model(A_ref, C, m), '--', label="P ∝ A^m")
    plt.xlabel("Cluster area A")
    plt.ylabel("Perimeter P")
    plt.title(
        f"Slope m = {m:.7f} ± {m_err:.5f}, "
        f"Boundary fractal dimension D_p = {2*m:.3f}, "
        f"L = {L}, p = {p:.7f}, f = {f:.9f}, R^2 = {r2:.3f}"
    )
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return C, m, C_err, m_err, r2
