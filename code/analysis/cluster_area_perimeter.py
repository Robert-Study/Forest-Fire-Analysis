import numpy as np
from scipy.optimize import curve_fit

from code.constants import EMPTY, TREE


def analyse_clusters(board, L):
    """Return the area and perimeter of every four-connected tree cluster."""
    visited = np.zeros_like(board, dtype=bool)
    clusters = []
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(L):
        for x in range(L):
            if board[y, x] != TREE or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            cells = []

            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))

                for dy, dx in neighbours:
                    ny = (cy + dy) % L
                    nx = (cx + dx) % L
                    if board[ny, nx] == TREE and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            area = len(cells)
            perimeter = 0
            for cy, cx in cells:
                for dy, dx in neighbours:
                    ny = (cy + dy) % L
                    nx = (cx + dx) % L
                    if board[ny, nx] == EMPTY:
                        perimeter += 1

            clusters.append({"area": area, "perimeter": perimeter})

    return clusters


def model(A, C, m):
    """Power-law perimeter model: P = C * A**m."""
    return C * A**m


def fit_area_perimeter(areas, perimeters):
    """Fit P = C*A**m, returning (C, m, C_err, m_err, r2)."""
    areas = np.asarray(areas, float)
    perimeters = np.asarray(perimeters, float)
    mask = (areas > 0) & (perimeters > 0)
    A_fit = areas[mask]
    P_fit = perimeters[mask]

    params, covariance = curve_fit(model, A_fit, P_fit)
    C, m = params
    C_err = float(np.sqrt(covariance[0, 0]))
    m_err = float(np.sqrt(covariance[1, 1]))

    logA = np.log(A_fit)
    logP = np.log(P_fit)
    logP_pred = np.log(model(A_fit, C, m))
    r2 = 1 - np.sum((logP - logP_pred) ** 2) / np.sum(
        (logP - np.mean(logP)) ** 2
    )

    return C, m, C_err, m_err, float(r2)

