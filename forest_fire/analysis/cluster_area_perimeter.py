import numpy as np
from forest_fire.constants import TREE


def analyse_clusters(board, L):
    """Count tree sites and exposed edges using periodic four-neighbour connectivity."""
    if board.shape != (L, L):
        raise ValueError('L must match both board dimensions.')
    visited = np.zeros_like(board, dtype=bool)     # Boolean grid to ensure each tree cell is only assigned to one cluster

    clusters = []  # List to store properties of all detected clusters

    neighbours = [(-1,0), (1,0), (0,-1), (0,1)] # Relative coordinates for 4 neighbours

    # Loop over every cell in the grid
    for y in range(L):
        for x in range(L):

            # Skip cells that are not trees or already assigned to a cluster
            if board[y, x] != TREE or visited[y, x]:
                continue

            stack = [(y, x)] # Search from this tree
            visited[y, x] = True   
            cells = []  # Store all (y,x) positions belonging to this cluster

            while stack:  # While there are still sites to explore
                cy, cx = stack.pop()    # Remove the last indexed entry into positional coordinates
                cells.append((cy, cx))  # Add said position to a list to "remember" all paths

                for dy, dx in neighbours:
                    ny = (cy + dy) % L # Add a small increment dy or dx (from neighbours)
                    nx = (cx + dx) % L # % L is to wrap aroud the boarder

                    if board[ny, nx] == TREE and not visited[ny, nx]:
                        visited[ny, nx] = True # Update status so no repeats
                        stack.append((ny, nx)) # Add neighbours to stack (cells to investigate list)

    
            area = len(cells) # Number of tree cells in this connected cluster
 
    
            perimeter = 0   # Count how many cluster edges border non-tree cells
            for cy, cx in cells:
                for dy, dx in neighbours:
                    ny = (cy + dy) % L   # Same logic as before
                    nx = (cx + dx) % L
                    if board[ny, nx] != TREE:
                        perimeter += 1        # Untill cluster is enclosed keep adding 1

            clusters.append({"area": area, "perimeter": perimeter})

            
    return clusters


from scipy.optimize import curve_fit


def model(A, C, m):
    """Power-law perimeter model: P = C * A^m."""
    return C * A**m


def fit_area_perimeter(areas, perimeters):
    """Fit P = C*A^m, returning (C, m, C_err, m_err, r2)."""
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

    r2 = 1 - np.sum((logP - logP_pred)**2) / np.sum((logP - np.mean(logP))**2)

    return C, m, C_err, m_err, float(r2)
