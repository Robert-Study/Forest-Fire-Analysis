import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit


def decaying_sine(t, C, A, d, w, phi):
    return C + A*np.exp(-d*t) * np.sin(w*t + phi)


def analyse_decaying_sine(tree_counts, burn_counts, L, show_plots):
    
       # Cropping data
    N = len(tree_counts)
    t_full = np.arange(N)
    crop = int(0.018 * N) # Crop out the startup region where behaviour is linear rather than oscillatory (low chance for fire to spread initially)
    t_fit = np.arange(N - crop) # Zeroed time axis to use curve_fit function
    tree_pct = 100 * np.array(tree_counts, float) / (L * L) # Convert raw tree counts to population percentages (comparible over differnt grid sizes)
    tree_fit_y = tree_pct[crop:]
    
    # A clear oscillatory behaviour is visible in Plot 1a/c, which led to the creation of the below decaying sine model.
    # High tree growth rate -> Strong fires outbreaks -> Fuel is deprived, Fires calm -> High tree growth rate (repeat)
    # Randomness over a large grid causes fire outbreaks to become out of phase (self-organises consistent populations).
    # Parameters = [Vertical offset (C), Amlitude (A), Decay coefficient (d), Angular Freq (w), offset (phi)]
    
       # Modelling function
    bounds = ([0,0,0,0,-np.pi], [100,100,1,1,np.pi]) # ([Lower bound parameter], [Upper bound parameter])

       # Fit the model using curve_fit (minimises RMS between model and data).
    params_tree, cov_tree = curve_fit(decaying_sine, t_fit, tree_fit_y, bounds=bounds, maxfev=20000)
    errors_tree = np.sqrt(np.diag(cov_tree))  # Extract parameter errors (sigma)

    tree_model_segment = decaying_sine(t_fit, *params_tree) # unpacks parameters (*) and uses them to generate the model
    r2_tree = 1 - np.sum((tree_fit_y - tree_model_segment)**2) / np.sum((tree_fit_y - np.mean(tree_fit_y))**2)  # Comparing of data and model 

    if show_plots:  # Helpful for the report. Fits and plots are kept separate so the program runs faster.

           # Plot 1 — Tree percentage with fitted curve
        plt.figure(figsize=(12,5))
        plt.plot(t_full, tree_pct, 'g-',  label="Tree Population Simulated", linewidth=2, alpha=0.4)
        plt.plot(t_full[crop:], tree_model_segment, 'k--', label="Decaying Sine Wave Model", linewidth=1, alpha=0.6)
        # Extracting individual parameters to include in title
        tree_params = f"(C={params_tree[0]:.1f}±{errors_tree[0]:.2f}, A={params_tree[1]:.1f}±{errors_tree[1]:.2f}, d={params_tree[2]:.5f}±{errors_tree[2]:.5f}, ω={params_tree[3]:.3f}±{errors_tree[3]:.3f}, φ={params_tree[4]:.2f}±{errors_tree[4]:.2f}, R²={r2_tree:.3f})"
        plt.title(f"Tree Population % — Decaying Sine Fit\n{tree_params}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlabel("Time (frames)")
        plt.axhline(params_tree[0], color='k', linestyle='--', linewidth=0.8, alpha=0.2)
        plt.xlim(0, t_full[-1])
        plt.ylabel("Tree %")
        plt.tight_layout()
        plt.savefig(f"Tree_Coverage_{datetime.now().strftime('%H%M%S')}.png", dpi=500, bbox_inches="tight")
        plt.show()

            
        if burn_counts is not None: # Fit and plot burn signal if available (used once in report to compare phase differences)

            burn_pct = 100 * np.array(burn_counts, float) / (L * L) # Same process as before for tree modelling
            burn_fit_y = burn_pct[crop:]
            bounds_burn = ([0, 0, 0, 0, -np.pi], [100, 100, 1, 1, np.pi]) 
            params_burn, cov_burn = curve_fit(decaying_sine, t_fit, burn_fit_y, bounds=bounds_burn, maxfev=20000)
            burn_errors = np.sqrt(np.diag(cov_burn))
            burn_model_segment = decaying_sine(t_fit, *params_burn)
            r2_burn = 1 - np.sum((burn_fit_y - burn_model_segment)**2) / np.sum((burn_fit_y - np.mean(burn_fit_y))**2) #fits close to 1 are accurate

               # Plot 2 — Burn percentage with fitted curve
            plt.figure(figsize=(12,5))
            plt.plot(t_full, burn_pct, 'r-',  label="Tree Population Simulated", linewidth=2, alpha=0.4)
            plt.plot(t_full[crop:], burn_model_segment, 'k--', label="Decaying Sine Wave Model", linewidth=1, alpha=0.6)
            burn_params = f"(C={params_burn[0]:.1f}±{burn_errors[0]:.2f}, A={params_burn[1]:.1f}±{burn_errors[1]:.2f}, d={params_burn[2]:.5f}±{burn_errors[2]:.5f}, ω={params_burn[3]:.3f}±{burn_errors[3]:.3f}, φ={params_burn[4]:.2f}±{burn_errors[4]:.2f}, R²={r2_burn:.3f})"
            plt.title(f"Burn Population % — Decaying Sine Fit\n{burn_params}")
            plt.grid(alpha=0.3)
            plt.xlim(0, t_full[-1])
            plt.legend()
            plt.xlabel("Time (frames)")
            plt.axhline(params_burn[0], color='k', linestyle='--', linewidth=0.8, alpha=0.2)
            plt.ylabel("Burn %")
            plt.tight_layout()
            plt.savefig(f"Burn_Coverage_{datetime.now().strftime('%H%M%S')}.png", dpi=500, bbox_inches="tight")
            plt.show()

    return params_tree, errors_tree, r2_tree

