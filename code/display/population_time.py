import matplotlib.pyplot as plt
import numpy as np

from code.constants import TREE, FIRE


def population_plot(tree_counts, burn_counts, frames, L, p, f, title, save_2_file):   
    
    tree_pct = (np.array(tree_counts) / (L * L)) * 100  # Convert counts → % population (comparable across different L sizes)
    burn_pct = (np.array(burn_counts) / (L * L)) * 100  # Same logic for burning population percentages

    fig, ax1 = plt.subplots(figsize=(12, 5))  # My standard figure size for reports (wide, short)

       # Tree Coverage (%)
    tree_line, = ax1.plot(tree_pct, 'g-', linewidth=1.8, label="Tree Coverage (%)")  #'g-' = green line, standard linestyle
    ax1.set_ylabel("Tree Coverage (%)", color='green')
    ax1.set_xlabel("Time (Frames)")
    ax1.set_xlim(0, frames)
    ax1.grid(alpha=0.3)  # Grid lines should not be visually overwhelming

       # Burning (%)
    ax2 = ax1.twinx()  # Second y-axis (burning population is much smaller so needs its own scale)
    burn_line, = ax2.plot(burn_pct, 'r-', alpha=0.4, linewidth=1.5)  #'r-' = red line
    ax2.set_ylabel("Burning (%)", color='red')
    
       # Layout Customisation (these values will work across most plots, my report plots vary)                 
    ax1.set_ylim(0, 70)                                      # Tree coverage upper bound suitable for most runs
    ax2.set_ylim(0, burn_pct.max() * 2.5)                    # Scaling by ×2.5 prevents both curves from overlapping visually
    ax2.set_yticks(np.linspace(0, burn_pct.max() * 2.5, 8))  # Ticks are lined up with gridlines
        
       # Legend setup (needed because ax1 and ax2 normally have separate legends)
    ax1.plot([], [], 'r-', alpha=0.4, linewidth=1.5, label="Burning (%)")  # Blank red entry added to ax1 for unified legend
    ax1.legend(loc="upper right")  # Legend now shows both Tree % and Burning % on the same plot

    plt.title(title)    # Finishing and saving plot
    plt.tight_layout()  # Ensures consistent plot shape across different parameter sets
    
    if save_2_file:
        plt.savefig(f"Tree_Coverage_{datetime.now().strftime('%H%M%S')}.png", dpi=500)  # High-res 500 dpi figure for report
        
    plt.show()
