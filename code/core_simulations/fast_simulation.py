import numpy as np
from datetime import datetime

from code.constants import EMPTY, TREE, FIRE, cmap, fps


def update_forest(board, p, f):
    new_board = board.copy() # previous frame state
    
      # Create 3 resuable boolean masks (grids of True/False) where if the condition is met then True
      # These are far quicker for the processor to read (1 bit per cell) rather than the board every time (1 byte per cell)
    Empty_idx = (board == EMPTY)  #if EMPTY(0) -> True else False
    Tree_idx = (board == TREE)    #if TREE (1) -> True else False 
    Burning_idx = (board == FIRE) #if FIRE (2) -> True else False
    
      # Creates a grid of uint16 [possible values] values (2^16), used to check against probability thresholds 'p', 'f'
    random_grid = np.random.randint(0, 65536, size=board.shape, dtype=np.uint16) # Probabilities will be expressed in units of 1/65536
    
      # Condition 1 — any cell burning in the previous frame becomes empty.
    new_board[Burning_idx] = EMPTY
    
      # Condition 2 — fire spreads to neighbouring trees.
      # np.roll creates shifts the burning mask to the adjacent 4 neighbours (for any cell check if it has burning neighbours)
    up    = np.roll(Burning_idx, -1, axis=0) 
    down  = np.roll(Burning_idx,  1, axis=0) #axis=0, vertical shifts
    left  = np.roll(Burning_idx, -1, axis=1) #axis=1, horizontal shifts
    right = np.roll(Burning_idx,  1, axis=1)
                                           
      # Combine neighbour checks. "|" is boolean OR.
    fire_neighbours = up | down | left | right # True if any neighbour is burning
    
    fire_spread_idx = Tree_idx & fire_neighbours # "&" is boolean AND.
    new_board[fire_spread_idx] = FIRE # Fire spreads only where a tree exists AND any neighbour is burning.

      # Condition 3 — for each tree in the grid, if a random number (1-65336) is below threshold '65536f'
      # Then the tree has been stuck by lightning (with probability f).
    lightning_idx = Tree_idx & (random_grid < int(65536*f)) 
    new_board[lightning_idx] = FIRE
    
      # Condition 4 — new tree growth with probability p on empty tiles.
    grow_idx = Empty_idx & (random_grid < int(p*65536))
    new_board[grow_idx] = TREE    # Identical logic to above 
    
    return new_board

def run_simulation(L, frames, p, f, gif, save_2_file):
    import pandas as pd
    from tqdm import tqdm
    
    if gif: # if gif (bool) is True record gif data
        import imageio

        gif_name = f"forest_fire_{datetime.now().strftime('%H%M%S')}.gif"  #datetime.now() provides time stamp for files (uniquely idenitifyible) helpful for debugging
        writer = imageio.get_writer(gif_name, mode="I", fps=fps)  #"I" = multi-image gif, fps capped at 60

    tree_counts, burn_counts = [], []        # Population trackers (trees and burning trees through time)
    board = np.zeros((L, L), dtype=np.uint8) # Initialise board/grid: fully 0's (EMPTY), 1 byte << default 8 bytes 
  
    for t in tqdm(range(frames)): # Loop over every frame of the simulation, change to 'for t in tqdm(range(frames))' for simulation progress bar
        board = update_forest(board, p, f) # Advance 1 frame (≈+9.5 bytes/cell/frame)

        tree_counts.append(np.sum(board == TREE))  # Board accessed twice (≈+2 bytes/cell/frame)
        burn_counts.append(np.sum(board == FIRE))  # Fire and tree population tracking
        
        if gif:   # Generating gif (Resource heavy / very helpful for debugging)
            snapshot = cmap[board]            # Generates a board snapshot (frame)                      
            writer.append_data(snapshot)      # Append directly to gif (avoids RAM buildup unlike provided code) (less crashing / larger simulations possible)
                                              # Old method crashed for 8GB of RAM < 53 frames for a 5000x5000 (3 bytes for RBG per cell and per frame)
    if gif:  
        from IPython.display import HTML, display

        writer.close()      # Finalise gif (enables arbitrarily long animations)
        print(f"Saved animation: {gif_name}") 
        display(HTML(f"""<style>img.pixelated {{image-rendering: pixelated;image-rendering: crisp-edges;}}</style><div style='text-align:center;'><img class='pixelated'src='{gif_name}' width='{600}'></div>"""))
        # Pixelated/crisp-edges prevent blur from scaling a width of 1200 is possible on my notebook, but to be safe it is 600 for other users
              
    if save_2_file:
        file_name = f"Tree_Population_data_{datetime.now().strftime('%H%M%S')}.csv" # Saves the simulation data to a file for future analysis (safe against crashes)
        df = pd.DataFrame()                  # Setup dataframe (df) to save file
        df["tree_counts"] = tree_counts      # Append tree population per frame to df
        df["burn_counts"] = burn_counts      # Append burning population per frame to df
        df.to_csv(file_name, index=False) #Saves file (index is disabled as it can easily be recreated using np.arrange after cropping data to suitable range)
        print(f"Saved Population data to: {file_name}")
        
    return tree_counts, burn_counts
