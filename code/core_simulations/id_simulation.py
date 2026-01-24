import numpy as np
import imageio
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from IPython.display import HTML, display

from code.constants import EMPTY, TREE, FIRE, cmap, fps
from code.analysis.critical_time import oscillatory_regime_end, critical_regime_start, simulation_length as simulation_length_frames


def update_forest_ID(board, fire_id, p, f, next_id, t):

    # Copy current state
    new_board = board.copy()
    new_id    = fire_id.copy()    # fire_id grid stores an integer ID for each burning cell
                                  # -1 indicates "no fire present"

    Empty_idx   = (board == EMPTY) # Same logic as before
    Tree_idx    = (board == TREE)
    Burning_idx = (board == FIRE)
    random_grid = np.random.randint(0, 65536, size=board.shape, dtype=np.uint16) 

    # Condition 1 — Burning cells become empty
    new_board[Burning_idx] = EMPTY
    new_id[Burning_idx]    = -1    # Remove fire ID once burning has finished (-1 represents no fire)

    # Condition 2 — Fire spreads to neighbouring trees (ID propagation)
    up    = np.roll(Burning_idx, -1, axis=0)
    down  = np.roll(Burning_idx,  1, axis=0) # Neighbour detection (same spatial logic as base model)
    left  = np.roll(Burning_idx, -1, axis=1)
    right = np.roll(Burning_idx,  1, axis=1)
    fire_neighbours = up | down | left | right

        # Shift the fire-ID grid in the same way to determine which fire caused the spread
    upID    = np.roll(fire_id, -1, axis=0)
    downID  = np.roll(fire_id,  1, axis=0)
    leftID  = np.roll(fire_id, -1, axis=1)
    rightID = np.roll(fire_id,  1, axis=1)

        # Stack neighbour IDs and select the smallest valid ID (if multiple fires reach the same point choose the fire assigned an ID first)
    neighbour_ids = np.stack([upID, downID, leftID, rightID])
    
        # Ignore invalid IDs (-1), take all neighbouring fire IDs and a value for inf
    valid = np.where(neighbour_ids >= 0, neighbour_ids, np.inf) # If there is no fire spread inf is picked
    min_ids = np.min(valid, axis=0)                                  
    min_ids[min_ids == np.inf] = -1                             # If there is no fire spread set to -1

        # Apply fire spread
    fire_spread_idx = Tree_idx & fire_neighbours
    new_board[fire_spread_idx] = FIRE
    new_id[fire_spread_idx]    = min_ids[fire_spread_idx].astype(np.int32) # If not then take the smallest neighbouring ID and assign it
                                           # int32 has 4.3 billion values, RAM would be maximised well before this could ever be reached

    # Condition 3 — Lightning ignition (unique ID assignment)
    lightning_idx = Tree_idx & (random_grid < int(f * 65536))

    if np.any(lightning_idx):
        # Each lightning strike starts a new fire event
        # Assign sequential IDs to all new ignitions in this frame
        ids = np.arange(next_id, next_id + np.sum(lightning_idx), dtype=np.int32)

        new_board[lightning_idx] = FIRE
        new_id[lightning_idx]    = ids

        next_id += np.sum(lightning_idx)  # advance ID counter

    # Condition 4 — Tree growth on empty cells
    grow_idx = Empty_idx & (random_grid < int(p * 65536))
    new_board[grow_idx] = TREE

    return new_board, new_id, next_id

def run_simulation_ID(L, p, f, gif):


    #   Non-critical region: exp(-D t) = 0.1:  t = 0               -> t = [ln(10)/A] f^-B
    #   Critical region:   exp(-D t) = 0.001:  t=[ln(1000)/A] f^-B -> t=2*[ln(1000)/A] f^-B

    non_critical = oscillatory_regime_end(f)
    critical = critical_regime_start(f)
    simulation_length = simulation_length_frames(f)                     # Ensure long sampling in critical regime

    board   = np.zeros((L, L), dtype=np.uint8)           # start fully EMPTY
    fire_id = np.full((L, L), -1, dtype=np.int32)        # -1 = no fire ID assigned

    next_id = 0                                          # first fire ID
    early_history       = defaultdict(list)              # fire sizes in non-critical region
    equilibrium_history = defaultdict(list)              # fire sizes in critical region

    # GIF setup
    if gif:
        fire_colors   = {}                               # fire_id provides an RGB colour (for gif)
        gif_name      = f"fire_sim_{datetime.now().strftime('%H%M%S')}.gif"
        writer        = imageio.get_writer(gif_name, mode="I", fps=fps)

    # Main simulation loop
    for t in tqdm(range(simulation_length)):     

        board, fire_id, next_id = update_forest_ID(board, fire_id, p, f, next_id, t) # Same as before

        for fid in np.unique(fire_id):          # For every unique positive fire ID (i.e. ignoring -1)
            if fid >= 0:
                size = np.sum(fire_id == fid)   # Calculate the amount of cell that are on fire this frame from that fire

                if t < non_critical:            # Store the history of the fire sizes through time while the population in the oscillatory population region
                    early_history[fid].append(size)

                elif t >= critical:             # Store the history of the fire sizes through time in the critical region
                    equilibrium_history[fid].append(size)

        if gif:
            frame = np.zeros((L, L, 3), dtype=np.uint8) # Gif generation is same as before
            frame[board == EMPTY] = cmap[EMPTY]
            frame[board == TREE]  = cmap[TREE]

            ys, xs = np.where(board == FIRE)
            for y, x in zip(ys, xs):            # Iterate over every combination of x,y (all cells in the grid)
                fid = fire_id[y, x]
                if fid not in fire_colors:      # If there is a new fire ID assign it a random colour
                    fire_colors[fid] = np.array([
                        np.random.randint(50, 256),
                        np.random.randint(0, 160),
                        np.random.randint(50, 256)
                    ], dtype=np.uint8)
                frame[y, x] = fire_colors[fid] # Then set all of the fires with that ID to that colour
            writer.append_data(frame) # Add the entire image to the gif
            
    if gif:     # Close GIF writer and display it
        writer.close()
        print("Saved:", gif_name)
        display(HTML(f"""<style>img.pixelated {{image-rendering: pixelated;image-rendering: crisp-edges;}}</style><div style='text-align:center;'><img class='pixelated' src='{gif_name}' width='600'></div>"""))

    return early_history, equilibrium_history
