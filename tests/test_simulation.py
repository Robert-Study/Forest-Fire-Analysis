import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from forest_fire.constants import EMPTY, FIRE, TREE
from forest_fire.core_simulations.fast_simulation import update_forest, run_simulation, validate_probabilities
from forest_fire.core_simulations.id_simulation import update_forest_ID, run_simulation_ID
from forest_fire.analysis.cluster_area_perimeter import analyse_clusters
from forest_fire.analysis.event_statistics import completed_fire_sizes


class SimulationTests(unittest.TestCase):
    def test_synchronous_burning_cell_cannot_regrow_in_same_step(self):
        board = np.full((3, 3), FIRE, dtype=np.uint8)
        np.testing.assert_array_equal(update_forest(board, 1, 0), np.full_like(board, EMPTY))

    def test_vectorised_step_agrees_with_scalar_reference(self):
        rng = np.random.default_rng(3)
        board = rng.integers(0, 3, (7, 8), dtype=np.uint8)
        draws = np.random.default_rng(9).integers(0, 2**32, board.shape, dtype=np.uint32)
        expected = board.copy()
        for y in range(7):
            for x in range(8):
                near_fire = any(board[(y+dy)%7, (x+dx)%8] == FIRE for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)])
                if board[y,x] == FIRE:
                    expected[y,x] = EMPTY
                elif board[y,x] == TREE and (near_fire or draws[y,x] < int(.05*2**32)):
                    expected[y,x] = FIRE
                elif board[y,x] == EMPTY and draws[y,x] < int(.4*2**32):
                    expected[y,x] = TREE
        np.testing.assert_array_equal(update_forest(board, .4, .05, np.random.default_rng(9)), expected)

    def test_low_probability_does_not_silently_round_to_zero(self):
        validate_probabilities(.001, 1e-7, 32)
        with self.assertRaisesRegex(ValueError, 'resolution'):
            validate_probabilities(.001, 1e-7, 16)

    def test_seeded_counts_reproduce_and_csv_is_complete(self):
        expected = run_simulation(8, 30, .2, .05, seed=7)
        with tempfile.TemporaryDirectory() as temp:
            actual = run_simulation(8, 30, .2, .05, seed=7, save_2_file=True, output_dir=temp)
            self.assertEqual(actual, expected)
            self.assertEqual(len(next(Path(temp).glob('*.csv')).read_text().splitlines()), 31)

    def test_lightning_does_not_overwrite_a_spreading_lineage(self):
        board = np.full((4, 4), EMPTY, dtype=np.uint8)
        board[1,1], board[1,2] = FIRE, TREE
        ids = np.full((4, 4), -1, dtype=np.int32)
        ids[1,1] = 7
        _, after, next_id = update_forest_ID(board, ids, 0, 1, 8, 0)
        self.assertEqual(after[1,2], 7)
        self.assertEqual(next_id, 8)

    def test_complete_large_events_retained_active_events_excluded(self):
        sizes, perimeters = completed_fire_sizes({1:150, 2:80, 3:4}, {1:11, 2:12, 3:2}, {2}, 10, {1:60})
        np.testing.assert_array_equal(sizes, [150])
        np.testing.assert_array_equal(perimeters, [60])

    def test_cluster_edges_include_burning_neighbours(self):
        board = np.full((3, 3), FIRE, dtype=np.uint8)
        board[1,1] = TREE
        self.assertEqual(analyse_clusters(board, 3), [{'area':1, 'perimeter':4}])

    def test_event_spanning_window_keeps_full_history(self):
        # Event 0 starts early, persists across the boundary and ends. Event 1
        # remains active at the run end and must not enter either distribution.
        module = 'forest_fire.core_simulations.id_simulation'
        def step(board, ids, p, f, next_id, t, **kwargs):
            new_ids = np.full((2,2), -1, np.int32)
            if t < 3:
                new_ids[0,0] = 0
            if t == 3:
                new_ids[0,1] = 1
            return (new_ids >= 0).astype(np.uint8)*FIRE, new_ids, 2
        with patch(module+'.oscillatory_regime_end', return_value=1), patch(module+'.critical_regime_start', return_value=2), patch(module+'.simulation_length_frames', return_value=4), patch(module+'.update_forest_ID', side_effect=step):
            early, late = run_simulation_ID(2, .2, .01)
        self.assertEqual(dict(early), {0:[1,1,1]})
        self.assertEqual(dict(late), {})


if __name__ == '__main__':
    unittest.main()
