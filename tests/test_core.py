import unittest

import numpy as np

from forest_fire.analysis.cluster_area_perimeter import analyse_clusters
from forest_fire.analysis.sampling_time import sample_when_fire_ends
from forest_fire.constants import EMPTY, FIRE, TREE
from forest_fire.core_simulations.fast_simulation import update_forest
from forest_fire.core_simulations.id_simulation import update_forest_ID


class ForestFireCoreTests(unittest.TestCase):
    def test_fire_spreads_across_periodic_boundary(self):
        edges = [((0, 1), (2, 1)), ((2, 1), (0, 1)),
                 ((1, 0), (1, 2)), ((1, 2), (1, 0))]
        for source, destination in edges:
            with self.subTest(source=source, destination=destination):
                board = np.full((3, 3), EMPTY, dtype=np.uint8)
                board[source], board[destination] = FIRE, TREE
                updated = update_forest(board, p=0.0, f=0.0)
                self.assertEqual(updated[source], EMPTY)
                self.assertEqual(updated[destination], FIRE)
                identifiers = np.full((3, 3), -1, dtype=np.int32)
                identifiers[source] = 7
                tracked, ids, next_id = update_forest_ID(
                    board, identifiers, 0.0, 0.0, 8, 0)
                self.assertEqual(tracked[destination], FIRE)
                self.assertEqual(ids[destination], 7)
                self.assertEqual(next_id, 8)

    def test_single_tree_cluster_has_four_edge_perimeter(self):
        board = np.full((3, 3), EMPTY, dtype=np.uint8)
        board[1, 1] = TREE

        self.assertEqual(analyse_clusters(board, L=3), [{"area": 1, "perimeter": 4}])

    def test_cluster_connectivity_wraps_across_boundary(self):
        board = np.full((3, 3), EMPTY, dtype=np.uint8)
        board[0, 1] = TREE
        board[2, 1] = TREE

        self.assertEqual(analyse_clusters(board, L=3), [{"area": 2, "perimeter": 6}])

    def test_fire_end_sampling_detects_transition(self):
        self.assertTrue(sample_when_fire_ends(3, 0))
        self.assertFalse(sample_when_fire_ends(0, 0))
        self.assertFalse(sample_when_fire_ends(3, 1))


if __name__ == "__main__":
    unittest.main()
