import copy
import unittest

import numpy as np
import torch

from gcs.analysis.reference_geometry import (
    count_knn_components,
    rebuild_geometry_variant,
)


class ReferenceGeometryTest(unittest.TestCase):
    def setUp(self):
        x_sample = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.2],
                [2.0, 1.0, 0.4],
                [1.0, 2.0, 0.6],
            ],
            dtype=torch.float64,
        )
        self.geometry = {
            "X_sample": x_sample,
            "X_prompts": torch.arange(18, dtype=torch.float64).reshape(6, 3),
            "X_embeddings": [
                torch.full((3, 3), float(index), dtype=torch.float64)
                for index in range(6)
            ],
            "knn_indices": np.zeros((6, 2), dtype=int),
            "knn_distances": np.zeros((6, 2), dtype=float),
            "all_input_tangent_bases": [
                torch.zeros((1, 3), dtype=torch.float64) for _ in range(6)
            ],
            "N": 6,
            "k": 2,
            "d": 1,
            "d_model": 3,
            "layer_idx": 0,
            "hook_point": "hook_resid_post",
            "model_type": "transformer",
            "embedding_type": None,
        }

    def test_variant_uses_prefix_for_smaller_n_and_rebuilds_shapes(self):
        original = copy.deepcopy(self.geometry)
        variant = rebuild_geometry_variant(
            self.geometry,
            N=5,
            k=2,
            d=2,
            device="cpu",
        )

        torch.testing.assert_close(
            variant["X_sample"],
            original["X_sample"][:5],
        )
        torch.testing.assert_close(
            variant["X_prompts"],
            original["X_prompts"][:5],
        )
        self.assertEqual(len(variant["X_embeddings"]), 5)
        self.assertEqual(variant["knn_indices"].shape, (5, 2))
        self.assertEqual(len(variant["all_input_tangent_bases"]), 5)
        self.assertTrue(
            all(tuple(basis.shape) == (2, 3) for basis in variant["all_input_tangent_bases"])
        )
        self.assertEqual((variant["N"], variant["k"], variant["d"]), (5, 2, 2))

        # The source geometry is an immutable reference for every variant.
        torch.testing.assert_close(self.geometry["X_sample"], original["X_sample"])
        self.assertEqual(self.geometry["N"], 6)

    def test_invalid_variant_dimensions_fail_closed(self):
        with self.assertRaises(ValueError):
            rebuild_geometry_variant(self.geometry, N=2, k=2, d=1, device="cpu")
        with self.assertRaises(ValueError):
            rebuild_geometry_variant(self.geometry, N=6, k=2, d=3, device="cpu")

    def test_component_counter_uses_symmetrized_neighbor_union(self):
        connected = np.asarray([[1], [0, 2], [1]], dtype=object)
        disconnected = np.asarray([[1], [0], [3], [2]], dtype=object)

        self.assertEqual(count_knn_components(connected), 1)
        self.assertEqual(count_knn_components(disconnected), 2)


if __name__ == "__main__":
    unittest.main()
