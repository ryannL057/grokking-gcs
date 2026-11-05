import unittest

import numpy as np
import torch

from gcs.analysis.batched_gcs import (
    build_coherence_matrix,
    compute_batched_transformed_bases,
)


class BatchedGCSTest(unittest.TestCase):
    def test_batched_jvps_match_independent_sample_jvps(self):
        embeddings = torch.tensor(
            [
                [[0.2, -0.1], [1.0, 2.0]],
                [[0.5, 0.3], [3.0, 4.0]],
                [[-0.2, 0.8], [5.0, 6.0]],
            ],
            dtype=torch.float64,
        )
        tangent_bases = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.6, 0.8], [-0.8, 0.6]],
                [[-1.0, 0.0], [0.0, -1.0]],
            ],
            dtype=torch.float64,
        )

        def batched_path(x):
            return x[:, -1, :] ** 2 + 0.25 * x[:, 0, :]

        observed = compute_batched_transformed_bases(
            batched_path,
            embeddings,
            tangent_bases,
            token_probe="final",
        )

        expected_samples = []
        for sample_idx in range(embeddings.shape[0]):
            sample = embeddings[sample_idx]

            def single_path(x):
                return x[-1, :] ** 2 + 0.25 * x[0, :]

            expected_ranks = []
            for direction in tangent_bases[sample_idx]:
                tangent = torch.zeros_like(sample)
                tangent[-1, :] = direction
                _, transformed = torch.autograd.functional.jvp(
                    single_path,
                    sample,
                    tangent,
                )
                expected_ranks.append(transformed)
            expected_samples.append(torch.stack(expected_ranks))

        expected = torch.stack(expected_samples)
        torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)

    def test_operand_probe_averages_the_two_independent_jvps(self):
        embeddings = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]],
            ],
            dtype=torch.float64,
        )
        tangent_bases = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ],
            dtype=torch.float64,
        )

        def path(x):
            return 2.0 * x[:, 0, :] + 4.0 * x[:, 1, :] + x[:, 2, :]

        observed = compute_batched_transformed_bases(
            path,
            embeddings,
            tangent_bases,
            token_probe="operands",
        )
        expected = torch.tensor(
            [
                [[3.0, 0.0]],
                [[0.0, 3.0]],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(observed, expected)

    def test_coherence_matrix_matches_rankwise_absolute_cosine(self):
        transformed = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[-1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=torch.float64,
        )
        neighbors = np.asarray([[1], [0, 2], [1]], dtype=object)

        matrix = build_coherence_matrix(
            transformed,
            neighbors,
            alignment_method="cosine",
        )

        expected = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(matrix.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
