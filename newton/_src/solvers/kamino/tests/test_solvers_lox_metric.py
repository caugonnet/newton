# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX mass-split constraint metrics."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat66f
from newton._src.solvers.kamino._src.solvers.lox.metric import (
    METRIC_STATUS_INVALID,
    METRIC_STATUS_VALID,
    compute_mass_split_metric,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


@wp.kernel
def _evaluate_block_metric(
    delassus: wp.array[mat66f],
    dimension: wp.array[wp.int32],
    multiplicity: wp.array[wp.int32],
    scale: wp.array[wp.float32],
    inverse: wp.array[mat66f],
    penalty: wp.array[mat66f],
    status: wp.array[wp.int32],
):
    index = wp.tid()
    result = compute_mass_split_metric(delassus[index], dimension[index], multiplicity[index], scale[index])
    inverse[index] = result.inverse
    penalty[index] = result.penalty
    status[index] = result.status


class TestLOXMetric(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_full_block_inverse_multiplicity_and_scale(self):
        delassus = np.asarray(
            [
                [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[3.0, 0.8, -0.4], [0.8, 2.5, 0.6], [-0.4, 0.6, 1.7]],
            ],
            dtype=np.float32,
        )
        padded = np.zeros((2, 6, 6), dtype=np.float32)
        padded[0, 0, 0] = delassus[0, 0, 0]
        padded[1, :3, :3] = delassus[1]
        inverse = wp.zeros(2, dtype=mat66f, device=self.device)
        penalty = wp.zeros_like(inverse)
        status = wp.zeros(2, dtype=wp.int32, device=self.device)
        wp.launch(
            _evaluate_block_metric,
            dim=2,
            inputs=[
                wp.array(padded, dtype=mat66f, device=self.device),
                wp.array([1, 3], dtype=wp.int32, device=self.device),
                wp.array([1, 2], dtype=wp.int32, device=self.device),
                wp.array([7.0, 5.0], dtype=wp.float32, device=self.device),
            ],
            outputs=[inverse, penalty, status],
            device=self.device,
        )

        expected_first = 0.25
        self.assertAlmostEqual(float(inverse.numpy()[0, 0, 0]), expected_first, places=6)
        expected_second = np.linalg.inv(2.0 * delassus[1])
        np.testing.assert_allclose(inverse.numpy()[1, :3, :3], expected_second, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(penalty.numpy()[1, :3, :3], 5.0 * expected_second, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(inverse.numpy()[1], inverse.numpy()[1].T, atol=1.0e-7)
        np.testing.assert_array_equal(status.numpy(), [METRIC_STATUS_VALID, METRIC_STATUS_VALID])

    def test_invalid_block_reports_invalid_status(self):
        """Reject a singular active block."""
        invalid = wp.zeros(1, dtype=mat66f, device=self.device)
        invalid_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            _evaluate_block_metric,
            dim=1,
            inputs=[
                invalid,
                wp.array([2], dtype=wp.int32, device=self.device),
                wp.array([1], dtype=wp.int32, device=self.device),
                wp.array([1.0], dtype=wp.float32, device=self.device),
            ],
            outputs=[wp.zeros_like(invalid), wp.zeros_like(invalid), invalid_status],
            device=self.device,
        )
        self.assertEqual(int(invalid_status.numpy()[0]), METRIC_STATUS_INVALID)

    def test_timestep_scale_is_applied_only_to_structural_penalty(self):
        physical = np.zeros((1, 6, 6), dtype=np.float32)
        physical[0, :2, :2] = np.asarray([[2.0, 0.5], [0.5, 1.25]], dtype=np.float32)
        time_step = 0.02
        gamma = 4.0
        inverse = wp.zeros(1, dtype=mat66f, device=self.device)
        penalty = wp.zeros_like(inverse)
        status = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            _evaluate_block_metric,
            dim=1,
            inputs=[
                wp.array(physical, dtype=mat66f, device=self.device),
                wp.array([2], dtype=wp.int32, device=self.device),
                wp.array([1], dtype=wp.int32, device=self.device),
                wp.array([gamma / time_step**2], dtype=wp.float32, device=self.device),
            ],
            outputs=[inverse, penalty, status],
            device=self.device,
        )
        expected_inverse = np.linalg.inv(physical[0, :2, :2])
        np.testing.assert_allclose(inverse.numpy()[0, :2, :2], expected_inverse, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(
            time_step**2 * penalty.numpy()[0, :2, :2], gamma * expected_inverse, rtol=2.0e-5, atol=2.0e-6
        )


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
