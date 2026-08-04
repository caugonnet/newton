# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX splitting state updates."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox import PROJECTION_STATUS_VALID
from newton._src.solvers.kamino._src.solvers.lox.iteration import SplittingState
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _world_dt(world_count: int, value: float, device: wp.DeviceLike) -> wp.array[wp.float32]:
    """Construct an explicit uniform per-world time-step array."""
    return wp.full(world_count, value, dtype=wp.float32, device=device)


class TestLOXIteration(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_cross_iterate_update_and_per_world_freeze(self):
        """Scale cross-iterate residuals and freeze converged worlds."""
        state = SplittingState([1, 2], device=self.device)
        initial = wp.zeros(3, dtype=vec6f, device=self.device)
        state.begin(initial, reset_dual=True)
        solution = wp.array(
            [[0.05, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6],
            dtype=vec6f,
            device=self.device,
        )
        state.prepare_projection(solution)
        state.projected_twist.assign(
            np.asarray([[0.04, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6], dtype=np.float32)
        )
        status = wp.full(2, PROJECTION_STATUS_VALID, dtype=wp.int32, device=self.device)
        state.finish_iteration(
            status,
            time_step=_world_dt(state.num_worlds, 0.01, self.device),
            position_tolerance=0.001,
            rotation_tolerance=0.001,
            velocity_tolerance=0.1,
        )

        np.testing.assert_allclose(state.residual_change.numpy(), [0.5, 2.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(state.residual_split.numpy(), [0.1, 1.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(state.residual_cross_iterate.numpy(), [0.5, 2.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(state.residual_total.numpy(), [0.5, 2.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(state.world_active.numpy(), [False, True])
        np.testing.assert_array_equal(state.world_converged.numpy(), [True, False])
        np.testing.assert_allclose(state.splitting_dual.numpy()[0, 0], -0.01, rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(state.splitting_dual.numpy()[1, 0], -0.1, rtol=0.0, atol=1.0e-7)

        state.prepare_projection(solution)
        projected = state.projected_twist.numpy()
        projected[1, 0] = 0.2
        state.projected_twist.assign(projected)
        state.finish_iteration(
            status,
            time_step=_world_dt(state.num_worlds, 0.01, self.device),
            position_tolerance=0.001,
            rotation_tolerance=0.001,
            velocity_tolerance=0.1,
        )
        np.testing.assert_array_equal(state.world_converged.numpy(), [True, True])
        np.testing.assert_array_equal(state.iteration_count.numpy(), [1, 2])

    def test_projection_failure_deactivates_world(self):
        state = SplittingState([1], device=self.device)
        initial = wp.zeros(1, dtype=vec6f, device=self.device)
        state.begin(initial)
        state.prepare_projection(initial)
        state.finish_iteration(
            wp.zeros(1, dtype=wp.int32, device=self.device),
            time_step=_world_dt(state.num_worlds, 0.01, self.device),
            position_tolerance=1.0e-5,
            rotation_tolerance=1.0e-5,
            velocity_tolerance=1.0e-5,
        )
        np.testing.assert_array_equal(state.world_active.numpy(), [False])
        np.testing.assert_array_equal(state.world_failed.numpy(), [True])
        np.testing.assert_array_equal(state.iteration_count.numpy(), [1])

    def test_cross_iterate_residual_uses_pose_scale(self):
        """Normalize cross-iterate motion with the pose tolerance."""
        state = SplittingState([1], device=self.device)
        initial = wp.zeros(1, dtype=vec6f, device=self.device)
        candidate = wp.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=vec6f, device=self.device)
        status = wp.full(1, PROJECTION_STATUS_VALID, dtype=wp.int32, device=self.device)

        state.begin(initial)
        state.prepare_projection(candidate)
        state.finish_iteration(
            status,
            time_step=_world_dt(state.num_worlds, 0.01, self.device),
            position_tolerance=0.001,
            rotation_tolerance=0.001,
            velocity_tolerance=0.1,
        )

        np.testing.assert_allclose(state.residual_change.numpy(), [0.1], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(state.residual_cross_iterate.numpy(), [0.1], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(state.residual_total.numpy(), [0.1], rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(state.world_converged.numpy(), [True])

    def test_split_residual_uses_velocity_tolerance(self):
        """Normalize the rigid v-p residual directly in velocity space."""
        state = SplittingState([1], device=self.device)
        initial = wp.zeros(1, dtype=vec6f, device=self.device)
        candidate = wp.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=vec6f, device=self.device)
        state.begin(initial)
        state.prepare_projection(candidate)
        state.projected_twist.assign(np.asarray([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))

        state.finish_iteration(
            wp.full(1, PROJECTION_STATUS_VALID, dtype=wp.int32, device=self.device),
            time_step=_world_dt(state.num_worlds, 0.01, self.device),
            position_tolerance=1.0,
            rotation_tolerance=1.0,
            velocity_tolerance=1.0e-3,
        )

        np.testing.assert_allclose(state.residual_split.numpy(), [5.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(state.world_converged.numpy(), [False])

    def test_lagged_velocity_residual_requires_a_second_iteration(self):
        """Gate convergence until a required lagged velocity residual is valid."""
        state = SplittingState([1], device=self.device)
        initial = wp.zeros(1, dtype=vec6f, device=self.device)
        status = wp.full(1, PROJECTION_STATUS_VALID, dtype=wp.int32, device=self.device)
        residual = wp.array([0.5], dtype=wp.float32, device=self.device)
        required = wp.array([1], dtype=wp.int32, device=self.device)
        state.begin(initial)

        state.prepare_projection(initial)
        state.finish_iteration(
            status,
            time_step=_world_dt(state.num_worlds, 0.01, self.device),
            position_tolerance=1.0e-5,
            rotation_tolerance=1.0e-5,
            velocity_tolerance=1.0e-5,
            lagged_velocity_residual=residual,
            lagged_velocity_required=required,
        )
        np.testing.assert_array_equal(state.world_active.numpy(), [True])
        np.testing.assert_array_equal(state.world_converged.numpy(), [False])
        np.testing.assert_allclose(state.residual_lagged_velocity.numpy(), [0.5])

        state.prepare_projection(initial)
        state.finish_iteration(
            status,
            time_step=_world_dt(state.num_worlds, 0.01, self.device),
            position_tolerance=1.0e-5,
            rotation_tolerance=1.0e-5,
            velocity_tolerance=1.0e-5,
            lagged_velocity_residual=residual,
            lagged_velocity_required=required,
        )
        np.testing.assert_array_equal(state.world_converged.numpy(), [True])
        np.testing.assert_array_equal(state.iteration_count.numpy(), [2])

    def test_structural_residual_participates_in_convergence(self):
        state = SplittingState([1], device=self.device)
        initial = wp.zeros(1, dtype=vec6f, device=self.device)
        status = wp.full(1, PROJECTION_STATUS_VALID, dtype=wp.int32, device=self.device)
        state.begin(initial)
        state.prepare_projection(initial)
        state.finish_iteration(
            status,
            time_step=_world_dt(state.num_worlds, 0.01, self.device),
            position_tolerance=1.0e-5,
            rotation_tolerance=1.0e-5,
            velocity_tolerance=1.0e-5,
            structural_residual=wp.array([2.0], dtype=wp.float32, device=self.device),
            projected_structural_residual=wp.array([3.0], dtype=wp.float32, device=self.device),
        )
        np.testing.assert_array_equal(state.world_active.numpy(), [True])
        np.testing.assert_array_equal(state.world_converged.numpy(), [False])
        np.testing.assert_array_equal(state.residual_structural.numpy(), [2.0])
        np.testing.assert_array_equal(state.residual_structural_projected.numpy(), [3.0])
        np.testing.assert_array_equal(state.residual_total.numpy(), [2.0])

        state.prepare_projection(initial)
        state.finish_iteration(
            status,
            time_step=_world_dt(state.num_worlds, 0.01, self.device),
            position_tolerance=1.0e-5,
            rotation_tolerance=1.0e-5,
            velocity_tolerance=1.0e-5,
            structural_residual=wp.zeros(1, dtype=wp.float32, device=self.device),
        )
        np.testing.assert_array_equal(state.world_converged.numpy(), [True])

    def test_nonfinite_global_candidate_fails_contact_free_world(self):
        state = SplittingState([1], device=self.device)
        state.begin(wp.zeros(1, dtype=vec6f, device=self.device))
        candidate = wp.array([[float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=vec6f, device=self.device)
        state.prepare_projection(candidate)
        state.finish_iteration(
            wp.full(1, PROJECTION_STATUS_VALID, dtype=wp.int32, device=self.device),
            time_step=_world_dt(state.num_worlds, 0.01, self.device),
            position_tolerance=1.0e-5,
            rotation_tolerance=1.0e-5,
            velocity_tolerance=1.0e-5,
        )

        np.testing.assert_array_equal(state.world_active.numpy(), [False])
        np.testing.assert_array_equal(state.world_failed.numpy(), [True])

    def test_dual_impulse_survives_weight_change(self):
        state = SplittingState([2], device=self.device)
        dual = np.asarray(
            [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], [6.0, -5.0, 4.0, -3.0, 2.0, -1.0]],
            dtype=np.float32,
        )
        old_weight = np.stack((2.0 * np.eye(6), 3.0 * np.eye(6))).astype(np.float32)
        new_inverse_weight = np.stack((0.25 * np.eye(6), 0.2 * np.eye(6))).astype(np.float32)
        body_has_unilateral = wp.array([1, 0], dtype=wp.int32, device=self.device)

        state.splitting_dual.assign(dual)
        state.store_dual_impulse(
            wp.array(old_weight, dtype=mat66f, device=self.device),
            body_has_unilateral,
        )
        np.testing.assert_allclose(state.splitting_dual_impulse.numpy()[0], 2.0 * dual[0])
        np.testing.assert_allclose(state.splitting_dual_impulse.numpy()[1], 0.0)

        state.splitting_dual.fill_(42.0)
        state.begin(wp.zeros(2, dtype=vec6f, device=self.device), reset_dual=False)
        state.restore_dual_from_impulse(
            wp.array(new_inverse_weight, dtype=mat66f, device=self.device),
            body_has_unilateral,
        )
        np.testing.assert_allclose(state.splitting_dual.numpy()[0], 0.5 * dual[0])
        np.testing.assert_allclose(state.splitting_dual.numpy()[1], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
