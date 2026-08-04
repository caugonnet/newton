# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Composed regression for the LOX splitting loop."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat36f, vec6f
from newton._src.solvers.kamino._src.solvers.lox import (
    BatchedPrimalBodySystem,
    SplittingState,
    project_constraints_sequential,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


class TestLOXLoop(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_one_body_hard_contact_returns_projected_velocity(self):
        system = BatchedPrimalBodySystem([1], device=self.device)
        time_step = wp.full(1, 0.01, dtype=wp.float32, device=self.device)
        initial_twist = wp.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=vec6f, device=self.device)
        system.assemble_bodies(
            wp.ones(1, dtype=wp.float32, device=self.device),
            wp.array([np.eye(3, dtype=np.float32)], dtype=wp.mat33f, device=self.device),
            initial_twist,
            wp.zeros(1, dtype=vec6f, device=self.device),
            time_step=time_step,
        )
        system.build_weighted_matrix()
        system.factorize()

        state = SplittingState([1], device=self.device)
        state.begin(initial_twist, reset_dual=True)
        world_offset = wp.zeros(1, dtype=wp.int32, device=self.device)
        world_contact_count = wp.ones(1, dtype=wp.int32, device=self.device)
        world_limit_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        contact_body_first = wp.zeros(1, dtype=wp.int32, device=self.device)
        contact_body_second = wp.full(1, -1, dtype=wp.int32, device=self.device)
        contact_jacobian = np.zeros((1, 3, 6), dtype=np.float32)
        contact_jacobian[0, 0, 1] = 1.0
        contact_jacobian[0, 1, 2] = 1.0
        contact_jacobian[0, 2, 0] = 1.0
        contact_jacobian_first = wp.array(contact_jacobian, dtype=mat36f, device=self.device)
        contact_jacobian_second = wp.zeros(1, dtype=mat36f, device=self.device)
        contact_bias = wp.zeros(1, dtype=wp.vec3f, device=self.device)
        contact_friction = wp.array([0.5], dtype=wp.float32, device=self.device)
        contact_reaction = wp.zeros(1, dtype=wp.vec3f, device=self.device)
        contact_velocity = wp.zeros(1, dtype=wp.vec3f, device=self.device)
        empty_body = wp.empty(0, dtype=wp.int32, device=self.device)
        empty_jacobian = wp.empty(0, dtype=vec6f, device=self.device)
        empty_scalar = wp.empty(0, dtype=wp.float32, device=self.device)
        projection_status = wp.zeros(1, dtype=wp.int32, device=self.device)

        for _iteration in range(25):
            system.solve_candidate(state.projected_twist, state.splitting_dual)
            state.prepare_projection(system.solution.view(dtype=vec6f))
            project_constraints_sequential(
                10,
                state.world_active,
                world_offset,
                world_limit_count,
                empty_body,
                empty_body,
                empty_jacobian,
                empty_jacobian,
                empty_scalar,
                world_offset,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_bias,
                contact_friction,
                world_offset,
                world_limit_count,
                empty_body,
                empty_body,
                empty_jacobian,
                empty_jacobian,
                empty_scalar,
                system.inverse_weight,
                state.projected_twist,
                empty_scalar,
                empty_scalar,
                contact_reaction,
                contact_velocity,
                empty_scalar,
                empty_scalar,
                projection_status,
            )
            state.finish_iteration(
                projection_status,
                time_step=time_step,
                position_tolerance=1.0e-3,
                rotation_tolerance=1.0e-3,
                velocity_tolerance=0.1,
            )
        state.mark_iteration_limit()

        np.testing.assert_allclose(state.projected_twist.numpy(), 0.0, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(contact_reaction.numpy()[0], [0.0, 0.0, 1.0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(state.world_converged.numpy(), [True])
        np.testing.assert_array_equal(state.world_failed.numpy(), [False])
        self.assertLessEqual(int(state.iteration_count.numpy()[0]), 25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
