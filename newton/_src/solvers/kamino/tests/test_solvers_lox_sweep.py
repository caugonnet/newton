# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX body-space projection sweeps."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat36f, mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox.projection import PROJECTION_STATUS_VALID
from newton._src.solvers.kamino._src.solvers.lox.sweep import (
    compute_projection_residuals,
    prepare_jacobi_projection_data,
    project_constraints_jacobi,
    project_constraints_sequential,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


class TestLOXSweep(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_heterogeneous_world_contact_and_limit_sweeps(self):
        identity = np.eye(6, dtype=np.float32)
        inverse_weight = wp.array([identity, identity], dtype=mat66f, device=self.device)
        projected_twist = wp.array(
            [[0.0, 0.2, 0.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=vec6f,
            device=self.device,
        )

        contact_jacobian = np.zeros((1, 3, 6), dtype=np.float32)
        contact_jacobian[0, 0, 1] = 1.0
        contact_jacobian[0, 1, 2] = 1.0
        contact_jacobian[0, 2, 0] = 1.0
        zero_contact_jacobian = np.zeros((1, 3, 6), dtype=np.float32)
        limit_jacobian = np.zeros((1, 6), dtype=np.float32)
        limit_jacobian[0, 0] = 1.0
        zero_limit_jacobian = np.zeros((1, 6), dtype=np.float32)

        world_contact_offset = wp.array([0, 1], dtype=wp.int32, device=self.device)
        world_contact_count = wp.array([1, 0], dtype=wp.int32, device=self.device)
        contact_world = wp.array([0], dtype=wp.int32, device=self.device)
        contact_local = wp.array([0], dtype=wp.int32, device=self.device)
        contact_body_first = wp.array([0], dtype=wp.int32, device=self.device)
        contact_body_second = wp.array([-1], dtype=wp.int32, device=self.device)
        contact_jacobian_first = wp.array(contact_jacobian, dtype=mat36f, device=self.device)
        contact_jacobian_second = wp.array(zero_contact_jacobian, dtype=mat36f, device=self.device)
        contact_bias = wp.array([[0.0, 0.0, -1.0]], dtype=wp.vec3f, device=self.device)
        contact_friction = wp.array([0.5], dtype=wp.float32, device=self.device)
        contact_reaction = wp.array([[0.0, 0.0, 0.5]], dtype=wp.vec3f, device=self.device)
        contact_velocity = wp.zeros(1, dtype=wp.vec3f, device=self.device)

        world_limit_offset = wp.array([0, 0], dtype=wp.int32, device=self.device)
        world_limit_count = wp.array([0, 1], dtype=wp.int32, device=self.device)
        limit_world = wp.array([1], dtype=wp.int32, device=self.device)
        limit_local = wp.array([0], dtype=wp.int32, device=self.device)
        limit_body_first = wp.array([1], dtype=wp.int32, device=self.device)
        limit_body_second = wp.array([-1], dtype=wp.int32, device=self.device)
        limit_jacobian_first = wp.array(limit_jacobian, dtype=vec6f, device=self.device)
        limit_jacobian_second = wp.array(zero_limit_jacobian, dtype=vec6f, device=self.device)
        limit_bias = wp.array([0.0], dtype=wp.float32, device=self.device)
        limit_reaction = wp.array([0.5], dtype=wp.float32, device=self.device)
        limit_velocity = wp.zeros(1, dtype=wp.float32, device=self.device)
        world_status = wp.zeros(2, dtype=wp.int32, device=self.device)
        world_active = wp.ones(2, dtype=wp.bool, device=self.device)
        empty_int = wp.empty(0, dtype=wp.int32, device=self.device)
        empty_vec6 = wp.empty(0, dtype=vec6f, device=self.device)
        empty_float = wp.empty(0, dtype=wp.float32, device=self.device)
        world_friction_offset = wp.zeros(2, dtype=wp.int32, device=self.device)
        world_friction_count = wp.zeros(2, dtype=wp.int32, device=self.device)
        contact_residual = wp.zeros(1, dtype=wp.float32, device=self.device)
        limit_residual = wp.zeros(1, dtype=wp.float32, device=self.device)
        world_contact_residual_max = wp.zeros(2, dtype=wp.float32, device=self.device)
        world_limit_residual_max = wp.zeros(2, dtype=wp.float32, device=self.device)

        project_constraints_sequential(
            3,
            world_active,
            world_friction_offset,
            world_friction_count,
            empty_int,
            empty_int,
            empty_vec6,
            empty_vec6,
            empty_float,
            world_contact_offset,
            world_contact_count,
            contact_body_first,
            contact_body_second,
            contact_jacobian_first,
            contact_jacobian_second,
            contact_bias,
            contact_friction,
            world_limit_offset,
            world_limit_count,
            limit_body_first,
            limit_body_second,
            limit_jacobian_first,
            limit_jacobian_second,
            limit_bias,
            inverse_weight,
            projected_twist,
            empty_float,
            empty_float,
            contact_reaction,
            contact_velocity,
            limit_reaction,
            limit_velocity,
            world_status,
        )
        compute_projection_residuals(
            world_active,
            world_status,
            empty_int,
            empty_int,
            world_friction_count,
            empty_int,
            empty_int,
            empty_vec6,
            empty_vec6,
            empty_float,
            empty_float,
            contact_world,
            contact_local,
            world_contact_count,
            contact_body_first,
            contact_body_second,
            contact_jacobian_first,
            contact_jacobian_second,
            contact_bias,
            contact_friction,
            contact_reaction,
            limit_world,
            limit_local,
            world_limit_count,
            limit_body_first,
            limit_body_second,
            limit_jacobian_first,
            limit_jacobian_second,
            limit_bias,
            limit_reaction,
            inverse_weight,
            projected_twist,
            empty_float,
            contact_velocity,
            limit_velocity,
            contact_residual,
            limit_residual,
            empty_float,
            world_contact_residual_max,
            world_limit_residual_max,
            wp.zeros(2, dtype=wp.float32, device=self.device),
        )

        np.testing.assert_allclose(projected_twist.numpy()[0, :2], [1.0, 0.0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(projected_twist.numpy()[1, 0], 0.0, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(contact_reaction.numpy()[0], [-0.2, 0.0, 1.0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(limit_reaction.numpy()[0], 2.0, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(contact_velocity.numpy()[0], [0.0, 0.0, 1.0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(limit_velocity.numpy()[0], 0.0, rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(world_status.numpy(), np.full(2, PROJECTION_STATUS_VALID, dtype=np.int32))
        np.testing.assert_allclose(contact_residual.numpy(), 0.0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(limit_residual.numpy(), 0.0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(world_contact_residual_max.numpy(), 0.0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(world_limit_residual_max.numpy(), 0.0, rtol=0.0, atol=1.0e-6)

    def test_invalid_iteration_count(self):
        with self.assertRaises(ValueError):
            project_constraints_sequential(0, *([None] * 32))

    def test_mass_split_jacobi_scales_body_contributions(self):
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)] * 2, dtype=mat66f, device=self.device)
        projected_twist = wp.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6], dtype=vec6f, device=self.device)
        twist_delta = wp.zeros(2, dtype=vec6f, device=self.device)
        body_world = wp.zeros(2, dtype=wp.int32, device=self.device)
        body_constraint_count = wp.array([2, 1], dtype=wp.int32, device=self.device)
        static_body_constraint_count = wp.zeros(2, dtype=wp.int32, device=self.device)

        contact_jacobian = np.zeros((2, 3, 6), dtype=np.float32)
        contact_jacobian[:, 0, 1] = 1.0
        contact_jacobian[:, 1, 2] = 1.0
        contact_jacobian[:, 2, 0] = 1.0
        contact_world = wp.zeros(2, dtype=wp.int32, device=self.device)
        contact_local = wp.array([0, 1], dtype=wp.int32, device=self.device)
        world_contact_count = wp.array([2], dtype=wp.int32, device=self.device)
        contact_body_first = wp.zeros(2, dtype=wp.int32, device=self.device)
        contact_body_second = wp.array([1, -1], dtype=wp.int32, device=self.device)
        contact_jacobian_first = wp.array(contact_jacobian, dtype=mat36f, device=self.device)
        contact_jacobian_second_values = np.zeros_like(contact_jacobian)
        contact_jacobian_second_values[0] = contact_jacobian[0]
        contact_jacobian_second = wp.array(contact_jacobian_second_values, dtype=mat36f, device=self.device)
        contact_bias = wp.zeros(2, dtype=wp.vec3f, device=self.device)
        contact_friction = wp.zeros(2, dtype=wp.float32, device=self.device)
        contact_delassus = wp.zeros(2, dtype=wp.mat33f, device=self.device)
        contact_delassus_normal_first = wp.zeros(2, dtype=wp.mat33f, device=self.device)
        contact_reaction = wp.zeros(2, dtype=wp.vec3f, device=self.device)

        empty_int = wp.empty(0, dtype=wp.int32, device=self.device)
        empty_vec6 = wp.empty(0, dtype=vec6f, device=self.device)
        empty_float = wp.empty(0, dtype=wp.float32, device=self.device)
        world_limit_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        prepared_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        projection_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        world_active = wp.ones(1, dtype=wp.bool, device=self.device)

        prepare_jacobi_projection_data(
            empty_int,
            empty_int,
            world_limit_count,
            empty_int,
            empty_int,
            empty_vec6,
            empty_vec6,
            contact_world,
            contact_local,
            world_contact_count,
            contact_body_first,
            contact_body_second,
            contact_jacobian_first,
            contact_jacobian_second,
            contact_bias,
            contact_friction,
            empty_int,
            empty_int,
            world_limit_count,
            empty_int,
            empty_int,
            empty_vec6,
            empty_vec6,
            body_constraint_count,
            static_body_constraint_count,
            inverse_weight,
            empty_float,
            contact_delassus,
            contact_delassus_normal_first,
            empty_float,
            prepared_status,
        )
        project_constraints_jacobi(
            1,
            world_active,
            body_world,
            empty_int,
            empty_int,
            world_limit_count,
            empty_int,
            empty_int,
            empty_vec6,
            empty_vec6,
            empty_float,
            empty_float,
            contact_world,
            contact_local,
            world_contact_count,
            contact_body_first,
            contact_body_second,
            contact_jacobian_first,
            contact_jacobian_second,
            contact_bias,
            contact_friction,
            contact_delassus,
            contact_delassus_normal_first,
            empty_int,
            empty_int,
            world_limit_count,
            empty_int,
            empty_int,
            empty_vec6,
            empty_vec6,
            empty_float,
            empty_float,
            inverse_weight,
            projected_twist,
            twist_delta,
            contact_reaction,
            empty_float,
            empty_float,
            prepared_status,
            projection_status,
        )

        np.testing.assert_allclose(contact_delassus.numpy()[:, 2, 2], [3.0, 2.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(contact_reaction.numpy()[:, 2], [1.0 / 3.0, 0.5], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            projected_twist.numpy(),
            [[-1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(projection_status.numpy(), [PROJECTION_STATUS_VALID])


if __name__ == "__main__":
    unittest.main(verbosity=2)
