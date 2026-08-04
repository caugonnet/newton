# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX body-space problem primitives."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox.problem import (
    compute_augmented_joint_multiplier,
    compute_augmented_joint_row,
    compute_body_explicit_wrench,
    compute_body_inertial_system,
    compute_dynamic_joint_row,
    compute_velocity_distance,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


@wp.kernel
def _evaluate_problem_primitives(
    mass: wp.float32,
    inertia: wp.mat33f,
    velocity: vec6f,
    wrench: vec6f,
    jacobian: vec6f,
    time_step: wp.float32,
    body_matrix: wp.array[mat66f],
    body_rhs: wp.array[vec6f],
    dynamic_matrix: wp.array[mat66f],
    dynamic_rhs: wp.array[vec6f],
    augmented_matrix: wp.array[mat66f],
    augmented_rhs: wp.array[vec6f],
    multiplier: wp.array[wp.float32],
    distance: wp.array[wp.float32],
    explicit_wrench: wp.array[vec6f],
):
    body = compute_body_inertial_system(mass, inertia, velocity, wrench, time_step)
    dynamic = compute_dynamic_joint_row(jacobian, 0.7, -1.25)
    augmented = compute_augmented_joint_row(jacobian, -0.03, 2.5, 40.0, time_step, 0.7)
    body_matrix[0] = body.matrix
    body_rhs[0] = body.right_hand_side
    dynamic_matrix[0] = dynamic.matrix
    dynamic_rhs[0] = dynamic.right_hand_side
    augmented_matrix[0] = augmented.matrix
    augmented_rhs[0] = augmented.right_hand_side
    multiplier[0] = compute_augmented_joint_multiplier(2.5, 40.0, -0.01)
    distance[0] = compute_velocity_distance(velocity, velocity + jacobian, time_step, 2.0e-3, 4.0e-3)
    explicit_wrench[0] = compute_body_explicit_wrench(
        mass,
        inertia,
        velocity,
        vec6f(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        vec6f(-1.0, 1.0, -2.0, 2.0, -1.0, 0.5),
        wp.vec3f(0.0, 0.0, -9.81),
    )


class TestLOXProblem(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_primal_contributions_and_residual_scaling(self):
        mass = np.float32(2.0)
        inertia = np.asarray([[0.5, 0.1, 0.0], [0.1, 0.8, -0.05], [0.0, -0.05, 1.2]], dtype=np.float32)
        velocity = np.asarray([1.0, -2.0, 0.5, 0.2, -0.4, 0.7], dtype=np.float32)
        wrench = np.asarray([3.0, -1.0, 2.0, 0.5, 1.5, -0.25], dtype=np.float32)
        jacobian = np.asarray([0.25, -0.5, 1.0, -0.75, 0.1, 0.4], dtype=np.float32)
        time_step = np.float32(0.02)

        body_matrix = wp.empty(1, dtype=mat66f, device=self.device)
        body_rhs = wp.empty(1, dtype=vec6f, device=self.device)
        dynamic_matrix = wp.empty(1, dtype=mat66f, device=self.device)
        dynamic_rhs = wp.empty(1, dtype=vec6f, device=self.device)
        augmented_matrix = wp.empty(1, dtype=mat66f, device=self.device)
        augmented_rhs = wp.empty(1, dtype=vec6f, device=self.device)
        multiplier = wp.empty(1, dtype=wp.float32, device=self.device)
        distance = wp.empty(1, dtype=wp.float32, device=self.device)
        explicit_wrench = wp.empty(1, dtype=vec6f, device=self.device)

        wp.launch(
            _evaluate_problem_primitives,
            dim=1,
            inputs=[mass, wp.mat33f(inertia), vec6f(velocity), vec6f(wrench), vec6f(jacobian), time_step],
            outputs=[
                body_matrix,
                body_rhs,
                dynamic_matrix,
                dynamic_rhs,
                augmented_matrix,
                augmented_rhs,
                multiplier,
                distance,
                explicit_wrench,
            ],
            device=self.device,
        )

        spatial_mass = np.zeros((6, 6), dtype=np.float32)
        spatial_mass[:3, :3] = mass * np.eye(3, dtype=np.float32)
        spatial_mass[3:, 3:] = inertia
        np.testing.assert_allclose(body_matrix.numpy()[0], spatial_mass, rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(
            body_rhs.numpy()[0], spatial_mass @ velocity + time_step * wrench, rtol=1.0e-6, atol=1.0e-7
        )

        outer = np.outer(jacobian, jacobian)
        np.testing.assert_allclose(dynamic_matrix.numpy()[0], 0.7 * outer, rtol=1.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(dynamic_rhs.numpy()[0], 0.7 * -1.25 * jacobian, rtol=1.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(augmented_matrix.numpy()[0], time_step**2 * 40.0 * outer, rtol=1.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(
            augmented_rhs.numpy()[0],
            (-time_step * (2.5 + 40.0 * -0.03) + time_step**2 * 40.0 * 0.7) * jacobian,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        self.assertAlmostEqual(float(multiplier.numpy()[0]), 2.1, delta=1.0e-6)

        expected_linear = time_step * np.max(np.abs(jacobian[:3])) / 2.0e-3
        expected_angular = time_step * np.max(np.abs(jacobian[3:])) / 4.0e-3
        self.assertAlmostEqual(float(distance.numpy()[0]), max(expected_linear, expected_angular), delta=1.0e-6)
        np.testing.assert_allclose(
            explicit_wrench.numpy()[0],
            [0.0, 3.0, -18.62, 6.1095, 4.13, 6.543],
            rtol=0.0,
            atol=2.0e-6,
        )


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
