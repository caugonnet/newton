# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX batched dense primal body systems."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox import (
    BODY_WEIGHT_STATUS_VALID,
    BatchedPrimalBodySystem,
)
from newton._src.solvers.kamino._src.solvers.lox.linear import HybridLLTBlockedSolver
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _spatial_mass(mass: float, inertia: np.ndarray) -> np.ndarray:
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = mass * np.eye(3)
    result[3:, 3:] = inertia
    return result


def _body_arrays(
    masses: np.ndarray,
    inertias: np.ndarray,
    velocities: np.ndarray | None = None,
    wrenches: np.ndarray | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array, wp.array, wp.array, wp.array]:
    body_count = len(masses)
    if velocities is None:
        velocities = np.zeros((body_count, 6), dtype=np.float32)
    if wrenches is None:
        wrenches = np.zeros((body_count, 6), dtype=np.float32)
    return (
        wp.array(masses, dtype=wp.float32, device=device),
        wp.array(inertias, dtype=wp.mat33f, device=device),
        wp.array(velocities, dtype=vec6f, device=device),
        wp.array(wrenches, dtype=vec6f, device=device),
    )


def _row_arrays(
    worlds: np.ndarray,
    bodies_first: np.ndarray,
    bodies_second: np.ndarray,
    jacobians_first: np.ndarray,
    jacobians_second: np.ndarray,
    device: wp.DeviceLike,
) -> tuple[wp.array, wp.array, wp.array, wp.array, wp.array]:
    return (
        wp.array(worlds, dtype=wp.int32, device=device),
        wp.array(bodies_first, dtype=wp.int32, device=device),
        wp.array(bodies_second, dtype=wp.int32, device=device),
        wp.array(jacobians_first, dtype=vec6f, device=device),
        wp.array(jacobians_second, dtype=vec6f, device=device),
    )


def _world_dt(system: BatchedPrimalBodySystem, value: float, device: wp.DeviceLike) -> wp.array[wp.float32]:
    """Construct an explicit uniform per-world time-step array."""
    return wp.full(system.num_worlds, value, dtype=wp.float32, device=device)


def _extract_world_matrices(system: BatchedPrimalBodySystem, values: wp.array) -> list[np.ndarray]:
    flat = values.numpy()
    matrices = []
    offset = 0
    for body_count in system.body_counts:
        dimension = 6 * body_count
        matrices.append(flat[offset : offset + dimension * dimension].reshape(dimension, dimension))
        offset += dimension * dimension
    return matrices


def _extract_world_vectors(system: BatchedPrimalBodySystem, values: wp.array) -> list[np.ndarray]:
    flat = values.numpy()
    vectors = []
    offset = 0
    for body_count in system.body_counts:
        dimension = 6 * body_count
        vectors.append(flat[offset : offset + dimension])
        offset += dimension
    return vectors


class TestLOXSystem(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_free_body_weighted_llt_solve(self):
        system = BatchedPrimalBodySystem([1], device=self.device)
        mass = np.asarray([2.0], dtype=np.float32)
        inertia = np.asarray([[[0.5, 0.1, 0.0], [0.1, 0.8, -0.05], [0.0, -0.05, 1.2]]], dtype=np.float32)
        velocity = np.asarray([[1.0, -2.0, 0.5, 0.2, -0.4, 0.7]], dtype=np.float32)
        wrench = np.asarray([[3.0, -1.0, 2.0, 0.5, 1.5, -0.25]], dtype=np.float32)
        time_step = 0.02
        body_inputs = _body_arrays(mass, inertia, velocity, wrench, self.device)

        system.assemble_bodies(*body_inputs, time_step=_world_dt(system, time_step, self.device))
        system.build_weighted_matrix()
        system.factorize_and_solve()

        spatial_mass = _spatial_mass(float(mass[0]), inertia[0])
        right_hand_side = spatial_mass @ velocity[0] + time_step * wrench[0]
        weighted = 2.0 * spatial_mass
        expected_solution = np.linalg.solve(weighted, right_hand_side)
        np.testing.assert_allclose(system.smooth_matrix.numpy().reshape(6, 6), spatial_mass, rtol=1.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(system.weight.numpy()[0], spatial_mass, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(system.weighted_matrix.numpy().reshape(6, 6), weighted, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(system.right_hand_side.numpy(), right_hand_side, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(system.solution.numpy(), expected_solution, rtol=2.0e-5, atol=2.0e-6)
        self.assertEqual(system.weight_status.numpy()[0], BODY_WEIGHT_STATUS_VALID)

        projected = np.asarray([[0.4, -0.3, 0.2, 0.1, 0.0, -0.2]], dtype=np.float32)
        splitting_dual = np.asarray([[0.1, 0.2, -0.1, 0.0, 0.3, 0.2]], dtype=np.float32)
        system.solve_candidate(
            wp.array(projected, dtype=vec6f, device=self.device),
            wp.array(splitting_dual, dtype=vec6f, device=self.device),
        )
        expected_candidate_rhs = right_hand_side + spatial_mass @ (projected[0] + splitting_dual[0])
        np.testing.assert_allclose(
            system.candidate_right_hand_side.numpy(), expected_candidate_rhs, rtol=2.0e-5, atol=2.0e-6
        )
        np.testing.assert_allclose(
            system.solution.numpy(), np.linalg.solve(weighted, expected_candidate_rhs), rtol=2.0e-5, atol=2.0e-6
        )

    def test_weight_skips_bodies_without_unilaterals(self):
        """Penalize only bodies incident to unilateral constraints."""
        components = ((0,), (1, 2))
        masses = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
        inertias = np.asarray(
            [np.diag([0.5, 0.7, 0.9]), np.diag([0.6, 0.8, 1.1]), np.diag([0.7, 1.0, 1.3])],
            dtype=np.float32,
        )
        velocities = np.asarray(
            [[0.1, -0.2, 0.3, 0.0, 0.1, -0.1], [-0.3, 0.2, 0.1, 0.2, 0.0, -0.2], [0.2, 0.1, -0.4, 0.1, -0.3, 0.2]],
            dtype=np.float32,
        )
        wrenches = np.asarray(
            [[1.0, 0.0, -2.0, 0.0, 0.2, 0.0], [0.0, 1.0, -3.0, -0.1, 0.0, 0.3], [0.5, -0.2, 1.0, 0.2, 0.0, -0.1]],
            dtype=np.float32,
        )
        time_step = 0.01
        has_unilateral = wp.array([0, 1, 0], dtype=wp.int32, device=self.device)
        projected = wp.array(np.ones((3, 6), dtype=np.float32), dtype=vec6f, device=self.device)
        splitting_dual = wp.array(2.0 * np.ones((3, 6), dtype=np.float32), dtype=vec6f, device=self.device)

        for anisotropic in (False, True):
            with self.subTest(anisotropic=anisotropic):
                system = BatchedPrimalBodySystem([3], body_components=components, device=self.device)
                system.assemble_bodies(
                    *_body_arrays(masses, inertias, velocities, wrenches, self.device),
                    time_step=_world_dt(system, time_step, self.device),
                )
                system.selective_body_weights = True
                build = system.build_anisotropic_weighted_matrix if anisotropic else system.build_weighted_matrix
                build(body_has_unilateral=has_unilateral)

                np.testing.assert_array_equal(system.body_weight_enabled.numpy(), [0, 1, 0])
                np.testing.assert_array_equal(system.weight.numpy()[0], np.zeros((6, 6), dtype=np.float32))
                self.assertGreater(float(np.linalg.norm(system.weight.numpy()[1])), 0.0)
                np.testing.assert_array_equal(system.weight.numpy()[2], np.zeros((6, 6), dtype=np.float32))
                np.testing.assert_array_equal(system.weighted_matrix.numpy()[:36], system.smooth_matrix.numpy()[:36])

                system.factorize_and_solve()
                system.solve_candidate(projected, splitting_dual)
                spatial_mass = _spatial_mass(float(masses[0]), inertias[0])
                expected = velocities[0] + time_step * np.linalg.solve(spatial_mass, wrenches[0])
                body_offset = system.body_vector_index_host[0]
                np.testing.assert_allclose(
                    system.solution.numpy()[body_offset : body_offset + 6],
                    expected,
                    rtol=2.0e-5,
                    atol=2.0e-6,
                )

                system.selective_body_weights = False
                build(body_has_unilateral=has_unilateral)
                np.testing.assert_array_equal(system.block_has_unilateral.numpy(), [0, 1])
                np.testing.assert_array_equal(system.body_weight_enabled.numpy(), [0, 1, 1])
                self.assertGreater(float(np.linalg.norm(system.weight.numpy()[2])), 0.0)

    def test_independent_components_factorize_separately(self):
        components = ((0, 2), (1,), (3,))
        component_system = BatchedPrimalBodySystem([4], body_components=components, device=self.device)
        world_system = BatchedPrimalBodySystem([4], device=self.device)
        masses = np.asarray([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
        inertias = np.asarray(
            [np.diag([0.3 + 0.1 * body, 0.5 + 0.1 * body, 0.7 + 0.1 * body]) for body in range(4)],
            dtype=np.float32,
        )
        velocities = np.asarray(
            [
                [0.2, -0.1, 0.3, 0.4, -0.2, 0.1],
                [-0.3, 0.5, -0.4, 0.2, 0.1, -0.2],
                [0.6, -0.2, 0.1, -0.1, 0.3, 0.2],
                [-0.1, 0.4, 0.2, 0.5, -0.3, 0.1],
            ],
            dtype=np.float32,
        )
        wrenches = np.asarray(
            [
                [1.0, -0.5, 0.2, 0.1, 0.3, -0.2],
                [-0.3, 0.7, -0.4, 0.2, -0.1, 0.5],
                [0.6, 0.1, -0.8, -0.2, 0.4, 0.3],
                [0.2, -0.6, 0.5, 0.3, 0.2, -0.4],
            ],
            dtype=np.float32,
        )
        rows = _row_arrays(
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([2], dtype=np.int32),
            np.asarray([[0.4, -0.2, 0.1, 0.5, -0.3, 0.2]], dtype=np.float32),
            np.asarray([[-0.1, 0.6, -0.4, 0.2, 0.3, -0.5]], dtype=np.float32),
            self.device,
        )
        for system in (component_system, world_system):
            system.assemble_bodies(
                *_body_arrays(masses, inertias, velocities, wrenches, self.device),
                time_step=_world_dt(system, 0.01, self.device),
            )
            system.add_dynamic_rows(
                *rows,
                wp.array([0.7], dtype=wp.float32, device=self.device),
                wp.array([-1.25], dtype=wp.float32, device=self.device),
            )
            system.build_weighted_matrix()
            system.factorize_and_solve()

        self.assertEqual(component_system.body_components, components)
        np.testing.assert_array_equal(component_system.info.dim.numpy(), [12, 6, 6])
        self.assertEqual(component_system.info.total_mat_size, 216)
        self.assertEqual(world_system.info.total_mat_size, 576)

        component_solution = component_system.solution.numpy()
        component_solution_by_body = np.stack(
            [component_solution[offset : offset + 6] for offset in component_system.body_vector_index_host]
        )
        np.testing.assert_allclose(
            component_solution_by_body,
            world_system.solution.numpy().reshape(4, 6),
            rtol=3.0e-5,
            atol=3.0e-6,
        )

    def test_weight_uses_constant_beta_for_all_bodies(self):
        masses = np.asarray([2.0, 3.0], dtype=np.float32)
        inertias = np.asarray([np.diag([0.5, 0.7, 0.9]), np.diag([0.6, 0.8, 1.1])], dtype=np.float32)
        velocities = np.asarray([[0.1, -0.2, 0.3, 0.0, 0.1, -0.1], [-0.3, 0.2, 0.1, 0.2, 0.0, -0.2]], dtype=np.float32)
        wrenches = np.asarray([[1.0, 0.0, -2.0, 0.0, 0.2, 0.0], [0.0, 1.0, -3.0, -0.1, 0.0, 0.3]], dtype=np.float32)
        time_step = 0.01
        for anisotropic in (False, True):
            with self.subTest(anisotropic=anisotropic):
                system = BatchedPrimalBodySystem([2], device=self.device)
                system.assemble_bodies(
                    *_body_arrays(masses, inertias, velocities, wrenches, self.device),
                    time_step=_world_dt(system, time_step, self.device),
                )
                smooth_worlds = _extract_world_matrices(system, system.smooth_matrix)
                metric_matrix = wp.array(10.0 * system.smooth_matrix.numpy(), dtype=wp.float32, device=self.device)
                build = system.build_anisotropic_weighted_matrix if anisotropic else system.build_weighted_matrix
                build(
                    metric_matrix=metric_matrix,
                    beta=5.0,
                )

                weighted_worlds = _extract_world_matrices(system, system.weighted_matrix)
                np.testing.assert_allclose(
                    weighted_worlds[0][:6, :6], 6.0 * smooth_worlds[0][:6, :6], rtol=2.0e-5, atol=2.0e-6
                )
                np.testing.assert_allclose(
                    weighted_worlds[0][6:, 6:], 6.0 * smooth_worlds[0][6:, 6:], rtol=2.0e-5, atol=2.0e-6
                )
                np.testing.assert_allclose(
                    system.weight.numpy()[0], 5.0 * smooth_worlds[0][:6, :6], rtol=2.0e-5, atol=2.0e-6
                )
                np.testing.assert_allclose(
                    system.weight.numpy()[1], 5.0 * smooth_worlds[0][6:, 6:], rtol=2.0e-5, atol=2.0e-6
                )
                self.assertGreater(float(np.linalg.norm(system.inverse_weight.numpy()[0])), 0.0)
                self.assertTrue(np.all(system.weight_status.numpy() >= BODY_WEIGHT_STATUS_VALID))

    def test_dynamic_joint_row_assembly(self):
        system = BatchedPrimalBodySystem([2], device=self.device)
        masses = np.asarray([1.5, 2.2], dtype=np.float32)
        inertias = np.asarray([np.diag([0.4, 0.7, 1.0]), np.diag([0.6, 0.9, 1.3])], dtype=np.float32)
        body_inputs = _body_arrays(masses, inertias, device=self.device)
        time_step = 0.01
        system.assemble_bodies(*body_inputs, time_step=_world_dt(system, time_step, self.device))

        jacobian_first = np.asarray([[0.4, -0.2, 0.1, 0.5, -0.3, 0.2]], dtype=np.float32)
        jacobian_second = np.asarray([[-0.1, 0.6, -0.4, 0.2, 0.3, -0.5]], dtype=np.float32)
        rows = _row_arrays(
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            jacobian_first,
            jacobian_second,
            self.device,
        )
        effective_inertia = 0.7
        free_velocity = -1.25
        system.add_dynamic_rows(
            *rows,
            wp.array([effective_inertia], dtype=wp.float32, device=self.device),
            wp.array([free_velocity], dtype=wp.float32, device=self.device),
        )

        base = np.zeros((12, 12), dtype=np.float64)
        base[:6, :6] = _spatial_mass(float(masses[0]), inertias[0])
        base[6:, 6:] = _spatial_mass(float(masses[1]), inertias[1])
        combined_jacobian = np.concatenate((jacobian_first[0], jacobian_second[0]))
        expected_matrix = base + effective_inertia * np.outer(combined_jacobian, combined_jacobian)
        expected_rhs = effective_inertia * free_velocity * combined_jacobian
        actual_matrix = system.smooth_matrix.numpy().reshape(12, 12)
        np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=2.0e-6, atol=2.0e-7)
        np.testing.assert_allclose(system.right_hand_side.numpy(), expected_rhs, rtol=2.0e-6, atol=2.0e-7)
        np.testing.assert_allclose(actual_matrix, actual_matrix.T, rtol=0.0, atol=2.0e-7)
        self.assertGreater(np.linalg.eigvalsh(actual_matrix)[0], 0.0)

    def test_prescribed_body_is_eliminated_from_dynamic_row(self):
        """Eliminate a prescribed endpoint and retain its known velocity in the row target."""
        system = BatchedPrimalBodySystem(
            [2],
            body_components=((1,),),
            dynamic_bodies=(1,),
            device=self.device,
        )
        masses = np.asarray([0.0, 2.2], dtype=np.float32)
        inertias = np.asarray([np.zeros((3, 3)), np.diag([0.6, 0.9, 1.3])], dtype=np.float32)
        prescribed_twist = np.asarray(
            [[0.3, -0.5, 0.2, 0.4, -0.1, 0.6], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        system.assemble_bodies(
            *_body_arrays(masses, inertias, velocities=prescribed_twist, device=self.device),
            time_step=_world_dt(system, 0.01, self.device),
        )

        jacobian_first = np.asarray([[0.4, -0.2, 0.1, 0.5, -0.3, 0.2]], dtype=np.float32)
        jacobian_second = np.asarray([[-0.1, 0.6, -0.4, 0.2, 0.3, -0.5]], dtype=np.float32)
        rows = _row_arrays(
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            jacobian_first,
            jacobian_second,
            self.device,
        )
        effective_inertia = 0.7
        free_velocity = -1.25
        system.add_dynamic_rows(
            *rows,
            wp.array([effective_inertia], dtype=wp.float32, device=self.device),
            wp.array([free_velocity], dtype=wp.float32, device=self.device),
            prescribed_twist=wp.array(prescribed_twist, dtype=vec6f, device=self.device),
        )

        dynamic_mass = _spatial_mass(float(masses[1]), inertias[1])
        expected_matrix = dynamic_mass + effective_inertia * np.outer(jacobian_second[0], jacobian_second[0])
        reduced_target = free_velocity - jacobian_first[0] @ prescribed_twist[0]
        expected_rhs = effective_inertia * reduced_target * jacobian_second[0]
        self.assertEqual(system.body_vector_index_host, (-1, 0))
        self.assertEqual(system.info.total_vec_size, 6)
        np.testing.assert_allclose(
            system.smooth_matrix.numpy().reshape(6, 6), expected_matrix, rtol=2.0e-6, atol=2.0e-7
        )
        np.testing.assert_allclose(system.right_hand_side.numpy(), expected_rhs, rtol=2.0e-6, atol=2.0e-7)
        self.assertGreater(np.linalg.eigvalsh(expected_matrix)[0], 0.0)

    def test_structural_row_penalty_and_assembly(self):
        system = BatchedPrimalBodySystem([2], device=self.device)
        masses = np.asarray([1.0, 1.8], dtype=np.float32)
        inertias = np.asarray([np.diag([0.3, 0.5, 0.9]), np.diag([0.4, 0.8, 1.1])], dtype=np.float32)
        system.assemble_bodies(
            *_body_arrays(masses, inertias, device=self.device),
            time_step=_world_dt(system, 0.02, self.device),
        )

        jacobian_first = np.asarray([[0.5, 0.1, -0.2, 0.3, -0.4, 0.6]], dtype=np.float32)
        jacobian_second = np.asarray([[-0.3, 0.4, 0.2, -0.5, 0.1, -0.2]], dtype=np.float32)
        rows = _row_arrays(
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            jacobian_first,
            jacobian_second,
            self.device,
        )
        time_step = 0.02
        effective_mass = 0.3
        penalty_scale = 4.0
        residual = -0.03
        multiplier = 2.5
        penalty = wp.zeros(1, dtype=wp.float32, device=self.device)
        linearization_twist = np.asarray(
            [[0.2, -0.1, 0.4, -0.3, 0.5, 0.1], [-0.4, 0.3, 0.2, 0.6, -0.2, 0.7]], dtype=np.float32
        )
        system.add_structural_rows(
            *rows,
            wp.array([residual], dtype=wp.float32, device=self.device),
            wp.array([multiplier], dtype=wp.float32, device=self.device),
            wp.array([effective_mass], dtype=wp.float32, device=self.device),
            wp.array(linearization_twist, dtype=vec6f, device=self.device),
            _world_dt(system, time_step, self.device),
            wp.array([penalty_scale], dtype=wp.float32, device=self.device),
            penalty,
        )

        rho = penalty_scale * effective_mass / (time_step * time_step)
        combined_jacobian = np.concatenate((jacobian_first[0], jacobian_second[0]))
        base = np.zeros((12, 12), dtype=np.float64)
        base[:6, :6] = _spatial_mass(float(masses[0]), inertias[0])
        base[6:, 6:] = _spatial_mass(float(masses[1]), inertias[1])
        expected_matrix = base + time_step * time_step * rho * np.outer(combined_jacobian, combined_jacobian)
        row_linearization_velocity = combined_jacobian @ linearization_twist.reshape(-1)
        expected_rhs = (
            -time_step * (multiplier + rho * residual) + time_step**2 * rho * row_linearization_velocity
        ) * combined_jacobian
        actual_matrix = system.smooth_matrix.numpy().reshape(12, 12)
        self.assertAlmostEqual(float(penalty.numpy()[0]), rho, delta=4.0e-4)
        np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=2.0e-6, atol=3.0e-7)
        np.testing.assert_allclose(system.right_hand_side.numpy(), expected_rhs, rtol=2.0e-6, atol=3.0e-7)
        np.testing.assert_allclose(actual_matrix, actual_matrix.T, rtol=0.0, atol=3.0e-7)
        self.assertGreater(np.linalg.eigvalsh(actual_matrix)[0], 0.0)

    def test_structural_block_assembly_preserves_row_coupling(self):
        system = BatchedPrimalBodySystem([2], device=self.device)
        masses = np.asarray([1.2, 1.9], dtype=np.float32)
        inertias = np.asarray([np.diag([0.3, 0.6, 0.8]), np.diag([0.5, 0.9, 1.4])], dtype=np.float32)
        time_step = 0.03
        system.assemble_bodies(
            *_body_arrays(masses, inertias, device=self.device),
            time_step=_world_dt(system, time_step, self.device),
        )

        jacobian_first = np.asarray(
            [[0.6, -0.2, 0.1, 0.3, -0.5, 0.4], [-0.1, 0.7, 0.2, -0.4, 0.2, 0.6]],
            dtype=np.float32,
        )
        jacobian_second = np.asarray(
            [[-0.3, 0.4, -0.2, 0.5, 0.1, -0.6], [0.5, -0.1, 0.3, 0.2, -0.7, 0.1]],
            dtype=np.float32,
        )
        residual = np.asarray([-0.04, 0.025], dtype=np.float32)
        multiplier = np.asarray([1.5, -0.8], dtype=np.float32)
        penalty_active = np.asarray([[1200.0, 350.0], [350.0, 900.0]], dtype=np.float32)
        penalty = np.zeros((1, 6, 6), dtype=np.float32)
        penalty[0, :2, :2] = penalty_active
        linearization_twist = np.asarray(
            [[0.2, -0.1, 0.4, 0.3, -0.2, 0.1], [-0.3, 0.5, -0.2, 0.4, 0.1, -0.6]],
            dtype=np.float32,
        )
        system.add_structural_blocks(
            wp.array([0], dtype=wp.int32, device=self.device),
            wp.array([0], dtype=wp.int32, device=self.device),
            wp.array([2], dtype=wp.int32, device=self.device),
            wp.array([0], dtype=wp.int32, device=self.device),
            wp.array([1], dtype=wp.int32, device=self.device),
            wp.array(jacobian_first, dtype=vec6f, device=self.device),
            wp.array(jacobian_second, dtype=vec6f, device=self.device),
            wp.array(residual, dtype=wp.float32, device=self.device),
            wp.array(multiplier, dtype=wp.float32, device=self.device),
            wp.array(penalty, dtype=mat66f, device=self.device),
            wp.array([1], dtype=wp.int32, device=self.device),
            wp.array(linearization_twist, dtype=vec6f, device=self.device),
            _world_dt(system, time_step, self.device),
        )

        combined_jacobian = np.concatenate((jacobian_first, jacobian_second), axis=1)
        base = np.zeros((12, 12), dtype=np.float64)
        base[:6, :6] = _spatial_mass(float(masses[0]), inertias[0])
        base[6:, 6:] = _spatial_mass(float(masses[1]), inertias[1])
        expected_matrix = base + time_step**2 * combined_jacobian.T @ penalty_active @ combined_jacobian
        linearization_velocity = combined_jacobian @ linearization_twist.reshape(-1)
        coefficient = -time_step * (multiplier + penalty_active @ residual)
        coefficient += time_step**2 * penalty_active @ linearization_velocity
        expected_rhs = combined_jacobian.T @ coefficient
        np.testing.assert_allclose(
            system.smooth_matrix.numpy().reshape(12, 12), expected_matrix, rtol=3.0e-6, atol=5.0e-7
        )
        np.testing.assert_allclose(system.right_hand_side.numpy(), expected_rhs, rtol=3.0e-6, atol=5.0e-7)

    def test_simple_joint_aware_weight_metric_applies_joint_scale(self):
        system = BatchedPrimalBodySystem([1], device=self.device)
        mass = np.asarray([2.0], dtype=np.float32)
        inertia = np.asarray([np.diag([0.4, 0.7, 1.1])], dtype=np.float32)
        system.assemble_bodies(
            *_body_arrays(mass, inertia, device=self.device),
            time_step=_world_dt(system, 0.01, self.device),
        )

        jacobian = np.asarray([[0.5, -0.2, 0.1, 0.3, -0.4, 0.6]], dtype=np.float32)
        arguments = (
            wp.array([0], dtype=wp.int32, device=self.device),
            wp.array([0], dtype=wp.int32, device=self.device),
            wp.array([-1], dtype=wp.int32, device=self.device),
            wp.array(jacobian, dtype=vec6f, device=self.device),
            wp.zeros(1, dtype=vec6f, device=self.device),
            wp.array([2.5], dtype=wp.float32, device=self.device),
        )

        unit_metric = (
            system.build_simple_joint_aware_weight_metric(
                *arguments, joint_metric_scale=wp.array([1.0], dtype=wp.float32, device=self.device)
            )
            .numpy()
            .reshape(6, 6)
            .copy()
        )
        scaled_metric = (
            system.build_simple_joint_aware_weight_metric(
                *arguments, joint_metric_scale=wp.array([4.0], dtype=wp.float32, device=self.device)
            )
            .numpy()
            .reshape(6, 6)
            .copy()
        )
        smooth = system.smooth_matrix.numpy().reshape(6, 6)
        contribution = 2.5 * np.outer(jacobian[0], jacobian[0])
        np.testing.assert_allclose(unit_metric, smooth + contribution, rtol=2.0e-6, atol=3.0e-7)
        np.testing.assert_allclose(scaled_metric, smooth + 4.0 * contribution, rtol=2.0e-6, atol=3.0e-7)

        with self.assertRaisesRegex(ValueError, "joint_metric_scale"):
            system.build_simple_joint_aware_weight_metric(
                *arguments, joint_metric_scale=wp.zeros(0, dtype=wp.float32, device=self.device)
            )

    def test_heterogeneous_multiple_worlds_and_llt(self):
        system = BatchedPrimalBodySystem([1, 2], device=self.device)
        self.assertIsInstance(system.linear_solver, HybridLLTBlockedSolver)
        self.assertEqual(system.linear_solver._sequential_num_blocks, 1)
        self.assertEqual(system.linear_solver._tiled_num_blocks, 1)
        masses = np.asarray([1.2, 1.5, 2.0], dtype=np.float32)
        inertias = np.asarray(
            [np.diag([0.2, 0.3, 0.5]), np.diag([0.4, 0.6, 0.8]), np.diag([0.5, 0.9, 1.4])],
            dtype=np.float32,
        )
        velocities = np.asarray(
            [[0.2, -0.1, 0.4, 0.3, -0.2, 0.1], [0.5, 0.2, -0.3, 0.1, 0.4, -0.2], [-0.2, 0.6, 0.1, -0.4, 0.3, 0.5]],
            dtype=np.float32,
        )
        wrenches = np.asarray(
            [[1.0, 0.0, -0.5, 0.2, 0.1, 0.0], [0.0, 0.5, 0.2, -0.1, 0.0, 0.3], [0.4, -0.2, 0.0, 0.1, 0.2, -0.3]],
            dtype=np.float32,
        )
        time_step = 0.015
        system.assemble_bodies(
            *_body_arrays(masses, inertias, velocities, wrenches, self.device),
            time_step=_world_dt(system, time_step, self.device),
        )

        jacobian_first = np.asarray([[0.3, -0.4, 0.2, 0.1, 0.5, -0.2]], dtype=np.float32)
        jacobian_second = np.asarray([[-0.5, 0.1, 0.4, -0.3, 0.2, 0.6]], dtype=np.float32)
        rows = _row_arrays(
            np.asarray([1], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            np.asarray([2], dtype=np.int32),
            jacobian_first,
            jacobian_second,
            self.device,
        )
        system.add_dynamic_rows(
            *rows,
            wp.array([0.45], dtype=wp.float32, device=self.device),
            wp.array([0.8], dtype=wp.float32, device=self.device),
        )
        system.build_weighted_matrix()
        system.factorize_and_solve()

        smooth_worlds = _extract_world_matrices(system, system.smooth_matrix)
        weighted_worlds = _extract_world_matrices(system, system.weighted_matrix)
        rhs_worlds = _extract_world_vectors(system, system.right_hand_side)
        solution_worlds = _extract_world_vectors(system, system.solution)
        expected_world0 = _spatial_mass(float(masses[0]), inertias[0])
        expected_world1 = np.zeros((12, 12), dtype=np.float64)
        expected_world1[:6, :6] = _spatial_mass(float(masses[1]), inertias[1])
        expected_world1[6:, 6:] = _spatial_mass(float(masses[2]), inertias[2])
        combined_jacobian = np.concatenate((jacobian_first[0], jacobian_second[0]))
        expected_world1 += 0.45 * np.outer(combined_jacobian, combined_jacobian)
        np.testing.assert_allclose(smooth_worlds[0], expected_world0, rtol=1.0e-6, atol=2.0e-7)
        np.testing.assert_allclose(smooth_worlds[1], expected_world1, rtol=2.0e-6, atol=3.0e-7)
        for weighted, rhs, solution in zip(weighted_worlds, rhs_worlds, solution_worlds, strict=True):
            np.testing.assert_allclose(weighted, weighted.T, rtol=0.0, atol=4.0e-7)
            self.assertGreater(np.linalg.eigvalsh(weighted)[0], 0.0)
            np.testing.assert_allclose(solution, np.linalg.solve(weighted, rhs), rtol=3.0e-5, atol=3.0e-6)
        np.testing.assert_array_equal(
            system.weight_status.numpy(), np.full(3, BODY_WEIGHT_STATUS_VALID, dtype=np.int32)
        )


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
