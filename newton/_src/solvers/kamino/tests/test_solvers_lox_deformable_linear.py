# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LOX cloth linear solve and incomplete preconditioner."""

import unittest

import numpy as np
import warp as wp
import warp.sparse as wps
from warp.optim import linear as wpl

import newton
from newton._src.solvers.kamino._src.solvers.lox import (
    DEFORMABLE_PRECONDITIONER_STATUS_FAILED,
    DEFORMABLE_PRECONDITIONER_STATUS_REGULARIZED,
    DEFORMABLE_PRECONDITIONER_STATUS_VALID,
    DEFORMABLE_WEIGHT_STATUS_REGULARIZED,
    DEFORMABLE_WEIGHT_STATUS_VALID,
    DeformableClothSystem,
    DeformableIncompleteLDLT,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_solvers_lox_deformable_system import (
    _bsr_to_dense,
    _build_grid_model,
    _world_dt,
)


def _preconditioner_factor_to_dense(preconditioner: DeformableIncompleteLDLT) -> np.ndarray:
    """Reconstruct the block incomplete LDLT matrix."""
    lower = np.eye(3 * preconditioner.row_count, dtype=np.float64)
    offsets = preconditioner.lower_matrix.offsets.numpy()
    columns = preconditioner.lower_matrix.columns.numpy()
    values = preconditioner.lower_matrix.values.numpy()
    for row in range(preconditioner.row_count):
        for slot in range(int(offsets[row]), int(offsets[row + 1])):
            column = int(columns[slot])
            lower[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] = values[slot]

    diagonal = np.zeros_like(lower)
    for row, block in enumerate(preconditioner.diagonal_values.numpy()):
        diagonal[3 * row : 3 * row + 3, 3 * row : 3 * row + 3] = block
    return lower @ diagonal @ lower.T


def _factor_to_dense(system: DeformableClothSystem) -> np.ndarray:
    """Reconstruct a cloth system's block incomplete LDLT matrix."""
    return _preconditioner_factor_to_dense(system.preconditioner)


def _make_cycle_system(device: wp.DeviceLike):
    """Build a four-row SPD block cycle with one level-one Cholesky fill."""
    coordinates = (
        (0, 0),
        (0, 1),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 2),
        (3, 3),
    )
    rows = wp.array([row for row, _ in coordinates], dtype=wp.int32, device=device)
    columns = wp.array([column for _, column in coordinates], dtype=wp.int32, device=device)
    values_np = np.empty((len(coordinates), 3, 3), dtype=np.float32)
    for slot, (row, column) in enumerate(coordinates):
        scale = 4.0 if row == column else -0.75
        values_np[slot] = scale * np.eye(3, dtype=np.float32)
    values = wp.array(values_np, dtype=wp.mat33, device=device)
    matrix = wps.bsr_zeros(4, 4, wp.mat33, device=device)
    wps.bsr_set_from_triplets(
        matrix,
        rows,
        columns,
        values,
        prune_numerical_zeros=False,
        topology="compact",
    )
    packed_world = wp.zeros(4, dtype=wp.int32, device=device)
    world_active = wp.ones(1, dtype=wp.int32, device=device)
    batch_offsets = wp.array([0, 12], dtype=wp.int32, device=device)
    return matrix, packed_world, world_active, batch_offsets


def _lower_coordinates(preconditioner: DeformableIncompleteLDLT) -> list[tuple[int, int]]:
    """Return lower-factor coordinates in storage order."""
    offsets = preconditioner.lower_matrix.offsets.numpy()
    columns = preconditioner.lower_matrix.columns.numpy()
    return [
        (row, int(columns[slot]))
        for row in range(preconditioner.row_count)
        for slot in range(int(offsets[row]), int(offsets[row + 1]))
    ]


def _candidate_rhs(system: DeformableClothSystem, center: np.ndarray) -> np.ndarray:
    """Evaluate the dense active-world candidate right-hand side."""
    return system.smooth_rhs.numpy().reshape(-1) + np.repeat(system.weight.numpy(), 3) * center.reshape(-1)


def _residual_norm(system: DeformableClothSystem, center: np.ndarray) -> float:
    """Evaluate the dense candidate residual norm."""
    matrix = _bsr_to_dense(system.system_matrix)
    solution = system.smooth_velocity.numpy().reshape(-1)
    return float(np.linalg.norm(matrix @ solution - _candidate_rhs(system, center)))


def _build_mixed_component_model(device: wp.DeviceLike) -> newton.Model:
    """Build one three-particle cloth and one nine-particle cloth in one world."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        density=1.0,
        tri_ke=100.0,
        tri_ka=80.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    builder.add_cloth_grid(
        pos=wp.vec3(3.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=2,
        cell_x=1.0,
        cell_y=1.0,
        mass=1.0,
        tri_ke=100.0,
        tri_ka=80.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=1.0,
        edge_kd=0.0,
    )
    builder.end_world()
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


class TestLOXDeformableLinearSolve(unittest.TestCase):
    """Test scalar consensus weights, preconditioned CR, and block incomplete LDLT."""

    def setUp(self):
        """Select the configured Newton test device."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_form_scalar_consensus_weight_and_system_matrix(self):
        """Form one scalar nodal weight and add it isotropically to each diagonal."""
        model = _build_grid_model(device=self.device, fix_left=True)
        state = model.state()
        positions = state.particle_q.numpy()
        positions[3] += np.array((0.08, -0.04, 0.2), dtype=np.float32)
        state.particle_q.assign(positions)

        system = DeformableClothSystem(model, weight_sigma=0.2, weight_beta=3.0)
        system.assemble(state, _world_dt(model, 0.01, self.device))

        smooth_matrix = _bsr_to_dense(system.smooth_matrix)
        system_matrix = _bsr_to_dense(system.system_matrix)
        weights = system.weight.numpy()
        inverse_weights = system.inverse_weight.numpy()
        statuses = system.weight_status.numpy()
        packed_to_newton = system.topology.packed_to_newton.numpy()
        masses = model.particle_mass.numpy()
        flags = model.particle_flags.numpy()

        expected_difference = np.zeros_like(system_matrix)
        for packed, original in enumerate(packed_to_newton):
            active = (flags[original] & int(newton.ParticleFlags.ACTIVE)) != 0 and masses[original] > 0.0
            if active:
                diagonal = smooth_matrix[3 * packed : 3 * packed + 3, 3 * packed : 3 * packed + 3]
                minimum_eigenvalue = float(np.linalg.eigvalsh(0.5 * (diagonal + diagonal.T))[0])
                eigenvalue_floor = max(1.0e-8, 1.0e-6 * float(masses[original]))
                eta = max(minimum_eigenvalue, eigenvalue_floor)
                expected_weight = max(0.2 * eta, min(3.0 * float(masses[original]), eta))
                self.assertAlmostEqual(float(weights[packed]), expected_weight, delta=2.0e-5)
                self.assertAlmostEqual(float(inverse_weights[packed]), 1.0 / expected_weight, delta=2.0e-5)
                expected_status = (
                    DEFORMABLE_WEIGHT_STATUS_REGULARIZED
                    if minimum_eigenvalue < eigenvalue_floor
                    else DEFORMABLE_WEIGHT_STATUS_VALID
                )
                self.assertEqual(int(statuses[packed]), expected_status)
                expected_difference[
                    3 * packed : 3 * packed + 3,
                    3 * packed : 3 * packed + 3,
                ] = expected_weight * np.eye(3)
            else:
                self.assertEqual(float(weights[packed]), 0.0)
                self.assertEqual(float(inverse_weights[packed]), 0.0)

        np.testing.assert_allclose(system_matrix - smooth_matrix, expected_difference, rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_array_equal(system.system_matrix.offsets.numpy(), system.smooth_matrix.offsets.numpy())
        np.testing.assert_array_equal(system.system_matrix.columns.numpy(), system.smooth_matrix.columns.numpy())

    def test_restrict_consensus_operator_but_keep_full_weight_preconditioner(self):
        """Restrict nodal consensus while retaining broad preconditioner weights."""
        model = _build_grid_model(device=self.device)
        system = DeformableClothSystem(model)
        system.selective_consensus = True
        system.assemble(model.state(), _world_dt(model, 0.01, self.device))

        incidence_np = np.zeros(model.particle_count, dtype=np.int32)
        incidence_np[1] = 1
        incidence = wp.array(incidence_np, dtype=wp.int32, device=self.device)
        system.set_unilateral_incidence(incidence)

        full_weight = system.full_weight.numpy()
        selected_weight = system.weight.numpy()
        np.testing.assert_array_equal(system.consensus_enabled.numpy(), incidence_np)
        np.testing.assert_array_equal(selected_weight[incidence_np == 0], 0.0)
        np.testing.assert_allclose(selected_weight[incidence_np != 0], full_weight[incidence_np != 0])

        smooth_matrix = _bsr_to_dense(system.smooth_matrix)
        system_matrix = _bsr_to_dense(system.system_matrix)
        preconditioner_matrix = _bsr_to_dense(system.preconditioner_matrix)
        selected_diagonal = np.repeat(selected_weight, 3)
        full_diagonal = np.repeat(full_weight, 3)
        np.testing.assert_allclose(system_matrix - smooth_matrix, np.diag(selected_diagonal), atol=2.0e-6)
        np.testing.assert_allclose(preconditioner_matrix - smooth_matrix, np.diag(full_diagonal), atol=2.0e-6)

    def test_apply_incomplete_ldlt_against_dense_factor(self):
        """Apply the public linear operator as the inverse of the reconstructed factor."""
        model = _build_grid_model(device=self.device)
        state = model.state()
        positions = state.particle_q.numpy()
        positions[3] += np.array((0.12, -0.07, 0.25), dtype=np.float32)
        state.particle_q.assign(positions)

        system = DeformableClothSystem(model)
        system.assemble(state, _world_dt(model, 0.015, self.device))
        factor = _factor_to_dense(system)
        self.assertEqual(system.preconditioner.uses_persistent_apply, self.device.is_cuda)

        np.testing.assert_allclose(factor, factor.T, rtol=2.0e-6, atol=2.0e-6)
        self.assertGreater(float(np.linalg.eigvalsh(factor)[0]), 0.0)
        self.assertEqual(
            int(system.preconditioner.world_status.numpy()[0]),
            DEFORMABLE_PRECONDITIONER_STATUS_VALID,
        )

        right_hand_side_np = np.linspace(-0.7, 0.9, 3 * model.particle_count, dtype=np.float32).reshape((-1, 3))
        right_hand_side = wp.array(right_hand_side_np, dtype=wp.vec3, device=self.device)
        addend_np = np.linspace(0.4, -0.2, 3 * model.particle_count, dtype=np.float32).reshape((-1, 3))
        addend = wp.array(addend_np, dtype=wp.vec3, device=self.device)
        result = wp.empty_like(right_hand_side)
        alpha = -0.6
        beta = 0.25
        system.preconditioner.linear_operator.matvec(right_hand_side, addend, result, alpha, beta)

        expected = alpha * np.linalg.solve(factor, right_hand_side_np.reshape(-1))
        expected += beta * addend_np.reshape(-1)
        np.testing.assert_allclose(result.numpy().reshape(-1), expected, rtol=2.0e-5, atol=2.0e-5)

    def test_reduce_one_step_cr_residual_with_ic1_fill(self):
        """Reduce a cycle-system one-step CR residual using IC(1) fill."""
        matrix, packed_world, world_active, batch_offsets = _make_cycle_system(self.device)
        incomplete_zero = DeformableIncompleteLDLT(
            matrix,
            packed_world,
            world_active,
            batch_offsets,
            fill_level=0,
        )
        incomplete_one = DeformableIncompleteLDLT(
            matrix,
            packed_world,
            world_active,
            batch_offsets,
            fill_level=1,
        )
        incomplete_zero.factorize()
        incomplete_one.factorize()
        operator = wpl.aslinearoperator(matrix, batch_offsets=batch_offsets)
        right_hand_side_np = np.linspace(-1.0, 0.7, 12, dtype=np.float32).reshape((-1, 3))
        right_hand_side = wp.array(right_hand_side_np, dtype=wp.vec3, device=self.device)
        solution_zero = wp.zeros_like(right_hand_side)
        solution_one = wp.zeros_like(right_hand_side)

        wpl.cg(
            operator,
            right_hand_side,
            solution_zero,
            maxiter=1,
            tol=0.0,
            atol=0.0,
            M=incomplete_zero.linear_operator,
            check_every=0,
            use_cuda_graph=False,
        )
        wpl.cg(
            operator,
            right_hand_side,
            solution_one,
            maxiter=1,
            tol=0.0,
            atol=0.0,
            M=incomplete_one.linear_operator,
            check_every=0,
            use_cuda_graph=False,
        )

        dense_matrix = _bsr_to_dense(matrix)
        rhs = right_hand_side_np.reshape(-1)
        residual_zero = np.linalg.norm(dense_matrix @ solution_zero.numpy().reshape(-1) - rhs)
        residual_one = np.linalg.norm(dense_matrix @ solution_one.numpy().reshape(-1) - rhs)
        self.assertGreater(residual_zero, 1.0e-5)
        self.assertLess(residual_one, 1.0e-4 * residual_zero)

    def test_report_preconditioner_regularization_and_failure(self):
        """Report regularized pivots and fall back to identity after non-finite data."""
        model = _build_grid_model(device=self.device)
        system = DeformableClothSystem(model)
        system.assemble(model.state(), _world_dt(model, 0.01, self.device))

        values = np.zeros_like(system.preconditioner_matrix.values.numpy())
        for diagonal_slot in system.diagonal_slots.numpy():
            values[int(diagonal_slot)] = -np.eye(3, dtype=np.float32)
        system.preconditioner_matrix.values.assign(values)
        system.preconditioner.factorize()
        self.assertEqual(
            int(system.preconditioner.world_status.numpy()[0]),
            DEFORMABLE_PRECONDITIONER_STATUS_REGULARIZED,
        )
        self.assertTrue(np.all(np.isfinite(system.preconditioner.inverse_diagonal_values.numpy())))

        values[int(system.diagonal_slots.numpy()[0]), 0, 0] = np.nan
        system.preconditioner_matrix.values.assign(values)
        system.preconditioner.factorize()
        self.assertEqual(
            int(system.preconditioner.world_status.numpy()[0]),
            DEFORMABLE_PRECONDITIONER_STATUS_FAILED,
        )

        right_hand_side_np = np.linspace(-1.0, 1.0, 3 * model.particle_count, dtype=np.float32).reshape((-1, 3))
        right_hand_side = wp.array(right_hand_side_np, dtype=wp.vec3, device=self.device)
        addend = wp.zeros_like(right_hand_side)
        result = wp.empty_like(right_hand_side)
        system.preconditioner.linear_operator.matvec(right_hand_side, addend, result, 1.0, 0.0)
        np.testing.assert_array_equal(result.numpy(), right_hand_side_np)

    def test_solve_batched_system_and_mask_inactive_world(self):
        """Match dense per-world solves and retain inactive-world warm starts."""
        model = _build_grid_model(device=self.device, world_count=2)
        state = model.state()
        positions = state.particle_q.numpy()
        positions[3] += np.array((0.1, -0.04, 0.2), dtype=np.float32)
        positions[7] += np.array((-0.08, 0.06, -0.15), dtype=np.float32)
        state.particle_q.assign(positions)

        system = DeformableClothSystem(model, cr_iterations=16)
        system.assemble(state, _world_dt(model, 0.015, self.device))
        center_np = np.linspace(-0.3, 0.5, 3 * model.particle_count, dtype=np.float32).reshape((-1, 3))
        center = wp.array(center_np, dtype=wp.vec3, device=self.device)
        iterations, _, _ = system.solve_candidate(center)

        np.testing.assert_array_equal(system.system_operator.batch_offsets.numpy(), [0, 12, 24])
        np.testing.assert_array_equal(iterations.numpy(), [16])
        matrix = _bsr_to_dense(system.system_matrix)
        right_hand_side = _candidate_rhs(system, center_np)
        expected = np.zeros_like(right_hand_side)
        for world in range(2):
            dof_start = 12 * world
            dof_end = dof_start + 12
            expected[dof_start:dof_end] = np.linalg.solve(
                matrix[dof_start:dof_end, dof_start:dof_end],
                right_hand_side[dof_start:dof_end],
            )
        np.testing.assert_allclose(system.smooth_velocity.numpy().reshape(-1), expected, rtol=3.0e-5, atol=3.0e-5)

        warm_start_np = np.linspace(0.1, 0.8, 3 * model.particle_count, dtype=np.float32).reshape((-1, 3))
        system.smooth_velocity.assign(warm_start_np)
        active = wp.array([1, 0], dtype=wp.int32, device=self.device)
        system.solve_candidate(center, world_active=active)
        np.testing.assert_array_equal(system.smooth_velocity.numpy()[4:], warm_start_np[4:])

    def test_reuse_factorization_and_warm_start(self):
        """Reuse one factorization across solves and improve an unchanged warm-started solve."""
        model = _build_grid_model(device=self.device)
        state = model.state()
        positions = state.particle_q.numpy()
        positions[3, 2] = 0.3
        state.particle_q.assign(positions)

        system = DeformableClothSystem(model, cr_iterations=2)
        system.assemble(state, _world_dt(model, 0.01, self.device))
        center_np = np.linspace(-0.2, 0.4, 3 * model.particle_count, dtype=np.float32).reshape((-1, 3))
        center = wp.array(center_np, dtype=wp.vec3, device=self.device)

        system.solve_candidate(center)
        first_residual = _residual_norm(system, center_np)
        system.solve_candidate(center)
        second_residual = _residual_norm(system, center_np)
        self.assertLessEqual(second_residual, first_residual + 2.0e-6)
        self.assertEqual(int(system.preconditioner.factorization_count.numpy()[0]), 1)

        system.assemble(state, _world_dt(model, 0.01, self.device))
        self.assertEqual(int(system.preconditioner.factorization_count.numpy()[0]), 2)

    def test_reduce_residual_more_than_jacobi_preconditioning(self):
        """Reduce one-iteration residual more than the preallocated Jacobi path."""
        model = _build_grid_model(device=self.device)
        state = model.state()
        positions = state.particle_q.numpy()
        positions[3] += np.array((0.2, -0.1, 0.35), dtype=np.float32)
        state.particle_q.assign(positions)

        jacobi_system = DeformableClothSystem(model, cr_iterations=1, preconditioner="jacobi")
        incomplete_system = DeformableClothSystem(model, cr_iterations=1)
        jacobi_system.assemble(state, _world_dt(model, 0.02, self.device))
        incomplete_system.assemble(state, _world_dt(model, 0.02, self.device))
        center_np = np.linspace(-0.5, 0.6, 3 * model.particle_count, dtype=np.float32).reshape((-1, 3))
        center = wp.array(center_np, dtype=wp.vec3, device=self.device)

        jacobi_system.solve_candidate(center)
        jacobi_residual = _residual_norm(jacobi_system, center_np)
        incomplete_system.solve_candidate(center)
        incomplete_residual = _residual_norm(incomplete_system, center_np)

        self.assertGreater(jacobi_residual, 0.0)
        self.assertLess(incomplete_residual, 0.1 * jacobi_residual)

    def test_capture_mixed_direct_and_iterative_components(self):
        """Capture batched Cholesky and CR component solves together."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model = _build_mixed_component_model(self.device)
        state = model.state()
        system = DeformableClothSystem(model, direct_max_particles=3, cr_iterations=4)
        center = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        system.assemble(state, _world_dt(model, 0.01, self.device))
        system.solve_candidate(center)

        with wp.ScopedCapture(device=self.device) as capture:
            system.assemble(state, _world_dt(model, 0.01, self.device))
            system.solve_candidate(center)
        wp.capture_launch(capture.graph)

        self.assertTrue(np.all(np.isfinite(system.smooth_velocity.numpy())))
        self.assertGreaterEqual(int(system.direct_solver.factorization_count.numpy()[0]), 2)
        self.assertGreaterEqual(int(system.preconditioner.factorization_count.numpy()[0]), 2)

    def test_capture_scalar_jacobi_and_batched_cr(self):
        """Capture and replay scalar Jacobi setup and batched CR."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model = _build_grid_model(device=self.device, world_count=2)
        state = model.state()
        system = DeformableClothSystem(
            model,
            cr_iterations=8,
            preconditioner="jacobi",
        )
        center = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        system.assemble(state, _world_dt(model, 0.01, self.device))
        system.solve_candidate(center)

        with wp.ScopedCapture(device=self.device) as capture:
            system.assemble(state, _world_dt(model, 0.01, self.device))
            system.solve_candidate(center)
        wp.capture_launch(capture.graph)

        self.assertTrue(np.all(np.isfinite(system.smooth_velocity.numpy())))
        self.assertTrue(np.all(np.isfinite(system.preconditioner.inverse_diagonal.numpy())))
        self.assertTrue(np.all(system.preconditioner.world_status.numpy() != DEFORMABLE_PRECONDITIONER_STATUS_FAILED))
        self.assertGreaterEqual(int(system.preconditioner.factorization_count.numpy()[0]), 2)

    def test_capture_factorization_and_batched_cr(self):
        """Capture and replay fixed-topology assembly, factorization, and CR."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model = _build_grid_model(device=self.device, world_count=2)
        state = model.state()
        system = DeformableClothSystem(model, cr_iterations=4, preconditioner_fill_level=1)
        center = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        system.assemble(state, _world_dt(model, 0.01, self.device))
        system.solve_candidate(center)

        with wp.ScopedCapture(device=self.device) as capture:
            system.assemble(state, _world_dt(model, 0.01, self.device))
            system.solve_candidate(center)
        wp.capture_launch(capture.graph)

        self.assertTrue(np.all(np.isfinite(system.smooth_velocity.numpy())))
        self.assertTrue(np.all(np.isfinite(system.preconditioner.inverse_diagonal_values.numpy())))
        self.assertTrue(np.all(system.preconditioner.world_status.numpy() != DEFORMABLE_PRECONDITIONER_STATUS_FAILED))
        self.assertEqual(system.preconditioner.fill_level, 1)
        self.assertGreaterEqual(int(system.preconditioner.factorization_count.numpy()[0]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
