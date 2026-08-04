# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for frozen-contact LOX orchestration."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.bodies import update_body_inertias
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.kinematics.constraints import (
    make_unilateral_constraints_info,
    update_constraints_info,
)
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians
from newton._src.solvers.kamino._src.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino._src.models.builders.basics import (
    build_box_on_plane,
    build_box_pendulum,
    build_boxes_hinged,
)
from newton._src.solvers.kamino._src.solvers.lox import (
    LOX_STATUS_ACTIVE,
    LOX_STATUS_CONVERGED,
    LOX_STATUS_ITERATION_LIMIT,
    LOXKaminoAdapter,
    LOXSolver,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


class TestLOXSolver(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_free_fall_matches_unconstrained_velocity(self):
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        data = model.data()
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)
        solver = LOXSolver(adapter)

        initial_velocity = np.asarray([[0.2, -0.1, 0.3, 0.4, -0.2, 0.1]], dtype=np.float32)
        data.bodies.u_i.assign(initial_velocity)
        time_step = 0.01
        solver.begin_time_step(time_step, reset_dual=True)
        solver.solve(time_step)

        gravity_data = model.gravity.vector.numpy()[0]
        expected = initial_velocity.copy()
        expected[0, :3] += time_step * gravity_data
        np.testing.assert_allclose(data.bodies.u_i.numpy(), expected, rtol=0.0, atol=1.0e-3)
        np.testing.assert_array_equal(solver.world_accepted.numpy(), [True])
        np.testing.assert_array_equal(solver.world_converged.numpy(), [True])
        np.testing.assert_array_equal(solver.world_failed.numpy(), [False])
        np.testing.assert_array_equal(solver.world_iteration_limit.numpy(), [False])
        np.testing.assert_array_equal(solver.world_status.numpy(), [LOX_STATUS_CONVERGED])
        self.assertLessEqual(int(solver.iteration_count.numpy()[0]), solver.max_iterations)

    def test_jointed_world_projects_one_static_contact(self):
        model = build_boxes_hinged(ground=False, dynamic_joints=True).finalize(device=self.device)
        data = model.data()
        contacts = ContactsKamino(capacity=[1], device=self.device)
        make_unilateral_constraints_info(model, data, contacts=contacts)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))

        velocity = np.zeros((2, 6), dtype=np.float32)
        velocity[1, 2] = -1.0
        data.bodies.u_i.assign(velocity)
        data.bodies.w_e_i.zero_()
        data.bodies.w_a_i.zero_()
        data.joints.m_j.assign(np.ones(data.joints.m_j.shape[0], dtype=np.float32))
        data.joints.dq_b_j.zero_()
        data.joints.lambda_j.zero_()

        body_pose = data.bodies.q_i.numpy()
        contact_positions = np.zeros((1, 3), dtype=np.float32)
        contact_positions[0] = body_pose[1, :3]
        contacts.model_active_contacts.assign(np.asarray([1], dtype=np.int32))
        contacts.world_active_contacts.assign(np.asarray([1], dtype=np.int32))
        contacts.wid.assign(np.asarray([0], dtype=np.int32))
        contacts.cid.assign(np.asarray([0], dtype=np.int32))
        contacts.bid_AB.assign(np.asarray([[-1, 1]], dtype=np.int32))
        contacts.position_A.assign(contact_positions)
        contacts.position_B.assign(contact_positions)
        contacts.gapfunc.assign(np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32))
        contacts.frame.assign(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        contacts.material.assign(np.asarray([[0.5, 0.0]], dtype=np.float32))
        update_constraints_info(model, data)

        jacobians = DenseSystemJacobians(model=model, contacts=contacts)
        jacobians.build(model=model, data=data, contacts=contacts)
        adapter = LOXKaminoAdapter(model, data, jacobians, contacts=contacts)
        solver = LOXSolver(adapter)
        self.assertEqual(solver._joint_metric_mode, 0)
        time_step = 0.01
        solver.begin_time_step(time_step, reset_dual=True)
        solver.solve(time_step)

        reaction = contacts.reaction.numpy()[0]
        velocity_contact = contacts.velocity.numpy()[0]
        self.assertGreater(float(reaction[2]), 0.0)
        self.assertLessEqual(float(np.linalg.norm(reaction[:2])), 0.5 * float(reaction[2]) + 1.0e-5)
        self.assertGreaterEqual(float(velocity_contact[2]), -2.0e-4)
        self.assertTrue(np.isfinite(data.bodies.u_i.numpy()).all())
        np.testing.assert_array_equal(solver.world_accepted.numpy(), [True])
        np.testing.assert_array_equal(solver.world_status.numpy(), [LOX_STATUS_CONVERGED])
        self.assertGreater(adapter.structural_row_count, 0)
        np.testing.assert_array_equal(adapter.body_has_unilateral.numpy(), [0, 1])
        self.assertGreater(float(np.linalg.norm(solver.system.weight.numpy()[0])), 0.0)
        self.assertGreater(float(np.linalg.norm(solver.system.weight.numpy()[1])), 0.0)

    def test_direct_structural_solve_enforces_joint_rows(self):
        model = build_boxes_hinged(ground=False, dynamic_joints=True).finalize(device=self.device)
        data = model.data()
        make_unilateral_constraints_info(model, data)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)
        solver = LOXSolver(adapter, max_iterations=1, _joint_solve_mode=1)

        time_step = 0.01
        solver.begin_time_step(time_step, reset_dual=True)
        solver.solve(time_step, linearization_twist=wp.zeros_like(data.bodies.u_i))

        self.assertIsNotNone(solver.structural_joint_solver)
        structural_solver = solver.structural_joint_solver
        np.testing.assert_array_equal(structural_solver.block_scaling_status.numpy(), 1)
        dimension = int(structural_solver.info.dim.numpy()[0])
        matrix_offset = int(structural_solver.info.mio.numpy()[0])
        vector_offset = int(structural_solver.info.vio.numpy()[0])
        transformed_schur = structural_solver.transformed_schur_matrix.numpy()[
            matrix_offset : matrix_offset + dimension * dimension
        ].reshape(dimension, dimension)
        row_vector_index = structural_solver.row_vector_index.numpy()
        for row_offset, row_count in zip(
            adapter.structural_block_row_offset.numpy(),
            adapter.structural_block_row_count.numpy(),
            strict=True,
        ):
            indices = row_vector_index[row_offset : row_offset + row_count] - vector_offset
            np.testing.assert_allclose(
                transformed_schur[np.ix_(indices, indices)],
                np.eye(row_count),
                rtol=3.0e-4,
                atol=3.0e-4,
            )
        self.assertLess(float(np.max(np.abs(adapter.structural_candidate_residual.numpy()))), 2.0e-5)
        self.assertTrue(np.isfinite(adapter.structural_multiplier.numpy()).all())
        self.assertTrue(np.isfinite(data.bodies.u_i.numpy()).all())

        active = wp.ones(adapter.num_worlds, dtype=wp.bool, device=self.device)
        structural_solver.body_system.candidate_right_hand_side.zero_()
        structural_solver.transformed_impulse.fill_(1.0)
        structural_solver.solve(
            time_step,
            wp.zeros_like(data.bodies.u_i),
            active,
            wp.zeros_like(adapter.structural_residual),
            adapter.structural_multiplier_index,
            adapter.structural_multiplier,
            adapter.data.joints.lambda_j,
        )
        progressive_impulse = structural_solver.transformed_impulse.numpy()
        self.assertGreater(float(np.linalg.norm(progressive_impulse)), 0.0)
        self.assertFalse(np.array_equal(progressive_impulse, np.ones_like(progressive_impulse)))

        multiplier = adapter.structural_multiplier.numpy().copy()
        structural_solver.factorize()
        np.testing.assert_array_equal(structural_solver.transformed_impulse.numpy(), 0.0)
        structural_solver.warmstart(time_step, adapter.structural_multiplier)
        expected_impulse = np.zeros_like(structural_solver.impulse.numpy())
        expected_impulse[structural_solver.row_vector_index.numpy()] = time_step * multiplier
        np.testing.assert_allclose(structural_solver.impulse.numpy(), expected_impulse, rtol=2.0e-6, atol=2.0e-7)
        transformed_impulse = structural_solver.transformed_impulse.numpy()
        factor = structural_solver.block_scaling_factor.numpy()
        for block, (row_offset, row_count) in enumerate(
            zip(
                adapter.structural_block_row_offset.numpy(),
                adapter.structural_block_row_count.numpy(),
                strict=True,
            )
        ):
            indices = structural_solver.row_vector_index.numpy()[row_offset : row_offset + row_count]
            np.testing.assert_allclose(
                factor[block, :row_count, :row_count] @ transformed_impulse[indices],
                expected_impulse[indices],
                rtol=3.0e-5,
                atol=3.0e-6,
            )

        projected_fraction = 0.35
        projected_twist = np.asarray(
            [[0.3, -0.2, 0.1, 0.4, -0.1, 0.2], [-0.4, 0.5, -0.3, 0.2, 0.6, -0.5]],
            dtype=np.float32,
        )
        global_twist = np.asarray(
            [[-0.1, 0.4, -0.2, 0.3, 0.2, -0.5], [0.5, -0.3, 0.2, -0.4, 0.1, 0.6]],
            dtype=np.float32,
        )
        structural_solver.refine_from_twists(
            time_step,
            wp.zeros_like(data.bodies.u_i),
            wp.array(global_twist, dtype=wp.spatial_vectorf, device=self.device),
            wp.array(projected_twist, dtype=wp.spatial_vectorf, device=self.device),
            projected_fraction,
            wp.ones(1, dtype=wp.bool, device=self.device),
            wp.ones(1, dtype=wp.bool, device=self.device),
            adapter.structural_residual,
            adapter.structural_multiplier_index,
            adapter.structural_multiplier,
            data.joints.lambda_j,
        )
        first = adapter.structural_body_first_global.numpy()
        second = adapter.structural_body_second_global.numpy()
        jacobian_first = adapter.structural_jacobian_first.numpy()
        jacobian_second = adapter.structural_jacobian_second.numpy()
        expected_rhs = adapter.structural_residual.numpy() / time_step
        for row in range(adapter.structural_row_count):
            blended_velocity = 0.0
            if first[row] >= 0:
                blended_velocity += np.dot(
                    jacobian_first[row],
                    (1.0 - projected_fraction) * global_twist[first[row]]
                    + projected_fraction * projected_twist[first[row]],
                )
            if second[row] >= 0:
                blended_velocity += np.dot(
                    jacobian_second[row],
                    (1.0 - projected_fraction) * global_twist[second[row]]
                    + projected_fraction * projected_twist[second[row]],
                )
            expected_rhs[row] += blended_velocity
        np.testing.assert_allclose(
            structural_solver.right_hand_side.numpy()[structural_solver.row_vector_index.numpy()],
            expected_rhs,
            rtol=2.0e-5,
            atol=2.0e-6,
        )

    def test_independent_articulations_use_separate_body_and_schur_factors(self):
        builder = build_boxes_hinged(ground=False, dynamic_joints=True)
        build_box_pendulum(
            builder=builder,
            z_offset=1.0,
            ground=False,
            dynamic_joints=True,
            new_world=False,
            world_index=0,
        )
        model = builder.finalize(device=self.device)
        data = model.data()
        make_unilateral_constraints_info(model, data)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)
        solver = LOXSolver(adapter, max_iterations=1, _joint_solve_mode=1)

        self.assertEqual(solver.system.body_components, ((0, 1), (2,)))
        np.testing.assert_array_equal(solver.system.info.dim.numpy(), [12, 6])
        self.assertEqual(solver.system.info.total_mat_size, 180)
        structural_solver = solver.structural_joint_solver
        self.assertIsNotNone(structural_solver)
        self.assertEqual(structural_solver.component_row_counts, (5, 5))
        np.testing.assert_array_equal(structural_solver.info.dim.numpy(), [5, 5])
        self.assertEqual(structural_solver.info.total_mat_size, 50)

        time_step = 0.01
        solver.begin_time_step(time_step, reset_dual=True)
        solver.solve(time_step, linearization_twist=wp.zeros_like(data.bodies.u_i))

        np.testing.assert_array_equal(structural_solver.block_scaling_status.numpy(), 1)
        self.assertLess(float(np.max(np.abs(adapter.structural_candidate_residual.numpy()))), 3.0e-5)
        self.assertTrue(np.isfinite(data.bodies.u_i.numpy()).all())

        alm = LOXSolver(adapter, joint_penalty_scale=123.0)
        seed = alm.joint_penalty_scale_seed(time_step)
        self.assertEqual(len(seed), 1)
        self.assertTrue(np.isfinite(seed[0]))
        self.assertGreater(seed[0], 0.0)
        self.assertNotEqual(seed[0], 123.0)

    def test_direct_blend_is_invariant_without_unilaterals(self):
        def solve(projected_fraction: float) -> np.ndarray:
            model = build_boxes_hinged(ground=False, dynamic_joints=True).finalize(device=self.device)
            data = model.data()
            make_unilateral_constraints_info(model, data)
            update_body_inertias(model.bodies, data.bodies)
            compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
            velocity = np.asarray(
                [[0.2, -0.1, 0.3, 0.4, -0.2, 0.1], [-0.3, 0.5, -0.4, 0.2, 0.1, -0.2]],
                dtype=np.float32,
            )
            data.bodies.u_i.assign(velocity)
            jacobians = DenseSystemJacobians(model=model)
            jacobians.build(model=model, data=data)
            adapter = LOXKaminoAdapter(model, data, jacobians)
            solver = LOXSolver(
                adapter,
                max_iterations=3,
                joint_multiplier_projected_fraction=projected_fraction,
                _joint_solve_mode=1,
            )
            time_step = 0.01
            solver.begin_time_step(time_step, reset_dual=True)
            solver.solve(time_step, linearization_twist=wp.zeros_like(data.bodies.u_i))
            return data.bodies.u_i.numpy()

        np.testing.assert_allclose(solve(1.0), solve(0.0), rtol=2.0e-5, atol=2.0e-6)

    def test_accepts_any_positive_fixed_iteration_counts(self):
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        data = model.data()
        update_body_inertias(model.bodies, data.bodies)
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)

        with self.assertRaises(ValueError):
            LOXSolver(adapter, projection_method="invalid")

        solver = LOXSolver(adapter, max_iterations=1, projection_iterations=1)
        self.assertEqual(solver.max_iterations, 1)
        self.assertEqual(solver.projection_iterations, 1)
        self.assertEqual(solver.projection_method, "jacobi")
        self.assertEqual(LOXSolver(adapter, projection_method="gauss_seidel").projection_method, "gauss_seidel")
        solver.begin_time_step(0.01, reset_dual=True)
        solver.solve(0.01)
        np.testing.assert_array_equal(solver.world_accepted.numpy(), [True])
        np.testing.assert_array_equal(solver.world_iteration_limit.numpy(), [True])
        np.testing.assert_array_equal(solver.world_status.numpy(), [LOX_STATUS_ITERATION_LIMIT])

        solver.splitting.splitting_dual.assign(np.ones((1, 6), dtype=np.float32))
        solver.reset()
        np.testing.assert_allclose(solver.splitting.splitting_dual.numpy(), 0.0)
        np.testing.assert_array_equal(solver.world_accepted.numpy(), [False])
        np.testing.assert_array_equal(solver.world_converged.numpy(), [False])
        np.testing.assert_array_equal(solver.world_failed.numpy(), [False])
        np.testing.assert_array_equal(solver.world_iteration_limit.numpy(), [False])
        np.testing.assert_array_equal(solver.world_status.numpy(), [LOX_STATUS_ACTIVE])


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
