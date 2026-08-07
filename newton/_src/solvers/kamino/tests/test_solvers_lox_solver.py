# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for frozen-contact LOX orchestration."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.bodies import update_body_inertias
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.kinematics.constraints import (
    make_unilateral_constraints_info,
    update_constraints_info,
)
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians
from newton._src.solvers.kamino._src.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino._src.models.builders.basics import (
    build_box_on_plane,
    build_boxes_hinged,
)
from newton._src.solvers.kamino._src.solvers.lox import (
    LOX_STATUS_ACTIVE,
    LOX_STATUS_CONVERGED,
    LOX_STATUS_ITERATION_LIMIT,
    LOXKaminoAdapter,
    LOXProblem,
    LOXSolver,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


class TestLOXSolver(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    @staticmethod
    def begin_time_step(solver: LOXSolver, model, time_step: float, **kwargs) -> None:
        """Configure and begin a uniform per-world LOX time step."""
        model.time.set_uniform_timestep(time_step)
        solver.begin_time_step(model.time.dt, model.time.inv_dt, **kwargs)

    def test_free_fall_matches_unconstrained_velocity(self):
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        data = model.data()
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)
        solver = LOXSolver(adapter, inertial_warmstart_fraction=1.0)

        initial_velocity = np.asarray([[0.2, -0.1, 0.3, 0.4, -0.2, 0.1]], dtype=np.float32)
        data.bodies.u_i.assign(initial_velocity)
        time_step = 0.01
        self.begin_time_step(solver, model, time_step, reset_dual=True)
        solver.solve(LOXProblem())

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

    def test_eager_solve_uses_conditional_loop(self):
        """Terminate uncaptured splitting iterations through the device condition."""
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        data = model.data()
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)
        solver = LOXSolver(
            adapter,
            max_iterations=25,
            use_graph_conditionals=True,
            inertial_warmstart_fraction=1.0,
        )

        self.begin_time_step(solver, model, 0.01, reset_dual=True)
        with patch.object(wp, "capture_while", wraps=wp.capture_while) as capture_while:
            solver.solve(LOXProblem())

        capture_while.assert_called_once()
        self.assertLess(int(solver.iteration_count.numpy()[0]), solver.max_iterations)

    def test_eager_solve_can_disable_conditional_loop(self):
        """Unroll uncaptured splitting iterations when graph conditionals are disabled."""
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        data = model.data()
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)
        solver = LOXSolver(
            adapter,
            max_iterations=2,
            use_graph_conditionals=False,
            inertial_warmstart_fraction=1.0,
        )

        self.begin_time_step(solver, model, 0.01, reset_dual=True)
        with patch.object(wp, "capture_while", wraps=wp.capture_while) as capture_while:
            solver.solve(LOXProblem())

        capture_while.assert_not_called()

    def test_eager_solve_can_run_fixed_iterations(self):
        """Run the requested rigid iteration count without convergence checks."""
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        data = model.data()
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)
        solver = LOXSolver(
            adapter,
            max_iterations=3,
            fixed_iterations=True,
            inertial_warmstart_fraction=1.0,
        )

        self.begin_time_step(solver, model, 0.01, reset_dual=True)
        with patch.object(wp, "capture_while", wraps=wp.capture_while) as capture_while:
            solver.solve(LOXProblem())

        capture_while.assert_not_called()
        np.testing.assert_array_equal(solver.iteration_count.numpy(), [solver.max_iterations])
        np.testing.assert_array_equal(solver.world_converged.numpy(), [False])
        np.testing.assert_array_equal(solver.world_iteration_limit.numpy(), [True])

    def test_initialize_rigid_inertial_warmstart_fraction(self):
        """Pre-apply a fraction of State wrench and gravity to the rigid initial guess."""
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        data = model.data()
        update_body_inertias(model.bodies, data.bodies)
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)
        fraction = 0.25
        solver = LOXSolver(adapter)

        initial_velocity = np.asarray([[0.2, -0.1, 0.3, 0.4, -0.2, 0.1]], dtype=np.float32)
        initial_wrench = np.asarray([[1.2, -0.7, 0.5, 0.08, -0.04, 0.12]], dtype=np.float32)
        data.bodies.u_i.assign(initial_velocity)
        data.bodies.w_e_i.assign(initial_wrench)
        time_step = 0.02
        self.begin_time_step(solver, model, time_step, reset_dual=True)
        np.testing.assert_array_equal(solver.projected_twist.numpy(), initial_velocity)

        solver.inertial_warmstart_fraction = fraction
        retained = np.full_like(initial_velocity, 17.0)
        next_velocity = np.asarray([[-0.3, 0.5, -0.4, 0.2, 0.1, -0.2]], dtype=np.float32)
        next_wrench = np.asarray([[-0.6, 0.9, 1.1, -0.03, 0.07, -0.05]], dtype=np.float32)
        solver.projected_twist.assign(retained)
        data.bodies.u_i.assign(next_velocity)
        data.bodies.w_e_i.assign(next_wrench)
        self.begin_time_step(solver, model, time_step, reset_dual=True)

        inverse_mass = model.bodies.inv_m_i.numpy()[:, None]
        world = model.bodies.wid.numpy()
        acceleration = np.empty_like(next_wrench)
        acceleration[:, :3] = inverse_mass * next_wrench[:, :3] + model.gravity.vector.numpy()[world]
        acceleration[:, 3:] = np.einsum("bij,bj->bi", data.bodies.inv_I_i.numpy(), next_wrench[:, 3:])
        np.testing.assert_allclose(
            solver.projected_twist.numpy(),
            next_velocity + fraction * time_step * acceleration,
            rtol=0.0,
            atol=1.0e-7,
        )

        solver.system.body_block.fill_(-1)
        solver.inertial_warmstart_fraction = 1.0
        solver.projected_twist.assign(retained)
        self.begin_time_step(solver, model, time_step, reset_dual=True)
        np.testing.assert_array_equal(solver.projected_twist.numpy(), next_velocity)

    def test_rigid_dual_impulse_round_trips_through_state_storage(self):
        """Round-trip the rigid consensus impulse through Newton state storage."""
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        data = model.data()
        update_body_inertias(model.bodies, data.bodies)
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        solver = LOXSolver(LOXKaminoAdapter(model, data, jacobians))
        input_impulse = wp.array(
            [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]],
            dtype=wp.spatial_vectorf,
            device=self.device,
        )
        output_impulse = wp.zeros_like(input_impulse)

        solver.splitting.splitting_dual_impulse.fill_(17.0)
        solver.load_state_dual_impulses(input_impulse, None)
        np.testing.assert_array_equal(solver.splitting.splitting_dual_impulse.numpy(), input_impulse.numpy())

        stored = -0.5 * input_impulse.numpy()
        solver.splitting.splitting_dual_impulse.assign(stored)
        solver.write_state_dual_impulses(output_impulse, None)
        np.testing.assert_array_equal(output_impulse.numpy(), stored)

    def test_masked_reset_preserves_unselected_rigid_world(self):
        """Reset persistent rigid state only in selected worlds."""
        builder = ModelBuilderKamino(default_world=False)
        builder.add_builder(build_boxes_hinged(ground=False, dynamic_joints=True, implicit_pd=True))
        builder.add_builder(build_boxes_hinged(ground=False, dynamic_joints=True, implicit_pd=True))
        model = builder.finalize(device=self.device)
        model.joints.tau_j_max.fill_(1.0)
        data = model.data()
        make_unilateral_constraints_info(model, data)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)
        solver = LOXSolver(adapter)

        body_world = adapter.splitting.body_world.numpy()
        structural_world = adapter.structural_row_world.numpy()
        effort_world = adapter.effort_world.numpy()
        body_values = np.arange(6 * body_world.size, dtype=np.float32).reshape((-1, 6)) + 1.0
        structural_values = np.arange(structural_world.size, dtype=np.float32) + 2.0
        effort_values = np.arange(effort_world.size, dtype=np.float32) + 3.0
        adapter.splitting.splitting_dual_impulse.assign(body_values)
        adapter.structural_multiplier.assign(structural_values)
        adapter.effort_counter_applied.assign(effort_values)
        solver.iteration_count.assign(np.asarray([4, 7], dtype=np.int32))
        solver.world_status.assign(np.asarray([LOX_STATUS_CONVERGED, LOX_STATUS_ITERATION_LIMIT], dtype=np.int32))

        solver.reset(world_mask=wp.array([True, False], dtype=wp.bool, device=self.device))

        np.testing.assert_array_equal(adapter.splitting.splitting_dual_impulse.numpy()[body_world == 0], 0.0)
        np.testing.assert_array_equal(
            adapter.splitting.splitting_dual_impulse.numpy()[body_world == 1], body_values[body_world == 1]
        )
        np.testing.assert_array_equal(adapter.structural_multiplier.numpy()[structural_world == 0], 0.0)
        np.testing.assert_array_equal(
            adapter.structural_multiplier.numpy()[structural_world == 1], structural_values[structural_world == 1]
        )
        np.testing.assert_array_equal(adapter.effort_counter_applied.numpy()[effort_world == 0], 0.0)
        np.testing.assert_array_equal(
            adapter.effort_counter_applied.numpy()[effort_world == 1], effort_values[effort_world == 1]
        )
        np.testing.assert_array_equal(solver.iteration_count.numpy(), [0, 7])
        np.testing.assert_array_equal(solver.world_status.numpy(), [LOX_STATUS_ACTIVE, LOX_STATUS_ITERATION_LIMIT])

    def test_deformable_projection_updates_proximal_before_projection(self):
        """Update elastic proxes before deformable contact projection."""
        events = []
        system = MagicMock()
        splitting = MagicMock()
        splitting.build_consensus_center.return_value = "center"
        system.solve_candidate.side_effect = lambda center: events.append("solve")
        system.update_proximal.side_effect = lambda time_step: events.append("proximal")
        splitting.prepare_projection.side_effect = lambda velocity: events.append("projection")

        solver = object.__new__(LOXSolver)
        solver.deformable_system = system
        solver.deformable_splitting = splitting
        solver.world_active = "world_active"
        solver._deformable_contacts_active = False
        solver.deformable_contacts = None

        solver._deformable_candidate_projection(0.01)

        self.assertEqual(events, ["solve", "proximal", "projection"])
        splitting.prepare_projection.assert_called_once_with(system.smooth_velocity)

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
        self.begin_time_step(solver, model, time_step, reset_dual=True)
        solver.solve(LOXProblem())

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

    def test_avbd_projects_one_rigid_contact_with_bounded_stationarity(self):
        """Project a rigid contact with AVBD and bound its primal stationarity."""
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        data = model.data()
        contacts = ContactsKamino(capacity=[1], device=self.device)
        make_unilateral_constraints_info(model, data, contacts=contacts)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))

        velocity = np.zeros((1, 6), dtype=np.float32)
        velocity[0, 2] = -1.0
        data.bodies.u_i.assign(velocity)
        data.bodies.w_e_i.zero_()
        data.bodies.w_a_i.zero_()
        body_position = data.bodies.q_i.numpy()[0, :3].reshape((1, 3))
        contacts.model_active_contacts.assign(np.asarray([1], dtype=np.int32))
        contacts.world_active_contacts.assign(np.asarray([1], dtype=np.int32))
        contacts.wid.assign(np.asarray([0], dtype=np.int32))
        contacts.cid.assign(np.asarray([0], dtype=np.int32))
        contacts.bid_AB.assign(np.asarray([[-1, 0]], dtype=np.int32))
        contacts.position_A.assign(body_position)
        contacts.position_B.assign(body_position)
        contacts.gapfunc.assign(np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32))
        contacts.frame.assign(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        contacts.material.assign(np.asarray([[0.5, 0.0]], dtype=np.float32))
        update_constraints_info(model, data)

        jacobians = DenseSystemJacobians(model=model, contacts=contacts)
        jacobians.build(model=model, data=data, contacts=contacts)
        adapter = LOXKaminoAdapter(model, data, jacobians, contacts=contacts)
        solver = LOXSolver(adapter, max_iterations=1, projection_method="avbd", projection_iterations=20)
        self.begin_time_step(solver, model, 0.01, reset_dual=True)
        solver.solve(LOXProblem())

        self.assertGreater(float(contacts.reaction.numpy()[0, 2]), 0.0)
        self.assertTrue(np.all(np.isfinite(data.bodies.u_i.numpy())))
        self.assertTrue(np.all(np.isfinite(adapter.world_avbd_stationarity_max.numpy())))
        self.assertLess(
            float(adapter.world_avbd_stationarity_max.numpy()[0]),
            1.0e-3,
        )
        np.testing.assert_array_equal(solver.world_accepted.numpy(), [True])

    def test_avbd_matches_anisotropic_coulomb_branches(self):
        """Match exact local separation, sticking, and sliding after twenty AVBD sweeps."""

        def solve(initial_linear_velocity, projection_method, projection_iterations):
            model = build_box_on_plane(ground=False).finalize(device=self.device)
            model.gravity.vector.zero_()
            data = model.data()
            contacts = ContactsKamino(capacity=[1], device=self.device)
            make_unilateral_constraints_info(model, data, contacts=contacts)
            update_body_inertias(model.bodies, data.bodies)
            compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))

            velocity = np.zeros((1, 6), dtype=np.float32)
            velocity[0, :3] = initial_linear_velocity
            data.bodies.u_i.assign(velocity)
            data.bodies.w_e_i.zero_()
            data.bodies.w_a_i.zero_()

            offset = np.asarray((0.25, -0.15, 0.20), dtype=np.float32)
            position = data.bodies.q_i.numpy()[0, :3].reshape((1, 3)) + offset
            contacts.model_active_contacts.assign(np.asarray([1], dtype=np.int32))
            contacts.world_active_contacts.assign(np.asarray([1], dtype=np.int32))
            contacts.wid.assign(np.asarray([0], dtype=np.int32))
            contacts.cid.assign(np.asarray([0], dtype=np.int32))
            contacts.bid_AB.assign(np.asarray([[-1, 0]], dtype=np.int32))
            contacts.position_A.assign(position)
            contacts.position_B.assign(position)
            contacts.gapfunc.assign(np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32))
            contacts.frame.assign(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
            contacts.material.assign(np.asarray([[0.5, 0.0]], dtype=np.float32))
            update_constraints_info(model, data)

            jacobians = DenseSystemJacobians(model=model, contacts=contacts)
            jacobians.build(model=model, data=data, contacts=contacts)
            adapter = LOXKaminoAdapter(model, data, jacobians, contacts=contacts)
            solver = LOXSolver(
                adapter,
                max_iterations=1,
                projection_method=projection_method,
                projection_iterations=projection_iterations,
                position_tolerance=1.0e-12,
                velocity_tolerance=1.0e-12,
            )
            self.begin_time_step(solver, model, 0.01, reset_dual=True)
            solver.solve(LOXProblem())
            return (
                solver.splitting.projected_twist.numpy()[0],
                adapter.contact_reaction.numpy()[0],
                adapter.contact_velocity.numpy()[0],
                float(adapter.world_contact_residual_max.numpy()[0]),
                float(adapter.world_avbd_stationarity_max.numpy()[0]),
            )

        cases = {
            "separating": np.asarray((0.3, -0.1, 0.2), dtype=np.float32),
            "sticking": np.asarray((0.7500008, -0.4500005, -1.3750013), dtype=np.float32),
            "sliding": np.asarray((1.0, 0.0, -1.0), dtype=np.float32),
        }
        for name, initial_velocity in cases.items():
            with self.subTest(name=name):
                reference = solve(initial_velocity, "gauss_seidel", 1)
                avbd = solve(initial_velocity, "avbd", 20)
                np.testing.assert_allclose(avbd[0], reference[0], rtol=0.0, atol=5.0e-6)
                np.testing.assert_allclose(avbd[1], reference[1], rtol=0.0, atol=1.0e-6)
                np.testing.assert_allclose(avbd[2], reference[2], rtol=0.0, atol=3.0e-6)
                self.assertLess(avbd[3], 1.0e-6)
                self.assertLess(avbd[4], 1.0e-6)

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
        self.begin_time_step(solver, model, time_step, reset_dual=True)
        problem = LOXProblem()
        problem.linearization_twist = wp.zeros_like(data.bodies.u_i)
        solver.solve(problem)

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
            self.begin_time_step(solver, model, time_step, reset_dual=True)
            problem = LOXProblem()
            problem.linearization_twist = wp.zeros_like(data.bodies.u_i)
            solver.solve(problem)
            return data.bodies.u_i.numpy()

        np.testing.assert_allclose(solve(1.0), solve(0.0), rtol=2.0e-5, atol=2.0e-6)

    def test_accepts_any_positive_fixed_iteration_counts(self):
        """Accept fixed iteration counts and every supported projection method."""
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        data = model.data()
        update_body_inertias(model.bodies, data.bodies)
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)

        with self.assertRaisesRegex(ValueError, "'jacobi'.*'gauss_seidel'.*'apgd'.*'avbd'"):
            LOXSolver(adapter, projection_method="invalid")
        for fraction in (-0.1, 1.1, np.nan):
            with self.subTest(inertial_warmstart_fraction=fraction):
                with self.assertRaisesRegex(ValueError, "inertial_warmstart_fraction"):
                    LOXSolver(adapter, inertial_warmstart_fraction=fraction)

        solver = LOXSolver(adapter, max_iterations=1, projection_iterations=1)
        self.assertEqual(solver.max_iterations, 1)
        self.assertEqual(solver.projection_iterations, 1)
        self.assertEqual(solver.projection_method, "jacobi")
        self.assertEqual(solver.gauss_seidel_max_colors, 0)
        for color_count in (0, 1, 2, 17):
            self.assertEqual(
                LOXSolver(
                    adapter,
                    projection_method="gauss_seidel",
                    gauss_seidel_max_colors=color_count,
                ).gauss_seidel_max_colors,
                color_count,
            )
        for color_count in (-1, 1.0, True):
            with self.assertRaisesRegex(ValueError, "gauss_seidel_max_colors"):
                LOXSolver(adapter, gauss_seidel_max_colors=color_count)
        with self.assertRaisesRegex(ValueError, "fixed_iterations"):
            LOXSolver(adapter, fixed_iterations=1)
        self.assertEqual(LOXSolver(adapter, projection_method="gauss_seidel").projection_method, "gauss_seidel")
        self.assertEqual(LOXSolver(adapter, projection_method="apgd").projection_method, "apgd")
        self.assertEqual(LOXSolver(adapter, projection_method="avbd").projection_method, "avbd")
        self.begin_time_step(solver, model, 0.01, reset_dual=True)
        solver.solve(LOXProblem())
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
