# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the LOX Kamino dynamics backend."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.detector import CollisionDetector
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from newton._src.solvers.kamino._src.models.builders.basics import (
    build_box_on_plane,
    build_box_pendulum,
    build_boxes_hinged,
    build_cartpole,
)
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino._src.solvers.lox import LOXSolver
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _build_revolute_dynamics_model(
    *, damping: float, friction: float, velocity: float, device: wp.DeviceLike = None
) -> newton.Model:
    """Build a gravity-free world-to-body hinge with consistent initial velocity."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Y,
        damping=damping,
        friction=friction,
    )
    builder.add_articulation([joint])
    builder.body_qd[body] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, velocity, 0.0)
    builder.joint_qd[builder.joint_qd_start[joint]] = velocity
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_binary_revolute_friction_model(*, device: wp.DeviceLike = None) -> newton.Model:
    """Build a gravity-free two-body hinge with relative speed two."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    bodies = [
        builder.add_link(
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            lock_inertia=True,
        )
        for _ in range(2)
    ]
    joint = builder.add_joint_revolute(
        parent=bodies[0],
        child=bodies[1],
        axis=newton.Axis.Y,
        friction=5.0,
    )
    builder.add_articulation([joint])
    builder.body_qd[bodies[0]] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    builder.body_qd[bodies[1]] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    builder.joint_qd[builder.joint_qd_start[joint]] = 2.0
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


class TestSolverKaminoLOX(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def make_config(
        self,
        nonlinear_iterations: int = 1,
        compute_solution_metrics: bool = False,
        sparse_jacobian: bool = False,
    ) -> SolverKamino.Config:
        return SolverKamino.Config(
            dynamics_solver="lox",
            compute_solution_metrics=compute_solution_metrics,
            sparse_jacobian=sparse_jacobian,
            lox=kamino_config.LOXSolverConfig(
                nonlinear_iterations=nonlinear_iterations,
                max_iterations=25,
                projection_iterations=5,
            ),
        )

    def test_free_fall_advances_projected_velocity_and_pose(self):
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        time_step = 0.01

        solver.step(state_previous, state_next, control, dt=time_step)

        self.assertIsNone(solver.problem_fd)
        self.assertIsInstance(solver.solver_fd, LOXSolver)
        self.assertIsInstance(solver._jacobians, DenseSystemJacobians)
        gravity = model.gravity.vector.numpy()[0]
        expected_velocity = time_step * gravity
        np.testing.assert_allclose(state_next.u_i.numpy()[0, :3], expected_velocity, rtol=0.0, atol=5.0e-4)
        expected_height = state_previous.q_i.numpy()[0, 2] + time_step * expected_velocity[2]
        self.assertAlmostEqual(float(state_next.q_i.numpy()[0, 2]), float(expected_height), places=5)
        self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())

    def test_joint_damping_is_implicit_in_the_smooth_row(self):
        time_step = 0.1
        damping = 4.0
        initial_velocity = 2.0
        model = _build_revolute_dynamics_model(damping=damping, friction=0.0, velocity=initial_velocity)
        solver = SolverKamino(model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=time_step)

        expected = initial_velocity / (1.0 + time_step * damping)
        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), expected, places=4)

    def test_joint_friction_slips_then_sticks_with_one_bounded_row(self):
        time_step = 0.1
        friction = 5.0
        variants = (("jacobi", True), ("gauss_seidel", False))
        for projection_method, sparse_jacobian in variants:
            results = []
            for initial_velocity in (2.0, 0.25):
                with self.subTest(projection_method=projection_method, initial_velocity=initial_velocity):
                    model = _build_revolute_dynamics_model(damping=0.0, friction=friction, velocity=initial_velocity)
                    config = self.make_config(sparse_jacobian=sparse_jacobian)
                    config.lox.projection_method = projection_method
                    solver = SolverKamino(model, config=config)
                    state_previous = model.state()
                    state_next = model.state()

                    solver.step(state_previous, state_next, model.control(), contacts=None, dt=time_step)
                    adapter = solver._solver_kamino._lox_adapter
                    results.append(
                        (
                            float(state_next.joint_qd.numpy()[0]),
                            float(adapter.friction_reaction.numpy()[0]),
                            adapter.friction_capacity,
                        )
                    )

            self.assertAlmostEqual(results[0][0], 1.5, places=4)
            self.assertAlmostEqual(results[0][1], -time_step * friction, places=5)
            self.assertAlmostEqual(results[1][0], 0.0, delta=1.0e-4)
            self.assertAlmostEqual(results[1][1], -0.25, delta=1.0e-4)
            self.assertEqual(results[0][2], 1)
            self.assertEqual(results[1][2], 1)

    def test_binary_joint_friction_sparse_matches_dense_and_counts_once_per_body(self):
        results = []
        for sparse_jacobian in (False, True):
            model = _build_binary_revolute_friction_model()
            solver = SolverKamino(model, config=self.make_config(sparse_jacobian=sparse_jacobian))
            state_previous = model.state()
            state_next = model.state()

            solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.1)
            adapter = solver._solver_kamino._lox_adapter
            results.append((state_next.body_qd.numpy(), state_next.joint_qd.numpy(), adapter.friction_reaction.numpy()))
            np.testing.assert_array_equal(adapter.body_constraint_count.numpy(), [2, 2])
            np.testing.assert_array_equal(adapter.static_body_constraint_count.numpy(), [1, 1])

        for dense, sparse in zip(results[0], results[1], strict=True):
            np.testing.assert_allclose(sparse, dense, rtol=1.0e-5, atol=2.0e-6)
        self.assertAlmostEqual(float(results[0][1][0]), 1.0, places=4)
        self.assertAlmostEqual(float(results[0][2][0]), -0.5, places=5)

    def test_free_fall_produces_solution_metrics(self):
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        solver = SolverKaminoImpl(model=model, config=self.make_config(compute_solution_metrics=True))
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), dt=0.01)

        self.assertIsNotNone(solver.metrics)
        self.assertIsNone(solver._problem_metrics)
        metric_values = (
            solver.metrics.data.r_eom,
            solver.metrics.data.r_kinematics,
            solver.metrics.data.r_cts_joints,
            solver.metrics.data.r_cts_limits,
            solver.metrics.data.r_cts_contacts,
            solver.metrics.data.r_v_plus,
            solver.metrics.data.r_ncp_primal,
            solver.metrics.data.r_ncp_dual,
            solver.metrics.data.r_ncp_compl,
            solver.metrics.data.r_vi_natmap,
        )
        self.assertTrue(all(np.isfinite(value.numpy()).all() for value in metric_values))
        self.assertLess(float(solver.metrics.data.r_eom.numpy()[0]), 1.0e-5)
        self.assertLess(float(solver.metrics.data.r_v_plus.numpy()[0]), 1.0e-5)

    def test_sparse_jacobian_matches_dense_step(self):
        results = []
        for sparse_jacobian in (False, True):
            model = build_boxes_hinged(z_offset=0.0, ground=True).finalize(device=self.device)
            detector = CollisionDetector(
                model,
                config=kamino_config.CollisionDetectorConfig(pipeline="primitive"),
            )
            contacts = detector.contacts
            solver = SolverKaminoImpl(
                model=model,
                contacts=contacts,
                config=self.make_config(
                    compute_solution_metrics=True,
                    sparse_jacobian=sparse_jacobian,
                ),
            )
            state_previous = model.state()
            state_next = model.state()
            solver.step(
                state_previous,
                state_next,
                model.control(),
                contacts=contacts,
                detector=detector,
                dt=0.01,
            )
            expected_type = SparseSystemJacobians if sparse_jacobian else DenseSystemJacobians
            self.assertIsInstance(solver._jacobians, expected_type)
            results.append(
                (
                    state_next.q_i.numpy(),
                    state_next.u_i.numpy(),
                    state_next.lambda_j.numpy(),
                    contacts.reaction.numpy(),
                    solver.metrics.data.r_eom.numpy(),
                    solver.metrics.data.r_vi_natmap.numpy(),
                )
            )

        for dense, sparse in zip(results[0], results[1], strict=True):
            np.testing.assert_allclose(sparse, dense, rtol=2.0e-5, atol=2.0e-6)

    def test_sparse_jacobian_direct_structural_step(self):
        model = build_boxes_hinged(z_offset=0.0, ground=False).finalize(device=self.device)
        config = self.make_config(sparse_jacobian=True)
        config.lox.joint_solve_direct = True
        solver = SolverKaminoImpl(model=model, config=config)
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), dt=0.01)

        self.assertIsInstance(solver._jacobians, SparseSystemJacobians)
        self.assertIsNotNone(solver.solver_fd.structural_joint_solver)
        self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])

    def test_revolute_joint_step_updates_structural_multipliers(self):
        model = build_box_pendulum(ground=False, dynamic_joints=True).finalize(device=self.device)
        solver = SolverKaminoImpl(
            model=model,
            config=self.make_config(nonlinear_iterations=1, compute_solution_metrics=True),
        )
        state_previous = model.state()
        state_next = model.state()
        state_previous.q_j.fill_(0.25)
        state_previous.q_j_p.fill_(-0.5)

        solver.step(state_previous, state_next, model.control(), dt=0.01)

        self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.lambda_j.numpy()).all())
        np.testing.assert_allclose(state_next.q_j_p.numpy(), state_previous.q_j.numpy(), rtol=0.0, atol=0.0)
        self.assertGreater(solver._lox_adapter.structural_row_count, 0)
        self.assertGreater(int(solver.solver_fd.iteration_count.numpy()[0]), 1)
        self.assertLessEqual(float(solver.solver_fd.splitting.residual_structural.numpy()[0]), 1.0)
        multiplier = solver._lox_adapter.structural_multiplier.numpy().copy()
        self.assertGreater(float(np.max(np.abs(multiplier))), 0.0)
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])

        solver.solver_fd.begin_time_step(0.01)
        np.testing.assert_allclose(
            solver._lox_adapter.structural_multiplier.numpy(),
            solver.solver_fd.joint_warmstart_factor * multiplier,
        )
        solver.solver_fd.reset()
        np.testing.assert_array_equal(
            solver._lox_adapter.structural_multiplier.numpy(),
            np.zeros_like(multiplier),
        )
        metric_values = (
            solver.metrics.data.r_eom.numpy(),
            solver.metrics.data.r_kinematics.numpy(),
            solver.metrics.data.r_cts_joints.numpy(),
            solver.metrics.data.r_v_plus.numpy(),
            solver.metrics.data.r_ncp_dual.numpy(),
        )
        self.assertTrue(all(np.isfinite(value).all() for value in metric_values))

    def test_direct_structural_config_builds_solver(self):
        model = build_boxes_hinged(z_offset=0.0, ground=False).finalize(device=self.device)
        config = self.make_config()
        config.lox.joint_solve_direct = True

        solver = SolverKaminoImpl(model=model, config=config)

        self.assertIsNotNone(solver.solver_fd.structural_joint_solver)

    def test_box_on_plane_projects_detected_contact(self):
        model = build_box_on_plane(ground=True).finalize(device=self.device)
        detector = CollisionDetector(
            model,
            config=kamino_config.CollisionDetectorConfig(pipeline="primitive"),
        )
        contacts = detector.contacts
        solver = SolverKaminoImpl(
            model=model,
            contacts=contacts,
            config=self.make_config(compute_solution_metrics=True),
        )
        state_previous = model.state()
        state_next = model.state()

        solver.step(
            state_previous,
            state_next,
            model.control(),
            contacts=contacts,
            detector=detector,
            dt=0.01,
        )

        self.assertGreater(int(contacts.model_active_contacts.numpy()[0]), 0)
        self.assertGreaterEqual(float(contacts.reaction.numpy()[0, 2]), 0.0)
        self.assertGreaterEqual(float(contacts.velocity.numpy()[0, 2]), -2.0e-4)
        self.assertGreaterEqual(float(state_next.u_i.numpy()[0, 2]), -2.0e-4)
        self.assertGreater(float(state_next.w_i.numpy()[0, 2]), 0.0)
        self.assertGreaterEqual(int(contacts.mode.numpy()[0]), 0)
        self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])
        contact_residual_max = solver.solver_fd.contact_residual_max.numpy()
        self.assertTrue(np.isfinite(contact_residual_max).all())
        self.assertLessEqual(float(contact_residual_max[0]), 1.0e-4)
        metric_values = np.asarray(
            [
                solver.metrics.data.r_eom.numpy()[0],
                solver.metrics.data.r_v_plus.numpy()[0],
                solver.metrics.data.r_ncp_primal.numpy()[0],
                solver.metrics.data.r_ncp_dual.numpy()[0],
                solver.metrics.data.r_ncp_compl.numpy()[0],
                solver.metrics.data.r_vi_natmap.numpy()[0],
            ]
        )
        self.assertTrue(np.isfinite(metric_values).all())
        self.assertLess(float(metric_values[0]), 1.0e-3)
        self.assertLess(float(metric_values[2]), 1.0e-5)

    def test_contact_metrics_use_projected_twist(self):
        model = build_box_on_plane(ground=True).finalize(device=self.device)
        detector = CollisionDetector(
            model,
            config=kamino_config.CollisionDetectorConfig(pipeline="primitive"),
        )
        contacts = detector.contacts
        solver = SolverKaminoImpl(
            model=model,
            contacts=contacts,
            config=self.make_config(compute_solution_metrics=True),
        )
        state_previous = model.state()
        state_next = model.state()

        solver.step(
            state_previous,
            state_next,
            model.control(),
            contacts=contacts,
            detector=detector,
            dt=0.01,
        )

        metrics = solver.metrics
        projected_vi = float(metrics.data.r_vi_natmap.numpy()[0])
        projected_compl = float(metrics.data.r_ncp_compl.numpy()[0])
        projected_f_ncp = float(metrics.data.f_ncp.numpy()[0])
        projected_f_ccp = float(metrics.data.f_ccp.numpy()[0])
        metrics.reset()
        metrics.evaluate(
            sigma=metrics._zero_sigma,
            lambdas=metrics._constraint_impulse,
            v_plus=metrics._constraint_velocity,
            model=model,
            data=solver._data,
            state_p=state_previous,
            problem=solver._problem_metrics,
            jacobians=solver._jacobians,
            limits=solver._limits,
            contacts=contacts,
        )
        dual_implied_vi = float(metrics.data.r_vi_natmap.numpy()[0])
        dual_implied_f_ncp = float(metrics.data.f_ncp.numpy()[0])
        dual_implied_f_ccp = float(metrics.data.f_ccp.numpy()[0])

        solver._lox_adapter.write_outputs(0.01, body_velocity=solver.solver_fd.projected_twist)
        metrics.reset()
        metrics.evaluate_from_constraint_forces(
            model=model,
            data=solver._data,
            state_p=state_previous,
            problem=solver._problem_metrics,
            jacobians=solver._jacobians,
            limits=solver._limits,
            contacts=contacts,
        )
        recomputed_projected_compl = float(metrics.data.r_ncp_compl.numpy()[0])

        solver._lox_adapter.write_outputs(0.01, body_velocity=solver.solver_fd.splitting.global_twist)
        metrics.reset()
        metrics.evaluate_from_constraint_forces(
            model=model,
            data=solver._data,
            state_p=state_previous,
            problem=solver._problem_metrics,
            jacobians=solver._jacobians,
            limits=solver._limits,
            contacts=contacts,
        )
        global_compl = float(metrics.data.r_ncp_compl.numpy()[0])

        self.assertGreater(dual_implied_vi, 1.0e-6)
        self.assertLess(projected_vi, 0.2 * dual_implied_vi)
        self.assertGreater(abs(projected_f_ncp - dual_implied_f_ncp), 1.0e-7)
        self.assertGreater(abs(projected_f_ccp - dual_implied_f_ccp), 1.0e-7)
        self.assertAlmostEqual(projected_compl, recomputed_projected_compl)
        self.assertGreater(global_compl, 5.0 * projected_compl)

    def test_product_space_structural_split_hinged_contact(self):
        model = build_boxes_hinged(z_offset=0.0, ground=True).finalize(device=self.device)
        detector = CollisionDetector(
            model,
            config=kamino_config.CollisionDetectorConfig(pipeline="primitive"),
        )
        contacts = detector.contacts
        config = self.make_config(compute_solution_metrics=True)
        config.lox.max_iterations = 10
        config.lox.projection_iterations = 3
        solver = SolverKaminoImpl(model=model, contacts=contacts, config=config)
        state_previous = model.state()
        state_next = model.state()

        for _ in range(20):
            solver.step(
                state_previous,
                state_next,
                model.control(),
                contacts=contacts,
                detector=detector,
                dt=0.01,
            )
            state_previous, state_next = state_next, state_previous

        splitting = solver.solver_fd.splitting
        self.assertGreater(int(contacts.model_active_contacts.numpy()[0]), 0)
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])
        self.assertTrue(np.isfinite(splitting.residual_structural.numpy()).all())
        self.assertTrue(np.isfinite(splitting.residual_structural_projected.numpy()).all())
        self.assertLess(float(solver.metrics.data.r_cts_joints.numpy()[0]), 1.0e-4)
        self.assertLess(float(solver.metrics.data.r_eom.numpy()[0]), 1.0e-3)

    def test_cartpole_projects_detected_joint_limit(self):
        model = build_cartpole(ground=False, limits=True).finalize(device=self.device)
        solver = SolverKaminoImpl(
            model=model,
            config=self.make_config(nonlinear_iterations=1, compute_solution_metrics=True),
        )
        state_previous = model.state()
        state_next = model.state()
        pose = state_previous.q_i.numpy()
        pose[:, 1] += 4.1
        state_previous.q_i.assign(pose)

        solver.step(state_previous, state_next, model.control(), dt=0.01)

        limits = solver._limits
        self.assertGreater(int(limits.model_active_limits.numpy()[0]), 0)
        self.assertGreater(float(limits.reaction.numpy()[0]), 0.0)
        self.assertGreaterEqual(float(limits.velocity.numpy()[0]), -2.0e-4)
        self.assertLess(float(state_next.dq_j.numpy()[0]), -0.09)
        self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])
        limit_residual_max = solver.solver_fd.limit_residual_max.numpy()
        self.assertTrue(np.isfinite(limit_residual_max).all())
        self.assertLessEqual(float(limit_residual_max[0]), 1.0e-4)
        metric_values = np.asarray(
            [
                solver.metrics.data.r_eom.numpy()[0],
                solver.metrics.data.r_v_plus.numpy()[0],
                solver.metrics.data.r_ncp_primal.numpy()[0],
                solver.metrics.data.r_ncp_dual.numpy()[0],
                solver.metrics.data.r_ncp_compl.numpy()[0],
                solver.metrics.data.r_vi_natmap.numpy()[0],
            ]
        )
        self.assertTrue(np.isfinite(metric_values).all())
        self.assertLess(float(metric_values[0]), 2.0e-2)
        self.assertLess(float(metric_values[2]), 1.0e-5)

    def test_cartpole_sustained_joint_force_remains_bounded(self):
        model = build_cartpole(ground=False, limits=True).finalize(device=self.device)
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.tau_j.assign(np.asarray([10.0, 0.0], dtype=np.float32))

        for _ in range(300):
            solver.step(state_previous, state_next, control, dt=0.001)
            state_previous, state_next = state_next, state_previous

        body_velocity = state_previous.u_i.numpy()
        joint_residual = solver._data.joints.r_j.numpy()
        self.assertTrue(np.isfinite(body_velocity).all())
        self.assertTrue(np.isfinite(joint_residual).all())
        self.assertLess(float(np.max(np.abs(body_velocity))), 100.0)
        self.assertLess(float(np.max(np.abs(joint_residual))), 1.0e-2)
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])

    def test_cuda_graph_capture(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        for sparse_jacobian in (False, True):
            with self.subTest(sparse_jacobian=sparse_jacobian):
                model = build_boxes_hinged(z_offset=0.0, ground=True).finalize(device=self.device)
                detector = CollisionDetector(
                    model,
                    config=kamino_config.CollisionDetectorConfig(pipeline="primitive"),
                )
                contacts = detector.contacts
                solver = SolverKaminoImpl(
                    model=model,
                    contacts=contacts,
                    config=self.make_config(
                        nonlinear_iterations=1,
                        compute_solution_metrics=True,
                        sparse_jacobian=sparse_jacobian,
                    ),
                )
                state_previous = model.state()
                state_next = model.state()
                control = model.control()
                solver.step(state_previous, state_next, control, contacts=contacts, detector=detector, dt=0.01)
                solver.reset(state_previous)

                with wp.ScopedCapture() as capture:
                    solver.step(state_previous, state_next, control, contacts=contacts, detector=detector, dt=0.01)
                wp.capture_launch(capture.graph)

                self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
                self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())
                self.assertTrue(np.isfinite(solver.metrics.data.r_eom.numpy()).all())
                self.assertTrue(np.isfinite(solver.metrics.data.r_vi_natmap.numpy()).all())

        newton_model = _build_revolute_dynamics_model(
            damping=1.0,
            friction=5.0,
            velocity=2.0,
            device=self.device,
        )
        model = ModelKamino.from_newton(newton_model)
        solver = SolverKaminoImpl(model=model, config=self.make_config(sparse_jacobian=True))
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        solver.step(state_previous, state_next, control, dt=0.1)
        solver.reset(state_previous)

        with wp.ScopedCapture() as capture:
            solver.step(state_previous, state_next, control, dt=0.1)
        wp.capture_launch(capture.graph)

        self.assertEqual(solver._lox_adapter.friction_capacity, 1)
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())
        self.assertTrue(np.isfinite(solver._lox_adapter.friction_reaction.numpy()).all())
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])

    def test_cuda_graph_capture_uses_conditional_loop(self):
        if not self.device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("CUDA conditional graph nodes require CUDA 12.4 or newer.")
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()

        with mock.patch.object(wp, "capture_while", wraps=wp.capture_while) as capture_while:
            with wp.ScopedCapture() as capture:
                solver.step(state_previous, state_next, control, dt=0.01)

        capture_while.assert_called_once()
        wp.capture_launch(capture.graph)
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
