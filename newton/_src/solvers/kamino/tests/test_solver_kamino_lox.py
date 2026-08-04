# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the LOX Kamino dynamics backend."""

import math
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.geometry.detector import CollisionDetector
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from newton._src.solvers.kamino._src.models.builders.basics import (
    build_box_on_plane,
    build_boxes_hinged,
    build_cartpole,
)
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino._src.solvers.lox import LOX_STATUS_ITERATION_LIMIT, LOXProblem, LOXSolver
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _build_revolute_dynamics_model(
    *,
    damping: float,
    friction: float,
    velocity: float,
    armature: float = 0.0,
    target_ke: float = 0.0,
    target_kd: float = 0.0,
    effort_limit: float = math.inf,
    actuator_mode: newton.JointTargetMode | None = None,
    device: wp.DeviceLike = None,
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
        armature=armature,
        target_ke=target_ke,
        target_kd=target_kd,
        effort_limit=effort_limit,
        actuator_mode=actuator_mode,
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


def _build_massless_fixed_child_drive_model(
    *,
    include_massless_fixed_child: bool = True,
    device: wp.DeviceLike = None,
) -> newton.Model:
    """Build a driven link with an optional massless fixed child."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    inertia = wp.mat33f(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    driven = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    revolute = builder.add_joint_revolute(
        parent=-1,
        child=driven,
        axis=newton.Axis.Z,
        armature=0.1,
        target_ke=650.0,
        target_kd=100.0,
        effort_limit=math.inf,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    joints = [revolute]
    if include_massless_fixed_child:
        child = builder.add_link(
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32)),
            mass=0.0,
            inertia=wp.mat33f(),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_fixed(
                parent=driven,
                child=child,
                parent_xform=wp.transformf(
                    wp.vec3f(0.0, 0.0, 0.1),
                    wp.quat_identity(dtype=wp.float32),
                ),
            )
        )
    builder.add_articulation(joints)
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_driven_link_with_grounded_fixed_base(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int]:
    """Build a driven link whose prescribed base overlaps the ground."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    inertia = wp.mat33f(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    base = builder.add_link(
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.05), wp.quat_identity(dtype=wp.float32)),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_box(
        base,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    driven = builder.add_link(
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.25), wp.quat_identity(dtype=wp.float32)),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    fixed = builder.add_joint_fixed(
        parent=-1,
        child=base,
        parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.05), wp.quat_identity(dtype=wp.float32)),
    )
    revolute = builder.add_joint_revolute(
        parent=base,
        child=driven,
        axis=newton.Axis.Z,
        armature=0.1,
        target_ke=650.0,
        target_kd=100.0,
        effort_limit=math.inf,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([fixed, revolute])
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    builder.end_world()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 16
    return model, driven


def _build_zero_mass_prismatic_anchor_model(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int, int]:
    """Build a zero-mass anchor connected to a dynamic prismatic child."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    anchor = builder.add_link()
    child = builder.add_link()
    builder.add_shape_box(anchor, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    builder.add_shape_box(child)
    fixed = builder.add_joint_fixed(parent=-1, child=anchor)
    prismatic = builder.add_joint_prismatic(parent=anchor, child=child, axis=newton.Axis.Z)
    builder.add_articulation([fixed, prismatic])
    builder.end_world()
    return builder.finalize(device=device), anchor, child


def _build_prescribed_only_model(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int]:
    """Build a world containing only one zero-mass body."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link()
    builder.add_shape_box(body, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    fixed = builder.add_joint_fixed(parent=-1, child=body)
    builder.add_articulation([fixed])
    builder.end_world()
    return builder.finalize(device=device), body


def _build_flagged_kinematic_model(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int]:
    """Build a massive free body whose motion is prescribed by its body flag."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    builder.end_world()
    return builder.finalize(device=device), body


def _build_cable_model(
    *,
    binary: bool = False,
    explicit_world: bool = True,
    enabled: bool = True,
    stretch_stiffness: float = 0.0,
    stretch_damping: float = 0.0,
    shear_stiffness: float = 0.0,
    shear_damping: float = 0.0,
    bend_stiffness: float = 0.0,
    bend_damping: float = 0.0,
    twist_stiffness: float = 0.0,
    twist_damping: float = 0.0,
    child_xform: wp.transformf | None = None,
    device: wp.DeviceLike = None,
) -> tuple[newton.Model, int, int]:
    """Build a gravity-free world or binary cable with coincident rest anchors."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    if explicit_world:
        builder.begin_world()
    inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    parent = -1
    if binary:
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    if child_xform is None:
        child_xform = wp.transform_identity(dtype=wp.float32)
    child = builder.add_link(xform=child_xform, mass=1.0, inertia=inertia, lock_inertia=True)
    joint = builder.add_joint_cable(
        parent,
        child,
        stretch_stiffness=stretch_stiffness,
        stretch_damping=stretch_damping,
        shear_stiffness=shear_stiffness,
        shear_damping=shear_damping,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        twist_stiffness=twist_stiffness,
        twist_damping=twist_damping,
        enabled=enabled,
    )
    builder.add_articulation([joint])
    if explicit_world:
        builder.end_world()
    return builder.finalize(device=device), parent, child


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

        self.assertIsInstance(solver.problem_fd, LOXProblem)
        self.assertIsInstance(solver.solver_fd, LOXSolver)
        self.assertIsInstance(solver._jacobians, DenseSystemJacobians)
        gravity = model.gravity.vector.numpy()[0]
        expected_velocity = time_step * gravity
        np.testing.assert_allclose(state_next.u_i.numpy()[0, :3], expected_velocity, rtol=0.0, atol=5.0e-4)
        expected_height = state_previous.q_i.numpy()[0, 2] + time_step * expected_velocity[2]
        self.assertAlmostEqual(float(state_next.q_i.numpy()[0, 2]), float(expected_height), places=5)
        self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())

    def test_free_fall_uses_each_world_time_step(self):
        """Advance each rigid world with its configured device-side time step."""
        builder = build_box_on_plane(ground=False)
        build_box_on_plane(builder=builder, ground=False)
        model = builder.finalize(device=self.device)
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        time_step = np.asarray([0.01, 0.025], dtype=np.float32)
        model.time.dt.assign(time_step)
        model.time.inv_dt.assign(1.0 / time_step)

        solver.step(state_previous, state_next, model.control(), dt=None)

        gravity = model.gravity.vector.numpy()
        expected_velocity = time_step[:, None] * gravity
        np.testing.assert_allclose(state_next.u_i.numpy()[:, :3], expected_velocity, rtol=0.0, atol=5.0e-4)
        expected_height = state_previous.q_i.numpy()[:, 2] + time_step * expected_velocity[:, 2]
        np.testing.assert_allclose(state_next.q_i.numpy()[:, 2], expected_height, rtol=0.0, atol=1.0e-6)

    def test_zero_mass_joint_anchor_is_prescribed_and_eliminated(self):
        """Treat a zero-mass joint anchor as prescribed without polluting the child matrix."""
        for direct_structural in (False, True):
            with self.subTest(direct_structural=direct_structural):
                model, anchor, child = _build_zero_mass_prismatic_anchor_model(device=self.device)
                config = self.make_config(nonlinear_iterations=2)
                config.use_collision_detector = False
                config.lox.joint_solve_direct = direct_structural
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                anchor_pose = state_previous.body_q.numpy()[anchor].copy()
                prescribed_velocity = np.zeros((2, 6), dtype=np.float32)
                prescribed_velocity[anchor, 0] = 0.25
                state_previous.body_qd.assign(prescribed_velocity)

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

                implementation = solver._solver_kamino
                adapter = implementation._lox_adapter
                self.assertEqual(adapter.system.body_vector_index_host, (-1, 0))
                self.assertEqual(adapter.system.info.total_vec_size, 6)
                expected_anchor_pose = anchor_pose.copy()
                expected_anchor_pose[0] += 0.0025
                np.testing.assert_allclose(
                    state_next.body_q.numpy()[anchor], expected_anchor_pose, rtol=0.0, atol=1.0e-7
                )
                np.testing.assert_allclose(
                    state_next.body_qd.numpy()[anchor], prescribed_velocity[anchor], rtol=0.0, atol=0.0
                )
                self.assertAlmostEqual(float(state_next.body_qd.numpy()[child, 0]), 0.25, places=4)
                self.assertLess(float(state_next.body_qd.numpy()[child, 2]), -1.0e-3)
                self.assertTrue(np.isfinite(adapter.system.weighted_matrix.numpy()).all())
                self.assertTrue(np.isfinite(state_next.body_q.numpy()).all())
                self.assertTrue(np.isfinite(state_next.body_qd.numpy()).all())
                np.testing.assert_array_equal(implementation._solver_fd.world_failed.numpy(), [False])

    def test_prescribed_only_world_advances_without_body_unknowns(self):
        """Advance a prescribed-only world with a zero-dimensional body system."""
        for direct_structural in (False, True):
            with self.subTest(direct_structural=direct_structural):
                model, body = _build_prescribed_only_model(device=self.device)
                config = self.make_config(nonlinear_iterations=2)
                config.use_collision_detector = False
                config.lox.joint_solve_direct = direct_structural
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                prescribed_velocity = np.asarray([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
                previous_pose = state_previous.body_q.numpy()[body].copy()
                state_previous.body_qd.assign(prescribed_velocity)

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

                implementation = solver._solver_kamino
                system = implementation._lox_adapter.system
                self.assertEqual(system.body_vector_index_host, (-1,))
                np.testing.assert_array_equal(system.info.dim.numpy(), [0])
                expected_pose = previous_pose.copy()
                expected_pose[0] += 0.002
                np.testing.assert_allclose(state_next.body_q.numpy()[body], expected_pose, rtol=0.0, atol=1.0e-7)
                np.testing.assert_allclose(
                    state_next.body_qd.numpy()[body], prescribed_velocity[body], rtol=0.0, atol=0.0
                )
                np.testing.assert_array_equal(implementation._solver_fd.world_failed.numpy(), [False])

    def test_body_flag_marks_massive_body_as_kinematic(self):
        """Prescribe a massive body's motion through BodyFlags.KINEMATIC."""
        model, body = _build_flagged_kinematic_model(device=self.device)
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        prescribed_velocity = np.asarray([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        previous_pose = state_previous.body_q.numpy()[body].copy()
        state_previous.body_qd.assign(prescribed_velocity)

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        system = solver._solver_kamino._lox_adapter.system
        self.assertEqual(system.body_vector_index_host, (-1,))
        expected_pose = previous_pose.copy()
        expected_pose[0] += 0.002
        np.testing.assert_allclose(state_next.body_q.numpy()[body], expected_pose, rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(state_next.body_qd.numpy()[body], prescribed_velocity[body], rtol=0.0, atol=0.0)

    def test_avbd_keeps_zero_mass_contact_body_prescribed(self):
        """Keep a zero-mass contact body fixed during AVBD projection."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverKamino.register_custom_attributes(builder)
        support = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
            label="zero_mass_support",
        )
        builder.add_shape_box(
            support,
            hx=1.0,
            hy=1.0,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0),
        )
        box = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
            label="dynamic_box",
        )
        builder.add_shape_box(
            box,
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=1.0),
        )
        model = builder.finalize(device=self.device)
        model.rigid_contact_max = 32
        collision_pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
        contacts = collision_pipeline.contacts()
        config = self.make_config()
        config.use_collision_detector = False
        config.lox.projection_method = "avbd"
        config.lox.projection_iterations = 20
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        support_pose = state_previous.body_q.numpy()[support].copy()

        collision_pipeline.collide(state_previous, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        solver.step(state_previous, state_next, model.control(), contacts, dt=0.001)

        np.testing.assert_allclose(state_next.body_q.numpy()[support], support_pose, rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(state_next.body_qd.numpy()[support], 0.0, rtol=0.0, atol=1.0e-7)
        self.assertLess(abs(float(state_next.body_qd.numpy()[box, 2])), 1.0e-3)
        np.testing.assert_array_equal(solver._solver_kamino._solver_fd.world_failed.numpy(), [False])

    def test_kinematic_joint_friction_is_projection_noop(self):
        """Ignore joint friction when both incident bodies are prescribed."""
        model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=1.0,
            velocity=0.25,
            device=self.device,
        )
        model.body_flags.fill_(int(newton.BodyFlags.KINEMATIC))
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        implementation = solver._solver_kamino
        self.assertEqual(implementation._lox_adapter.system.body_vector_index_host, (-1,))
        np.testing.assert_array_equal(implementation._solver_fd.world_failed.numpy(), [False])
        np.testing.assert_allclose(
            state_next.body_qd.numpy(),
            state_previous.body_qd.numpy(),
            rtol=0.0,
            atol=0.0,
        )

    def test_cable_world_parent_stretch_and_damping(self):
        """Restore stretch and reduce axial speed through cable damping."""
        model, _parent, child = _build_cable_model(stretch_stiffness=100.0, device=self.device)
        config = self.make_config(nonlinear_iterations=2)
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        state_previous.body_q.assign([wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32))])

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        self.assertLess(float(state_next.body_qd.numpy()[child, 2]), 0.0)
        adapter = solver._solver_kamino._lox_adapter
        self.assertEqual(adapter.cables.count, 1)
        self.assertEqual(adapter.dynamic_row_count, 0)
        self.assertEqual(adapter.structural_row_count, 0)
        self.assertTrue(np.isfinite(adapter.cables.stress.numpy()).all())

        speeds = []
        for damping in (0.0, 10.0):
            damped_model, _parent, damped_child = _build_cable_model(
                stretch_damping=damping,
                device=self.device,
            )
            damped_config = self.make_config(nonlinear_iterations=3)
            damped_config.use_collision_detector = False
            damped_solver = SolverKamino(damped_model, config=damped_config)
            damped_state_previous = damped_model.state()
            damped_state_next = damped_model.state()
            damped_state_previous.body_qd.assign([wp.spatial_vectorf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)])
            damped_solver.step(
                damped_state_previous,
                damped_state_next,
                damped_model.control(),
                contacts=None,
                dt=0.01,
            )
            speeds.append(float(damped_state_next.body_qd.numpy()[damped_child, 2]))
        self.assertLess(speeds[1], speeds[0])

    def test_cable_accepts_implicit_single_world(self):
        """Accept implicit ownership for a cable in a single-world model."""
        model, _parent, child = _build_cable_model(
            explicit_world=False,
            stretch_stiffness=100.0,
            device=self.device,
        )
        np.testing.assert_array_equal(model.joint_world.numpy(), [-1])
        np.testing.assert_array_equal(model.body_world.numpy(), [-1])

        config = self.make_config(nonlinear_iterations=2)
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        state_previous.body_q.assign([wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32))])

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        self.assertLess(float(state_next.body_qd.numpy()[child, 2]), 0.0)

    def test_cable_bend_and_twist_restore_rotation(self):
        """Restore isolated bend and twist rotations in their material modes."""
        cases = (
            ("bend", wp.vec3f(1.0, 0.0, 0.0), {"bend_stiffness": 20.0}, 3),
            ("twist", wp.vec3f(0.0, 0.0, 1.0), {"twist_stiffness": 20.0}, 5),
        )
        for name, axis, coefficients, velocity_index in cases:
            with self.subTest(mode=name):
                model, _parent, child = _build_cable_model(device=self.device, **coefficients)
                config = self.make_config(nonlinear_iterations=3)
                config.use_collision_detector = False
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                state_previous.body_q.assign(
                    [
                        wp.transformf(
                            wp.vec3f(0.0),
                            wp.quat_from_axis_angle(axis, 0.1),
                        )
                    ]
                )

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

                velocity = state_next.body_qd.numpy()[child]
                self.assertLess(float(velocity[velocity_index]), 0.0)
                self.assertTrue(np.isfinite(velocity).all())

        precurved_model, _parent, child = _build_cable_model(
            bend_stiffness=20.0,
            twist_stiffness=10.0,
            child_xform=wp.transformf(
                wp.vec3f(0.0),
                wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), 0.2),
            ),
            device=self.device,
        )
        precurved_config = self.make_config(nonlinear_iterations=2)
        precurved_config.use_collision_detector = False
        precurved_solver = SolverKamino(precurved_model, config=precurved_config)
        precurved_previous = precurved_model.state()
        precurved_next = precurved_model.state()

        precurved_solver.step(
            precurved_previous,
            precurved_next,
            precurved_model.control(),
            contacts=None,
            dt=0.01,
        )

        np.testing.assert_allclose(precurved_next.body_qd.numpy()[child], 0.0, atol=1.0e-7)
        cable = precurved_solver._solver_kamino._lox_adapter.cables
        self.assertGreater(float(np.linalg.norm(cable.rest_curvature_local.numpy()[0])), 0.0)
        np.testing.assert_allclose(cable.strain.numpy(), 0.0, atol=1.0e-6)

    def test_binary_cable_balances_wrenches_and_respects_enabled(self):
        """Balance binary cable wrenches and suppress disabled material forces."""
        model, parent, child = _build_cable_model(
            binary=True,
            shear_stiffness=50.0,
            device=self.device,
        )
        config = self.make_config(nonlinear_iterations=2, sparse_jacobian=True)
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        state_previous.body_q.assign(
            [
                wp.transform_identity(dtype=wp.float32),
                wp.transformf(wp.vec3f(0.1, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            ]
        )

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        velocity = state_next.body_qd.numpy()
        self.assertGreater(float(velocity[parent, 0]), 0.0)
        self.assertLess(float(velocity[child, 0]), 0.0)
        np.testing.assert_allclose(velocity[parent, :3] + velocity[child, :3], 0.0, atol=1.0e-6)
        wrench = solver._solver_kamino._lox_adapter.data.bodies.w_j_i.numpy()
        self.assertTrue(np.isfinite(wrench).all())
        np.testing.assert_allclose(wrench[parent] + wrench[child], 0.0, atol=1.0e-5)

        disabled_model, _parent, disabled_child = _build_cable_model(
            enabled=False,
            stretch_stiffness=100.0,
            device=self.device,
        )
        disabled_config = self.make_config(nonlinear_iterations=2)
        disabled_config.use_collision_detector = False
        disabled_solver = SolverKamino(disabled_model, config=disabled_config)
        disabled_previous = disabled_model.state()
        disabled_next = disabled_model.state()
        disabled_previous.body_q.assign([wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32))])
        disabled_solver.step(
            disabled_previous,
            disabled_next,
            disabled_model.control(),
            contacts=None,
            dt=0.01,
        )
        np.testing.assert_allclose(disabled_next.body_qd.numpy()[disabled_child], 0.0, atol=1.0e-7)

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

    def test_position_drive_with_massless_fixed_child(self):
        """Track a target when the driven body has a massless fixed child."""
        cases = ((False, False, False), (True, False, False), (True, False, True), (True, True, False))
        for include_massless_fixed_child, direct_structural, sparse_jacobian in cases:
            with self.subTest(
                include_massless_fixed_child=include_massless_fixed_child,
                direct_structural=direct_structural,
                sparse_jacobian=sparse_jacobian,
            ):
                model = _build_massless_fixed_child_drive_model(
                    include_massless_fixed_child=include_massless_fixed_child,
                    device=self.device,
                )
                config = self.make_config(sparse_jacobian=sparse_jacobian)
                config.use_collision_detector = False
                config.lox.joint_solve_direct = direct_structural
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                control = model.control()
                control.joint_target_q.assign([0.2])

                for _ in range(240):
                    solver.step(state_previous, state_next, control, contacts=None, dt=1.0 / 600.0)
                    state_previous, state_next = state_next, state_previous

                self.assertGreater(float(state_previous.joint_q.numpy()[0]), 0.1)
                expected_body_indices = (0, 6) if include_massless_fixed_child else (0,)
                self.assertEqual(
                    solver._solver_kamino._lox_adapter.system.body_vector_index_host, expected_body_indices
                )
                if include_massless_fixed_child and direct_structural:
                    self.assertIsNone(solver._solver_kamino._solver_fd.structural_joint_solver)
                if include_massless_fixed_child:
                    body_q = state_previous.body_q.numpy()
                    body_qd = state_previous.body_qd.numpy()
                    np.testing.assert_allclose(body_qd[1], body_qd[0], rtol=0.0, atol=1.0e-4)
                    np.testing.assert_allclose(body_q[1, 3:], body_q[0, 3:], rtol=0.0, atol=1.0e-5)
                    np.testing.assert_allclose(body_q[1, :2], body_q[0, :2], rtol=0.0, atol=1.0e-5)
                    self.assertAlmostEqual(float(body_q[1, 2] - body_q[0, 2]), 0.1, places=5)

    def test_prescribed_contact_does_not_reject_dynamic_world(self):
        """Ignore contacts whose two incident bodies are prescribed."""
        model, _driven = _build_driven_link_with_grounded_fixed_base(device=self.device)
        shape_pairs = wp.array([(0, 1)], dtype=wp.vec2i, device=self.device)
        collision_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=shape_pairs,
        )
        contacts = collision_pipeline.contacts()
        config = self.make_config(sparse_jacobian=True)
        config.use_collision_detector = False
        config.lox.projection_method = "gauss_seidel"
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.joint_target_q.assign([0.2])
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_previous)

        collision_pipeline.collide(state_previous, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        solver.step(state_previous, state_next, control, contacts, dt=1.0 / 600.0)
        adapter = solver._solver_kamino._lox_adapter
        self.assertTrue(np.any((adapter.contact_body_first.numpy() < 0) & (adapter.contact_body_second.numpy() < 0)))
        np.testing.assert_array_equal(solver._solver_kamino._solver_fd.world_failed.numpy(), [False])
        state_previous, state_next = state_next, state_previous

        for _ in range(119):
            collision_pipeline.collide(state_previous, contacts)
            solver.step(state_previous, state_next, control, contacts, dt=1.0 / 600.0)
            state_previous, state_next = state_next, state_previous

        np.testing.assert_array_equal(solver._solver_kamino._solver_fd.world_failed.numpy(), [False])
        self.assertGreater(float(state_previous.joint_q.numpy()[0]), 0.1)

    def test_joint_effort_limit_saturates_implicit_drive(self):
        """Clamp the internal implicit drive while retaining its full stiffness."""
        time_step = 0.1
        model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=0.0,
            velocity=0.0,
            target_ke=100.0,
            effort_limit=1.0,
            actuator_mode=newton.JointTargetMode.POSITION,
            device=self.device,
        )
        config = self.make_config()
        config.lox.max_iterations = 40
        config.lox.joint_solve_direct = True
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.joint_target_q.assign([1.0])

        solver.step(state_previous, state_next, control, contacts=None, dt=time_step)

        adapter = solver._solver_kamino._lox_adapter
        data = solver._solver_kamino._data
        multiplier = data.joints.lambda_j.numpy()[int(adapter.dynamic_multiplier_index.numpy()[0])]
        self.assertEqual(adapter.effort_capacity, 1)
        self.assertAlmostEqual(float(adapter.dynamic_effective_inertia.numpy()[0]), 1.0, places=6)
        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), 0.1, places=4)
        self.assertAlmostEqual(float(adapter.effort_net_applied.numpy()[0] / time_step), 1.0, places=4)
        self.assertAlmostEqual(float(multiplier), 1.0, places=3)
        self.assertLess(float(adapter.world_effort_defect_max.numpy()[0]), 1.0e-5)

    def test_joint_effort_outputs_use_applied_correction_at_iteration_limit(self):
        """Report the applied effort correction for an iteration-limited state."""
        time_step = 0.1
        model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=0.0,
            velocity=0.0,
            target_ke=100.0,
            effort_limit=1.0,
            actuator_mode=newton.JointTargetMode.POSITION,
            device=self.device,
        )
        config = self.make_config()
        config.lox.max_iterations = 1
        config.lox.joint_solve_direct = True
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.joint_target_q.assign([1.0])

        solver.step(state_previous, state_next, control, contacts=None, dt=time_step)

        implementation = solver._solver_kamino
        adapter = implementation._lox_adapter
        multiplier = implementation._data.joints.lambda_j.numpy()[int(adapter.dynamic_multiplier_index.numpy()[0])]
        self.assertEqual(int(implementation._solver_fd.world_status.numpy()[0]), LOX_STATUS_ITERATION_LIMIT)
        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), 5.0, places=4)
        self.assertAlmostEqual(float(adapter.effort_counter_applied.numpy()[0]), 0.0, places=6)
        self.assertAlmostEqual(float(adapter.effort_counter_next.numpy()[0]), -4.9, places=4)
        self.assertAlmostEqual(float(adapter.effort_net_applied.numpy()[0]), 5.0, places=4)
        self.assertAlmostEqual(float(adapter.effort_net_target.numpy()[0]), 0.1, places=4)
        self.assertAlmostEqual(float(multiplier), 50.0, places=3)

    def test_joint_effort_limit_excludes_external_joint_force(self):
        """Keep external joint forces outside the actuator effort bound."""
        time_step = 0.1
        model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=0.0,
            velocity=0.0,
            armature=1.0,
            effort_limit=0.5,
            actuator_mode=newton.JointTargetMode.EFFORT,
            device=self.device,
        )
        config = self.make_config()
        config.lox.joint_solve_direct = True
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.joint_f.assign([20.0])

        solver.step(state_previous, state_next, control, contacts=None, dt=time_step)

        adapter = solver._solver_kamino._lox_adapter
        multiplier = solver._solver_kamino._data.joints.lambda_j.numpy()[
            int(adapter.dynamic_multiplier_index.numpy()[0])
        ]
        self.assertEqual(adapter.effort_capacity, 1)
        self.assertAlmostEqual(float(adapter.effort_raw_impulse.numpy()[0]), 0.0, places=6)
        self.assertAlmostEqual(float(adapter.effort_net_applied.numpy()[0]), 0.0, places=6)
        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), 1.0, places=4)
        self.assertAlmostEqual(float(multiplier), 10.0, places=3)

    def test_unlimited_joint_effort_keeps_legacy_path(self):
        """Keep infinite-effort drives on the legacy LOX path."""
        time_step = 0.1
        model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=0.0,
            velocity=0.0,
            target_ke=100.0,
            effort_limit=math.inf,
            actuator_mode=newton.JointTargetMode.POSITION,
            device=self.device,
        )
        config = self.make_config()
        config.lox.joint_solve_direct = True
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.joint_target_q.assign([1.0])

        solver.step(state_previous, state_next, control, contacts=None, dt=time_step)

        adapter = solver._solver_kamino._lox_adapter
        self.assertEqual(adapter.effort_capacity, 0)
        self.assertFalse(adapter.has_bounded_effort)
        self.assertIsNone(adapter.effort_counter_applied)
        self.assertIsNone(adapter.world_effort_residual_max)
        self.assertEqual(adapter.body_constraint_count.numpy().tolist(), [adapter.structural_block_count])
        self.assertEqual(adapter.body_has_unilateral.numpy().tolist(), [0])
        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), 5.0, places=4)

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

    def test_one_color_gauss_seidel_matches_rigid_jacobi(self):
        """Dispatch one-color rigid Gauss--Seidel through the exact Jacobi path."""
        results = {}
        for projection_method, max_colors in (("jacobi", 0), ("gauss_seidel", 1)):
            model = _build_revolute_dynamics_model(
                damping=0.0,
                friction=5.0,
                velocity=2.0,
                device=self.device,
            )
            config = self.make_config()
            config.lox.projection_method = projection_method
            config.lox.gauss_seidel_max_colors = max_colors
            solver = SolverKamino(model, config=config)
            state_next = model.state()

            solver.step(model.state(), state_next, model.control(), contacts=None, dt=0.1)

            adapter = solver._solver_kamino._lox_adapter
            results[projection_method] = (
                state_next.body_qd.numpy(),
                state_next.joint_qd.numpy(),
                adapter.friction_reaction.numpy(),
            )

        for colored_value, jacobi_value in zip(results["gauss_seidel"], results["jacobi"], strict=True):
            np.testing.assert_array_equal(colored_value, jacobi_value)

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

    def test_direct_structural_config_builds_solver(self):
        model = build_boxes_hinged(z_offset=0.0, ground=False).finalize(device=self.device)
        config = self.make_config()
        config.lox.joint_solve_direct = True

        solver = SolverKaminoImpl(model=model, config=config)

        self.assertIsNotNone(solver.solver_fd.structural_joint_solver)

    def test_selective_weights_config_builds_solver(self):
        """Enable selective proximal weights through the public LOX config."""
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        config = self.make_config()
        config.lox.selective_weights = True

        solver = SolverKaminoImpl(model=model, config=config)

        self.assertTrue(solver.solver_fd.system.selective_body_weights)

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

    def test_cuda_graph_capture_can_unroll_conditional_loop(self):
        """Unroll captured LOX splitting iterations when graph conditionals are disabled."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model = build_box_on_plane(ground=False).finalize(device=self.device)
        config = self.make_config()
        config.lox.use_graph_conditionals = False
        solver = SolverKaminoImpl(model=model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()

        with mock.patch.object(wp, "capture_while", wraps=wp.capture_while) as capture_while:
            with wp.ScopedCapture() as capture:
                solver.step(state_previous, state_next, control, dt=0.01)

        capture_while.assert_not_called()
        wp.capture_launch(capture.graph)
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
