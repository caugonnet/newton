# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pose integration for projected LOX body twists."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import warp as wp

from ......core.types import override
from ...core.bodies import update_body_inertias
from ...core.control import ControlKamino
from ...core.data import DataKamino
from ...core.math import compute_body_pose_update_with_logmap
from ...core.model import ModelKamino
from ...core.state import StateKamino
from ...core.types import vec6f
from ...geometry.contacts import ContactsKamino, make_contact_frame_znorm
from ...geometry.detector import CollisionDetector
from ...integrators.integrator import IntegratorBase
from ...kinematics.limits import LimitsKamino
from .adapter import LOXKaminoAdapter
from .limit import refresh_active_joint_configuration_limits
from .solver import LOXProblem, LOXSolver

if TYPE_CHECKING:
    from ....config import ConstraintStabilizationConfig, LOXSolverConfig
    from ...dynamics.dual import DualProblem
    from ...kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
    from ..warmstart import WarmstarterContacts, WarmstarterLimits

__all__ = ["IntegratorLOX", "accept_projected_body_state", "integrate_projected_body_poses"]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _integrate_projected_body_poses(
    body_world: wp.array[wp.int32],
    world_time_step: wp.array[wp.float32],
    pose_previous: wp.array[wp.transformf],
    projected_twist: wp.array[vec6f],
    pose_candidate: wp.array[wp.transformf],
):
    body = wp.tid()
    twist = projected_twist[body]
    pose_candidate[body] = compute_body_pose_update_with_logmap(
        world_time_step[body_world[body]],
        pose_previous[body],
        wp.vec3f(twist[0], twist[1], twist[2]),
        wp.vec3f(twist[3], twist[4], twist[5]),
    )


@wp.kernel
def _accept_projected_body_state(
    body_world: wp.array[wp.int32],
    world_accepted: wp.array[wp.bool],
    pose_candidate: wp.array[wp.transformf],
    twist_candidate: wp.array[vec6f],
    pose_accepted: wp.array[wp.transformf],
    twist_accepted: wp.array[vec6f],
):
    body = wp.tid()
    if world_accepted[body_world[body]]:
        pose_accepted[body] = pose_candidate[body]
        twist_accepted[body] = twist_candidate[body]


@wp.kernel
def _capture_frozen_contact_kinematics(
    active_count: wp.array[wp.int32],
    bodies: wp.array[wp.vec2i],
    position_a: wp.array[wp.vec3f],
    position_b: wp.array[wp.vec3f],
    gap: wp.array[wp.vec4f],
    body_pose: wp.array[wp.transformf],
    local_position_a: wp.array[wp.vec3f],
    local_position_b: wp.array[wp.vec3f],
    local_normal_a: wp.array[wp.vec3f],
    gap_offset: wp.array[wp.float32],
):
    contact = wp.tid()
    if contact >= active_count[0]:
        return

    body_pair = bodies[contact]
    point_a = position_a[contact]
    point_b = position_b[contact]
    normal = wp.normalize(wp.vec3f(gap[contact][0], gap[contact][1], gap[contact][2]))
    local_position_a[contact] = point_a
    local_position_b[contact] = point_b
    local_normal_a[contact] = normal
    if body_pair[0] >= 0:
        inverse_pose_a = wp.transform_inverse(body_pose[body_pair[0]])
        local_position_a[contact] = wp.transform_point(inverse_pose_a, point_a)
        local_normal_a[contact] = wp.transform_vector(inverse_pose_a, normal)
    if body_pair[1] >= 0:
        local_position_b[contact] = wp.transform_point(wp.transform_inverse(body_pose[body_pair[1]]), point_b)
    gap_offset[contact] = gap[contact][3] - wp.dot(point_b - point_a, normal)


@wp.kernel
def _refresh_frozen_contact_kinematics(
    active_count: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    bodies: wp.array[wp.vec2i],
    body_pose: wp.array[wp.transformf],
    local_position_a: wp.array[wp.vec3f],
    local_position_b: wp.array[wp.vec3f],
    local_normal_a: wp.array[wp.vec3f],
    gap_offset: wp.array[wp.float32],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    position_a: wp.array[wp.vec3f],
    position_b: wp.array[wp.vec3f],
    gap: wp.array[wp.vec4f],
    frame: wp.array[wp.quatf],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
):
    contact = wp.tid()
    if contact >= active_count[0]:
        return

    body_pair = bodies[contact]
    point_a = local_position_a[contact]
    point_b = local_position_b[contact]
    normal = local_normal_a[contact]
    if body_pair[0] >= 0:
        pose_a = body_pose[body_pair[0]]
        point_a = wp.transform_point(pose_a, point_a)
        normal = wp.transform_vector(pose_a, normal)
    if body_pair[1] >= 0:
        point_b = wp.transform_point(body_pose[body_pair[1]], point_b)
    normal = wp.normalize(normal)

    old_frame = wp.quat_to_matrix(frame[contact])
    new_frame = make_contact_frame_znorm(normal)
    world = contact_world[contact]
    local = contact_local[contact]
    if world >= 0 and world < world_capacity.shape[0] and local >= 0 and local < world_capacity[world]:
        destination = world_offset[world] + local
        reaction[destination] = wp.transpose(new_frame) @ (old_frame @ reaction[destination])
        velocity[destination] = wp.transpose(new_frame) @ (old_frame @ velocity[destination])

    position_a[contact] = point_a
    position_b[contact] = point_b
    gap[contact] = wp.vec4f(
        normal[0],
        normal[1],
        normal[2],
        wp.dot(point_b - point_a, normal) + gap_offset[contact],
    )
    frame[contact] = wp.quat_from_matrix(new_frame)


def integrate_projected_body_poses(
    body_world: wp.array[wp.int32],
    world_time_step: wp.array[wp.float32],
    pose_previous: wp.array[wp.transformf],
    projected_twist: wp.array[vec6f],
    pose_candidate: wp.array[wp.transformf],
) -> None:
    """Integrate body poses from a fixed begin-of-step state.

    Args:
        body_world: World index of each packed body.
        world_time_step: Time step of each world [s].
        pose_previous: Begin-of-step body poses.
        projected_twist: End-of-step linear-first body twists [m/s, rad/s].
        pose_candidate: Output end-of-step body poses.
    """
    body_count = body_world.shape[0]
    if (
        pose_previous.shape[0] != body_count
        or projected_twist.shape[0] != body_count
        or pose_candidate.shape[0] != body_count
    ):
        raise ValueError("Pose, twist, and body-world arrays must have identical lengths.")
    wp.launch(
        _integrate_projected_body_poses,
        dim=body_count,
        inputs=[body_world, world_time_step, pose_previous, projected_twist],
        outputs=[pose_candidate],
        device=pose_candidate.device,
    )


def accept_projected_body_state(
    body_world: wp.array[wp.int32],
    world_accepted: wp.array[wp.bool],
    pose_candidate: wp.array[wp.transformf],
    twist_candidate: wp.array[vec6f],
    pose_accepted: wp.array[wp.transformf],
    twist_accepted: wp.array[vec6f],
) -> None:
    """Update the last finite body state in accepted worlds only."""
    body_count = body_world.shape[0]
    if world_accepted.shape[0] == 0:
        raise ValueError("world_accepted must contain at least one world.")
    if any(array.shape[0] != body_count for array in (pose_candidate, twist_candidate, pose_accepted, twist_accepted)):
        raise ValueError("Pose, twist, and body-world arrays must have identical lengths.")
    wp.launch(
        _accept_projected_body_state,
        dim=body_count,
        inputs=[body_world, world_accepted, pose_candidate, twist_candidate],
        outputs=[pose_accepted, twist_accepted],
        device=pose_accepted.device,
    )


class IntegratorLOX(IntegratorBase):
    """Integrate accepted LOX iterates over a frozen contact topology."""

    def __init__(
        self,
        model: ModelKamino,
        solver: LOXSolver,
        problem: LOXProblem,
        config: LOXSolverConfig,
        constraints_config: ConstraintStabilizationConfig,
        jacobians: DenseSystemJacobians | SparseSystemJacobians,
        problem_metrics: DualProblem | None,
        warmstarter_limits: WarmstarterLimits | None,
        warmstarter_contacts: WarmstarterContacts | None,
        contacts: ContactsKamino | None,
        update_joints_data: Callable,
        update_jacobians: Callable,
        update_actuation_wrenches: Callable,
        update_wrenches: Callable,
        run_midstep_callback: Callable,
    ):
        super().__init__(model)
        self.solver = solver
        self.problem = problem
        self.config = config
        self.constraints_config = constraints_config
        self.jacobians = jacobians
        self.problem_metrics = problem_metrics
        self.warmstarter_limits = warmstarter_limits
        self.warmstarter_contacts = warmstarter_contacts
        self.update_joints_data = update_joints_data
        self.update_jacobians = update_jacobians
        self.update_actuation_wrenches = update_actuation_wrenches
        self.update_wrenches = update_wrenches
        self.run_midstep_callback = run_midstep_callback
        self.pose_begin = wp.zeros(model.size.sum_of_num_bodies, dtype=wp.transformf, device=model.device)
        self.pose_candidate = wp.zeros_like(self.pose_begin)
        self.pose_accepted = wp.zeros_like(self.pose_begin)
        self.twist_accepted = wp.zeros_like(solver.projected_twist)
        self.joint_position_begin = wp.zeros(
            model.size.sum_of_num_joint_coords,
            dtype=wp.float32,
            device=model.device,
        )
        self.contact_local_position_a = None
        self.contact_local_position_b = None
        self.contact_local_normal_a = None
        self.contact_gap_offset = None
        if config.nonlinear_iterations > 1 and contacts is not None and contacts.model_max_contacts_host > 0:
            capacity = contacts.model_max_contacts_host
            self.contact_local_position_a = wp.empty(capacity, dtype=wp.vec3f, device=model.device)
            self.contact_local_position_b = wp.empty(capacity, dtype=wp.vec3f, device=model.device)
            self.contact_local_normal_a = wp.empty(capacity, dtype=wp.vec3f, device=model.device)
            self.contact_gap_offset = wp.empty(capacity, dtype=wp.float32, device=model.device)

    @property
    def adapter(self) -> LOXKaminoAdapter | None:
        """Return the current rigid-body adapter."""
        return self.solver.adapter

    def set_solver(self, solver: LOXSolver) -> None:
        """Replace solver references after a rigid-topology rebuild."""
        self.solver = solver
        self.twist_accepted = wp.zeros_like(solver.projected_twist)

    @override
    def integrate(
        self,
        forward: Callable,
        model: ModelKamino,
        data: DataKamino,
        state_in: StateKamino,
        state_out: StateKamino,
        control: ControlKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        detector: CollisionDetector | None = None,
    ):
        """Advance one LOX step while retaining detected contact IDs."""
        time_step = model.time.dt
        inverse_time_step = model.time.inv_dt

        solver = self.solver
        adapter = self.adapter
        config = self.config
        constraints_config = self.constraints_config
        self.problem.initial_twist = None
        self.problem.linearization_twist = None
        self.problem.limit_stabilization_fraction = constraints_config.beta
        self.problem.contact_stabilization_fraction = constraints_config.gamma
        self.problem.contact_dead_zone = constraints_config.delta
        self.problem.impact_velocity_threshold = config.impact_velocity_threshold
        self.problem.contact_recoverable_response = config.contact_recoverable_response
        wp.copy(self.pose_begin, state_in.q_i)
        wp.copy(self.joint_position_begin, data.joints.q_j)

        forward(
            state_in=state_in,
            state_out=state_out,
            control=control,
            limits=limits,
            contacts=contacts,
            detector=detector,
        )

        if self.warmstarter_limits is not None:
            self.warmstarter_limits.warmstart(limits)
        if self.warmstarter_contacts is not None and contacts is not None:
            self.warmstarter_contacts.warmstart(model, data, contacts)

        if self.problem_metrics is not None:
            self.problem_metrics.build(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                jacobians=self.jacobians,
                reset_to_zero=True,
            )

        if self.contact_local_position_a is not None and contacts is not None:
            wp.launch(
                _capture_frozen_contact_kinematics,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    contacts.model_active_contacts,
                    contacts.bid_AB,
                    contacts.position_A,
                    contacts.position_B,
                    contacts.gapfunc,
                    data.bodies.q_i,
                ],
                outputs=[
                    self.contact_local_position_a,
                    self.contact_local_position_b,
                    self.contact_local_normal_a,
                    self.contact_gap_offset,
                ],
                device=model.device,
            )

        self.problem.begin_newton_deformable_time_step(time_step, inverse_time_step)
        solver.warmstart(problem=self.problem, model=model, data=data, limits=limits, contacts=contacts)
        wp.copy(self.pose_accepted, self.pose_begin)
        if adapter is not None:
            wp.copy(self.twist_accepted, adapter.body_velocity_begin)
            adapter.body_linearization_twist.zero_()
        for nonlinear_iteration in range(config.nonlinear_iterations):
            self.problem.linearization_twist = adapter.body_linearization_twist if adapter is not None else None
            solver.solve(problem=self.problem)
            if adapter is not None:
                integrate_projected_body_poses(
                    model.bodies.wid,
                    model.time.dt,
                    self.pose_begin,
                    solver.projected_twist,
                    self.pose_candidate,
                )
                accept_projected_body_state(
                    model.bodies.wid,
                    solver.world_accepted,
                    self.pose_candidate,
                    solver.projected_twist,
                    self.pose_accepted,
                    self.twist_accepted,
                )
                wp.copy(data.bodies.q_i, self.pose_accepted)
                wp.copy(solver.projected_twist, self.twist_accepted)
                adapter.write_outputs(time_step, inverse_time_step, body_velocity=self.twist_accepted)
                self.update_joints_data(q_j_p=self.joint_position_begin)
                update_body_inertias(model=model.bodies, data=data.bodies)
                adapter.gather_structural_candidate_residuals()

            if nonlinear_iteration + 1 < config.nonlinear_iterations:
                if adapter is not None:
                    wp.copy(adapter.body_linearization_twist, solver.projected_twist)
                    refresh_active_joint_configuration_limits(model, limits, data.joints.q_j)
                    if self.contact_local_position_a is not None and contacts is not None:
                        wp.launch(
                            _refresh_frozen_contact_kinematics,
                            dim=contacts.model_max_contacts_host,
                            inputs=[
                                contacts.model_active_contacts,
                                contacts.wid,
                                contacts.cid,
                                contacts.bid_AB,
                                data.bodies.q_i,
                                self.contact_local_position_a,
                                self.contact_local_position_b,
                                self.contact_local_normal_a,
                                self.contact_gap_offset,
                                adapter.world_contact_capacity,
                                adapter.world_contact_offset,
                            ],
                            outputs=[
                                contacts.position_A,
                                contacts.position_B,
                                contacts.gapfunc,
                                contacts.frame,
                                adapter.contact_reaction,
                                adapter.contact_velocity,
                            ],
                            device=model.device,
                        )
                    self.update_jacobians(contacts=contacts)
                    self.update_actuation_wrenches()
                if solver.deformable_system is not None:
                    solver.prepare_deformable_nonlinear_iteration(
                        time_step,
                        body_pose=data.bodies.q_i,
                        body_velocity_begin=adapter.body_velocity_begin if adapter is not None else None,
                    )

        wp.copy(data.joints.q_j_p, self.joint_position_begin)
        if adapter is not None:
            adapter.write_constraint_wrenches(time_step, inverse_time_step)
        self.problem.finish_newton_step(time_step, inverse_time_step)
        self.update_wrenches()
        if self.warmstarter_limits is not None:
            self.warmstarter_limits.update(limits)
        if self.warmstarter_contacts is not None:
            self.warmstarter_contacts.update(contacts)
        self.run_midstep_callback(state_in, state_out, control, contacts)
