# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free APGD projection for LOX unilateral reactions."""

from __future__ import annotations

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact import solve_contact_coulomb_newton
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    apply_contact_desaxce_correction,
    convert_contact_vector_normal_first_to_last,
    convert_contact_vector_normal_last_to_first,
    project_contact_coulomb_cone_orthogonal,
)

__all__ = [
    "project_constraints_apgd",
    "project_deformable_constraints_apgd",
]

wp.set_module_options({"enable_backward": False})


@wp.func
def _is_finite_vec3(value: wp.vec3f) -> wp.bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.kernel
def _initialize_world_state(
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    theta: wp.array[wp.float32],
    beta: wp.array[wp.float32],
    restart_dot: wp.array[wp.float32],
    projection_status: wp.array[wp.int32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    theta[world] = 1.0
    beta[world] = 0.0
    restart_dot[world] = 0.0
    projection_status[world] = prepared_status[world]


@wp.kernel
def _initialize_rigid_trials(
    friction_capacity: int,
    contact_capacity: int,
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_bound: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_friction: wp.array[wp.float32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    friction_reaction: wp.array[wp.float32],
    friction_trial: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_trial: wp.array[wp.float32],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if friction_local[constraint] >= world_friction_count[world] or not world_active[world]:
            return
        friction_value = wp.clamp(
            friction_reaction[constraint],
            -friction_bound[constraint],
            friction_bound[constraint],
        )
        friction_reaction[constraint] = friction_value
        friction_trial[constraint] = friction_value
        return

    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if contact_local[constraint] >= world_contact_count[world] or not world_active[world]:
            return
        contact_value = wp.vec3f(0.0)
        if contact_body_first[constraint] >= 0 or contact_body_second[constraint] >= 0:
            contact_value = project_contact_coulomb_cone_orthogonal(
                contact_reaction[constraint],
                contact_friction[constraint],
            )
        contact_reaction[constraint] = contact_value
        contact_trial[constraint] = contact_value
        return

    constraint -= contact_capacity
    world = limit_world[constraint]
    if limit_local[constraint] >= world_limit_count[world] or not world_active[world]:
        return
    limit_value = wp.max(0.0, limit_reaction[constraint])
    limit_reaction[constraint] = limit_value
    limit_trial[constraint] = limit_value


@wp.kernel
def _scatter_rigid_reactions_fused(
    friction_capacity: int,
    contact_capacity: int,
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    friction_reaction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if (
            friction_local[constraint] >= world_friction_count[world]
            or not world_active[world]
            or projection_status[world] != PROJECTION_STATUS_VALID
        ):
            return
        friction_impulse = friction_reaction[constraint]
        first = friction_body_first[constraint]
        second = friction_body_second[constraint]
        if first >= 0:
            wp.atomic_add(
                twist_delta,
                first,
                inverse_weight[first] @ (friction_impulse * friction_jacobian_first[constraint]),
            )
        if second >= 0:
            wp.atomic_add(
                twist_delta,
                second,
                inverse_weight[second] @ (friction_impulse * friction_jacobian_second[constraint]),
            )
        return

    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if (
            contact_local[constraint] >= world_contact_count[world]
            or not world_active[world]
            or projection_status[world] != PROJECTION_STATUS_VALID
        ):
            return
        contact_impulse = contact_reaction[constraint]
        first = contact_body_first[constraint]
        second = contact_body_second[constraint]
        if first >= 0:
            wp.atomic_add(
                twist_delta,
                first,
                inverse_weight[first] @ (wp.transpose(contact_jacobian_first[constraint]) @ contact_impulse),
            )
        if second >= 0:
            wp.atomic_add(
                twist_delta,
                second,
                inverse_weight[second] @ (wp.transpose(contact_jacobian_second[constraint]) @ contact_impulse),
            )
        return

    constraint -= contact_capacity
    world = limit_world[constraint]
    if (
        limit_local[constraint] >= world_limit_count[world]
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    limit_impulse = limit_reaction[constraint]
    first = limit_body_first[constraint]
    second = limit_body_second[constraint]
    if first >= 0:
        wp.atomic_add(
            twist_delta,
            first,
            inverse_weight[first] @ (limit_impulse * limit_jacobian_first[constraint]),
        )
    if second >= 0:
        wp.atomic_add(
            twist_delta,
            second,
            inverse_weight[second] @ (limit_impulse * limit_jacobian_second[constraint]),
        )


@wp.kernel
def _reconstruct_twist(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    baseline_twist: wp.array[vec6f],
    twist_delta: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    value = baseline_twist[body]
    if world_active[world] and projection_status[world] == PROJECTION_STATUS_VALID:
        value += twist_delta[body]
    projected_twist[body] = value
    twist_delta[body] = vec6f(0.0)


@wp.kernel
def _project_rigid_steps_fused(
    friction_capacity: int,
    contact_capacity: int,
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_bound: wp.array[wp.float32],
    friction_preconditioner: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_delassus: wp.array[wp.mat33f],
    contact_delassus_normal_first: wp.array[wp.mat33f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_preconditioner: wp.array[wp.float32],
    world_active: wp.array[wp.bool],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    friction_trial: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_trial: wp.array[wp.float32],
    friction_next: wp.array[wp.float32],
    friction_velocity: wp.array[wp.float32],
    contact_next: wp.array[wp.vec3f],
    contact_velocity: wp.array[wp.vec3f],
    limit_next: wp.array[wp.float32],
    limit_velocity: wp.array[wp.float32],
    restart_dot: wp.array[wp.float32],
    projection_status: wp.array[wp.int32],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if (
            friction_local[constraint] >= world_friction_count[world]
            or not world_active[world]
            or projection_status[world] != PROJECTION_STATUS_VALID
        ):
            return
        friction_value = float(0.0)
        first = friction_body_first[constraint]
        second = friction_body_second[constraint]
        if first >= 0:
            friction_value += wp.dot(friction_jacobian_first[constraint], projected_twist[first])
        if second >= 0:
            friction_value += wp.dot(friction_jacobian_second[constraint], projected_twist[second])
        metric = friction_preconditioner[constraint]
        if not wp.isfinite(metric) or metric <= 0.0 or not wp.isfinite(friction_value):
            projection_status[world] = PROJECTION_STATUS_INVALID
            return
        friction_next_value = wp.clamp(
            friction_trial[constraint] - friction_value / metric,
            -friction_bound[constraint],
            friction_bound[constraint],
        )
        if not wp.isfinite(friction_next_value):
            projection_status[world] = PROJECTION_STATUS_INVALID
            return
        friction_next[constraint] = friction_next_value
        friction_velocity[constraint] = friction_value
        wp.atomic_add(
            restart_dot,
            world,
            (friction_next_value - friction_reaction[constraint]) * (-friction_value),
        )
        return

    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if (
            contact_local[constraint] >= world_contact_count[world]
            or not world_active[world]
            or projection_status[world] != PROJECTION_STATUS_VALID
        ):
            return
        first = contact_body_first[constraint]
        second = contact_body_second[constraint]
        if first < 0 and second < 0:
            contact_next[constraint] = wp.vec3f(0.0)
            contact_velocity[constraint] = wp.vec3f(0.0)
            return
        contact_value = contact_bias[constraint]
        if first >= 0:
            contact_value += contact_jacobian_first[constraint] @ projected_twist[first]
        if second >= 0:
            contact_value += contact_jacobian_second[constraint] @ projected_twist[second]
        corrected = apply_contact_desaxce_correction(contact_value, contact_friction[constraint])
        if not _is_finite_vec3(corrected):
            projection_status[world] = PROJECTION_STATUS_INVALID
            return
        free_velocity_normal_first = convert_contact_vector_normal_last_to_first(
            contact_value - contact_delassus[constraint] @ contact_trial[constraint]
        )
        contact_next_value = convert_contact_vector_normal_first_to_last(
            solve_contact_coulomb_newton(
                contact_delassus_normal_first[constraint],
                free_velocity_normal_first,
                contact_friction[constraint],
            )
        )
        if not _is_finite_vec3(contact_next_value):
            projection_status[world] = PROJECTION_STATUS_INVALID
            return
        contact_next[constraint] = contact_next_value
        contact_velocity[constraint] = contact_value - contact_bias[constraint]
        wp.atomic_add(
            restart_dot,
            world,
            wp.dot(contact_next_value - contact_reaction[constraint], -corrected),
        )
        return

    constraint -= contact_capacity
    world = limit_world[constraint]
    if (
        limit_local[constraint] >= world_limit_count[world]
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = limit_body_first[constraint]
    second = limit_body_second[constraint]
    if first < 0 and second < 0:
        limit_next[constraint] = 0.0
        limit_velocity[constraint] = 0.0
        return
    limit_value = limit_bias[constraint]
    if first >= 0:
        limit_value += wp.dot(limit_jacobian_first[constraint], projected_twist[first])
    if second >= 0:
        limit_value += wp.dot(limit_jacobian_second[constraint], projected_twist[second])
    metric = limit_preconditioner[constraint]
    if not wp.isfinite(metric) or metric <= 0.0 or not wp.isfinite(limit_value):
        projection_status[world] = PROJECTION_STATUS_INVALID
        return
    limit_next_value = wp.max(0.0, limit_trial[constraint] - limit_value / metric)
    if not wp.isfinite(limit_next_value):
        projection_status[world] = PROJECTION_STATUS_INVALID
        return
    limit_next[constraint] = limit_next_value
    limit_velocity[constraint] = limit_value - limit_bias[constraint]
    wp.atomic_add(restart_dot, world, (limit_next_value - limit_reaction[constraint]) * (-limit_value))


@wp.kernel
def _finalize_acceleration(
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    restart_dot: wp.array[wp.float32],
    theta: wp.array[wp.float32],
    beta: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    value = restart_dot[world]
    restart_dot[world] = 0.0
    current = theta[world]
    if (
        projection_status[world] != PROJECTION_STATUS_VALID
        or not wp.isfinite(value)
        or value <= 0.0
        or not wp.isfinite(current)
        or current <= 0.0
    ):
        theta[world] = 1.0
        beta[world] = 0.0
        return
    next_theta = 2.0 * current / (wp.sqrt(current * current + 4.0) + current)
    beta[world] = current * (1.0 - current) / (current * current + next_theta)
    theta[world] = next_theta


@wp.kernel
def _extrapolate_rigid_reactions_fused(
    friction_capacity: int,
    contact_capacity: int,
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    beta: wp.array[wp.float32],
    friction_next: wp.array[wp.float32],
    contact_next: wp.array[wp.vec3f],
    limit_next: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    friction_trial: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_trial: wp.array[wp.float32],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if (
            friction_local[constraint] >= world_friction_count[world]
            or not world_active[world]
            or projection_status[world] != PROJECTION_STATUS_VALID
        ):
            return
        friction_old = friction_reaction[constraint]
        friction_value = friction_next[constraint]
        friction_reaction[constraint] = friction_value
        friction_trial[constraint] = friction_value + beta[world] * (friction_value - friction_old)
        return

    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if (
            contact_local[constraint] >= world_contact_count[world]
            or not world_active[world]
            or projection_status[world] != PROJECTION_STATUS_VALID
        ):
            return
        contact_old = contact_reaction[constraint]
        contact_value = contact_next[constraint]
        contact_reaction[constraint] = contact_value
        contact_trial[constraint] = contact_value + beta[world] * (contact_value - contact_old)
        return

    constraint -= contact_capacity
    world = limit_world[constraint]
    if (
        limit_local[constraint] >= world_limit_count[world]
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    limit_old = limit_reaction[constraint]
    limit_value = limit_next[constraint]
    limit_reaction[constraint] = limit_value
    limit_trial[constraint] = limit_value + beta[world] * (limit_value - limit_old)


def _scatter_rigid_reactions(adapter, world_active, inverse_weight, reaction_fields, twist_delta) -> None:
    friction_reaction, contact_reaction, limit_reaction = reaction_fields
    capacity = adapter.friction_capacity + adapter.contact_capacity + adapter.limit_capacity
    if capacity == 0:
        return
    wp.launch(
        _scatter_rigid_reactions_fused,
        dim=capacity,
        inputs=[
            adapter.friction_capacity,
            adapter.contact_capacity,
            adapter.friction_world,
            adapter.friction_local,
            adapter.world_friction_count,
            adapter.friction_body_first,
            adapter.friction_body_second,
            adapter.friction_jacobian_first,
            adapter.friction_jacobian_second,
            adapter.contact_world,
            adapter.contact_local,
            adapter.world_contact_count,
            adapter.contact_body_first,
            adapter.contact_body_second,
            adapter.contact_jacobian_first,
            adapter.contact_jacobian_second,
            adapter.limit_world,
            adapter.limit_local,
            adapter.world_limit_count,
            adapter.limit_body_first,
            adapter.limit_body_second,
            adapter.limit_jacobian_first,
            adapter.limit_jacobian_second,
            world_active,
            adapter.projection_status,
            inverse_weight,
            friction_reaction,
            contact_reaction,
            limit_reaction,
        ],
        outputs=[twist_delta],
        device=adapter.device,
    )


def project_constraints_apgd(
    projection_iterations: int,
    adapter,
    world_active: wp.array[wp.bool],
    body_world: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    body_baseline: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    theta: wp.array[wp.float32],
    beta: wp.array[wp.float32],
    restart_dot: wp.array[wp.float32],
    deformable_contacts=None,
    particle_baseline: wp.array[wp.vec3] | None = None,
    projected_velocity: wp.array[wp.vec3] | None = None,
) -> None:
    """Run fixed-count matrix-free APGD and return its final feasible iterate."""
    if (deformable_contacts is None) != (projected_velocity is None):
        raise ValueError("APGD deformable contacts and projected velocity must be supplied together.")
    if deformable_contacts is not None and particle_baseline is None:
        raise ValueError("APGD deformable projection requires a particle baseline.")
    wp.copy(body_baseline, projected_twist)
    adapter.projection_twist_delta.zero_()
    if deformable_contacts is not None:
        wp.copy(particle_baseline, projected_velocity)
        deformable_contacts.begin_apgd_scatter()
    wp.launch(
        _initialize_world_state,
        dim=world_active.shape[0],
        inputs=[world_active, adapter.world_jacobi_projection_status],
        outputs=[theta, beta, restart_dot, adapter.projection_status],
        device=adapter.device,
    )
    rigid_capacity = adapter.friction_capacity + adapter.contact_capacity + adapter.limit_capacity
    if rigid_capacity > 0:
        wp.launch(
            _initialize_rigid_trials,
            dim=rigid_capacity,
            inputs=[
                adapter.friction_capacity,
                adapter.contact_capacity,
                adapter.friction_world,
                adapter.friction_local,
                adapter.world_friction_count,
                adapter.friction_impulse_bound,
                adapter.contact_world,
                adapter.contact_local,
                adapter.world_contact_count,
                adapter.contact_body_first,
                adapter.contact_body_second,
                adapter.contact_friction,
                adapter.limit_world,
                adapter.limit_local,
                adapter.world_limit_count,
                world_active,
            ],
            outputs=[
                adapter.friction_reaction,
                adapter.friction_apgd_trial,
                adapter.contact_reaction,
                adapter.contact_apgd_trial,
                adapter.limit_reaction,
                adapter.limit_apgd_trial,
            ],
            device=adapter.device,
        )
    if deformable_contacts is not None:
        deformable_contacts.initialize_apgd(world_active, rigid_coordinates=True)

    for _iteration in range(projection_iterations):
        _scatter_rigid_reactions(
            adapter,
            world_active,
            inverse_weight,
            (adapter.friction_apgd_trial, adapter.contact_apgd_trial, adapter.limit_apgd_trial),
            adapter.projection_twist_delta,
        )
        if deformable_contacts is not None:
            deformable_contacts.scatter_apgd(
                world_active,
                inverse_weight,
                adapter.projection_twist_delta,
                rigid_coordinates=True,
                use_trial=True,
                projection_status=adapter.projection_status,
            )
        wp.launch(
            _reconstruct_twist,
            dim=projected_twist.shape[0],
            inputs=[body_world, world_active, adapter.projection_status, body_baseline],
            outputs=[adapter.projection_twist_delta, projected_twist],
            device=adapter.device,
        )
        if deformable_contacts is not None:
            deformable_contacts.reconstruct_apgd_particle_velocity(
                world_active,
                adapter.projection_status,
                particle_baseline,
                projected_velocity,
            )

        if rigid_capacity > 0:
            wp.launch(
                _project_rigid_steps_fused,
                dim=rigid_capacity,
                inputs=[
                    adapter.friction_capacity,
                    adapter.contact_capacity,
                    adapter.friction_world,
                    adapter.friction_local,
                    adapter.world_friction_count,
                    adapter.friction_body_first,
                    adapter.friction_body_second,
                    adapter.friction_jacobian_first,
                    adapter.friction_jacobian_second,
                    adapter.friction_impulse_bound,
                    adapter.friction_projection_delassus,
                    adapter.contact_world,
                    adapter.contact_local,
                    adapter.world_contact_count,
                    adapter.contact_body_first,
                    adapter.contact_body_second,
                    adapter.contact_jacobian_first,
                    adapter.contact_jacobian_second,
                    adapter.contact_bias,
                    adapter.contact_friction,
                    adapter.contact_projection_delassus,
                    adapter.contact_projection_delassus_normal_first,
                    adapter.limit_world,
                    adapter.limit_local,
                    adapter.world_limit_count,
                    adapter.limit_body_first,
                    adapter.limit_body_second,
                    adapter.limit_jacobian_first,
                    adapter.limit_jacobian_second,
                    adapter.limit_bias,
                    adapter.limit_projection_delassus,
                    world_active,
                    projected_twist,
                    adapter.friction_reaction,
                    adapter.friction_apgd_trial,
                    adapter.contact_reaction,
                    adapter.contact_apgd_trial,
                    adapter.limit_reaction,
                    adapter.limit_apgd_trial,
                ],
                outputs=[
                    adapter.friction_apgd_next,
                    adapter.friction_velocity,
                    adapter.contact_apgd_next,
                    adapter.contact_velocity,
                    adapter.limit_apgd_next,
                    adapter.limit_velocity,
                    restart_dot,
                    adapter.projection_status,
                ],
                device=adapter.device,
            )
        if deformable_contacts is not None:
            deformable_contacts.project_apgd(
                world_active,
                projected_velocity,
                projected_twist,
                restart_dot,
                adapter.projection_status,
                rigid_coordinates=True,
            )
        wp.launch(
            _finalize_acceleration,
            dim=world_active.shape[0],
            inputs=[world_active, adapter.projection_status],
            outputs=[restart_dot, theta, beta],
            device=adapter.device,
        )
        if rigid_capacity > 0:
            wp.launch(
                _extrapolate_rigid_reactions_fused,
                dim=rigid_capacity,
                inputs=[
                    adapter.friction_capacity,
                    adapter.contact_capacity,
                    adapter.friction_world,
                    adapter.friction_local,
                    adapter.world_friction_count,
                    adapter.contact_world,
                    adapter.contact_local,
                    adapter.world_contact_count,
                    adapter.limit_world,
                    adapter.limit_local,
                    adapter.world_limit_count,
                    world_active,
                    adapter.projection_status,
                    beta,
                    adapter.friction_apgd_next,
                    adapter.contact_apgd_next,
                    adapter.limit_apgd_next,
                ],
                outputs=[
                    adapter.friction_reaction,
                    adapter.friction_apgd_trial,
                    adapter.contact_reaction,
                    adapter.contact_apgd_trial,
                    adapter.limit_reaction,
                    adapter.limit_apgd_trial,
                ],
                device=adapter.device,
            )
        if deformable_contacts is not None:
            deformable_contacts.extrapolate_apgd(world_active, beta, adapter.projection_status, rigid_coordinates=True)

    _scatter_rigid_reactions(
        adapter,
        world_active,
        inverse_weight,
        (adapter.friction_reaction, adapter.contact_reaction, adapter.limit_reaction),
        adapter.projection_twist_delta,
    )
    if deformable_contacts is not None:
        deformable_contacts.scatter_apgd(
            world_active,
            inverse_weight,
            adapter.projection_twist_delta,
            rigid_coordinates=True,
            use_trial=False,
            projection_status=adapter.projection_status,
        )
    wp.launch(
        _reconstruct_twist,
        dim=projected_twist.shape[0],
        inputs=[body_world, world_active, adapter.projection_status, body_baseline],
        outputs=[adapter.projection_twist_delta, projected_twist],
        device=adapter.device,
    )
    if deformable_contacts is not None:
        deformable_contacts.reconstruct_apgd_particle_velocity(
            world_active,
            adapter.projection_status,
            particle_baseline,
            projected_velocity,
        )


def project_deformable_constraints_apgd(
    projection_iterations: int,
    contact_system,
    world_active: wp.array[wp.bool],
    particle_baseline: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
    theta: wp.array[wp.float32],
    beta: wp.array[wp.float32],
    restart_dot: wp.array[wp.float32],
) -> None:
    """Run matrix-free APGD for a pure deformable contact system."""
    status = contact_system.sequential_projection_status
    wp.copy(particle_baseline, projected_velocity)
    contact_system.begin_apgd_scatter()
    wp.launch(
        _initialize_world_state,
        dim=world_active.shape[0],
        inputs=[world_active, status],
        outputs=[theta, beta, restart_dot, status],
        device=projected_velocity.device,
    )
    contact_system.initialize_apgd(world_active, rigid_coordinates=False)
    for _iteration in range(projection_iterations):
        contact_system.scatter_apgd(
            world_active,
            None,
            None,
            rigid_coordinates=False,
            use_trial=True,
        )
        contact_system.reconstruct_apgd_particle_velocity(
            world_active,
            status,
            particle_baseline,
            projected_velocity,
        )
        contact_system.project_apgd(
            world_active,
            projected_velocity,
            None,
            restart_dot,
            status,
            rigid_coordinates=False,
        )
        wp.launch(
            _finalize_acceleration,
            dim=world_active.shape[0],
            inputs=[world_active, status],
            outputs=[restart_dot, theta, beta],
            device=projected_velocity.device,
        )
        contact_system.extrapolate_apgd(world_active, beta, status, rigid_coordinates=False)

    contact_system.scatter_apgd(
        world_active,
        None,
        None,
        rigid_coordinates=False,
        use_trial=False,
    )
    contact_system.reconstruct_apgd_particle_velocity(
        world_active,
        status,
        particle_baseline,
        projected_velocity,
    )
