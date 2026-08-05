# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Colored AVBD-style projection for LOX unilateral constraints."""

from __future__ import annotations

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact import _solve_contact_coulomb_newton_normal_last
from .deformable_contact import DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE, DEFORMABLE_CONTACT_STATUS_VALID
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    project_contact_coulomb_cone_orthogonal,
)

__all__ = [
    "prepare_constraints_avbd",
    "project_constraints_avbd",
    "project_deformable_constraints_avbd",
]

wp.set_module_options({"enable_backward": False})


@wp.func
def _is_finite_vec3(value: wp.vec3f) -> wp.bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _is_finite_vec6(value: vec6f) -> wp.bool:
    finite = wp.bool(True)
    for index in range(6):
        finite = finite and wp.isfinite(value[index])
    return finite


@wp.func
def _is_finite_mat33(value: wp.mat33f) -> wp.bool:
    finite = wp.bool(True)
    for row in range(3):
        for column in range(3):
            finite = finite and wp.isfinite(value[row, column])
    return finite


@wp.func
def _outer66(left: vec6f, right: vec6f) -> mat66f:
    value = mat66f(0.0)
    for row in range(6):
        for column in range(6):
            value[row, column] = left[row] * right[column]
    return value


@wp.func
def _contact_metric_hessian(jacobian: mat36f, inverse_delassus: wp.mat33f) -> mat66f:
    value = mat66f(0.0)
    for row in range(6):
        for column in range(6):
            entry = wp.float32(0.0)
            for axis_row in range(3):
                for axis_column in range(3):
                    entry += (
                        jacobian[axis_row, row]
                        * inverse_delassus[axis_row, axis_column]
                        * jacobian[axis_column, column]
                    )
            value[row, column] = entry
    return value


@wp.func
def _contact_row_sum_bound(value: wp.mat33f) -> wp.float32:
    bound = wp.float32(0.0)
    for row in range(3):
        row_sum = wp.float32(0.0)
        for column in range(3):
            row_sum += wp.abs(value[row, column])
        bound = wp.max(bound, row_sum)
    return bound


@wp.func
def _maximum_absolute_vec6(value: vec6f) -> wp.float32:
    result = wp.float32(0.0)
    for index in range(6):
        result = wp.max(result, wp.abs(value[index]))
    return result


@wp.func
def _maximum_absolute_diagonal(value: mat66f) -> wp.float32:
    result = wp.float32(0.0)
    for index in range(6):
        result = wp.max(result, wp.abs(value[index, index]))
    return result


@wp.kernel
def _reset_active_diagnostics(
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    world_stationarity: wp.array[wp.float32],
    world_update: wp.array[wp.float32],
):
    world = wp.tid()
    if world_active[world] and projection_status[world] == PROJECTION_STATUS_VALID:
        world_stationarity[world] = 0.0
        world_update[world] = 0.0


@wp.kernel
def _prepare_rigid_penalties(
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
    contact_delassus: wp.array[wp.mat33f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    friction_penalty: wp.array[wp.float32],
    contact_inverse_delassus: wp.array[wp.mat33f],
    limit_penalty: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        friction_penalty[constraint] = 0.0
        if friction_local[constraint] >= world_friction_count[world]:
            return
        value = wp.float32(0.0)
        first = friction_body_first[constraint]
        second = friction_body_second[constraint]
        if first >= 0:
            jacobian = friction_jacobian_first[constraint]
            value += wp.dot(jacobian, inverse_weight[first] @ jacobian)
        if second >= 0:
            jacobian = friction_jacobian_second[constraint]
            value += wp.dot(jacobian, inverse_weight[second] @ jacobian)
        if not wp.isfinite(value) or value <= 0.0:
            world_status[world] = PROJECTION_STATUS_INVALID
            return
        friction_penalty[constraint] = 1.0 / value
        return

    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        contact_inverse_delassus[constraint] = wp.mat33f(0.0)
        if contact_local[constraint] >= world_contact_count[world]:
            return
        if contact_body_first[constraint] < 0 and contact_body_second[constraint] < 0:
            return
        inverse_delassus = wp.inverse(contact_delassus[constraint])
        if not _is_finite_mat33(inverse_delassus):
            world_status[world] = PROJECTION_STATUS_INVALID
            return
        contact_inverse_delassus[constraint] = inverse_delassus
        return

    constraint -= contact_capacity
    world = limit_world[constraint]
    limit_penalty[constraint] = 0.0
    if limit_local[constraint] >= world_limit_count[world]:
        return
    value = wp.float32(0.0)
    first = limit_body_first[constraint]
    second = limit_body_second[constraint]
    if first >= 0:
        jacobian = limit_jacobian_first[constraint]
        value += wp.dot(jacobian, inverse_weight[first] @ jacobian)
    if second >= 0:
        jacobian = limit_jacobian_second[constraint]
        value += wp.dot(jacobian, inverse_weight[second] @ jacobian)
    if not wp.isfinite(value) or value <= 0.0:
        world_status[world] = PROJECTION_STATUS_INVALID
        return
    limit_penalty[constraint] = 1.0 / value


@wp.kernel
def _initialize_rigid_reactions(
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
    projection_status: wp.array[wp.int32],
    friction_reaction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if friction_local[constraint] < world_friction_count[world] and world_active[world]:
            friction_reaction[constraint] = wp.clamp(
                friction_reaction[constraint], -friction_bound[constraint], friction_bound[constraint]
            )
        return
    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if contact_local[constraint] < world_contact_count[world] and world_active[world]:
            value = wp.vec3f(0.0)
            if contact_body_first[constraint] >= 0 or contact_body_second[constraint] >= 0:
                value = project_contact_coulomb_cone_orthogonal(
                    contact_reaction[constraint], contact_friction[constraint]
                )
            if not _is_finite_vec3(value):
                projection_status[world] = PROJECTION_STATUS_INVALID
            else:
                contact_reaction[constraint] = value
        return
    constraint -= contact_capacity
    world = limit_world[constraint]
    if limit_local[constraint] < world_limit_count[world] and world_active[world]:
        limit_reaction[constraint] = wp.max(0.0, limit_reaction[constraint])


@wp.kernel
def _evaluate_rigid_contact_resolvent(
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
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    projected_twist: wp.array[vec6f],
    contact_reaction: wp.array[wp.vec3f],
    resolved_reaction: wp.array[wp.vec3f],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_local[contact] >= world_contact_count[world]
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        resolved_reaction[contact] = wp.vec3f(0.0)
        return
    velocity = contact_bias[contact]
    if first >= 0:
        velocity += contact_jacobian_first[contact] @ projected_twist[first]
    if second >= 0:
        velocity += contact_jacobian_second[contact] @ projected_twist[second]
    free_velocity = velocity - contact_delassus[contact] @ contact_reaction[contact]
    value = _solve_contact_coulomb_newton_normal_last(
        contact_delassus[contact],
        free_velocity,
        contact_friction[contact],
    )
    if not _is_finite_vec3(value):
        projection_status[world] = PROJECTION_STATUS_INVALID
        return
    resolved_reaction[contact] = value


@wp.kernel
def _accumulate_rigid_constraints(
    friction_capacity: int,
    contact_capacity: int,
    current_color: int,
    body_color: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_bound: wp.array[wp.float32],
    friction_penalty: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_inverse_delassus: wp.array[wp.mat33f],
    contact_augmented_reaction: wp.array[wp.vec3f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_penalty: wp.array[wp.float32],
    world_active: wp.array[wp.bool],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    limit_reaction: wp.array[wp.float32],
    body_force: wp.array[vec6f],
    body_hessian: wp.array[mat66f],
    projection_status: wp.array[wp.int32],
):
    constraint = wp.tid()
    first = int(-1)
    second = int(-1)
    force_first = vec6f(0.0)
    force_second = vec6f(0.0)
    hessian_first = mat66f(0.0)
    hessian_second = mat66f(0.0)
    world = int(-1)

    if constraint < friction_capacity:
        world = friction_world[constraint]
        if friction_local[constraint] >= world_friction_count[world]:
            return
        first = friction_body_first[constraint]
        second = friction_body_second[constraint]
        friction_velocity = wp.float32(0.0)
        if first >= 0:
            friction_velocity += wp.dot(friction_jacobian_first[constraint], projected_twist[first])
        if second >= 0:
            friction_velocity += wp.dot(friction_jacobian_second[constraint], projected_twist[second])
        penalty = friction_penalty[constraint]
        friction_projected = wp.clamp(
            friction_reaction[constraint] - penalty * friction_velocity,
            -friction_bound[constraint],
            friction_bound[constraint],
        )
        if first >= 0:
            friction_jacobian = friction_jacobian_first[constraint]
            force_first = friction_projected * friction_jacobian
            hessian_first = penalty * _outer66(friction_jacobian, friction_jacobian)
        if second >= 0:
            friction_jacobian = friction_jacobian_second[constraint]
            force_second = friction_projected * friction_jacobian
            hessian_second = penalty * _outer66(friction_jacobian, friction_jacobian)
    elif constraint < friction_capacity + contact_capacity:
        contact = constraint - friction_capacity
        world = contact_world[contact]
        if contact_local[contact] >= world_contact_count[world]:
            return
        first = contact_body_first[contact]
        second = contact_body_second[contact]
        inverse_delassus = contact_inverse_delassus[contact]
        contact_projected = contact_augmented_reaction[contact]
        if first >= 0:
            contact_jacobian = contact_jacobian_first[contact]
            force_first = wp.transpose(contact_jacobian) @ contact_projected
            hessian_first = _contact_metric_hessian(contact_jacobian, inverse_delassus)
        if second >= 0:
            contact_jacobian = contact_jacobian_second[contact]
            force_second = wp.transpose(contact_jacobian) @ contact_projected
            hessian_second = _contact_metric_hessian(contact_jacobian, inverse_delassus)
    else:
        limit = constraint - friction_capacity - contact_capacity
        world = limit_world[limit]
        if limit_local[limit] >= world_limit_count[world]:
            return
        first = limit_body_first[limit]
        second = limit_body_second[limit]
        limit_velocity = limit_bias[limit]
        if first >= 0:
            limit_velocity += wp.dot(limit_jacobian_first[limit], projected_twist[first])
        if second >= 0:
            limit_velocity += wp.dot(limit_jacobian_second[limit], projected_twist[second])
        penalty = limit_penalty[limit]
        limit_projected = wp.max(0.0, limit_reaction[limit] - penalty * limit_velocity)
        if first >= 0:
            limit_jacobian = limit_jacobian_first[limit]
            force_first = limit_projected * limit_jacobian
            hessian_first = penalty * _outer66(limit_jacobian, limit_jacobian)
        if second >= 0:
            limit_jacobian = limit_jacobian_second[limit]
            force_second = limit_projected * limit_jacobian
            hessian_second = penalty * _outer66(limit_jacobian, limit_jacobian)

    if world < 0 or not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return
    if first >= 0 and body_color[first] == current_color:
        wp.atomic_add(body_force, first, force_first)
        wp.atomic_add(body_hessian, first, hessian_first)
    if second >= 0 and body_color[second] == current_color:
        wp.atomic_add(body_force, second, force_second)
        wp.atomic_add(body_hessian, second, hessian_second)


@wp.func
def _solve_spatial_spd(hessian: mat66f, rhs: vec6f) -> vec6f:
    h_ll = wp.mat33f(0.0)
    h_al = wp.mat33f(0.0)
    h_aa = wp.mat33f(0.0)
    rhs_lin = wp.vec3f(rhs[0], rhs[1], rhs[2])
    rhs_ang = wp.vec3f(rhs[3], rhs[4], rhs[5])
    for row in range(3):
        for column in range(3):
            h_ll[row, column] = hessian[row, column]
            h_al[row, column] = hessian[row + 3, column]
            h_aa[row, column] = hessian[row + 3, column + 3]
    inverse_ll = wp.inverse(h_ll)
    schur = h_aa - h_al @ inverse_ll @ wp.transpose(h_al)
    angular = wp.inverse(schur) @ (rhs_ang - h_al @ (inverse_ll @ rhs_lin))
    linear = inverse_ll @ (rhs_lin - wp.transpose(h_al) @ angular)
    return vec6f(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])


@wp.kernel
def _solve_body_color(
    body_group: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    weight: wp.array[mat66f],
    baseline: wp.array[vec6f],
    body_force: wp.array[vec6f],
    body_hessian: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
):
    local = wp.tid()
    body = body_group[local]
    world = body_world[body]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return
    current = projected_twist[body]
    metric = weight[body]
    if _maximum_absolute_diagonal(metric) == 0.0:
        # A zero LOX metric has no primal coordinate to minimize. Its contact
        # endpoint is prescribed, just as in the inverse-weight projections.
        projected_twist[body] = baseline[body]
        return
    hessian = metric + body_hessian[body]
    rhs = metric @ (baseline[body] - current) + body_force[body]
    delta = _solve_spatial_spd(hessian, rhs)
    if not _is_finite_vec6(delta):
        projection_status[world] = PROJECTION_STATUS_INVALID
        return
    projected_twist[body] = current + delta


@wp.kernel
def _update_rigid_duals(
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
    friction_penalty: wp.array[wp.float32],
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
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_penalty: wp.array[wp.float32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
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
        friction_velocity = wp.float32(0.0)
        first = friction_body_first[constraint]
        second = friction_body_second[constraint]
        if first >= 0:
            friction_velocity += wp.dot(friction_jacobian_first[constraint], projected_twist[first])
        if second >= 0:
            friction_velocity += wp.dot(friction_jacobian_second[constraint], projected_twist[second])
        friction_reaction[constraint] = wp.clamp(
            friction_reaction[constraint] - friction_penalty[constraint] * friction_velocity,
            -friction_bound[constraint],
            friction_bound[constraint],
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
        contact_velocity = contact_bias[constraint]
        first = contact_body_first[constraint]
        second = contact_body_second[constraint]
        if first >= 0:
            contact_velocity += contact_jacobian_first[constraint] @ projected_twist[first]
        if second >= 0:
            contact_velocity += contact_jacobian_second[constraint] @ projected_twist[second]
        free_velocity = contact_velocity - contact_delassus[constraint] @ contact_reaction[constraint]
        value = _solve_contact_coulomb_newton_normal_last(
            contact_delassus[constraint],
            free_velocity,
            contact_friction[constraint],
        )
        if not _is_finite_vec3(value):
            projection_status[world] = PROJECTION_STATUS_INVALID
            return
        contact_reaction[constraint] = value
        return
    constraint -= contact_capacity
    world = limit_world[constraint]
    if (
        limit_local[constraint] >= world_limit_count[world]
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    limit_velocity = limit_bias[constraint]
    first = limit_body_first[constraint]
    second = limit_body_second[constraint]
    if first >= 0:
        limit_velocity += wp.dot(limit_jacobian_first[constraint], projected_twist[first])
    if second >= 0:
        limit_velocity += wp.dot(limit_jacobian_second[constraint], projected_twist[second])
    limit_reaction[constraint] = wp.max(0.0, limit_reaction[constraint] - limit_penalty[constraint] * limit_velocity)


@wp.kernel
def _prepare_deformable_penalties(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    body_jacobian: wp.array[mat36f],
    contact_status: wp.array[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    body_inverse_weight: wp.array[mat66f],
    include_rigid: bool,
    delassus: wp.array[wp.mat33f],
    inverse_delassus: wp.array[wp.mat33f],
    contact_world_status: wp.array[wp.int32],
    projection_status: wp.array[wp.int32],
):
    contact = wp.tid()
    delassus[contact] = wp.mat33f(0.0)
    inverse_delassus[contact] = wp.mat33f(0.0)
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return
    particle_value = wp.float32(0.0)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            coefficient = coefficients[contact, slot]
            particle_value += coefficient * coefficient * particle_inverse_weight[particle]
    block = particle_value * wp.identity(3, dtype=wp.float32)
    body = contact_body[contact]
    if include_rigid and body >= 0:
        jacobian = body_jacobian[contact]
        block += jacobian @ body_inverse_weight[body] @ wp.transpose(jacobian)
    block_bound = _contact_row_sum_bound(block)
    if not wp.isfinite(block_bound) or block_bound <= 0.0:
        world = contact_world[contact]
        contact_world_status[world] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
        projection_status[world] = PROJECTION_STATUS_INVALID
        return
    block_inverse = wp.inverse(block)
    if not _is_finite_mat33(block_inverse):
        world = contact_world[contact]
        contact_world_status[world] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
        projection_status[world] = PROJECTION_STATUS_INVALID
        return
    delassus[contact] = block
    inverse_delassus[contact] = block_inverse


@wp.kernel
def _initialize_deformable_reactions(
    contact_world: wp.array[wp.int32],
    normal: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    rigid_coordinates: bool,
    particle_reaction: wp.array[wp.vec3f],
    rigid_reaction: wp.array[wp.vec3f],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or world < 0
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    value = particle_reaction[contact]
    if rigid_coordinates:
        value = project_contact_coulomb_cone_orthogonal(rigid_reaction[contact], friction[contact])
    else:
        contact_normal = normal[contact]
        normal_value = wp.dot(contact_normal, value)
        tangent = value - normal_value * contact_normal
        tangent_length = wp.length(tangent)
        if friction[contact] * tangent_length <= -normal_value:
            value = wp.vec3f(0.0)
        elif tangent_length > friction[contact] * normal_value:
            projected_normal = (friction[contact] * tangent_length + normal_value) / (
                friction[contact] * friction[contact] + 1.0
            )
            value = projected_normal * contact_normal
            if tangent_length > 0.0:
                value += friction[contact] * projected_normal * tangent / tangent_length
    if not _is_finite_vec3(value):
        projection_status[world] = PROJECTION_STATUS_INVALID
        return
    if rigid_coordinates:
        rigid_reaction[contact] = value
    else:
        particle_reaction[contact] = value


@wp.kernel
def _evaluate_deformable_contact_resolvent(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    rigid_bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    delassus: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    rigid_coordinates: bool,
    projected_velocity: wp.array[wp.vec3f],
    projected_twist: wp.array[vec6f],
    particle_reaction: wp.array[wp.vec3f],
    rigid_reaction: wp.array[wp.vec3f],
    augmented_reaction: wp.array[wp.vec3f],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or world < 0
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    contact_frame = frame[contact]
    velocity = bias[contact]
    if rigid_coordinates:
        velocity = rigid_bias[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            value = projected_velocity[particle]
            if rigid_coordinates:
                value = wp.transpose(contact_frame) @ value
            velocity += coefficients[contact, slot] * value
    body = contact_body[contact]
    if rigid_coordinates and body >= 0:
        velocity += body_jacobian[contact] @ projected_twist[body]
    reaction = rigid_reaction[contact]
    if not rigid_coordinates:
        velocity = wp.transpose(contact_frame) @ velocity
        reaction = wp.transpose(contact_frame) @ particle_reaction[contact]
    free_velocity = velocity - delassus[contact] @ reaction
    resolved = _solve_contact_coulomb_newton_normal_last(
        delassus[contact],
        free_velocity,
        friction[contact],
    )
    if not _is_finite_vec3(resolved):
        projection_status[world] = PROJECTION_STATUS_INVALID
        return
    augmented_reaction[contact] = resolved


@wp.kernel
def _accumulate_deformable_particle_color(
    current_color: int,
    particle_color: wp.array[wp.int32],
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
    contact_world: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    inverse_delassus: wp.array[wp.mat33f],
    augmented_reaction: wp.array[wp.vec3f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    particle_force: wp.array[wp.vec3f],
    particle_hessian: wp.array[wp.mat33f],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or world < 0
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    contact_frame = frame[contact]
    reaction = augmented_reaction[contact]
    world_force = contact_frame @ reaction
    world_metric = contact_frame @ inverse_delassus[contact] @ wp.transpose(contact_frame)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        coefficient = coefficients[contact, slot]
        if particle >= 0 and particle_color[particle] == current_color:
            wp.atomic_add(particle_force, particle, coefficient * world_force)
            wp.atomic_add(particle_hessian, particle, coefficient * coefficient * world_metric)


@wp.kernel
def _accumulate_deformable_body_color(
    current_color: int,
    body_color: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    body_jacobian: wp.array[mat36f],
    inverse_delassus: wp.array[wp.mat33f],
    augmented_reaction: wp.array[wp.vec3f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    body_force: wp.array[vec6f],
    body_hessian: wp.array[mat66f],
):
    contact = wp.tid()
    body = contact_body[contact]
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or body < 0
        or body_color[body] != current_color
        or world < 0
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    jacobian = body_jacobian[contact]
    projected_reaction = augmented_reaction[contact]
    wp.atomic_add(body_force, body, wp.transpose(jacobian) @ projected_reaction)
    wp.atomic_add(body_hessian, body, _contact_metric_hessian(jacobian, inverse_delassus[contact]))


@wp.kernel
def _solve_particle_color(
    particle_group: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    inverse_weight: wp.array[wp.float32],
    baseline: wp.array[wp.vec3f],
    particle_force: wp.array[wp.vec3f],
    particle_hessian: wp.array[wp.mat33f],
    projected_velocity: wp.array[wp.vec3f],
):
    local = wp.tid()
    particle = particle_group[local]
    world = packed_world[particle]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return
    inverse_metric = inverse_weight[particle]
    if inverse_metric <= 0.0:
        return
    metric = 1.0 / inverse_metric
    current = projected_velocity[particle]
    hessian = metric * wp.identity(3, dtype=wp.float32) + particle_hessian[particle]
    delta = wp.inverse(hessian) @ (metric * (baseline[particle] - current) + particle_force[particle])
    if not _is_finite_vec3(delta) or not _is_finite_mat33(hessian):
        projection_status[world] = PROJECTION_STATUS_INVALID
        return
    projected_velocity[particle] = current + delta


@wp.kernel
def _update_deformable_duals(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    rigid_bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    delassus: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    rigid_coordinates: bool,
    projected_velocity: wp.array[wp.vec3f],
    projected_twist: wp.array[vec6f],
    particle_reaction: wp.array[wp.vec3f],
    rigid_reaction: wp.array[wp.vec3f],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or world < 0
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    contact_frame = frame[contact]
    velocity = bias[contact]
    if rigid_coordinates:
        velocity = rigid_bias[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            value = projected_velocity[particle]
            if rigid_coordinates:
                value = wp.transpose(contact_frame) @ value
            velocity += coefficients[contact, slot] * value
    body = contact_body[contact]
    if rigid_coordinates and body >= 0:
        velocity += body_jacobian[contact] @ projected_twist[body]
    reaction = rigid_reaction[contact]
    if not rigid_coordinates:
        velocity = wp.transpose(contact_frame) @ velocity
        reaction = wp.transpose(contact_frame) @ particle_reaction[contact]
    free_velocity = velocity - delassus[contact] @ reaction
    resolved = _solve_contact_coulomb_newton_normal_last(
        delassus[contact],
        free_velocity,
        friction[contact],
    )
    if not _is_finite_vec3(resolved):
        projection_status[world] = PROJECTION_STATUS_INVALID
        return
    if rigid_coordinates:
        rigid_reaction[contact] = resolved
    else:
        particle_reaction[contact] = contact_frame @ resolved


@wp.kernel
def _initialize_particle_stationarity(
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    inverse_weight: wp.array[wp.float32],
    baseline: wp.array[wp.vec3f],
    projected_velocity: wp.array[wp.vec3f],
    stationarity: wp.array[wp.vec3f],
    world_max_update: wp.array[wp.float32],
):
    particle = wp.tid()
    world = packed_world[particle]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        stationarity[particle] = wp.vec3f(0.0)
        return
    inverse_metric = inverse_weight[particle]
    value = wp.vec3f(0.0)
    if inverse_metric > 0.0:
        value = (projected_velocity[particle] - baseline[particle]) / inverse_metric
    stationarity[particle] = value
    update = projected_velocity[particle] - baseline[particle]
    maximum = wp.max(wp.abs(update[0]), wp.max(wp.abs(update[1]), wp.abs(update[2])))
    wp.atomic_max(world_max_update, world, maximum)


@wp.kernel
def _scatter_deformable_stationarity(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
    contact_world: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    rigid_coordinates: bool,
    particle_reaction: wp.array[wp.vec3f],
    rigid_reaction: wp.array[wp.vec3f],
    stationarity: wp.array[wp.vec3f],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or world < 0
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    reaction = particle_reaction[contact]
    if rigid_coordinates:
        reaction = frame[contact] @ rigid_reaction[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            wp.atomic_sub(stationarity, particle, coefficients[contact, slot] * reaction)


@wp.kernel
def _reduce_particle_stationarity(
    packed_world: wp.array[wp.int32],
    stationarity: wp.array[wp.vec3f],
    world_stationarity: wp.array[wp.float32],
):
    particle = wp.tid()
    value = stationarity[particle]
    maximum = wp.max(wp.abs(value[0]), wp.max(wp.abs(value[1]), wp.abs(value[2])))
    wp.atomic_max(world_stationarity, packed_world[particle], maximum)


@wp.kernel
def _initialize_body_stationarity(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    weight: wp.array[mat66f],
    baseline: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    stationarity: wp.array[vec6f],
    world_max_update: wp.array[wp.float32],
):
    body = wp.tid()
    world = body_world[body]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        stationarity[body] = vec6f(0.0)
        return
    delta = projected_twist[body] - baseline[body]
    stationarity[body] = weight[body] @ delta
    wp.atomic_max(world_max_update, world, _maximum_absolute_vec6(delta))


@wp.kernel
def _scatter_rigid_stationarity(
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
    friction_reaction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    stationarity: wp.array[vec6f],
):
    constraint = wp.tid()
    first = int(-1)
    second = int(-1)
    force_first = vec6f(0.0)
    force_second = vec6f(0.0)
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if friction_local[constraint] >= world_friction_count[world]:
            return
        first = friction_body_first[constraint]
        second = friction_body_second[constraint]
        friction_value = friction_reaction[constraint]
        if first >= 0:
            force_first = friction_value * friction_jacobian_first[constraint]
        if second >= 0:
            force_second = friction_value * friction_jacobian_second[constraint]
    elif constraint < friction_capacity + contact_capacity:
        contact = constraint - friction_capacity
        world = contact_world[contact]
        if contact_local[contact] >= world_contact_count[world]:
            return
        first = contact_body_first[contact]
        second = contact_body_second[contact]
        contact_value = contact_reaction[contact]
        if first >= 0:
            force_first = wp.transpose(contact_jacobian_first[contact]) @ contact_value
        if second >= 0:
            force_second = wp.transpose(contact_jacobian_second[contact]) @ contact_value
    else:
        limit = constraint - friction_capacity - contact_capacity
        world = limit_world[limit]
        if limit_local[limit] >= world_limit_count[world]:
            return
        first = limit_body_first[limit]
        second = limit_body_second[limit]
        limit_value = limit_reaction[limit]
        if first >= 0:
            force_first = limit_value * limit_jacobian_first[limit]
        if second >= 0:
            force_second = limit_value * limit_jacobian_second[limit]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return
    if first >= 0:
        wp.atomic_sub(stationarity, first, force_first)
    if second >= 0:
        wp.atomic_sub(stationarity, second, force_second)


@wp.kernel
def _scatter_deformable_body_stationarity(
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    body_jacobian: wp.array[mat36f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    reaction: wp.array[wp.vec3f],
    stationarity: wp.array[vec6f],
):
    contact = wp.tid()
    body = contact_body[contact]
    world = contact_world[contact]
    if (
        contact_status[contact] == DEFORMABLE_CONTACT_STATUS_VALID
        and body >= 0
        and world >= 0
        and world_active[world]
        and projection_status[world] == PROJECTION_STATUS_VALID
    ):
        wp.atomic_sub(stationarity, body, wp.transpose(body_jacobian[contact]) @ reaction[contact])


@wp.kernel
def _reduce_body_stationarity(
    body_world: wp.array[wp.int32],
    weight: wp.array[mat66f],
    stationarity: wp.array[vec6f],
    world_stationarity: wp.array[wp.float32],
):
    body = wp.tid()
    if _maximum_absolute_diagonal(weight[body]) > 0.0:
        wp.atomic_max(world_stationarity, body_world[body], _maximum_absolute_vec6(stationarity[body]))


def prepare_constraints_avbd(adapter, inverse_weight: wp.array[mat66f]) -> None:
    """Prepare automatic true-diagonal penalties for rigid AVBD constraints."""
    adapter.world_avbd_projection_status.fill_(PROJECTION_STATUS_VALID)
    capacity = adapter.friction_capacity + adapter.contact_capacity + adapter.limit_capacity
    if capacity == 0:
        return
    wp.launch(
        _prepare_rigid_penalties,
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
            adapter.contact_projection_delassus,
            adapter.limit_world,
            adapter.limit_local,
            adapter.world_limit_count,
            adapter.limit_body_first,
            adapter.limit_body_second,
            adapter.limit_jacobian_first,
            adapter.limit_jacobian_second,
            inverse_weight,
        ],
        outputs=[
            adapter.friction_avbd_penalty,
            adapter.contact_avbd_inverse_delassus,
            adapter.limit_avbd_penalty,
            adapter.world_avbd_projection_status,
        ],
        device=adapter.device,
    )


def _prepare_deformable(contact_system, world_active, projection_status, body_inverse_weight, include_rigid) -> None:
    inverse_weight = body_inverse_weight if include_rigid else contact_system._empty_body_inverse_weight
    wp.launch(
        _prepare_deformable_penalties,
        dim=contact_system.contact_capacity,
        inputs=[
            contact_system.particle_indices,
            contact_system.coefficients,
            contact_system.contact_world,
            contact_system.body,
            contact_system.body_jacobian,
            contact_system.status,
            contact_system.cloth_system.inverse_weight,
            inverse_weight,
            include_rigid,
        ],
        outputs=[
            contact_system.avbd_delassus,
            contact_system.avbd_inverse_delassus,
            contact_system.world_status,
            projection_status,
        ],
        device=contact_system.device,
    )
    wp.launch(
        _initialize_deformable_reactions,
        dim=contact_system.contact_capacity,
        inputs=[
            contact_system.contact_world,
            contact_system.normal,
            contact_system.friction,
            contact_system.status,
            world_active,
            projection_status,
            include_rigid,
        ],
        outputs=[contact_system.reaction, contact_system.rigid_reaction],
        device=contact_system.device,
    )


def _accumulate_particle_colors(
    contact_system,
    world_active,
    projection_status,
    baseline,
    projected_velocity,
) -> None:
    topology = contact_system.cloth_system.topology
    for color, group in enumerate(contact_system.cloth_system.particle_color_groups):
        contact_system.avbd_particle_force.zero_()
        contact_system.avbd_particle_hessian.zero_()
        wp.launch(
            _accumulate_deformable_particle_color,
            dim=contact_system.contact_capacity,
            inputs=[
                color,
                topology.packed_color,
                contact_system.particle_indices,
                contact_system.coefficients,
                contact_system.contact_world,
                contact_system.frame,
                contact_system.avbd_inverse_delassus,
                contact_system.avbd_augmented_reaction,
                contact_system.status,
                world_active,
                projection_status,
            ],
            outputs=[contact_system.avbd_particle_force, contact_system.avbd_particle_hessian],
            device=contact_system.device,
        )
        wp.launch(
            _solve_particle_color,
            dim=group.shape[0],
            inputs=[
                group,
                topology.packed_world,
                world_active,
                projection_status,
                contact_system.cloth_system.inverse_weight,
                baseline,
                contact_system.avbd_particle_force,
                contact_system.avbd_particle_hessian,
            ],
            outputs=[projected_velocity],
            device=contact_system.device,
        )


def _accumulate_body_colors(
    adapter,
    world_active,
    weight,
    baseline,
    projected_twist,
    deformable_contacts,
) -> None:
    rigid_capacity = adapter.friction_capacity + adapter.contact_capacity + adapter.limit_capacity
    for color, group in enumerate(adapter.avbd_body_color_groups):
        adapter.avbd_body_force.zero_()
        adapter.avbd_body_hessian.zero_()
        if rigid_capacity > 0:
            wp.launch(
                _accumulate_rigid_constraints,
                dim=rigid_capacity,
                inputs=[
                    adapter.friction_capacity,
                    adapter.contact_capacity,
                    color,
                    adapter.avbd_body_color,
                    adapter.friction_world,
                    adapter.friction_local,
                    adapter.world_friction_count,
                    adapter.friction_body_first,
                    adapter.friction_body_second,
                    adapter.friction_jacobian_first,
                    adapter.friction_jacobian_second,
                    adapter.friction_impulse_bound,
                    adapter.friction_avbd_penalty,
                    adapter.contact_world,
                    adapter.contact_local,
                    adapter.world_contact_count,
                    adapter.contact_body_first,
                    adapter.contact_body_second,
                    adapter.contact_jacobian_first,
                    adapter.contact_jacobian_second,
                    adapter.contact_avbd_inverse_delassus,
                    adapter.contact_avbd_augmented_reaction,
                    adapter.limit_world,
                    adapter.limit_local,
                    adapter.world_limit_count,
                    adapter.limit_body_first,
                    adapter.limit_body_second,
                    adapter.limit_jacobian_first,
                    adapter.limit_jacobian_second,
                    adapter.limit_bias,
                    adapter.limit_avbd_penalty,
                    world_active,
                    projected_twist,
                    adapter.friction_reaction,
                    adapter.limit_reaction,
                ],
                outputs=[adapter.avbd_body_force, adapter.avbd_body_hessian, adapter.projection_status],
                device=adapter.device,
            )
        if deformable_contacts is not None:
            wp.launch(
                _accumulate_deformable_body_color,
                dim=deformable_contacts.contact_capacity,
                inputs=[
                    color,
                    adapter.avbd_body_color,
                    deformable_contacts.contact_world,
                    deformable_contacts.body,
                    deformable_contacts.body_jacobian,
                    deformable_contacts.avbd_inverse_delassus,
                    deformable_contacts.avbd_augmented_reaction,
                    deformable_contacts.status,
                    world_active,
                    adapter.projection_status,
                ],
                outputs=[adapter.avbd_body_force, adapter.avbd_body_hessian],
                device=adapter.device,
            )
        wp.launch(
            _solve_body_color,
            dim=group.shape[0],
            inputs=[
                group,
                adapter.system.body_world,
                world_active,
                adapter.projection_status,
                weight,
                baseline,
                adapter.avbd_body_force,
                adapter.avbd_body_hessian,
            ],
            outputs=[projected_twist],
            device=adapter.device,
        )


def _update_rigid(adapter, world_active, projected_twist) -> None:
    capacity = adapter.friction_capacity + adapter.contact_capacity + adapter.limit_capacity
    if capacity == 0:
        return
    wp.launch(
        _update_rigid_duals,
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
            adapter.friction_impulse_bound,
            adapter.friction_avbd_penalty,
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
            adapter.limit_world,
            adapter.limit_local,
            adapter.world_limit_count,
            adapter.limit_body_first,
            adapter.limit_body_second,
            adapter.limit_jacobian_first,
            adapter.limit_jacobian_second,
            adapter.limit_bias,
            adapter.limit_avbd_penalty,
            world_active,
            adapter.projection_status,
            projected_twist,
        ],
        outputs=[adapter.friction_reaction, adapter.contact_reaction, adapter.limit_reaction],
        device=adapter.device,
    )


def _evaluate_rigid_contacts(adapter, world_active, projected_twist) -> None:
    if adapter.contact_capacity == 0:
        return
    wp.launch(
        _evaluate_rigid_contact_resolvent,
        dim=adapter.contact_capacity,
        inputs=[
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
            world_active,
            adapter.projection_status,
            projected_twist,
            adapter.contact_reaction,
        ],
        outputs=[adapter.contact_avbd_augmented_reaction],
        device=adapter.device,
    )


def _update_deformable(
    contact_system, world_active, projection_status, projected_velocity, projected_twist, rigid
) -> None:
    wp.launch(
        _update_deformable_duals,
        dim=contact_system.contact_capacity,
        inputs=[
            contact_system.particle_indices,
            contact_system.coefficients,
            contact_system.contact_world,
            contact_system.body,
            contact_system.frame,
            contact_system.body_jacobian,
            contact_system.bias,
            contact_system.rigid_bias,
            contact_system.friction,
            contact_system.avbd_delassus,
            contact_system.status,
            world_active,
            projection_status,
            rigid,
            projected_velocity,
            projected_twist,
        ],
        outputs=[contact_system.reaction, contact_system.rigid_reaction],
        device=contact_system.device,
    )


def _evaluate_deformable_contacts(
    contact_system, world_active, projection_status, projected_velocity, projected_twist, rigid
) -> None:
    wp.launch(
        _evaluate_deformable_contact_resolvent,
        dim=contact_system.contact_capacity,
        inputs=[
            contact_system.particle_indices,
            contact_system.coefficients,
            contact_system.contact_world,
            contact_system.body,
            contact_system.frame,
            contact_system.body_jacobian,
            contact_system.bias,
            contact_system.rigid_bias,
            contact_system.friction,
            contact_system.avbd_delassus,
            contact_system.status,
            world_active,
            projection_status,
            rigid,
            projected_velocity,
            projected_twist,
            contact_system.reaction,
            contact_system.rigid_reaction,
        ],
        outputs=[contact_system.avbd_augmented_reaction],
        device=contact_system.device,
    )


def _compute_particle_diagnostics(
    contact_system, world_active, projection_status, baseline, projected_velocity, rigid_coordinates
) -> None:
    wp.launch(
        _reset_active_diagnostics,
        dim=world_active.shape[0],
        inputs=[world_active, projection_status],
        outputs=[contact_system.world_avbd_stationarity_max, contact_system.world_avbd_update_max],
        device=contact_system.device,
    )
    wp.launch(
        _initialize_particle_stationarity,
        dim=contact_system.cloth_system.particle_count,
        inputs=[
            contact_system.cloth_system.topology.packed_world,
            world_active,
            projection_status,
            contact_system.cloth_system.inverse_weight,
            baseline,
            projected_velocity,
        ],
        outputs=[contact_system.avbd_particle_stationarity, contact_system.world_avbd_update_max],
        device=contact_system.device,
    )
    wp.launch(
        _scatter_deformable_stationarity,
        dim=contact_system.contact_capacity,
        inputs=[
            contact_system.particle_indices,
            contact_system.coefficients,
            contact_system.contact_world,
            contact_system.frame,
            contact_system.status,
            world_active,
            projection_status,
            rigid_coordinates,
            contact_system.reaction,
            contact_system.rigid_reaction,
        ],
        outputs=[contact_system.avbd_particle_stationarity],
        device=contact_system.device,
    )
    wp.launch(
        _reduce_particle_stationarity,
        dim=contact_system.cloth_system.particle_count,
        inputs=[contact_system.cloth_system.topology.packed_world, contact_system.avbd_particle_stationarity],
        outputs=[contact_system.world_avbd_stationarity_max],
        device=contact_system.device,
    )


def _compute_body_diagnostics(adapter, world_active, weight, baseline, projected_twist, deformable_contacts) -> None:
    wp.launch(
        _reset_active_diagnostics,
        dim=world_active.shape[0],
        inputs=[world_active, adapter.projection_status],
        outputs=[adapter.world_avbd_stationarity_max, adapter.world_avbd_update_max],
        device=adapter.device,
    )
    wp.launch(
        _initialize_body_stationarity,
        dim=projected_twist.shape[0],
        inputs=[
            adapter.system.body_world,
            world_active,
            adapter.projection_status,
            weight,
            baseline,
            projected_twist,
        ],
        outputs=[adapter.avbd_body_stationarity, adapter.world_avbd_update_max],
        device=adapter.device,
    )
    capacity = adapter.friction_capacity + adapter.contact_capacity + adapter.limit_capacity
    if capacity > 0:
        wp.launch(
            _scatter_rigid_stationarity,
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
                adapter.friction_reaction,
                adapter.contact_reaction,
                adapter.limit_reaction,
                world_active,
                adapter.projection_status,
            ],
            outputs=[adapter.avbd_body_stationarity],
            device=adapter.device,
        )
    if deformable_contacts is not None:
        wp.launch(
            _scatter_deformable_body_stationarity,
            dim=deformable_contacts.contact_capacity,
            inputs=[
                deformable_contacts.contact_world,
                deformable_contacts.body,
                deformable_contacts.body_jacobian,
                deformable_contacts.status,
                world_active,
                adapter.projection_status,
                deformable_contacts.rigid_reaction,
            ],
            outputs=[adapter.avbd_body_stationarity],
            device=adapter.device,
        )
    wp.launch(
        _reduce_body_stationarity,
        dim=projected_twist.shape[0],
        inputs=[adapter.system.body_world, weight, adapter.avbd_body_stationarity],
        outputs=[adapter.world_avbd_stationarity_max],
        device=adapter.device,
    )


def project_constraints_avbd(
    projection_iterations: int,
    adapter,
    world_active: wp.array[wp.bool],
    weight: wp.array[mat66f],
    inverse_weight: wp.array[mat66f],
    body_baseline: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    deformable_contacts=None,
    particle_baseline: wp.array[wp.vec3f] | None = None,
    projected_velocity: wp.array[wp.vec3f] | None = None,
) -> None:
    """Run fixed-count colored AVBD sweeps for a rigid or mixed system."""
    if (deformable_contacts is None) != (projected_velocity is None):
        raise ValueError("AVBD deformable contacts and projected velocity must be supplied together.")
    if deformable_contacts is not None and particle_baseline is None:
        raise ValueError("AVBD deformable projection requires a particle baseline.")
    wp.copy(body_baseline, projected_twist)
    if deformable_contacts is not None:
        wp.copy(particle_baseline, projected_velocity)
    wp.copy(adapter.projection_status, adapter.world_avbd_projection_status)
    capacity = adapter.friction_capacity + adapter.contact_capacity + adapter.limit_capacity
    if capacity > 0:
        wp.launch(
            _initialize_rigid_reactions,
            dim=capacity,
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
                adapter.projection_status,
                adapter.friction_reaction,
                adapter.contact_reaction,
                adapter.limit_reaction,
            ],
            device=adapter.device,
        )
    if deformable_contacts is not None:
        _prepare_deformable(deformable_contacts, world_active, adapter.projection_status, inverse_weight, True)
    # Reactions remain warm-started, but a truncated local solve must begin at
    # its current proximal center. Reconstructing a primal guess from stale
    # reactions can make a fixed-count AVBD projection expansive.
    for _iteration in range(projection_iterations):
        _evaluate_rigid_contacts(adapter, world_active, projected_twist)
        if deformable_contacts is not None:
            _evaluate_deformable_contacts(
                deformable_contacts,
                world_active,
                adapter.projection_status,
                projected_velocity,
                projected_twist,
                True,
            )
        _accumulate_body_colors(
            adapter,
            world_active,
            weight,
            body_baseline,
            projected_twist,
            deformable_contacts,
        )
        if deformable_contacts is not None:
            _accumulate_particle_colors(
                deformable_contacts,
                world_active,
                adapter.projection_status,
                particle_baseline,
                projected_velocity,
            )
        _update_rigid(adapter, world_active, projected_twist)
        if deformable_contacts is not None:
            _update_deformable(
                deformable_contacts,
                world_active,
                adapter.projection_status,
                projected_velocity,
                projected_twist,
                True,
            )
    if deformable_contacts is not None:
        _compute_particle_diagnostics(
            deformable_contacts,
            world_active,
            adapter.projection_status,
            particle_baseline,
            projected_velocity,
            True,
        )
    _compute_body_diagnostics(adapter, world_active, weight, body_baseline, projected_twist, deformable_contacts)


def project_deformable_constraints_avbd(
    projection_iterations: int,
    contact_system,
    world_active: wp.array[wp.bool],
    particle_baseline: wp.array[wp.vec3f],
    projected_velocity: wp.array[wp.vec3f],
) -> None:
    """Run fixed-count colored AVBD sweeps for pure deformable contacts."""
    status = contact_system.sequential_projection_status
    status.fill_(PROJECTION_STATUS_VALID)
    wp.copy(particle_baseline, projected_velocity)
    _prepare_deformable(contact_system, world_active, status, None, False)
    empty_twist = contact_system._empty_body_twist
    for _iteration in range(projection_iterations):
        _evaluate_deformable_contacts(
            contact_system,
            world_active,
            status,
            projected_velocity,
            empty_twist,
            False,
        )
        _accumulate_particle_colors(
            contact_system,
            world_active,
            status,
            particle_baseline,
            projected_velocity,
        )
        _update_deformable(contact_system, world_active, status, projected_velocity, empty_twist, False)
    _compute_particle_diagnostics(
        contact_system,
        world_active,
        status,
        particle_baseline,
        projected_velocity,
        False,
    )
