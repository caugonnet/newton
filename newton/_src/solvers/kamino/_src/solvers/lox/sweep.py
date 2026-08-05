# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-space unilateral projection sweeps."""

from __future__ import annotations

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact import (
    CoulombSolveStatistics,
    _record_coulomb_solve_statistics,
    _solve_contact_coulomb_newton_normal_last,
    _solve_contact_coulomb_newton_normal_last_instrumented,
    compute_contact_scaled_alart_curnier_residual,
)
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    compute_contact_delassus,
    compute_limit_delassus,
    convert_contact_matrix_normal_last_to_first,
    convert_contact_vector_normal_last_to_first,
    prepare_contact_coulomb,
    project_contact_coulomb,
    project_contact_coulomb_prepared,
    project_joint_friction,
    project_limit_unilateral,
)

__all__ = [
    "compute_projection_residuals",
    "prepare_contact_projection_data",
    "prepare_jacobi_projection_data",
    "project_constraints_jacobi",
    "project_constraints_sequential",
    "sweep_constraints_sequential",
    "warm_start_constraints_sequential",
]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _prepare_contact_projection_data(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if contact_local[contact] >= world_contact_count[world]:
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        delassus[contact] = wp.mat33f(0.0)
        return
    if first >= 0:
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        inverse_weight_second = inverse_weight[second]
    data = prepare_contact_coulomb(
        contact_jacobian_first[contact],
        inverse_weight_first,
        contact_jacobian_second[contact],
        inverse_weight_second,
        contact_bias[contact],
        contact_friction[contact],
    )
    delassus[contact] = data.delassus
    if data.status == PROJECTION_STATUS_INVALID:
        world_status[world] = data.status


@wp.func
def _atomic_add_twist(values: wp.array[vec6f], body: wp.int32, increment: vec6f):
    if body >= 0:
        wp.atomic_add(values, body, increment)


@wp.func
def _is_finite_twist(value: vec6f) -> wp.bool:
    result = wp.bool(True)
    for index in range(6):
        result = result and wp.isfinite(value[index])
    return result


@wp.kernel
def _prepare_contacts_jacobi(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if contact_local[contact] >= world_contact_count[world]:
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        delassus[contact] = wp.mat33f(0.0)
        return
    if first >= 0:
        multiplicity = wp.max(1, body_constraint_count[first] - static_body_constraint_count[first])
        inverse_weight_first = wp.float32(multiplicity) * inverse_weight[first]
    if second >= 0:
        multiplicity = wp.max(1, body_constraint_count[second] - static_body_constraint_count[second])
        inverse_weight_second = wp.float32(multiplicity) * inverse_weight[second]
    data = prepare_contact_coulomb(
        contact_jacobian_first[contact],
        inverse_weight_first,
        contact_jacobian_second[contact],
        inverse_weight_second,
        contact_bias[contact],
        contact_friction[contact],
    )
    delassus[contact] = data.delassus
    if data.status == PROJECTION_STATUS_INVALID:
        world_status[world] = data.status


@wp.kernel
def _prepare_limits_jacobi(
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    limit = wp.tid()
    world = limit_world[limit]
    if limit_local[limit] >= world_limit_count[world]:
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = limit_body_first[limit]
    second = limit_body_second[limit]
    if first < 0 and second < 0:
        delassus[limit] = 0.0
        return
    if first >= 0:
        multiplicity = wp.max(1, body_constraint_count[first] - static_body_constraint_count[first])
        inverse_weight_first = wp.float32(multiplicity) * inverse_weight[first]
    if second >= 0:
        multiplicity = wp.max(1, body_constraint_count[second] - static_body_constraint_count[second])
        inverse_weight_second = wp.float32(multiplicity) * inverse_weight[second]
    value = compute_limit_delassus(
        limit_jacobian_first[limit],
        inverse_weight_first,
        limit_jacobian_second[limit],
        inverse_weight_second,
    )
    delassus[limit] = value
    if not wp.isfinite(value) or value <= 0.0:
        world_status[world] = PROJECTION_STATUS_INVALID


@wp.kernel
def _prepare_frictions_jacobi(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    friction = wp.tid()
    world = friction_world[friction]
    if friction_local[friction] >= world_friction_count[world]:
        return
    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = friction_body_first[friction]
    second = friction_body_second[friction]
    if first >= 0:
        multiplicity = wp.max(1, body_constraint_count[first] - static_body_constraint_count[first])
        inverse_weight_first = wp.float32(multiplicity) * inverse_weight[first]
    if second >= 0:
        multiplicity = wp.max(1, body_constraint_count[second] - static_body_constraint_count[second])
        inverse_weight_second = wp.float32(multiplicity) * inverse_weight[second]
    value = compute_limit_delassus(
        friction_jacobian_first[friction],
        inverse_weight_first,
        friction_jacobian_second[friction],
        inverse_weight_second,
    )
    delassus[friction] = value
    if not wp.isfinite(value) or value <= 0.0:
        world_status[world] = PROJECTION_STATUS_INVALID


@wp.kernel
def _initialize_jacobi_projection_status(
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    if world_active[world]:
        world_status[world] = prepared_status[world]


@wp.kernel
def _warmstart_contacts_jacobi(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    inverse_weight: wp.array[mat66f],
    reaction: wp.array[wp.vec3f],
    twist_delta: wp.array[vec6f],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_local[contact] >= world_contact_count[world]
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    impulse = reaction[contact]
    if first >= 0:
        _atomic_add_twist(
            twist_delta,
            first,
            inverse_weight[first] @ (wp.transpose(contact_jacobian_first[contact]) @ impulse),
        )
    if second >= 0:
        _atomic_add_twist(
            twist_delta,
            second,
            inverse_weight[second] @ (wp.transpose(contact_jacobian_second[contact]) @ impulse),
        )


@wp.kernel
def _warmstart_limits_jacobi(
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
):
    limit = wp.tid()
    world = limit_world[limit]
    if (
        limit_local[limit] >= world_limit_count[world]
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = limit_body_first[limit]
    second = limit_body_second[limit]
    impulse = reaction[limit]
    if first >= 0:
        _atomic_add_twist(twist_delta, first, (inverse_weight[first] @ limit_jacobian_first[limit]) * impulse)
    if second >= 0:
        _atomic_add_twist(twist_delta, second, (inverse_weight[second] @ limit_jacobian_second[limit]) * impulse)


@wp.kernel
def _warmstart_frictions_jacobi(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
):
    friction = wp.tid()
    world = friction_world[friction]
    if (
        friction_local[friction] >= world_friction_count[world]
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = friction_body_first[friction]
    second = friction_body_second[friction]
    impulse = reaction[friction]
    if first >= 0:
        _atomic_add_twist(twist_delta, first, (inverse_weight[first] @ friction_jacobian_first[friction]) * impulse)
    if second >= 0:
        _atomic_add_twist(twist_delta, second, (inverse_weight[second] @ friction_jacobian_second[friction]) * impulse)


@wp.kernel
def _project_contacts_jacobi(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_delassus: wp.array[wp.mat33f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.vec3f],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_local[contact] >= world_contact_count[world]
        or not world_active[world]
        or world_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        reaction[contact] = wp.vec3f(0.0)
        return
    twist_first = vec6f(0.0)
    twist_second = vec6f(0.0)
    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    if first >= 0:
        twist_first = projected_twist[first]
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        twist_second = projected_twist[second]
        inverse_weight_second = inverse_weight[second]
    reaction_old = reaction[contact]
    current_velocity = (
        contact_jacobian_first[contact] @ twist_first
        + contact_jacobian_second[contact] @ twist_second
        + contact_bias[contact]
    )
    contact_block = contact_delassus[contact]
    free_velocity = current_velocity - contact_block @ reaction_old
    reaction_new = _solve_contact_coulomb_newton_normal_last(
        contact_block,
        free_velocity,
        contact_friction[contact],
    )
    reaction_delta = reaction_new - reaction_old
    if (
        not wp.isfinite(reaction_new[0])
        or not wp.isfinite(reaction_new[1])
        or not wp.isfinite(reaction_new[2])
        or not wp.isfinite(reaction_delta[0])
        or not wp.isfinite(reaction_delta[1])
        or not wp.isfinite(reaction_delta[2])
    ):
        world_status[world] = PROJECTION_STATUS_INVALID
        return

    correction_first = vec6f(0.0)
    correction_second = vec6f(0.0)
    if first >= 0:
        correction_first = inverse_weight_first @ (wp.transpose(contact_jacobian_first[contact]) @ reaction_delta)
    if second >= 0:
        correction_second = inverse_weight_second @ (wp.transpose(contact_jacobian_second[contact]) @ reaction_delta)
    if not _is_finite_twist(correction_first) or not _is_finite_twist(correction_second):
        world_status[world] = PROJECTION_STATUS_INVALID
        return

    reaction[contact] = reaction_new
    _atomic_add_twist(twist_delta, first, correction_first)
    _atomic_add_twist(twist_delta, second, correction_second)


@wp.kernel
def _project_contacts_jacobi_instrumented(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_delassus: wp.array[wp.mat33f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.vec3f],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
    branch_histogram: wp.array2d[wp.int64],
    expansion_histogram: wp.array2d[wp.int64],
    root_histogram: wp.array2d[wp.int64],
    failure_counts: wp.array2d[wp.int64],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_local[contact] >= world_contact_count[world]
        or not world_active[world]
        or world_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        reaction[contact] = wp.vec3f(0.0)
        return
    twist_first = vec6f(0.0)
    twist_second = vec6f(0.0)
    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    if first >= 0:
        twist_first = projected_twist[first]
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        twist_second = projected_twist[second]
        inverse_weight_second = inverse_weight[second]
    reaction_old = reaction[contact]
    current_velocity = (
        contact_jacobian_first[contact] @ twist_first
        + contact_jacobian_second[contact] @ twist_second
        + contact_bias[contact]
    )
    contact_block = contact_delassus[contact]
    free_velocity = current_velocity - contact_block @ reaction_old
    solve_result = _solve_contact_coulomb_newton_normal_last_instrumented(
        contact_block,
        free_velocity,
        contact_friction[contact],
    )
    _record_coulomb_solve_statistics(
        solve_result,
        wp.int32(0),
        branch_histogram,
        expansion_histogram,
        root_histogram,
        failure_counts,
    )
    reaction_new = solve_result.reaction
    reaction_delta = reaction_new - reaction_old
    if (
        not wp.isfinite(reaction_new[0])
        or not wp.isfinite(reaction_new[1])
        or not wp.isfinite(reaction_new[2])
        or not wp.isfinite(reaction_delta[0])
        or not wp.isfinite(reaction_delta[1])
        or not wp.isfinite(reaction_delta[2])
    ):
        world_status[world] = PROJECTION_STATUS_INVALID
        return

    correction_first = vec6f(0.0)
    correction_second = vec6f(0.0)
    if first >= 0:
        correction_first = inverse_weight_first @ (wp.transpose(contact_jacobian_first[contact]) @ reaction_delta)
    if second >= 0:
        correction_second = inverse_weight_second @ (wp.transpose(contact_jacobian_second[contact]) @ reaction_delta)
    if not _is_finite_twist(correction_first) or not _is_finite_twist(correction_second):
        world_status[world] = PROJECTION_STATUS_INVALID
        return

    reaction[contact] = reaction_new
    _atomic_add_twist(twist_delta, first, correction_first)
    _atomic_add_twist(twist_delta, second, correction_second)


@wp.kernel
def _project_limits_jacobi(
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
):
    limit = wp.tid()
    world = limit_world[limit]
    if (
        limit_local[limit] >= world_limit_count[world]
        or not world_active[world]
        or world_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    first = limit_body_first[limit]
    second = limit_body_second[limit]
    if first < 0 and second < 0:
        reaction[limit] = 0.0
        return
    current_velocity = limit_bias[limit]
    if first >= 0:
        current_velocity += wp.dot(limit_jacobian_first[limit], projected_twist[first])
    if second >= 0:
        current_velocity += wp.dot(limit_jacobian_second[limit], projected_twist[second])
    reaction_old = reaction[limit]
    split_delassus = limit_delassus[limit]
    free_velocity = current_velocity - split_delassus * reaction_old
    reaction_new = wp.max(-free_velocity / split_delassus, 0.0)
    reaction_delta = reaction_new - reaction_old
    if not wp.isfinite(reaction_new) or not wp.isfinite(reaction_delta):
        world_status[world] = PROJECTION_STATUS_INVALID
        return

    correction_first = vec6f(0.0)
    correction_second = vec6f(0.0)
    if first >= 0:
        correction_first = (inverse_weight[first] @ limit_jacobian_first[limit]) * reaction_delta
    if second >= 0:
        correction_second = (inverse_weight[second] @ limit_jacobian_second[limit]) * reaction_delta
    if not _is_finite_twist(correction_first) or not _is_finite_twist(correction_second):
        world_status[world] = PROJECTION_STATUS_INVALID
        return

    reaction[limit] = reaction_new
    _atomic_add_twist(twist_delta, first, correction_first)
    _atomic_add_twist(twist_delta, second, correction_second)


@wp.kernel
def _project_frictions_jacobi(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
):
    friction = wp.tid()
    world = friction_world[friction]
    if (
        friction_local[friction] >= world_friction_count[world]
        or not world_active[world]
        or world_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = friction_body_first[friction]
    second = friction_body_second[friction]
    current_velocity = wp.float32(0.0)
    if first >= 0:
        current_velocity += wp.dot(friction_jacobian_first[friction], projected_twist[first])
    if second >= 0:
        current_velocity += wp.dot(friction_jacobian_second[friction], projected_twist[second])
    reaction_old = reaction[friction]
    split_delassus = friction_delassus[friction]
    free_velocity = current_velocity - split_delassus * reaction_old
    reaction_new = wp.clamp(
        -free_velocity / split_delassus,
        -friction_impulse_bound[friction],
        friction_impulse_bound[friction],
    )
    reaction_delta = reaction_new - reaction_old
    if not wp.isfinite(reaction_new) or not wp.isfinite(reaction_delta):
        world_status[world] = PROJECTION_STATUS_INVALID
        return
    correction_first = vec6f(0.0)
    correction_second = vec6f(0.0)
    if first >= 0:
        correction_first = (inverse_weight[first] @ friction_jacobian_first[friction]) * reaction_delta
    if second >= 0:
        correction_second = (inverse_weight[second] @ friction_jacobian_second[friction]) * reaction_delta
    if not _is_finite_twist(correction_first) or not _is_finite_twist(correction_second):
        world_status[world] = PROJECTION_STATUS_INVALID
        return
    reaction[friction] = reaction_new
    _atomic_add_twist(twist_delta, first, correction_first)
    _atomic_add_twist(twist_delta, second, correction_second)


@wp.kernel
def _apply_jacobi_twist_delta(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    twist_delta: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
        projected_twist[body] += twist_delta[body]
    twist_delta[body] = vec6f(0.0)


@wp.kernel
def _project_constraints_sequential(
    projection_iterations: wp.int32,
    world_active: wp.array[wp.bool],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    friction_velocity: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_velocity: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_velocity: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    world_status[world] = PROJECTION_STATUS_VALID
    friction_start = world_friction_offset[world]
    friction_end = friction_start + world_friction_count[world]
    limit_start = world_limit_offset[world]
    limit_end = limit_start + world_limit_count[world]
    contact_start = world_contact_offset[world]
    contact_end = contact_start + world_contact_count[world]

    for friction in range(friction_start, friction_end):
        first = friction_body_first[friction]
        second = friction_body_second[friction]
        friction_impulse = friction_reaction[friction]
        if first >= 0:
            projected_twist[first] += inverse_weight[first] @ (friction_impulse * friction_jacobian_first[friction])
        if second >= 0:
            projected_twist[second] += inverse_weight[second] @ (friction_impulse * friction_jacobian_second[friction])
    for limit in range(limit_start, limit_end):
        first = limit_body_first[limit]
        second = limit_body_second[limit]
        limit_impulse = limit_reaction[limit]
        if first >= 0:
            projected_twist[first] += inverse_weight[first] @ (limit_impulse * limit_jacobian_first[limit])
        if second >= 0:
            projected_twist[second] += inverse_weight[second] @ (limit_impulse * limit_jacobian_second[limit])
    for contact in range(contact_start, contact_end):
        first = contact_body_first[contact]
        second = contact_body_second[contact]
        contact_impulse = contact_reaction[contact]
        if first >= 0:
            projected_twist[first] += inverse_weight[first] @ (
                wp.transpose(contact_jacobian_first[contact]) @ contact_impulse
            )
        if second >= 0:
            projected_twist[second] += inverse_weight[second] @ (
                wp.transpose(contact_jacobian_second[contact]) @ contact_impulse
            )

    for _sweep in range(projection_iterations):
        for friction in range(friction_start, friction_end):
            first = friction_body_first[friction]
            second = friction_body_second[friction]
            twist_first = vec6f(0.0)
            twist_second = vec6f(0.0)
            inverse_weight_first = mat66f(0.0)
            inverse_weight_second = mat66f(0.0)
            if first >= 0:
                twist_first = projected_twist[first]
                inverse_weight_first = inverse_weight[first]
            if second >= 0:
                twist_second = projected_twist[second]
                inverse_weight_second = inverse_weight[second]
            result_friction = project_joint_friction(
                friction_jacobian_first[friction],
                inverse_weight_first,
                twist_first,
                friction_jacobian_second[friction],
                inverse_weight_second,
                twist_second,
                friction_reaction[friction],
                friction_impulse_bound[friction],
            )
            if result_friction.status != PROJECTION_STATUS_VALID:
                world_status[world] = result_friction.status
                return
            if first >= 0:
                projected_twist[first] = result_friction.twist_first
            if second >= 0:
                projected_twist[second] = result_friction.twist_second
            friction_reaction[friction] = result_friction.reaction
            friction_velocity[friction] = result_friction.velocity

        for limit in range(limit_start, limit_end):
            first = limit_body_first[limit]
            second = limit_body_second[limit]
            if first < 0 and second < 0:
                limit_reaction[limit] = 0.0
                limit_velocity[limit] = 0.0
                continue
            twist_first = vec6f(0.0)
            twist_second = vec6f(0.0)
            inverse_weight_first = mat66f(0.0)
            inverse_weight_second = mat66f(0.0)
            if first >= 0:
                twist_first = projected_twist[first]
                inverse_weight_first = inverse_weight[first]
            if second >= 0:
                twist_second = projected_twist[second]
                inverse_weight_second = inverse_weight[second]

            result_limit = project_limit_unilateral(
                limit_jacobian_first[limit],
                inverse_weight_first,
                twist_first,
                limit_jacobian_second[limit],
                inverse_weight_second,
                twist_second,
                limit_bias[limit],
                limit_reaction[limit],
            )
            if result_limit.status != PROJECTION_STATUS_VALID:
                world_status[world] = result_limit.status
                return
            if first >= 0:
                projected_twist[first] = result_limit.twist_first
            if second >= 0:
                projected_twist[second] = result_limit.twist_second
            limit_reaction[limit] = result_limit.reaction
            limit_velocity[limit] = result_limit.velocity - limit_bias[limit]

        for contact in range(contact_start, contact_end):
            first = contact_body_first[contact]
            second = contact_body_second[contact]
            if first < 0 and second < 0:
                contact_reaction[contact] = wp.vec3f(0.0)
                contact_velocity[contact] = wp.vec3f(0.0)
                continue
            twist_first = vec6f(0.0)
            twist_second = vec6f(0.0)
            inverse_weight_first = mat66f(0.0)
            inverse_weight_second = mat66f(0.0)
            if first >= 0:
                twist_first = projected_twist[first]
                inverse_weight_first = inverse_weight[first]
            if second >= 0:
                twist_second = projected_twist[second]
                inverse_weight_second = inverse_weight[second]

            result_contact = project_contact_coulomb(
                contact_jacobian_first[contact],
                inverse_weight_first,
                twist_first,
                contact_jacobian_second[contact],
                inverse_weight_second,
                twist_second,
                contact_bias[contact],
                contact_reaction[contact],
                contact_friction[contact],
            )
            if result_contact.status == PROJECTION_STATUS_INVALID:
                world_status[world] = result_contact.status
                return
            if first >= 0:
                projected_twist[first] = result_contact.twist_first
            if second >= 0:
                projected_twist[second] = result_contact.twist_second
            contact_reaction[contact] = result_contact.reaction
            contact_velocity[contact] = result_contact.velocity - contact_bias[contact]


@wp.kernel
def _project_constraints_sequential_prepared(
    projection_iterations: wp.int32,
    world_active: wp.array[wp.bool],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_delassus: wp.array[wp.mat33f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_projection_status: wp.array[wp.int32],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    friction_velocity: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_velocity: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_velocity: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    world_status[world] = contact_projection_status[world]
    if contact_projection_status[world] != PROJECTION_STATUS_VALID:
        return

    friction_start = world_friction_offset[world]
    friction_end = friction_start + world_friction_count[world]
    limit_start = world_limit_offset[world]
    limit_end = limit_start + world_limit_count[world]
    contact_start = world_contact_offset[world]
    contact_end = contact_start + world_contact_count[world]

    for friction in range(friction_start, friction_end):
        first = friction_body_first[friction]
        second = friction_body_second[friction]
        impulse = friction_reaction[friction]
        if first >= 0:
            projected_twist[first] += inverse_weight[first] @ (impulse * friction_jacobian_first[friction])
        if second >= 0:
            projected_twist[second] += inverse_weight[second] @ (impulse * friction_jacobian_second[friction])
    for limit in range(limit_start, limit_end):
        first = limit_body_first[limit]
        second = limit_body_second[limit]
        limit_impulse = limit_reaction[limit]
        if first >= 0:
            projected_twist[first] += inverse_weight[first] @ (limit_impulse * limit_jacobian_first[limit])
        if second >= 0:
            projected_twist[second] += inverse_weight[second] @ (limit_impulse * limit_jacobian_second[limit])
    for contact in range(contact_start, contact_end):
        first = contact_body_first[contact]
        second = contact_body_second[contact]
        contact_impulse = contact_reaction[contact]
        if first >= 0:
            projected_twist[first] += inverse_weight[first] @ (
                wp.transpose(contact_jacobian_first[contact]) @ contact_impulse
            )
        if second >= 0:
            projected_twist[second] += inverse_weight[second] @ (
                wp.transpose(contact_jacobian_second[contact]) @ contact_impulse
            )

    for _sweep in range(projection_iterations):
        for friction in range(friction_start, friction_end):
            first = friction_body_first[friction]
            second = friction_body_second[friction]
            twist_first = vec6f(0.0)
            twist_second = vec6f(0.0)
            inverse_weight_first = mat66f(0.0)
            inverse_weight_second = mat66f(0.0)
            if first >= 0:
                twist_first = projected_twist[first]
                inverse_weight_first = inverse_weight[first]
            if second >= 0:
                twist_second = projected_twist[second]
                inverse_weight_second = inverse_weight[second]
            result_friction = project_joint_friction(
                friction_jacobian_first[friction],
                inverse_weight_first,
                twist_first,
                friction_jacobian_second[friction],
                inverse_weight_second,
                twist_second,
                friction_reaction[friction],
                friction_impulse_bound[friction],
            )
            if result_friction.status != PROJECTION_STATUS_VALID:
                world_status[world] = result_friction.status
                return
            if first >= 0:
                projected_twist[first] = result_friction.twist_first
            if second >= 0:
                projected_twist[second] = result_friction.twist_second
            friction_reaction[friction] = result_friction.reaction
            friction_velocity[friction] = result_friction.velocity

        for limit in range(limit_start, limit_end):
            first = limit_body_first[limit]
            second = limit_body_second[limit]
            if first < 0 and second < 0:
                limit_reaction[limit] = 0.0
                limit_velocity[limit] = 0.0
                continue
            twist_first = vec6f(0.0)
            twist_second = vec6f(0.0)
            inverse_weight_first = mat66f(0.0)
            inverse_weight_second = mat66f(0.0)
            if first >= 0:
                twist_first = projected_twist[first]
                inverse_weight_first = inverse_weight[first]
            if second >= 0:
                twist_second = projected_twist[second]
                inverse_weight_second = inverse_weight[second]

            result_limit = project_limit_unilateral(
                limit_jacobian_first[limit],
                inverse_weight_first,
                twist_first,
                limit_jacobian_second[limit],
                inverse_weight_second,
                twist_second,
                limit_bias[limit],
                limit_reaction[limit],
            )
            if result_limit.status != PROJECTION_STATUS_VALID:
                world_status[world] = result_limit.status
                return
            if first >= 0:
                projected_twist[first] = result_limit.twist_first
            if second >= 0:
                projected_twist[second] = result_limit.twist_second
            limit_reaction[limit] = result_limit.reaction
            limit_velocity[limit] = result_limit.velocity - limit_bias[limit]

        for contact in range(contact_start, contact_end):
            first = contact_body_first[contact]
            second = contact_body_second[contact]
            if first < 0 and second < 0:
                contact_reaction[contact] = wp.vec3f(0.0)
                contact_velocity[contact] = wp.vec3f(0.0)
                continue
            twist_first = vec6f(0.0)
            twist_second = vec6f(0.0)
            inverse_weight_first = mat66f(0.0)
            inverse_weight_second = mat66f(0.0)
            if first >= 0:
                twist_first = projected_twist[first]
                inverse_weight_first = inverse_weight[first]
            if second >= 0:
                twist_second = projected_twist[second]
                inverse_weight_second = inverse_weight[second]

            result_contact = project_contact_coulomb_prepared(
                contact_jacobian_first[contact],
                inverse_weight_first,
                twist_first,
                contact_jacobian_second[contact],
                inverse_weight_second,
                twist_second,
                contact_bias[contact],
                contact_reaction[contact],
                contact_friction[contact],
                contact_delassus[contact],
            )
            if result_contact.status == PROJECTION_STATUS_INVALID:
                world_status[world] = result_contact.status
                return
            if first >= 0:
                projected_twist[first] = result_contact.twist_first
            if second >= 0:
                projected_twist[second] = result_contact.twist_second
            contact_reaction[contact] = result_contact.reaction
            contact_velocity[contact] = result_contact.velocity - contact_bias[contact]


@wp.kernel
def _warm_start_constraints_sequential(
    world_active: wp.array[wp.bool],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    prepared_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    world_status[world] = prepared_status[world]
    if prepared_status[world] != PROJECTION_STATUS_VALID:
        return

    friction_start = world_friction_offset[world]
    friction_end = friction_start + world_friction_count[world]
    for friction in range(friction_start, friction_end):
        first = friction_body_first[friction]
        second = friction_body_second[friction]
        impulse = friction_reaction[friction]
        if first >= 0:
            projected_twist[first] += inverse_weight[first] @ (impulse * friction_jacobian_first[friction])
        if second >= 0:
            projected_twist[second] += inverse_weight[second] @ (impulse * friction_jacobian_second[friction])

    limit_start = world_limit_offset[world]
    limit_end = limit_start + world_limit_count[world]
    for limit in range(limit_start, limit_end):
        first = limit_body_first[limit]
        second = limit_body_second[limit]
        limit_impulse = limit_reaction[limit]
        if first >= 0:
            projected_twist[first] += inverse_weight[first] @ (limit_impulse * limit_jacobian_first[limit])
        if second >= 0:
            projected_twist[second] += inverse_weight[second] @ (limit_impulse * limit_jacobian_second[limit])

    contact_start = world_contact_offset[world]
    contact_end = contact_start + world_contact_count[world]
    for contact in range(contact_start, contact_end):
        first = contact_body_first[contact]
        second = contact_body_second[contact]
        contact_impulse = contact_reaction[contact]
        if first >= 0:
            projected_twist[first] += inverse_weight[first] @ (
                wp.transpose(contact_jacobian_first[contact]) @ contact_impulse
            )
        if second >= 0:
            projected_twist[second] += inverse_weight[second] @ (
                wp.transpose(contact_jacobian_second[contact]) @ contact_impulse
            )


@wp.kernel
def _sweep_constraints_sequential_prepared(
    world_active: wp.array[wp.bool],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_delassus: wp.array[wp.mat33f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    friction_velocity: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_velocity: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_velocity: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    if not world_active[world] or world_status[world] != PROJECTION_STATUS_VALID:
        return

    friction_start = world_friction_offset[world]
    friction_end = friction_start + world_friction_count[world]
    for friction in range(friction_start, friction_end):
        first = friction_body_first[friction]
        second = friction_body_second[friction]
        twist_first = vec6f(0.0)
        twist_second = vec6f(0.0)
        inverse_weight_first = mat66f(0.0)
        inverse_weight_second = mat66f(0.0)
        if first >= 0:
            twist_first = projected_twist[first]
            inverse_weight_first = inverse_weight[first]
        if second >= 0:
            twist_second = projected_twist[second]
            inverse_weight_second = inverse_weight[second]
        result_friction = project_joint_friction(
            friction_jacobian_first[friction],
            inverse_weight_first,
            twist_first,
            friction_jacobian_second[friction],
            inverse_weight_second,
            twist_second,
            friction_reaction[friction],
            friction_impulse_bound[friction],
        )
        if result_friction.status != PROJECTION_STATUS_VALID:
            world_status[world] = result_friction.status
            return
        if first >= 0:
            projected_twist[first] = result_friction.twist_first
        if second >= 0:
            projected_twist[second] = result_friction.twist_second
        friction_reaction[friction] = result_friction.reaction
        friction_velocity[friction] = result_friction.velocity

    limit_start = world_limit_offset[world]
    limit_end = limit_start + world_limit_count[world]
    for limit in range(limit_start, limit_end):
        first = limit_body_first[limit]
        second = limit_body_second[limit]
        if first < 0 and second < 0:
            limit_reaction[limit] = 0.0
            limit_velocity[limit] = 0.0
            continue
        twist_first = vec6f(0.0)
        twist_second = vec6f(0.0)
        inverse_weight_first = mat66f(0.0)
        inverse_weight_second = mat66f(0.0)
        if first >= 0:
            twist_first = projected_twist[first]
            inverse_weight_first = inverse_weight[first]
        if second >= 0:
            twist_second = projected_twist[second]
            inverse_weight_second = inverse_weight[second]
        result_limit = project_limit_unilateral(
            limit_jacobian_first[limit],
            inverse_weight_first,
            twist_first,
            limit_jacobian_second[limit],
            inverse_weight_second,
            twist_second,
            limit_bias[limit],
            limit_reaction[limit],
        )
        if result_limit.status != PROJECTION_STATUS_VALID:
            world_status[world] = result_limit.status
            return
        if first >= 0:
            projected_twist[first] = result_limit.twist_first
        if second >= 0:
            projected_twist[second] = result_limit.twist_second
        limit_reaction[limit] = result_limit.reaction
        limit_velocity[limit] = result_limit.velocity - limit_bias[limit]

    contact_start = world_contact_offset[world]
    contact_end = contact_start + world_contact_count[world]
    for contact in range(contact_start, contact_end):
        first = contact_body_first[contact]
        second = contact_body_second[contact]
        if first < 0 and second < 0:
            contact_reaction[contact] = wp.vec3f(0.0)
            contact_velocity[contact] = wp.vec3f(0.0)
            continue
        twist_first = vec6f(0.0)
        twist_second = vec6f(0.0)
        inverse_weight_first = mat66f(0.0)
        inverse_weight_second = mat66f(0.0)
        if first >= 0:
            twist_first = projected_twist[first]
            inverse_weight_first = inverse_weight[first]
        if second >= 0:
            twist_second = projected_twist[second]
            inverse_weight_second = inverse_weight[second]
        result_contact = project_contact_coulomb_prepared(
            contact_jacobian_first[contact],
            inverse_weight_first,
            twist_first,
            contact_jacobian_second[contact],
            inverse_weight_second,
            twist_second,
            contact_bias[contact],
            contact_reaction[contact],
            contact_friction[contact],
            contact_delassus[contact],
        )
        if result_contact.status == PROJECTION_STATUS_INVALID:
            world_status[world] = result_contact.status
            return
        if first >= 0:
            projected_twist[first] = result_contact.twist_first
        if second >= 0:
            projected_twist[second] = result_contact.twist_second
        contact_reaction[contact] = result_contact.reaction
        contact_velocity[contact] = result_contact.velocity - contact_bias[contact]


@wp.kernel
def _initialize_projection_residuals(
    world_contact_residual_max: wp.array[wp.float32],
    world_limit_residual_max: wp.array[wp.float32],
    world_friction_residual_max: wp.array[wp.float32],
):
    world = wp.tid()
    world_contact_residual_max[world] = 0.0
    world_limit_residual_max[world] = 0.0
    world_friction_residual_max[world] = 0.0


@wp.kernel
def _compute_friction_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_velocity: wp.array[wp.float32],
    friction_residual: wp.array[wp.float32],
    world_friction_residual_max: wp.array[wp.float32],
):
    friction = wp.tid()
    world = friction_world[friction]
    if (
        friction_local[friction] >= world_friction_count[world]
        or not world_mask[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = friction_body_first[friction]
    second = friction_body_second[friction]
    if first >= 0:
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        inverse_weight_second = inverse_weight[second]
    value = wp.float32(0.0)
    if first >= 0:
        value += wp.dot(friction_jacobian_first[friction], projected_twist[first])
    if second >= 0:
        value += wp.dot(friction_jacobian_second[friction], projected_twist[second])
    friction_velocity[friction] = value
    delassus = compute_limit_delassus(
        friction_jacobian_first[friction],
        inverse_weight_first,
        friction_jacobian_second[friction],
        inverse_weight_second,
    )
    scale = wp.sqrt(delassus)
    scaled_reaction = scale * friction_reaction[friction]
    scaled_velocity = value / scale
    scaled_bound = scale * friction_impulse_bound[friction]
    residual = wp.abs(scaled_reaction - wp.clamp(scaled_reaction - scaled_velocity, -scaled_bound, scaled_bound))
    friction_residual[friction] = residual
    wp.atomic_max(world_friction_residual_max, world, residual)


@wp.kernel
def _compute_contact_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    contact_velocity: wp.array[wp.vec3f],
    contact_residual: wp.array[wp.float32],
    world_contact_residual_max: wp.array[wp.float32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_local[contact] >= world_contact_count[world]
        or not world_mask[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        contact_velocity[contact] = wp.vec3f(0.0)
        contact_residual[contact] = 0.0
        return
    if first >= 0:
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        inverse_weight_second = inverse_weight[second]
    contact_velocity_final = wp.vec3f(0.0)
    if first >= 0:
        contact_velocity_final += contact_jacobian_first[contact] @ projected_twist[first]
    if second >= 0:
        contact_velocity_final += contact_jacobian_second[contact] @ projected_twist[second]
    contact_velocity[contact] = contact_velocity_final
    contact_delassus = compute_contact_delassus(
        contact_jacobian_first[contact],
        inverse_weight_first,
        contact_jacobian_second[contact],
        inverse_weight_second,
    )
    contact_residual_vector = compute_contact_scaled_alart_curnier_residual(
        convert_contact_matrix_normal_last_to_first(contact_delassus),
        convert_contact_vector_normal_last_to_first(contact_reaction[contact]),
        convert_contact_vector_normal_last_to_first(contact_velocity_final + contact_bias[contact]),
        contact_friction[contact],
    )
    residual_max = wp.max(wp.abs(contact_residual_vector[0]), wp.abs(contact_residual_vector[1]))
    residual_max = wp.max(residual_max, wp.abs(contact_residual_vector[2]))
    contact_residual[contact] = residual_max
    wp.atomic_max(world_contact_residual_max, world, residual_max)


@wp.kernel
def _compute_limit_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_reaction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    limit_velocity: wp.array[wp.float32],
    limit_residual: wp.array[wp.float32],
    world_limit_residual_max: wp.array[wp.float32],
):
    limit = wp.tid()
    world = limit_world[limit]
    if (
        limit_local[limit] >= world_limit_count[world]
        or not world_mask[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = limit_body_first[limit]
    second = limit_body_second[limit]
    if first < 0 and second < 0:
        limit_velocity[limit] = 0.0
        limit_residual[limit] = 0.0
        return
    if first >= 0:
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        inverse_weight_second = inverse_weight[second]
    limit_velocity_final = wp.float32(0.0)
    if first >= 0:
        limit_velocity_final += wp.dot(limit_jacobian_first[limit], projected_twist[first])
    if second >= 0:
        limit_velocity_final += wp.dot(limit_jacobian_second[limit], projected_twist[second])
    limit_velocity[limit] = limit_velocity_final
    limit_delassus = compute_limit_delassus(
        limit_jacobian_first[limit],
        inverse_weight_first,
        limit_jacobian_second[limit],
        inverse_weight_second,
    )
    scale = wp.sqrt(limit_delassus)
    scaled_reaction = scale * limit_reaction[limit]
    scaled_velocity = (limit_velocity_final + limit_bias[limit]) / scale
    limit_residual_value = wp.abs(scaled_reaction - wp.max(scaled_reaction - scaled_velocity, 0.0))
    limit_residual[limit] = limit_residual_value
    wp.atomic_max(world_limit_residual_max, world, limit_residual_value)


def compute_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_reaction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_velocity: wp.array[wp.float32],
    contact_velocity: wp.array[wp.vec3f],
    limit_velocity: wp.array[wp.float32],
    contact_residual: wp.array[wp.float32],
    limit_residual: wp.array[wp.float32],
    friction_residual: wp.array[wp.float32],
    world_contact_residual_max: wp.array[wp.float32],
    world_limit_residual_max: wp.array[wp.float32],
    world_friction_residual_max: wp.array[wp.float32],
) -> None:
    """Recompute final unilateral velocities and per-world natural-map residual maxima."""
    world_count = world_mask.shape[0]
    if (
        projection_status.shape[0] != world_count
        or world_friction_count.shape[0] != world_count
        or world_contact_count.shape[0] != world_count
        or world_limit_count.shape[0] != world_count
        or world_contact_residual_max.shape[0] != world_count
        or world_limit_residual_max.shape[0] != world_count
        or world_friction_residual_max.shape[0] != world_count
    ):
        raise ValueError("Projection diagnostic world arrays must have identical lengths.")
    if (
        friction_world.shape[0] != friction_local.shape[0]
        or contact_world.shape[0] != contact_local.shape[0]
        or limit_world.shape[0] != limit_local.shape[0]
    ):
        raise ValueError("Projection diagnostic world and local arrays must have identical lengths.")
    wp.launch(
        _initialize_projection_residuals,
        dim=world_count,
        inputs=[],
        outputs=[
            world_contact_residual_max,
            world_limit_residual_max,
            world_friction_residual_max,
        ],
        device=projected_twist.device,
    )
    if friction_world.shape[0] > 0:
        wp.launch(
            _compute_friction_projection_residuals,
            dim=friction_world.shape[0],
            inputs=[
                world_mask,
                projection_status,
                friction_world,
                friction_local,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                friction_impulse_bound,
                friction_reaction,
                inverse_weight,
                projected_twist,
            ],
            outputs=[friction_velocity, friction_residual, world_friction_residual_max],
            device=projected_twist.device,
        )
    if contact_world.shape[0] > 0:
        wp.launch(
            _compute_contact_projection_residuals,
            dim=contact_world.shape[0],
            inputs=[
                world_mask,
                projection_status,
                contact_world,
                contact_local,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_bias,
                contact_friction,
                contact_reaction,
                inverse_weight,
                projected_twist,
            ],
            outputs=[contact_velocity, contact_residual, world_contact_residual_max],
            device=projected_twist.device,
        )
    if limit_world.shape[0] > 0:
        wp.launch(
            _compute_limit_projection_residuals,
            dim=limit_world.shape[0],
            inputs=[
                world_mask,
                projection_status,
                limit_world,
                limit_local,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                limit_bias,
                limit_reaction,
                inverse_weight,
                projected_twist,
            ],
            outputs=[limit_velocity, limit_residual, world_limit_residual_max],
            device=projected_twist.device,
        )


def prepare_contact_projection_data(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
) -> None:
    """Prepare fixed body-space contact data once per solve."""
    if contact_world.shape[0] != contact_local.shape[0]:
        raise ValueError("Contact world and local arrays must have identical lengths.")
    if world_contact_count.shape[0] != world_status.shape[0]:
        raise ValueError("Contact count and status world arrays must have identical lengths.")
    world_status.fill_(PROJECTION_STATUS_VALID)
    if contact_world.shape[0] == 0:
        return
    wp.launch(
        _prepare_contact_projection_data,
        dim=contact_world.shape[0],
        inputs=[
            contact_world,
            contact_local,
            world_contact_count,
            contact_body_first,
            contact_body_second,
            contact_jacobian_first,
            contact_jacobian_second,
            contact_bias,
            contact_friction,
            inverse_weight,
        ],
        outputs=[delassus, world_status],
        device=inverse_weight.device,
    )


def prepare_jacobi_projection_data(
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
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    friction_delassus: wp.array[wp.float32],
    contact_delassus: wp.array[wp.mat33f],
    limit_delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
) -> None:
    """Prepare mass-split body-space blocks for true Jacobi projection."""
    world_count = world_status.shape[0]
    if (
        world_friction_count.shape[0] != world_count
        or world_contact_count.shape[0] != world_count
        or world_limit_count.shape[0] != world_count
    ):
        raise ValueError("Friction, contact, limit, and status world arrays must have identical lengths.")
    if body_constraint_count.shape[0] != static_body_constraint_count.shape[0]:
        raise ValueError("Body incidence arrays must have identical lengths.")
    world_status.fill_(PROJECTION_STATUS_VALID)
    if friction_world.shape[0] > 0:
        wp.launch(
            _prepare_frictions_jacobi,
            dim=friction_world.shape[0],
            inputs=[
                friction_world,
                friction_local,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                body_constraint_count,
                static_body_constraint_count,
                inverse_weight,
            ],
            outputs=[friction_delassus, world_status],
            device=inverse_weight.device,
        )
    if contact_world.shape[0] > 0:
        wp.launch(
            _prepare_contacts_jacobi,
            dim=contact_world.shape[0],
            inputs=[
                contact_world,
                contact_local,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_bias,
                contact_friction,
                body_constraint_count,
                static_body_constraint_count,
                inverse_weight,
            ],
            outputs=[contact_delassus, world_status],
            device=inverse_weight.device,
        )
    if limit_world.shape[0] > 0:
        wp.launch(
            _prepare_limits_jacobi,
            dim=limit_world.shape[0],
            inputs=[
                limit_world,
                limit_local,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                body_constraint_count,
                static_body_constraint_count,
                inverse_weight,
            ],
            outputs=[limit_delassus, world_status],
            device=inverse_weight.device,
        )


def project_constraints_jacobi(
    projection_iterations: int,
    world_active: wp.array[wp.bool],
    body_world: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
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
    limit_delassus: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    twist_delta: wp.array[vec6f],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    prepared_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    deformable_contacts=None,
    deformable_projected_velocity: wp.array[wp.vec3] | None = None,
    warm_start: bool = True,
    coulomb_statistics: CoulombSolveStatistics | None = None,
) -> None:
    """Run mass-split Jacobi sweeps over all body-space unilaterals.

    Set ``warm_start`` to false when the projected state already contains the
    current reactions, such as for a final smoothing sweep after Gauss--Seidel.
    """
    if (
        not isinstance(projection_iterations, int)
        or isinstance(projection_iterations, bool)
        or projection_iterations < 1
    ):
        raise ValueError("projection_iterations must be an integer greater than or equal to one.")
    world_count = world_active.shape[0]
    if prepared_status.shape[0] != world_count or world_status.shape[0] != world_count:
        raise ValueError("Active, prepared-status, and status world arrays must have identical lengths.")
    if body_world.shape[0] != projected_twist.shape[0] or twist_delta.shape[0] != projected_twist.shape[0]:
        raise ValueError("Body world, projected twist, and Jacobi delta arrays must have identical lengths.")
    if (deformable_contacts is None) != (deformable_projected_velocity is None):
        raise ValueError("Deformable contacts and projected velocity must be supplied together.")
    if not isinstance(warm_start, bool):
        raise ValueError("warm_start must be a boolean.")

    if warm_start:
        wp.launch(
            _initialize_jacobi_projection_status,
            dim=world_count,
            inputs=[world_active, prepared_status],
            outputs=[world_status],
            device=projected_twist.device,
        )
    twist_delta.zero_()
    if warm_start and deformable_contacts is not None:
        deformable_contacts.begin_rigid_jacobi_accumulation()
    if warm_start and friction_world.shape[0] > 0:
        wp.launch(
            _warmstart_frictions_jacobi,
            dim=friction_world.shape[0],
            inputs=[
                friction_world,
                friction_local,
                world_active,
                prepared_status,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                inverse_weight,
                friction_reaction,
            ],
            outputs=[twist_delta],
            device=projected_twist.device,
        )
    if warm_start and contact_world.shape[0] > 0:
        wp.launch(
            _warmstart_contacts_jacobi,
            dim=contact_world.shape[0],
            inputs=[
                contact_world,
                contact_local,
                world_active,
                prepared_status,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                inverse_weight,
                contact_reaction,
            ],
            outputs=[twist_delta],
            device=projected_twist.device,
        )
    if warm_start and limit_world.shape[0] > 0:
        wp.launch(
            _warmstart_limits_jacobi,
            dim=limit_world.shape[0],
            inputs=[
                limit_world,
                limit_local,
                world_active,
                prepared_status,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                inverse_weight,
                limit_reaction,
            ],
            outputs=[twist_delta],
            device=projected_twist.device,
        )
    if warm_start and deformable_contacts is not None:
        deformable_contacts.accumulate_rigid_reaction_warm_start(
            world_active,
            prepared_status,
            deformable_contacts.cloth_system.inverse_weight,
            inverse_weight,
            deformable_projected_velocity,
            projected_twist,
            twist_delta,
            world_status,
        )
    if warm_start:
        wp.launch(
            _apply_jacobi_twist_delta,
            dim=projected_twist.shape[0],
            inputs=[body_world, world_active, world_status, twist_delta],
            outputs=[projected_twist],
            device=projected_twist.device,
        )
    if warm_start and deformable_contacts is not None:
        deformable_contacts.apply_rigid_particle_delta(
            world_active,
            world_status,
            deformable_projected_velocity,
        )

    for _sweep in range(projection_iterations):
        twist_delta.zero_()
        if deformable_contacts is not None:
            deformable_contacts.begin_rigid_jacobi_accumulation()
        if friction_world.shape[0] > 0:
            wp.launch(
                _project_frictions_jacobi,
                dim=friction_world.shape[0],
                inputs=[
                    friction_world,
                    friction_local,
                    world_active,
                    world_friction_count,
                    friction_body_first,
                    friction_body_second,
                    friction_jacobian_first,
                    friction_jacobian_second,
                    friction_impulse_bound,
                    friction_delassus,
                    inverse_weight,
                    projected_twist,
                ],
                outputs=[friction_reaction, twist_delta, world_status],
                device=projected_twist.device,
            )
        if contact_world.shape[0] > 0:
            contact_inputs = [
                contact_world,
                contact_local,
                world_active,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_delassus,
                contact_bias,
                contact_friction,
                inverse_weight,
                projected_twist,
            ]
            if coulomb_statistics is None:
                wp.launch(
                    _project_contacts_jacobi,
                    dim=contact_world.shape[0],
                    inputs=contact_inputs,
                    outputs=[contact_reaction, twist_delta, world_status],
                    device=projected_twist.device,
                )
            else:
                wp.launch(
                    _project_contacts_jacobi_instrumented,
                    dim=contact_world.shape[0],
                    inputs=contact_inputs,
                    outputs=[
                        contact_reaction,
                        twist_delta,
                        world_status,
                        coulomb_statistics.branch_histogram,
                        coulomb_statistics.expansion_histogram,
                        coulomb_statistics.root_histogram,
                        coulomb_statistics.failure_counts,
                    ],
                    device=projected_twist.device,
                )
        if limit_world.shape[0] > 0:
            wp.launch(
                _project_limits_jacobi,
                dim=limit_world.shape[0],
                inputs=[
                    limit_world,
                    limit_local,
                    world_active,
                    world_limit_count,
                    limit_body_first,
                    limit_body_second,
                    limit_jacobian_first,
                    limit_jacobian_second,
                    limit_bias,
                    limit_delassus,
                    inverse_weight,
                    projected_twist,
                ],
                outputs=[limit_reaction, twist_delta, world_status],
                device=projected_twist.device,
            )
        if deformable_contacts is not None:
            deformable_contacts.project_rigid_jacobi(
                world_active,
                deformable_contacts.cloth_system.inverse_weight,
                inverse_weight,
                deformable_projected_velocity,
                projected_twist,
                twist_delta,
                world_status,
                coulomb_statistics=coulomb_statistics,
            )
        wp.launch(
            _apply_jacobi_twist_delta,
            dim=projected_twist.shape[0],
            inputs=[body_world, world_active, world_status, twist_delta],
            outputs=[projected_twist],
            device=projected_twist.device,
        )
        if deformable_contacts is not None:
            deformable_contacts.apply_rigid_particle_delta(
                world_active,
                world_status,
                deformable_projected_velocity,
            )


def project_constraints_sequential(
    projection_iterations: int,
    world_active: wp.array[wp.bool],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    friction_velocity: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_velocity: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_velocity: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
    contact_delassus: wp.array[wp.mat33f] | None = None,
    contact_projection_status: wp.array[wp.int32] | None = None,
) -> None:
    """Run fixed sequential limit/contact sweeps with one worker per world."""
    if (
        not isinstance(projection_iterations, int)
        or isinstance(projection_iterations, bool)
        or projection_iterations < 1
    ):
        raise ValueError("projection_iterations must be an integer greater than or equal to one.")
    world_count = world_contact_offset.shape[0]
    if (
        world_active.shape[0] != world_count
        or world_friction_offset.shape[0] != world_count
        or world_limit_offset.shape[0] != world_count
        or world_status.shape[0] != world_count
    ):
        raise ValueError("Active, contact, limit, and status world arrays must have identical lengths.")
    prepared_arrays = (
        contact_delassus,
        contact_projection_status,
    )
    prepared = any(array is not None for array in prepared_arrays)
    if prepared and any(array is None for array in prepared_arrays):
        raise ValueError("All prepared contact projection arrays must be supplied together.")
    common_prefix = [
        projection_iterations,
        world_active,
        world_friction_offset,
        world_friction_count,
        friction_body_first,
        friction_body_second,
        friction_jacobian_first,
        friction_jacobian_second,
        friction_impulse_bound,
        world_contact_offset,
        world_contact_count,
        contact_body_first,
        contact_body_second,
        contact_jacobian_first,
        contact_jacobian_second,
    ]
    if prepared:
        inputs = [
            *common_prefix,
            contact_delassus,
            contact_bias,
            contact_friction,
            contact_projection_status,
        ]
        kernel = _project_constraints_sequential_prepared
    else:
        inputs = [*common_prefix, contact_bias, contact_friction]
        kernel = _project_constraints_sequential
    inputs += [
        world_limit_offset,
        world_limit_count,
        limit_body_first,
        limit_body_second,
        limit_jacobian_first,
        limit_jacobian_second,
        limit_bias,
        inverse_weight,
        projected_twist,
    ]
    wp.launch(
        kernel,
        dim=world_count,
        inputs=inputs,
        outputs=[
            friction_reaction,
            friction_velocity,
            contact_reaction,
            contact_velocity,
            limit_reaction,
            limit_velocity,
            world_status,
        ],
        device=projected_twist.device,
    )


def warm_start_constraints_sequential(
    world_active: wp.array[wp.bool],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    prepared_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
) -> None:
    """Apply sequential rigid reaction warm starts once per projection call."""
    world_count = world_active.shape[0]
    if prepared_status.shape[0] != world_count or world_status.shape[0] != world_count:
        raise ValueError("Active, prepared-status, and status world arrays must have identical lengths.")
    wp.launch(
        _warm_start_constraints_sequential,
        dim=world_count,
        inputs=[
            world_active,
            world_friction_offset,
            world_friction_count,
            friction_body_first,
            friction_body_second,
            friction_jacobian_first,
            friction_jacobian_second,
            world_contact_offset,
            world_contact_count,
            contact_body_first,
            contact_body_second,
            contact_jacobian_first,
            contact_jacobian_second,
            world_limit_offset,
            world_limit_count,
            limit_body_first,
            limit_body_second,
            limit_jacobian_first,
            limit_jacobian_second,
            inverse_weight,
            projected_twist,
            friction_reaction,
            contact_reaction,
            limit_reaction,
            prepared_status,
        ],
        outputs=[world_status],
        device=projected_twist.device,
    )


def sweep_constraints_sequential(
    world_active: wp.array[wp.bool],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_delassus: wp.array[wp.mat33f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    friction_velocity: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_velocity: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_velocity: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
) -> None:
    """Run one prepared sequential rigid sweep without reapplying warm starts."""
    wp.launch(
        _sweep_constraints_sequential_prepared,
        dim=world_active.shape[0],
        inputs=[
            world_active,
            world_friction_offset,
            world_friction_count,
            friction_body_first,
            friction_body_second,
            friction_jacobian_first,
            friction_jacobian_second,
            friction_impulse_bound,
            world_contact_offset,
            world_contact_count,
            contact_body_first,
            contact_body_second,
            contact_jacobian_first,
            contact_jacobian_second,
            contact_delassus,
            contact_bias,
            contact_friction,
            world_limit_offset,
            world_limit_count,
            limit_body_first,
            limit_body_second,
            limit_jacobian_first,
            limit_jacobian_second,
            limit_bias,
            inverse_weight,
            projected_twist,
        ],
        outputs=[
            friction_reaction,
            friction_velocity,
            contact_reaction,
            contact_velocity,
            limit_reaction,
            limit_velocity,
            world_status,
        ],
        device=projected_twist.device,
    )
