# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kamino container adapter for the LOX rigid-body backend.

The adapter owns persistent row and capacity mappings. Construction may read
immutable model arrays on the host; :meth:`LOXKaminoAdapter.update`
uses only fixed-size device operations so it remains suitable for graph
capture. Kamino contact vectors remain normal-last throughout this boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import warp as wp

from ...core.data import DataKamino
from ...core.joints import JointActuationType
from ...core.model import ModelKamino
from ...core.types import mat36f, mat66f, vec6f
from ...geometry.contacts import ContactMode, ContactsKamino
from ...kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians, SystemJacobiansType
from ...kinematics.limits import LimitsKamino
from .bias import compute_contact_velocity_target, compute_limit_velocity_target
from .iteration import SplittingState
from .metric import (
    METRIC_STATUS_VALID,
    compute_mass_split_metric,
)
from .problem import compute_body_explicit_wrench
from .projection import PROJECTION_STATUS_VALID
from .system import BatchedPrimalBodySystem

__all__ = ["LOXKaminoAdapter"]

wp.set_module_options({"enable_backward": False})


def _capacity_offsets(capacities: Sequence[int]) -> list[int]:
    offsets = [0]
    for capacity in capacities:
        offsets.append(offsets[-1] + capacity)
    return offsets


@wp.func
def _copy_screw(value: wp.spatial_vectorf) -> vec6f:
    result = vec6f(0.0)
    for index in range(6):
        result[index] = value[index]
    return result


@wp.func
def _make_spatial(value: vec6f) -> wp.spatial_vectorf:
    return wp.spatial_vectorf(value[0], value[1], value[2], value[3], value[4], value[5])


@wp.func
def _load_dense_jacobian_row(
    world: wp.int32,
    row: wp.int32,
    body_local: wp.int32,
    body_dofs: wp.array[wp.int32],
    jacobian_offsets: wp.array[wp.int32],
    jacobian_data: wp.array[wp.float32],
) -> vec6f:
    result = vec6f(0.0)
    if body_local >= 0:
        start = jacobian_offsets[world] + body_dofs[world] * row + 6 * body_local
        for index in range(6):
            result[index] = jacobian_data[start + index]
    return result


@wp.func
def _load_sparse_jacobian_row(index: wp.int32, jacobian_data: wp.array[vec6f]) -> vec6f:
    if index >= 0:
        return jacobian_data[index]
    return vec6f(0.0)


@wp.func
def _inverse_mass_quadratic_form(
    jacobian: vec6f,
    body: wp.int32,
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
) -> wp.float32:
    if body < 0:
        return 0.0
    linear = wp.vec3f(jacobian[0], jacobian[1], jacobian[2])
    angular = wp.vec3f(jacobian[3], jacobian[4], jacobian[5])
    return inverse_mass[body] * wp.dot(linear, linear) + wp.dot(angular, inverse_inertia_world[body] @ angular)


@wp.kernel
def _capture_body_velocity(
    source: wp.array[wp.spatial_vectorf],
    destination: wp.array[vec6f],
):
    body = wp.tid()
    destination[body] = _copy_screw(source[body])


@wp.kernel
def _capture_dynamic_joint_velocity(
    dof_index: wp.array[wp.int32],
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    row = wp.tid()
    destination[row] = source[dof_index[row]]


@wp.kernel
def _gather_body_explicit_wrenches(
    body_world: wp.array[wp.int32],
    mass: wp.array[wp.float32],
    inertia_world: wp.array[wp.mat33f],
    evaluation_velocity: wp.array[wp.spatial_vectorf],
    external_wrench: wp.array[wp.spatial_vectorf],
    actuation_wrench: wp.array[wp.spatial_vectorf],
    gravity: wp.array[wp.vec3f],
    explicit_wrench: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    gravity_vector = gravity[world]
    velocity_body = _copy_screw(evaluation_velocity[body])
    explicit_wrench[body] = compute_body_explicit_wrench(
        mass[body],
        inertia_world[body],
        velocity_body,
        _copy_screw(external_wrench[body]),
        _copy_screw(actuation_wrench[body]),
        gravity_vector,
    )


@wp.kernel
def _gather_dynamic_rows(
    row_world: wp.array[wp.int32],
    row_joint: wp.array[wp.int32],
    jacobian_row: wp.array[wp.int32],
    body_first_local: wp.array[wp.int32],
    body_second_local: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    value_index: wp.array[wp.int32],
    dof_index: wp.array[wp.int32],
    body_dofs: wp.array[wp.int32],
    jacobian_offsets: wp.array[wp.int32],
    jacobian_data: wp.array[wp.float32],
    sparse_jacobian: wp.bool,
    sparse_first_index: wp.array[wp.int32],
    sparse_second_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    joint_inertia: wp.array[wp.float32],
    joint_free_velocity: wp.array[wp.float32],
    joint_armature: wp.array[wp.float32],
    joint_position_stiffness: wp.array[wp.float32],
    joint_actuation_type: wp.array[wp.int32],
    joint_velocity: wp.array[wp.float32],
    joint_velocity_begin: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    time_step: wp.float32,
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    if sparse_jacobian:
        jacobian_first[row] = _load_sparse_jacobian_row(sparse_first_index[row], sparse_jacobian_data)
        jacobian_second[row] = _load_sparse_jacobian_row(sparse_second_index[row], sparse_jacobian_data)
    else:
        dense_row = jacobian_row[row]
        jacobian_first[row] = _load_dense_jacobian_row(
            world, dense_row, body_first_local[row], body_dofs, jacobian_offsets, jacobian_data
        )
        jacobian_second[row] = _load_dense_jacobian_row(
            world, dense_row, body_second_local[row], body_dofs, jacobian_offsets, jacobian_data
        )
    source = value_index[row]
    dof = dof_index[row]
    joint = row_joint[row]
    inertia = joint_inertia[source]
    effective_inertia[row] = inertia
    velocity = joint_free_velocity[source]
    if inertia > 0.0:
        velocity += joint_armature[dof] * (joint_velocity_begin[row] - joint_velocity[dof]) / inertia
        actuation_type = joint_actuation_type[joint]
        if (
            actuation_type == JointActuationType.POSITION
            or actuation_type == JointActuationType.POSITION_VELOCITY
            or actuation_type == JointActuationType.POSITION_VELOCITY_FORCE
        ):
            linearization_velocity = wp.float32(0.0)
            first = body_first_global[row]
            second = body_second_global[row]
            if first >= 0:
                linearization_velocity += wp.dot(jacobian_first[row], linearization_twist[first])
            if second >= 0:
                linearization_velocity += wp.dot(jacobian_second[row], linearization_twist[second])
            velocity += time_step * time_step * joint_position_stiffness[dof] * linearization_velocity / inertia
    free_velocity[row] = velocity


@wp.kernel
def _gather_joint_frictions(
    row_world: wp.array[wp.int32],
    jacobian_row: wp.array[wp.int32],
    body_first_local: wp.array[wp.int32],
    body_second_local: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    dof_index: wp.array[wp.int32],
    body_dofs: wp.array[wp.int32],
    jacobian_offsets: wp.array[wp.int32],
    jacobian_data: wp.array[wp.float32],
    sparse_jacobian: wp.bool,
    sparse_first_index: wp.array[wp.int32],
    sparse_second_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    friction_force: wp.array[wp.float32],
    body_velocity_begin: wp.array[vec6f],
    time_step: wp.float32,
    initialize: wp.bool,
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    impulse_bound: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    if sparse_jacobian:
        first_jacobian = _load_sparse_jacobian_row(sparse_first_index[row], sparse_jacobian_data)
        second_jacobian = _load_sparse_jacobian_row(sparse_second_index[row], sparse_jacobian_data)
    else:
        dense_row = jacobian_row[row]
        first_jacobian = _load_dense_jacobian_row(
            world, dense_row, body_first_local[row], body_dofs, jacobian_offsets, jacobian_data
        )
        second_jacobian = _load_dense_jacobian_row(
            world, dense_row, body_second_local[row], body_dofs, jacobian_offsets, jacobian_data
        )
    bound = time_step * friction_force[dof_index[row]]
    jacobian_first[row] = first_jacobian
    jacobian_second[row] = second_jacobian
    impulse_bound[row] = bound
    reaction[row] = wp.clamp(reaction[row], -bound, bound)
    if initialize:
        value = wp.float32(0.0)
        first = body_first_global[row]
        second = body_second_global[row]
        if first >= 0:
            value += wp.dot(first_jacobian, body_velocity_begin[first])
        if second >= 0:
            value += wp.dot(second_jacobian, body_velocity_begin[second])
        velocity[row] = value


@wp.kernel
def _gather_structural_rows(
    row_world: wp.array[wp.int32],
    jacobian_row: wp.array[wp.int32],
    body_first_local: wp.array[wp.int32],
    body_second_local: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    residual_index: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    body_dofs: wp.array[wp.int32],
    jacobian_offsets: wp.array[wp.int32],
    jacobian_data: wp.array[wp.float32],
    sparse_jacobian: wp.bool,
    sparse_first_index: wp.array[wp.int32],
    sparse_second_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    joint_residual: wp.array[wp.float32],
    joint_multiplier: wp.array[wp.float32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    residual: wp.array[wp.float32],
    multiplier: wp.array[wp.float32],
    effective_mass: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    if sparse_jacobian:
        first_jacobian = _load_sparse_jacobian_row(sparse_first_index[row], sparse_jacobian_data)
        second_jacobian = _load_sparse_jacobian_row(sparse_second_index[row], sparse_jacobian_data)
    else:
        dense_row = jacobian_row[row]
        first_jacobian = _load_dense_jacobian_row(
            world, dense_row, body_first_local[row], body_dofs, jacobian_offsets, jacobian_data
        )
        second_jacobian = _load_dense_jacobian_row(
            world, dense_row, body_second_local[row], body_dofs, jacobian_offsets, jacobian_data
        )
    jacobian_first[row] = first_jacobian
    jacobian_second[row] = second_jacobian
    residual[row] = joint_residual[residual_index[row]]
    # Kamino stores the reaction applied as +J^T lambda. The augmented
    # Lagrangian dual has the opposite sign in the smooth primal equation.
    multiplier[row] = -joint_multiplier[multiplier_index[row]]

    inverse_effective_mass = _inverse_mass_quadratic_form(
        first_jacobian, body_first_global[row], inverse_mass, inverse_inertia_world
    ) + _inverse_mass_quadratic_form(second_jacobian, body_second_global[row], inverse_mass, inverse_inertia_world)
    effective_mass[row] = 1.0 / inverse_effective_mass if inverse_effective_mass > 1.0e-12 else 0.0


@wp.func
def _inverse_mass_bilinear_form(
    first: vec6f,
    second: vec6f,
    body: wp.int32,
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
) -> wp.float32:
    if body < 0:
        return 0.0
    first_linear = wp.vec3f(first[0], first[1], first[2])
    second_linear = wp.vec3f(second[0], second[1], second[2])
    first_angular = wp.vec3f(first[3], first[4], first[5])
    second_angular = wp.vec3f(second[3], second[4], second[5])
    return inverse_mass[body] * wp.dot(first_linear, second_linear) + wp.dot(
        first_angular, inverse_inertia_world[body] @ second_angular
    )


@wp.kernel
def _compute_structural_metrics(
    block_world: wp.array[wp.int32],
    block_row_offset: wp.array[wp.int32],
    block_row_count: wp.array[wp.int32],
    block_body_first: wp.array[wp.int32],
    block_body_second: wp.array[wp.int32],
    body_incidence: wp.array[wp.int32],
    apply_mass_split: wp.bool,
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    time_step: wp.float32,
    penalty_scale: wp.array[wp.float32],
    block_body_first_multiplicity: wp.array[wp.int32],
    block_body_second_multiplicity: wp.array[wp.int32],
    block_multiplicity: wp.array[wp.int32],
    physical_delassus: wp.array[mat66f],
    inverse_metric: wp.array[mat66f],
    penalty_metric: wp.array[mat66f],
    metric_status: wp.array[wp.int32],
    row_effective_mass: wp.array[wp.float32],
    row_penalty: wp.array[wp.float32],
):
    block = wp.tid()
    world = block_world[block]
    offset = block_row_offset[block]
    count = block_row_count[block]
    first_body = block_body_first[block]
    second_body = block_body_second[block]
    first_multiplicity = int(0)
    second_multiplicity = int(0)
    if first_body >= 0:
        first_multiplicity = 1
        if apply_mass_split:
            first_multiplicity = wp.max(1, body_incidence[first_body])
    if second_body >= 0:
        second_multiplicity = 1
        if apply_mass_split:
            second_multiplicity = wp.max(1, body_incidence[second_body])
    block_body_first_multiplicity[block] = first_multiplicity
    block_body_second_multiplicity[block] = second_multiplicity
    block_multiplicity[block] = wp.max(1, wp.max(first_multiplicity, second_multiplicity))

    delassus = mat66f(0.0)
    split_delassus = mat66f(0.0)
    for row in range(6):
        if row < count:
            first_row = jacobian_first[offset + row]
            second_row = jacobian_second[offset + row]
            for col in range(6):
                if col < count:
                    first_contribution = _inverse_mass_bilinear_form(
                        first_row, jacobian_first[offset + col], first_body, inverse_mass, inverse_inertia_world
                    )
                    second_contribution = _inverse_mass_bilinear_form(
                        second_row, jacobian_second[offset + col], second_body, inverse_mass, inverse_inertia_world
                    )
                    delassus[row, col] = first_contribution + second_contribution
                    split_delassus[row, col] = (
                        wp.float32(first_multiplicity) * first_contribution
                        + wp.float32(second_multiplicity) * second_contribution
                    )

    result = compute_mass_split_metric(split_delassus, count, 1, penalty_scale[world] / (time_step * time_step))
    physical_delassus[block] = 0.5 * (delassus + wp.transpose(delassus))
    inverse_metric[block] = result.inverse
    penalty_metric[block] = result.penalty
    metric_status[block] = result.status
    for row in range(6):
        if row < count:
            row_effective_mass[offset + row] = result.inverse[row, row]
            row_penalty[offset + row] = result.penalty[row, row]


@wp.kernel
def _accumulate_constraint_incidence(
    entity_world: wp.array[wp.int32],
    entity_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    body_constraint_count: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
):
    entity = wp.tid()
    world = entity_world[entity]
    if entity_local[entity] >= world_count[world]:
        return
    first = body_first[entity]
    second = body_second[entity]
    if first >= 0:
        wp.atomic_add(body_constraint_count, first, 1)
        wp.atomic_max(body_has_unilateral, first, 1)
    if second >= 0 and second != first:
        wp.atomic_add(body_constraint_count, second, 1)
        wp.atomic_max(body_has_unilateral, second, 1)


@wp.kernel
def _clear_inactive_limits(
    entity_world: wp.array[wp.int32],
    entity_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    limit = wp.tid()
    if entity_local[limit] >= world_count[entity_world[limit]]:
        body_first[limit] = -1
        body_second[limit] = -1
        reaction[limit] = 0.0
        velocity[limit] = 0.0


@wp.kernel
def _clear_inactive_contacts(
    entity_world: wp.array[wp.int32],
    entity_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
):
    contact = wp.tid()
    if entity_local[contact] >= world_count[entity_world[contact]]:
        body_first[contact] = -1
        body_second[contact] = -1
        reaction[contact] = wp.vec3f(0.0)
        velocity[contact] = wp.vec3f(0.0)


@wp.kernel
def _copy_clamped_world_counts(
    source: wp.array[wp.int32],
    capacity: wp.array[wp.int32],
    destination: wp.array[wp.int32],
):
    world = wp.tid()
    destination[world] = wp.min(wp.max(source[world], 0), capacity[world])


@wp.kernel
def _mark_worlds_with_unilaterals(
    contact_count: wp.array[wp.int32],
    limit_count: wp.array[wp.int32],
    friction_count: wp.array[wp.int32],
    world_has_unilateral: wp.array[wp.bool],
):
    world = wp.tid()
    world_has_unilateral[world] = contact_count[world] > 0 or limit_count[world] > 0 or friction_count[world] > 0


@wp.kernel
def _gather_limits(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_local: wp.array[wp.int32],
    source_bodies: wp.array[wp.vec2i],
    source_violation: wp.array[wp.float32],
    source_reaction: wp.array[wp.float32],
    body_velocity_begin: wp.array[vec6f],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    body_offset: wp.array[wp.int32],
    body_dofs: wp.array[wp.int32],
    constraint_group_offset: wp.array[wp.int32],
    jacobian_offsets: wp.array[wp.int32],
    jacobian_data: wp.array[wp.float32],
    sparse_jacobian: wp.bool,
    sparse_jacobian_offsets: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    time_step: wp.float32,
    stabilization_fraction: wp.float32,
    import_reactions: wp.bool,
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    bias: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    world = source_world[source]
    local = source_local[source]
    if world < 0 or world >= world_capacity.shape[0] or local < 0 or local >= world_capacity[world]:
        return

    destination = world_offset[world] + local
    bodies = source_bodies[source]
    first_global = bodies[0]
    second_global = bodies[1]
    first_local = first_global - body_offset[world] if first_global >= 0 else -1
    second_local = second_global - body_offset[world] if second_global >= 0 else -1
    if sparse_jacobian:
        sparse_offset = sparse_jacobian_offsets[source]
        first_sparse_index = sparse_offset + 1 if first_global >= 0 else -1
        first_jacobian = _load_sparse_jacobian_row(first_sparse_index, sparse_jacobian_data)
        second_jacobian = _load_sparse_jacobian_row(sparse_offset, sparse_jacobian_data)
    else:
        dense_row = constraint_group_offset[world] + local
        first_jacobian = _load_dense_jacobian_row(
            world, dense_row, first_local, body_dofs, jacobian_offsets, jacobian_data
        )
        second_jacobian = _load_dense_jacobian_row(
            world, dense_row, second_local, body_dofs, jacobian_offsets, jacobian_data
        )
    body_first[destination] = first_global
    body_second[destination] = second_global
    jacobian_first[destination] = first_jacobian
    jacobian_second[destination] = second_jacobian
    velocity_previous = wp.float32(0.0)
    if first_global >= 0:
        velocity_previous += wp.dot(first_jacobian, body_velocity_begin[first_global])
    if second_global >= 0:
        velocity_previous += wp.dot(second_jacobian, body_velocity_begin[second_global])
    target = compute_limit_velocity_target(source_violation[source], time_step, stabilization_fraction)
    bias[destination] = -target
    if import_reactions:
        reaction[destination] = time_step * source_reaction[source]
        velocity[destination] = velocity_previous


@wp.kernel
def _gather_contacts(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_local: wp.array[wp.int32],
    source_bodies: wp.array[wp.vec2i],
    source_gap: wp.array[wp.vec4f],
    source_material: wp.array[wp.vec2f],
    source_reaction: wp.array[wp.vec3f],
    body_velocity_begin: wp.array[vec6f],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    body_offset: wp.array[wp.int32],
    body_dofs: wp.array[wp.int32],
    constraint_group_offset: wp.array[wp.int32],
    jacobian_offsets: wp.array[wp.int32],
    jacobian_data: wp.array[wp.float32],
    sparse_jacobian: wp.bool,
    sparse_jacobian_offsets: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    time_step: wp.float32,
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
    impact_velocity_threshold: wp.float32,
    import_reactions: wp.bool,
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[mat36f],
    jacobian_second: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    world = source_world[source]
    local = source_local[source]
    if world < 0 or world >= world_capacity.shape[0] or local < 0 or local >= world_capacity[world]:
        return

    destination = world_offset[world] + local
    bodies = source_bodies[source]
    first_global = bodies[0]
    second_global = bodies[1]
    first_local = first_global - body_offset[world] if first_global >= 0 else -1
    second_local = second_global - body_offset[world] if second_global >= 0 else -1
    first_jacobian = mat36f(0.0)
    second_jacobian = mat36f(0.0)
    sparse_offset = wp.int32(0)
    dense_row = wp.int32(0)
    if sparse_jacobian:
        sparse_offset = sparse_jacobian_offsets[source]
    else:
        dense_row = constraint_group_offset[world] + 3 * local
    for component in range(3):
        first_row = vec6f(0.0)
        second_row = vec6f(0.0)
        if sparse_jacobian:
            first_sparse_index = sparse_offset + 3 + component if first_global >= 0 else -1
            first_row = _load_sparse_jacobian_row(first_sparse_index, sparse_jacobian_data)
            second_row = _load_sparse_jacobian_row(sparse_offset + component, sparse_jacobian_data)
        else:
            first_row = _load_dense_jacobian_row(
                world, dense_row + component, first_local, body_dofs, jacobian_offsets, jacobian_data
            )
            second_row = _load_dense_jacobian_row(
                world, dense_row + component, second_local, body_dofs, jacobian_offsets, jacobian_data
            )
        for dof in range(6):
            first_jacobian[component, dof] = first_row[dof]
            second_jacobian[component, dof] = second_row[dof]

    velocity_previous = wp.vec3f(0.0)
    if first_global >= 0:
        velocity_previous += first_jacobian @ body_velocity_begin[first_global]
    if second_global >= 0:
        velocity_previous += second_jacobian @ body_velocity_begin[second_global]
    gap = source_gap[source]
    material = source_material[source]
    target = compute_contact_velocity_target(
        gap[3],
        velocity_previous[2],
        material[1],
        time_step,
        stabilization_fraction,
        dead_zone,
        impact_velocity_threshold,
    )

    body_first[destination] = first_global
    body_second[destination] = second_global
    jacobian_first[destination] = first_jacobian
    jacobian_second[destination] = second_jacobian
    bias[destination] = wp.vec3f(0.0, 0.0, -target)
    friction[destination] = material[0]
    if import_reactions:
        reaction[destination] = time_step * source_reaction[source]
        velocity[destination] = velocity_previous


@wp.kernel
def _unpack_body_vector(
    body_vector_index: wp.array[wp.int32],
    source: wp.array[wp.float32],
    destination: wp.array[vec6f],
):
    body = wp.tid()
    source_offset = body_vector_index[body]
    value = vec6f(0.0)
    for index in range(6):
        value[index] = source[source_offset + index]
    destination[body] = value


@wp.kernel
def _write_body_velocity(
    source: wp.array[vec6f],
    destination: wp.array[wp.spatial_vectorf],
):
    body = wp.tid()
    value = source[body]
    destination[body] = wp.spatial_vectorf(value[0], value[1], value[2], value[3], value[4], value[5])


@wp.kernel
def _write_structural_multipliers(
    multiplier_index: wp.array[wp.int32],
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    row = wp.tid()
    destination[multiplier_index[row]] = -source[row]


@wp.kernel
def _scale_structural_multipliers(
    scale: wp.float32,
    multiplier_index: wp.array[wp.int32],
    multiplier: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    row = wp.tid()
    scaled = scale * multiplier[row]
    multiplier[row] = scaled
    destination[multiplier_index[row]] = -scaled


@wp.kernel
def _gather_structural_residuals(
    residual_index: wp.array[wp.int32],
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    row = wp.tid()
    destination[row] = source[residual_index[row]]


@wp.kernel
def _write_dynamic_multipliers(
    inverse_time_step: wp.float32,
    multiplier_index: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    body_velocity: wp.array[vec6f],
    destination: wp.array[wp.float32],
):
    row = wp.tid()
    velocity = wp.float32(0.0)
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        velocity += wp.dot(jacobian_first[row], body_velocity[first])
    if second >= 0:
        velocity += wp.dot(jacobian_second[row], body_velocity[second])
    destination[multiplier_index[row]] = inverse_time_step * effective_inertia[row] * (free_velocity[row] - velocity)


@wp.kernel
def _accumulate_joint_wrenches(
    multiplier_index: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    multiplier: wp.array[wp.float32],
    destination: wp.array[wp.spatial_vectorf],
):
    row = wp.tid()
    scale = multiplier[multiplier_index[row]]
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        wp.atomic_add(destination, first, _make_spatial(scale * jacobian_first[row]))
    if second >= 0:
        wp.atomic_add(destination, second, _make_spatial(scale * jacobian_second[row]))


@wp.kernel
def _accumulate_joint_friction_wrenches(
    inverse_time_step: wp.float32,
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    destination: wp.array[wp.spatial_vectorf],
):
    row = wp.tid()
    force = inverse_time_step * reaction[row]
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        wp.atomic_add(destination, first, _make_spatial(force * jacobian_first[row]))
    if second >= 0:
        wp.atomic_add(destination, second, _make_spatial(force * jacobian_second[row]))


@wp.kernel
def _accumulate_limit_wrenches(
    inverse_time_step: wp.float32,
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_reaction: wp.array[wp.float32],
    limit_wrench: wp.array[wp.spatial_vectorf],
):
    limit = wp.tid()
    world = limit_world[limit]
    if limit_local[limit] >= world_count[world]:
        return
    limit_force = inverse_time_step * limit_reaction[limit]
    first = limit_body_first[limit]
    second = limit_body_second[limit]
    if first >= 0:
        wp.atomic_add(limit_wrench, first, _make_spatial(limit_force * limit_jacobian_first[limit]))
    if second >= 0:
        wp.atomic_add(limit_wrench, second, _make_spatial(limit_force * limit_jacobian_second[limit]))


@wp.kernel
def _accumulate_contact_wrenches(
    inverse_time_step: wp.float32,
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_reaction: wp.array[wp.vec3f],
    contact_wrench: wp.array[wp.spatial_vectorf],
):
    contact = wp.tid()
    world = contact_world[contact]
    if contact_local[contact] >= world_count[world]:
        return
    contact_force = inverse_time_step * contact_reaction[contact]
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first >= 0:
        wp.atomic_add(
            contact_wrench,
            first,
            _make_spatial(wp.transpose(contact_jacobian_first[contact]) @ contact_force),
        )
    if second >= 0:
        wp.atomic_add(
            contact_wrench,
            second,
            _make_spatial(wp.transpose(contact_jacobian_second[contact]) @ contact_force),
        )


@wp.kernel
def _update_structural_multipliers_from_twist_rows(
    time_step: wp.float32,
    structural_tolerance: wp.float32,
    row_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    residual: wp.array[wp.float32],
    penalty: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    projected_fraction: wp.float32,
    body_vector_index: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    multiplier: wp.array[wp.float32],
    candidate_residual: wp.array[wp.float32],
    world_residual: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return

    first_global = body_first_global[row]
    second_global = body_second_global[row]
    global_velocity = wp.float32(0.0)
    projected_velocity = wp.float32(0.0)
    linearization_velocity = wp.float32(0.0)
    if first_global >= 0:
        global_velocity += wp.dot(jacobian_first[row], global_twist[first_global])
        if projected_fraction > 0.0:
            projected_velocity += wp.dot(jacobian_first[row], projected_twist[first_global])
        linearization_velocity += wp.dot(jacobian_first[row], linearization_twist[first_global])
    if second_global >= 0:
        global_velocity += wp.dot(jacobian_second[row], global_twist[second_global])
        if projected_fraction > 0.0:
            projected_velocity += wp.dot(jacobian_second[row], projected_twist[second_global])
        linearization_velocity += wp.dot(jacobian_second[row], linearization_twist[second_global])

    global_residual = residual[row] + time_step * (global_velocity - linearization_velocity)
    candidate_residual[row] = global_residual
    wp.atomic_max(world_residual, world, wp.abs(global_residual) / structural_tolerance)

    update_residual = global_residual
    if projected_fraction > 0.0:
        update_residual += projected_fraction * time_step * (projected_velocity - global_velocity)
    multiplier_delta = penalty[row] * update_residual
    updated = multiplier[row] + multiplier_delta
    multiplier[row] = updated
    destination[multiplier_index[row]] = -updated

    for axis in range(6):
        if first_global >= 0:
            wp.atomic_add(
                right_hand_side,
                body_vector_index[first_global] + axis,
                -time_step * multiplier_delta * jacobian_first[row][axis],
            )
        if second_global >= 0:
            wp.atomic_add(
                right_hand_side,
                body_vector_index[second_global] + axis,
                -time_step * multiplier_delta * jacobian_second[row][axis],
            )


@wp.kernel
def _update_structural_multipliers_from_twist(
    time_step: wp.float32,
    structural_tolerance: wp.float32,
    block_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    block_row_offset: wp.array[wp.int32],
    block_row_count: wp.array[wp.int32],
    block_body_first_global: wp.array[wp.int32],
    block_body_second_global: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    residual: wp.array[wp.float32],
    penalty: wp.array[mat66f],
    metric_status: wp.array[wp.int32],
    linearization_twist: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    projected_fraction: wp.float32,
    body_vector_index: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    multiplier: wp.array[wp.float32],
    candidate_residual: wp.array[wp.float32],
    projected_residual: wp.array[wp.float32],
    world_residual: wp.array[wp.float32],
    world_projected_residual: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    block = wp.tid()
    world = block_world[block]
    if (
        not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
        or metric_status[block] != METRIC_STATUS_VALID
    ):
        return

    offset = block_row_offset[block]
    count = block_row_count[block]
    first_global = block_body_first_global[block]
    second_global = block_body_second_global[block]
    global_block_residual = vec6f(0.0)
    projected_block_residual = vec6f(0.0)
    for row in range(6):
        if row < count:
            global_velocity = wp.float32(0.0)
            projected_velocity = wp.float32(0.0)
            linearization_velocity = wp.float32(0.0)
            if first_global >= 0:
                global_velocity += wp.dot(jacobian_first[offset + row], global_twist[first_global])
                projected_velocity += wp.dot(jacobian_first[offset + row], projected_twist[first_global])
                linearization_velocity += wp.dot(jacobian_first[offset + row], linearization_twist[first_global])
            if second_global >= 0:
                global_velocity += wp.dot(jacobian_second[offset + row], global_twist[second_global])
                projected_velocity += wp.dot(jacobian_second[offset + row], projected_twist[second_global])
                linearization_velocity += wp.dot(jacobian_second[offset + row], linearization_twist[second_global])
            global_value = residual[offset + row] + time_step * (global_velocity - linearization_velocity)
            projected_value = residual[offset + row] + time_step * (projected_velocity - linearization_velocity)
            global_block_residual[row] = global_value
            projected_block_residual[row] = projected_value
            candidate_residual[offset + row] = global_value
            projected_residual[offset + row] = projected_value
            wp.atomic_max(world_residual, world, wp.abs(global_value) / structural_tolerance)
            wp.atomic_max(world_projected_residual, world, wp.abs(projected_value) / structural_tolerance)

    update_residual = global_block_residual + projected_fraction * (projected_block_residual - global_block_residual)
    multiplier_delta = penalty[block] @ update_residual
    for row in range(6):
        if row < count:
            updated = multiplier[offset + row] + multiplier_delta[row]
            multiplier[offset + row] = updated
            destination[multiplier_index[offset + row]] = -updated
            for axis in range(6):
                if first_global >= 0:
                    wp.atomic_add(
                        right_hand_side,
                        body_vector_index[first_global] + axis,
                        -time_step * multiplier_delta[row] * jacobian_first[offset + row][axis],
                    )
                if second_global >= 0:
                    wp.atomic_add(
                        right_hand_side,
                        body_vector_index[second_global] + axis,
                        -time_step * multiplier_delta[row] * jacobian_second[offset + row][axis],
                    )


@wp.kernel
def _evaluate_structural_residual_from_twist(
    time_step: wp.float32,
    structural_tolerance: wp.float32,
    row_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    residual: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    twist: wp.array[vec6f],
    candidate_residual: wp.array[wp.float32],
    world_residual: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return

    candidate_velocity = wp.float32(0.0)
    linearization_velocity = wp.float32(0.0)
    first = body_first_global[row]
    second = body_second_global[row]
    if first >= 0:
        candidate_velocity += wp.dot(jacobian_first[row], twist[first])
        linearization_velocity += wp.dot(jacobian_first[row], linearization_twist[first])
    if second >= 0:
        candidate_velocity += wp.dot(jacobian_second[row], twist[second])
        linearization_velocity += wp.dot(jacobian_second[row], linearization_twist[second])

    row_residual = residual[row] + time_step * (candidate_velocity - linearization_velocity)
    candidate_residual[row] = row_residual
    wp.atomic_max(world_residual, world, wp.abs(row_residual) / structural_tolerance)


@wp.kernel
def _write_limit_outputs(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_local: wp.array[wp.int32],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    inverse_time_step: wp.float32,
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
    destination_reaction: wp.array[wp.float32],
    destination_velocity: wp.array[wp.float32],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    world = source_world[source]
    local = source_local[source]
    if world < 0 or world >= world_capacity.shape[0] or local < 0 or local >= world_capacity[world]:
        return
    internal = world_offset[world] + local
    destination_reaction[source] = inverse_time_step * reaction[internal]
    destination_velocity[source] = velocity[internal]


@wp.kernel
def _write_contact_outputs(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_local: wp.array[wp.int32],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    inverse_time_step: wp.float32,
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
    destination_reaction: wp.array[wp.vec3f],
    destination_velocity: wp.array[wp.vec3f],
    destination_mode: wp.array[wp.int32],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    world = source_world[source]
    local = source_local[source]
    if world < 0 or world >= world_capacity.shape[0] or local < 0 or local >= world_capacity[world]:
        return
    internal = world_offset[world] + local
    destination_reaction[source] = inverse_time_step * reaction[internal]
    destination_velocity[source] = velocity[internal]
    destination_mode[source] = wp.static(ContactMode.make_compute_mode_func())(velocity[internal])


class LOXKaminoAdapter:
    """Persistent device adapter from Kamino containers to solver arrays."""

    def __init__(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: SystemJacobiansType,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        if not isinstance(model, ModelKamino):
            raise TypeError("model must be a ModelKamino instance.")
        if not isinstance(data, DataKamino):
            raise TypeError("data must be a DataKamino instance.")
        if not isinstance(jacobians, (DenseSystemJacobians, SparseSystemJacobians)):
            raise TypeError("jacobians must be a DenseSystemJacobians or SparseSystemJacobians instance.")
        if limits is not None and not isinstance(limits, LimitsKamino):
            raise TypeError("limits must be a LimitsKamino instance or None.")
        if contacts is not None and not isinstance(contacts, ContactsKamino):
            raise TypeError("contacts must be a ContactsKamino instance or None.")
        if data.device != model.device:
            raise ValueError("model and data must be allocated on the same device.")

        self.model = model
        self.data = data
        self.jacobians = jacobians
        self.limits = limits
        self.contacts = contacts
        self.device = wp.get_device(model.device)
        self.num_worlds = model.info.num_worlds
        self.sparse_jacobian = isinstance(jacobians, SparseSystemJacobians)

        empty_int = wp.empty(0, dtype=wp.int32, device=self.device)
        empty_float = wp.empty(0, dtype=wp.float32, device=self.device)
        empty_vec6 = wp.empty(0, dtype=vec6f, device=self.device)
        if isinstance(jacobians, DenseSystemJacobians):
            self._dense_jacobian_offsets = jacobians.data.J_cts_offsets
            self._dense_jacobian_data = jacobians.data.J_cts_data
            self._dense_dof_jacobian_offsets = jacobians.data.J_dofs_offsets
            self._dense_dof_jacobian_data = jacobians.data.J_dofs_data
            self._sparse_jacobian_data = empty_vec6
            self._sparse_dof_jacobian_data = empty_vec6
            self._sparse_limit_offsets = empty_int
            self._sparse_contact_offsets = empty_int
        else:
            if jacobians._J_cts is None or jacobians._J_dofs is None:
                raise RuntimeError("Sparse Jacobians must be finalized before constructing the adapter.")
            self._dense_jacobian_offsets = empty_int
            self._dense_jacobian_data = empty_float
            self._dense_dof_jacobian_offsets = empty_int
            self._dense_dof_jacobian_data = empty_float
            self._sparse_jacobian_data = jacobians._J_cts.bsm.nzb_values
            self._sparse_dof_jacobian_data = jacobians._J_dofs.bsm.nzb_values
            self._sparse_limit_offsets = jacobians._J_cts_limit_nzb_offsets
            self._sparse_contact_offsets = jacobians._J_cts_contact_nzb_offsets

        body_counts = tuple(int(value) for value in model.info.num_bodies.numpy().tolist())
        body_components = self._build_body_components()
        self.system = BatchedPrimalBodySystem(
            body_counts,
            body_components=body_components,
            device=self.device,
        )
        self.splitting = SplittingState(body_counts, device=self.device)
        self.world_active = self.splitting.world_active
        self.projected_twist = self.splitting.projected_twist
        self.system_solution_twist = wp.zeros(model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        self.body_velocity_begin = wp.zeros(model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        self.body_linearization_twist = wp.zeros(model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        self.body_explicit_wrench = wp.zeros(model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        self._limit_stabilization_fraction = 0.01
        self._contact_stabilization_fraction = 0.01
        self._contact_dead_zone = 1.0e-6
        self._impact_velocity_threshold = 1.0e-3
        self._block_joint_metrics = True
        self._uniform_joint_penalty_scale = wp.ones(self.num_worlds, dtype=wp.float32, device=self.device)

        self._allocate_joint_rows()
        self._allocate_joint_frictions()
        self._allocate_unilaterals()

    @staticmethod
    def _device_array(values: Sequence[int], device: wp.DeviceLike) -> wp.array[wp.int32]:
        return wp.array(values, dtype=wp.int32, device=device)

    def _build_body_components(self) -> tuple[tuple[int, ...], ...]:
        """Return joint-connected body components without contact edges."""
        body_count = self.model.size.sum_of_num_bodies
        parent = list(range(body_count))

        def find(body: int) -> int:
            root = body
            while parent[root] != root:
                root = parent[root]
            while parent[body] != body:
                next_body = parent[body]
                parent[body] = root
                body = next_body
            return root

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        joint_first = self.model.joints.bid_B.numpy().astype(int).tolist()
        joint_second = self.model.joints.bid_F.numpy().astype(int).tolist()
        for first, second in zip(joint_first, joint_second, strict=True):
            if first >= 0 and second >= 0:
                union(first, second)

        components: dict[int, list[int]] = {}
        for body in range(body_count):
            components.setdefault(find(body), []).append(body)
        return tuple(tuple(component) for component in components.values())

    def _allocate_joint_rows(self) -> None:
        model = self.model
        bodies_offset = model.info.bodies_offset.numpy().tolist()
        joint_world = model.joints.wid.numpy().tolist()
        joint_body_first = model.joints.bid_B.numpy().tolist()
        joint_body_second = model.joints.bid_F.numpy().tolist()
        joint_dof_count = model.joints.num_dofs.numpy().tolist()
        joint_dynamic_count = model.joints.num_dynamic_cts.numpy().tolist()
        joint_structural_count = model.joints.num_kinematic_cts.numpy().tolist()
        joint_dynamic_offset = model.joints.dynamic_cts_offset.numpy().tolist()
        joint_dof_offset = model.joints.dofs_offset.numpy().tolist()
        joint_structural_offset = model.joints.kinematic_cts_offset.numpy().tolist()
        world_dynamic_offset = model.info.joint_dynamic_cts_offset.numpy().tolist()
        world_structural_offset = model.info.joint_kinematic_cts_offset.numpy().tolist()
        world_joint_offset = model.info.joint_cts_offset.numpy().tolist()
        world_dynamic_count = model.info.num_joint_dynamic_cts.numpy().tolist()

        dynamic_world: list[int] = []
        dynamic_joint: list[int] = []
        dynamic_dense_row: list[int] = []
        dynamic_first_local: list[int] = []
        dynamic_second_local: list[int] = []
        dynamic_first_global: list[int] = []
        dynamic_second_global: list[int] = []
        dynamic_value_index: list[int] = []
        dynamic_dof_index: list[int] = []
        dynamic_multiplier_index: list[int] = []
        dynamic_sparse_first_index: list[int] = []
        dynamic_sparse_second_index: list[int] = []
        structural_world: list[int] = []
        structural_dense_row: list[int] = []
        structural_first_local: list[int] = []
        structural_second_local: list[int] = []
        structural_first_global: list[int] = []
        structural_second_global: list[int] = []
        structural_residual_index: list[int] = []
        structural_multiplier_index: list[int] = []
        structural_sparse_first_index: list[int] = []
        structural_sparse_second_index: list[int] = []
        structural_vector_index: list[int] = []
        structural_row_block: list[int] = []
        structural_block_world: list[int] = []
        structural_block_joint: list[int] = []
        structural_block_row_offset: list[int] = []
        structural_block_row_count: list[int] = []
        structural_block_first_local: list[int] = []
        structural_block_second_local: list[int] = []
        structural_block_first_global: list[int] = []
        structural_block_second_global: list[int] = []
        static_constraint_count = [0] * model.size.sum_of_num_bodies
        world_structural_count = model.info.num_joint_kinematic_cts.numpy().astype(int).tolist()
        structural_storage_offsets = [0]
        for count in world_structural_count:
            structural_storage_offsets.append(structural_storage_offsets[-1] + max(1, count))
        structural_world_local_count = [0] * self.num_worlds
        sparse_joint_offsets: list[int] | None = None
        if isinstance(self.jacobians, SparseSystemJacobians):
            sparse_joint_offsets = self.jacobians._J_cts_joint_nzb_offsets.numpy().tolist()

        for joint in range(model.size.sum_of_num_joints):
            world = int(joint_world[joint])
            body_offset = int(bodies_offset[world])
            first_global = int(joint_body_first[joint])
            second_global = int(joint_body_second[joint])
            first_local = first_global - body_offset if first_global >= 0 else -1
            second_local = second_global - body_offset if second_global >= 0 else -1
            sparse_joint_start = sparse_joint_offsets[joint] if sparse_joint_offsets is not None else -1
            adjacent_body_count = 2 if first_global >= 0 else 1
            dof_count = int(joint_dof_count[joint])

            dynamic_start = int(joint_dynamic_offset[joint])
            dynamic_start_local = dynamic_start - int(world_dynamic_offset[world])
            dynamic_count = int(joint_dynamic_count[joint])
            for local_row in range(dynamic_count):
                dynamic_world.append(world)
                dynamic_joint.append(joint)
                dynamic_dense_row.append(dynamic_start_local + local_row)
                dynamic_first_local.append(first_local)
                dynamic_second_local.append(second_local)
                dynamic_first_global.append(first_global)
                dynamic_second_global.append(second_global)
                dynamic_value_index.append(dynamic_start + local_row)
                dynamic_dof_index.append(int(joint_dof_offset[joint]) + local_row)
                dynamic_multiplier_index.append(int(world_joint_offset[world]) + dynamic_start_local + local_row)
                dynamic_sparse_first_index.append(
                    sparse_joint_start + dof_count + local_row
                    if sparse_joint_offsets is not None and first_global >= 0
                    else -1
                )
                dynamic_sparse_second_index.append(
                    sparse_joint_start + local_row if sparse_joint_offsets is not None else -1
                )

            structural_start = int(joint_structural_offset[joint])
            structural_start_local = structural_start - int(world_structural_offset[world])
            structural_group_start = int(world_dynamic_count[world])
            structural_count = int(joint_structural_count[joint])
            sparse_structural_start = -1
            if sparse_joint_offsets is not None:
                sparse_structural_start = sparse_joint_start
                if dynamic_count > 0:
                    sparse_structural_start += adjacent_body_count * dof_count
            if structural_count > 0:
                block = len(structural_block_world)
                structural_block_world.append(world)
                structural_block_joint.append(joint)
                structural_block_row_offset.append(len(structural_world))
                structural_block_row_count.append(structural_count)
                structural_block_first_local.append(first_local)
                structural_block_second_local.append(second_local)
                structural_block_first_global.append(first_global)
                structural_block_second_global.append(second_global)
                if first_global >= 0:
                    static_constraint_count[first_global] += 1
                if second_global >= 0 and second_global != first_global:
                    static_constraint_count[second_global] += 1
            for local_row in range(structural_count):
                structural_world.append(world)
                structural_vector_index.append(structural_storage_offsets[world] + structural_world_local_count[world])
                structural_world_local_count[world] += 1
                structural_dense_row.append(structural_group_start + structural_start_local + local_row)
                structural_first_local.append(first_local)
                structural_second_local.append(second_local)
                structural_first_global.append(first_global)
                structural_second_global.append(second_global)
                structural_residual_index.append(structural_start + local_row)
                structural_multiplier_index.append(
                    int(world_joint_offset[world]) + structural_group_start + structural_start_local + local_row
                )
                structural_sparse_first_index.append(
                    sparse_structural_start + structural_count + local_row
                    if sparse_joint_offsets is not None and first_global >= 0
                    else -1
                )
                structural_sparse_second_index.append(
                    sparse_structural_start + local_row if sparse_joint_offsets is not None else -1
                )
                structural_row_block.append(block)

        self.dynamic_row_count = len(dynamic_world)
        self.dynamic_row_world = self._device_array(dynamic_world, self.device)
        self.dynamic_row_joint = self._device_array(dynamic_joint, self.device)
        self.dynamic_jacobian_row = self._device_array(dynamic_dense_row, self.device)
        self.dynamic_body_first = self._device_array(dynamic_first_local, self.device)
        self.dynamic_body_second = self._device_array(dynamic_second_local, self.device)
        self.dynamic_body_first_global = self._device_array(dynamic_first_global, self.device)
        self.dynamic_body_second_global = self._device_array(dynamic_second_global, self.device)
        self.dynamic_value_index = self._device_array(dynamic_value_index, self.device)
        self.dynamic_dof_index = self._device_array(dynamic_dof_index, self.device)
        self.dynamic_multiplier_index = self._device_array(dynamic_multiplier_index, self.device)
        self.dynamic_sparse_first_index = self._device_array(dynamic_sparse_first_index, self.device)
        self.dynamic_sparse_second_index = self._device_array(dynamic_sparse_second_index, self.device)
        self.dynamic_jacobian_first = wp.zeros(self.dynamic_row_count, dtype=vec6f, device=self.device)
        self.dynamic_jacobian_second = wp.zeros(self.dynamic_row_count, dtype=vec6f, device=self.device)
        self.dynamic_effective_inertia = wp.zeros(self.dynamic_row_count, dtype=wp.float32, device=self.device)
        self.dynamic_free_velocity = wp.zeros(self.dynamic_row_count, dtype=wp.float32, device=self.device)
        self.dynamic_velocity_begin = wp.zeros(self.dynamic_row_count, dtype=wp.float32, device=self.device)

        self.structural_row_count = len(structural_world)
        self.structural_row_world = self._device_array(structural_world, self.device)
        self.structural_row_counts = tuple(world_structural_count)
        self.structural_vector_index = self._device_array(structural_vector_index, self.device)
        self.structural_jacobian_row = self._device_array(structural_dense_row, self.device)
        self.structural_body_first = self._device_array(structural_first_local, self.device)
        self.structural_body_second = self._device_array(structural_second_local, self.device)
        self.structural_body_first_global = self._device_array(structural_first_global, self.device)
        self.structural_body_second_global = self._device_array(structural_second_global, self.device)
        self.structural_residual_index = self._device_array(structural_residual_index, self.device)
        self.structural_multiplier_index = self._device_array(structural_multiplier_index, self.device)
        self.structural_sparse_first_index = self._device_array(structural_sparse_first_index, self.device)
        self.structural_sparse_second_index = self._device_array(structural_sparse_second_index, self.device)
        self.structural_row_block = self._device_array(structural_row_block, self.device)
        self.structural_jacobian_first = wp.zeros(self.structural_row_count, dtype=vec6f, device=self.device)
        self.structural_jacobian_second = wp.zeros(self.structural_row_count, dtype=vec6f, device=self.device)
        self.structural_residual = wp.zeros(self.structural_row_count, dtype=wp.float32, device=self.device)
        self.structural_candidate_residual = wp.zeros(self.structural_row_count, dtype=wp.float32, device=self.device)
        self.structural_projected_residual = wp.zeros(self.structural_row_count, dtype=wp.float32, device=self.device)
        self.world_structural_residual = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.world_projected_structural_residual = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.structural_multiplier = wp.zeros(self.structural_row_count, dtype=wp.float32, device=self.device)
        self.structural_effective_mass = wp.zeros(self.structural_row_count, dtype=wp.float32, device=self.device)
        self.structural_penalty = wp.zeros(self.structural_row_count, dtype=wp.float32, device=self.device)

        self.structural_block_count = len(structural_block_world)
        self.structural_block_world = self._device_array(structural_block_world, self.device)
        self.structural_block_joint = self._device_array(structural_block_joint, self.device)
        self.structural_block_row_offset = self._device_array(structural_block_row_offset, self.device)
        self.structural_block_row_count = self._device_array(structural_block_row_count, self.device)
        self.structural_block_body_first = self._device_array(structural_block_first_local, self.device)
        self.structural_block_body_second = self._device_array(structural_block_second_local, self.device)
        self.structural_block_body_first_global = self._device_array(structural_block_first_global, self.device)
        self.structural_block_body_second_global = self._device_array(structural_block_second_global, self.device)
        self.structural_block_body_first_multiplicity = wp.zeros(
            self.structural_block_count, dtype=wp.int32, device=self.device
        )
        self.structural_block_body_second_multiplicity = wp.zeros(
            self.structural_block_count, dtype=wp.int32, device=self.device
        )
        self.structural_block_multiplicity = wp.ones(self.structural_block_count, dtype=wp.int32, device=self.device)
        self.structural_delassus = wp.zeros(self.structural_block_count, dtype=mat66f, device=self.device)
        self.structural_inverse_metric = wp.zeros(self.structural_block_count, dtype=mat66f, device=self.device)
        self.structural_penalty_metric = wp.zeros(self.structural_block_count, dtype=mat66f, device=self.device)
        self.structural_metric_status = wp.zeros(self.structural_block_count, dtype=wp.int32, device=self.device)
        self.static_body_constraint_count = wp.array(static_constraint_count, dtype=wp.int32, device=self.device)
        self.body_constraint_count = wp.zeros(model.size.sum_of_num_bodies, dtype=wp.int32, device=self.device)

    def _allocate_joint_frictions(self) -> None:
        """Allocate one bounded scalar constraint for each frictional joint DOF."""
        source_model = self.model._model
        if source_model is None or source_model.joint_friction is None:
            friction_values: list[float] = [0.0] * self.model.size.sum_of_num_joint_dofs
            self._joint_friction_force = wp.empty(0, dtype=wp.float32, device=self.device)
        else:
            friction_values = source_model.joint_friction.numpy().astype(float).tolist()
            self._joint_friction_force = source_model.joint_friction
        if len(friction_values) != self.model.size.sum_of_num_joint_dofs:
            raise ValueError("Newton joint friction must contain one value per joint DOF.")
        if any(not math.isfinite(value) or value < 0.0 for value in friction_values):
            raise ValueError("Joint friction values must be finite and nonnegative.")

        body_offsets = self.model.info.bodies_offset.numpy().astype(int).tolist()
        world_dof_offsets = self.model.info.joint_dofs_offset.numpy().astype(int).tolist()
        joint_worlds = self.model.joints.wid.numpy().astype(int).tolist()
        joint_first = self.model.joints.bid_B.numpy().astype(int).tolist()
        joint_second = self.model.joints.bid_F.numpy().astype(int).tolist()
        joint_dof_offsets = self.model.joints.dofs_offset.numpy().astype(int).tolist()
        joint_dof_counts = self.model.joints.num_dofs.numpy().astype(int).tolist()
        sparse_offsets: list[int] | None = None
        if isinstance(self.jacobians, SparseSystemJacobians):
            sparse_offsets = self.jacobians._J_dofs_joint_nzb_offsets.numpy().astype(int).tolist()

        rows_by_world: list[list[tuple[int, ...]]] = [[] for _ in range(self.num_worlds)]
        for joint in range(self.model.size.sum_of_num_joints):
            world = joint_worlds[joint]
            first_global = joint_first[joint]
            second_global = joint_second[joint]
            first_local = first_global - body_offsets[world] if first_global >= 0 else -1
            second_local = second_global - body_offsets[world] if second_global >= 0 else -1
            dof_offset = joint_dof_offsets[joint]
            dof_count = joint_dof_counts[joint]
            sparse_start = sparse_offsets[joint] if sparse_offsets is not None else -1
            for local_dof in range(dof_count):
                dof = dof_offset + local_dof
                if friction_values[dof] <= 0.0:
                    continue
                rows_by_world[world].append(
                    (
                        dof,
                        dof - world_dof_offsets[world],
                        first_local,
                        second_local,
                        first_global,
                        second_global,
                        sparse_start + dof_count + local_dof
                        if sparse_offsets is not None and first_global >= 0
                        else -1,
                        sparse_start + local_dof if sparse_offsets is not None else -1,
                    )
                )

        counts = [len(rows) for rows in rows_by_world]
        offsets = _capacity_offsets(counts)
        rows = [row for world_rows in rows_by_world for row in world_rows]
        self.friction_capacity = len(rows)
        self.friction_capacities = tuple(counts)
        self.world_friction_offset = self._device_array(offsets[:-1], self.device)
        self.world_friction_count = self._device_array(counts, self.device)
        self.friction_world = self._device_array(
            [world for world, count in enumerate(counts) for _ in range(count)], self.device
        )
        self.friction_local = self._device_array([local for count in counts for local in range(count)], self.device)
        self.friction_dof_index = self._device_array([row[0] for row in rows], self.device)
        self.friction_jacobian_row = self._device_array([row[1] for row in rows], self.device)
        self.friction_body_first = self._device_array([row[4] for row in rows], self.device)
        self.friction_body_second = self._device_array([row[5] for row in rows], self.device)
        self.friction_body_first_local = self._device_array([row[2] for row in rows], self.device)
        self.friction_body_second_local = self._device_array([row[3] for row in rows], self.device)
        self.friction_sparse_first_index = self._device_array([row[6] for row in rows], self.device)
        self.friction_sparse_second_index = self._device_array([row[7] for row in rows], self.device)
        self.friction_jacobian_first = wp.zeros(self.friction_capacity, dtype=vec6f, device=self.device)
        self.friction_jacobian_second = wp.zeros(self.friction_capacity, dtype=vec6f, device=self.device)
        self.friction_impulse_bound = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.friction_reaction = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.friction_velocity = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.friction_residual = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.friction_projection_delassus = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.world_friction_residual_max = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)

    def _allocate_unilaterals(self) -> None:
        limit_capacities = [0] * self.num_worlds
        if self.limits is not None and self.limits.model_max_limits_host > 0:
            limit_capacities = list(self.limits.world_max_limits_host)
        contact_capacities = [0] * self.num_worlds
        if self.contacts is not None:
            try:
                if self.contacts.model_max_contacts_host > 0:
                    contact_capacities = list(self.contacts.world_max_contacts_host)
            except RuntimeError:
                self.contacts = None
        if len(limit_capacities) != self.num_worlds or len(contact_capacities) != self.num_worlds:
            raise ValueError("Unilateral container world capacities must match the model world count.")

        limit_offsets = _capacity_offsets(limit_capacities)
        contact_offsets = _capacity_offsets(contact_capacities)
        self.limit_capacities = tuple(limit_capacities)
        self.contact_capacities = tuple(contact_capacities)
        self.limit_capacity = limit_offsets[-1]
        self.contact_capacity = contact_offsets[-1]
        self.world_limit_capacity = self._device_array(limit_capacities, self.device)
        self.world_limit_offset = self._device_array(limit_offsets[:-1], self.device)
        self.world_limit_count = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.world_contact_capacity = self._device_array(contact_capacities, self.device)
        self.world_contact_offset = self._device_array(contact_offsets[:-1], self.device)
        self.world_contact_count = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.limit_world = self._device_array(
            [world for world, capacity in enumerate(limit_capacities) for _ in range(capacity)], self.device
        )
        self.limit_local = self._device_array(
            [local for capacity in limit_capacities for local in range(capacity)], self.device
        )
        self.contact_world = self._device_array(
            [world for world, capacity in enumerate(contact_capacities) for _ in range(capacity)], self.device
        )
        self.contact_local = self._device_array(
            [local for capacity in contact_capacities for local in range(capacity)], self.device
        )

        self.limit_body_first = wp.full(self.limit_capacity, -1, dtype=wp.int32, device=self.device)
        self.limit_body_second = wp.full(self.limit_capacity, -1, dtype=wp.int32, device=self.device)
        self.limit_jacobian_first = wp.zeros(self.limit_capacity, dtype=vec6f, device=self.device)
        self.limit_jacobian_second = wp.zeros(self.limit_capacity, dtype=vec6f, device=self.device)
        self.limit_bias = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.limit_reaction = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.limit_velocity = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.limit_residual = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.limit_projection_delassus = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.world_limit_residual_max = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)

        self.contact_body_first = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.contact_body_second = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.contact_jacobian_first = wp.zeros(self.contact_capacity, dtype=mat36f, device=self.device)
        self.contact_jacobian_second = wp.zeros(self.contact_capacity, dtype=mat36f, device=self.device)
        self.contact_bias = wp.zeros(self.contact_capacity, dtype=wp.vec3f, device=self.device)
        self.contact_friction = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.contact_reaction = wp.zeros(self.contact_capacity, dtype=wp.vec3f, device=self.device)
        self.contact_velocity = wp.zeros(self.contact_capacity, dtype=wp.vec3f, device=self.device)
        self.contact_residual = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.contact_projection_delassus = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.contact_projection_delassus_normal_first = wp.zeros(
            self.contact_capacity, dtype=wp.mat33f, device=self.device
        )
        self.world_contact_projection_status = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.world_jacobi_projection_status = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.projection_twist_delta = wp.zeros(self.model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        self.world_contact_residual_max = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.projection_status = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.world_has_unilateral = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.body_has_unilateral = wp.zeros(self.model.size.sum_of_num_bodies, dtype=wp.int32, device=self.device)

    def begin_time_step(
        self,
        time_step: float,
        limit_stabilization_fraction: float = 0.01,
        contact_stabilization_fraction: float = 0.01,
        contact_dead_zone: float = 1.0e-6,
        impact_velocity_threshold: float = 1.0e-3,
    ) -> None:
        """Cache begin-step velocities and import unilateral reaction impulses once."""
        if time_step <= 0.0:
            raise ValueError("time_step must be positive.")
        if not 0.0 <= limit_stabilization_fraction <= 1.0:
            raise ValueError("limit_stabilization_fraction must be in [0, 1].")
        if not 0.0 <= contact_stabilization_fraction <= 1.0:
            raise ValueError("contact_stabilization_fraction must be in [0, 1].")
        if contact_dead_zone < 0.0 or impact_velocity_threshold < 0.0:
            raise ValueError("Contact dead zone and impact velocity threshold must be nonnegative.")
        self._limit_stabilization_fraction = limit_stabilization_fraction
        self._contact_stabilization_fraction = contact_stabilization_fraction
        self._contact_dead_zone = contact_dead_zone
        self._impact_velocity_threshold = impact_velocity_threshold
        wp.launch(
            _capture_body_velocity,
            dim=self.model.size.sum_of_num_bodies,
            inputs=[self.data.bodies.u_i],
            outputs=[self.body_velocity_begin],
            device=self.device,
        )
        if self.dynamic_row_count > 0:
            wp.launch(
                _capture_dynamic_joint_velocity,
                dim=self.dynamic_row_count,
                inputs=[self.dynamic_dof_index, self.data.joints.dq_j],
                outputs=[self.dynamic_velocity_begin],
                device=self.device,
            )
        self._update_unilaterals(
            time_step,
            limit_stabilization_fraction,
            contact_stabilization_fraction,
            contact_dead_zone,
            impact_velocity_threshold,
        )
        self._freeze_constraint_multiplicities()

    def _freeze_constraint_multiplicities(self) -> None:
        """Build combined entity incidence without a device-to-host readback."""
        wp.copy(self.body_constraint_count, self.static_body_constraint_count)
        self.body_has_unilateral.zero_()
        for entity_world, entity_local, world_count, body_first, body_second in (
            (
                self.friction_world,
                self.friction_local,
                self.world_friction_count,
                self.friction_body_first,
                self.friction_body_second,
            ),
            (
                self.limit_world,
                self.limit_local,
                self.world_limit_count,
                self.limit_body_first,
                self.limit_body_second,
            ),
            (
                self.contact_world,
                self.contact_local,
                self.world_contact_count,
                self.contact_body_first,
                self.contact_body_second,
            ),
        ):
            if entity_world.shape[0] > 0:
                wp.launch(
                    _accumulate_constraint_incidence,
                    dim=entity_world.shape[0],
                    inputs=[entity_world, entity_local, world_count, body_first, body_second],
                    outputs=[self.body_constraint_count, self.body_has_unilateral],
                    device=self.device,
                )

    def update(
        self,
        time_step: float,
        joint_penalty_scale: float | wp.array[wp.float32] = 10.0,
        linearization_twist: wp.array[vec6f] | None = None,
        block_joint_metrics: bool = True,
        mass_split_joint_metrics: bool = True,
        assemble_structural_penalty: bool = True,
    ) -> None:
        """Gather one nonlinear evaluation and assemble the smooth primal system.

        :meth:`begin_time_step` must be called before the first nonlinear
        evaluation. It freezes the inertial and restitution velocities while
        this method continues to use the current body velocity for gyroscopic
        torque evaluation.

        If ``linearization_twist`` is omitted, the current body velocity is
        explicitly captured and used as the structural Newton linearization
        twist. Pass a packed zero twist for the first linearly implicit
        assembly about the begin-step pose.
        """
        if time_step <= 0.0:
            raise ValueError("time_step must be positive.")
        if isinstance(joint_penalty_scale, wp.array):
            if joint_penalty_scale.ndim != 1 or joint_penalty_scale.dtype != wp.float32:
                raise ValueError("joint_penalty_scale must be a one-dimensional float32 array.")
            if joint_penalty_scale.shape[0] != self.num_worlds:
                raise ValueError("joint_penalty_scale must contain one entry per world.")
            world_joint_penalty_scale = joint_penalty_scale
        else:
            if joint_penalty_scale <= 0.0:
                raise ValueError("joint_penalty_scale must be positive.")
            self._uniform_joint_penalty_scale.fill_(joint_penalty_scale)
            world_joint_penalty_scale = self._uniform_joint_penalty_scale
        if mass_split_joint_metrics and not block_joint_metrics:
            raise ValueError("mass_split_joint_metrics requires block_joint_metrics.")
        self._block_joint_metrics = block_joint_metrics

        data = self.data
        if linearization_twist is None:
            wp.launch(
                _capture_body_velocity,
                dim=self.model.size.sum_of_num_bodies,
                inputs=[data.bodies.u_i],
                outputs=[self.body_linearization_twist],
                device=self.device,
            )
            linearization_twist = self.body_linearization_twist
        elif linearization_twist.shape[0] != self.model.size.sum_of_num_bodies:
            raise ValueError("linearization_twist must contain one entry per model body.")
        wp.launch(
            _gather_body_explicit_wrenches,
            dim=self.model.size.sum_of_num_bodies,
            inputs=[
                self.model.bodies.wid,
                self.model.bodies.m_i,
                data.bodies.I_i,
                data.bodies.u_i,
                data.bodies.w_e_i,
                data.bodies.w_a_i,
                self.model.gravity.vector,
            ],
            outputs=[self.body_explicit_wrench],
            device=self.device,
        )
        self.system.assemble_bodies(
            self.model.bodies.m_i,
            data.bodies.I_i,
            self.body_velocity_begin,
            self.body_explicit_wrench,
            time_step,
        )
        self._update_unilaterals(
            time_step,
            self._limit_stabilization_fraction,
            self._contact_stabilization_fraction,
            self._contact_dead_zone,
            self._impact_velocity_threshold,
            import_reactions=False,
            update_counts=False,
        )

        if self.dynamic_row_count > 0:
            wp.launch(
                _gather_dynamic_rows,
                dim=self.dynamic_row_count,
                inputs=[
                    self.dynamic_row_world,
                    self.dynamic_row_joint,
                    self.dynamic_jacobian_row,
                    self.dynamic_body_first,
                    self.dynamic_body_second,
                    self.dynamic_body_first_global,
                    self.dynamic_body_second_global,
                    self.dynamic_value_index,
                    self.dynamic_dof_index,
                    self.model.info.num_body_dofs,
                    self._dense_jacobian_offsets,
                    self._dense_jacobian_data,
                    self.sparse_jacobian,
                    self.dynamic_sparse_first_index,
                    self.dynamic_sparse_second_index,
                    self._sparse_jacobian_data,
                    data.joints.m_j,
                    data.joints.dq_b_j,
                    self.model.joints.a_j,
                    self.model.joints.k_p_j,
                    self.model.joints.act_type,
                    data.joints.dq_j,
                    self.dynamic_velocity_begin,
                    linearization_twist,
                    time_step,
                ],
                outputs=[
                    self.dynamic_jacobian_first,
                    self.dynamic_jacobian_second,
                    self.dynamic_effective_inertia,
                    self.dynamic_free_velocity,
                ],
                device=self.device,
            )
            self.system.add_dynamic_rows(
                self.dynamic_row_world,
                self.dynamic_body_first_global,
                self.dynamic_body_second_global,
                self.dynamic_jacobian_first,
                self.dynamic_jacobian_second,
                self.dynamic_effective_inertia,
                self.dynamic_free_velocity,
            )

        if self.structural_row_count > 0:
            wp.launch(
                _gather_structural_rows,
                dim=self.structural_row_count,
                inputs=[
                    self.structural_row_world,
                    self.structural_jacobian_row,
                    self.structural_body_first,
                    self.structural_body_second,
                    self.structural_body_first_global,
                    self.structural_body_second_global,
                    self.structural_residual_index,
                    self.structural_multiplier_index,
                    self.model.info.num_body_dofs,
                    self._dense_jacobian_offsets,
                    self._dense_jacobian_data,
                    self.sparse_jacobian,
                    self.structural_sparse_first_index,
                    self.structural_sparse_second_index,
                    self._sparse_jacobian_data,
                    self.model.bodies.inv_m_i,
                    data.bodies.inv_I_i,
                    data.joints.r_j,
                    data.joints.lambda_j,
                ],
                outputs=[
                    self.structural_jacobian_first,
                    self.structural_jacobian_second,
                    self.structural_residual,
                    self.structural_multiplier,
                    self.structural_effective_mass,
                ],
                device=self.device,
            )
            if block_joint_metrics:
                wp.launch(
                    _compute_structural_metrics,
                    dim=self.structural_block_count,
                    inputs=[
                        self.structural_block_world,
                        self.structural_block_row_offset,
                        self.structural_block_row_count,
                        self.structural_block_body_first_global,
                        self.structural_block_body_second_global,
                        self.static_body_constraint_count,
                        mass_split_joint_metrics,
                        self.structural_jacobian_first,
                        self.structural_jacobian_second,
                        self.model.bodies.inv_m_i,
                        data.bodies.inv_I_i,
                        time_step,
                        world_joint_penalty_scale,
                    ],
                    outputs=[
                        self.structural_block_body_first_multiplicity,
                        self.structural_block_body_second_multiplicity,
                        self.structural_block_multiplicity,
                        self.structural_delassus,
                        self.structural_inverse_metric,
                        self.structural_penalty_metric,
                        self.structural_metric_status,
                        self.structural_effective_mass,
                        self.structural_penalty,
                    ],
                    device=self.device,
                )
                if assemble_structural_penalty:
                    self.system.add_structural_blocks(
                        self.structural_block_world,
                        self.structural_block_row_offset,
                        self.structural_block_row_count,
                        self.structural_block_body_first_global,
                        self.structural_block_body_second_global,
                        self.structural_jacobian_first,
                        self.structural_jacobian_second,
                        self.structural_residual,
                        self.structural_multiplier,
                        self.structural_penalty_metric,
                        self.structural_metric_status,
                        linearization_twist,
                        time_step,
                    )
            else:
                if assemble_structural_penalty:
                    self.system.add_structural_rows(
                        self.structural_row_world,
                        self.structural_body_first_global,
                        self.structural_body_second_global,
                        self.structural_jacobian_first,
                        self.structural_jacobian_second,
                        self.structural_residual,
                        self.structural_multiplier,
                        self.structural_effective_mass,
                        linearization_twist,
                        time_step,
                        world_joint_penalty_scale,
                        self.structural_penalty,
                    )

    def _update_unilaterals(
        self,
        time_step: float,
        limit_stabilization_fraction: float,
        contact_stabilization_fraction: float,
        contact_dead_zone: float,
        impact_velocity_threshold: float,
        import_reactions: bool = True,
        update_counts: bool = True,
    ) -> None:
        if self.friction_capacity > 0:
            wp.launch(
                _gather_joint_frictions,
                dim=self.friction_capacity,
                inputs=[
                    self.friction_world,
                    self.friction_jacobian_row,
                    self.friction_body_first_local,
                    self.friction_body_second_local,
                    self.friction_body_first,
                    self.friction_body_second,
                    self.friction_dof_index,
                    self.model.info.num_body_dofs,
                    self._dense_dof_jacobian_offsets,
                    self._dense_dof_jacobian_data,
                    self.sparse_jacobian,
                    self.friction_sparse_first_index,
                    self.friction_sparse_second_index,
                    self._sparse_dof_jacobian_data,
                    self._joint_friction_force,
                    self.body_velocity_begin,
                    time_step,
                    import_reactions,
                ],
                outputs=[
                    self.friction_jacobian_first,
                    self.friction_jacobian_second,
                    self.friction_impulse_bound,
                    self.friction_reaction,
                    self.friction_velocity,
                ],
                device=self.device,
            )

        if self.limit_capacity > 0 and self.limits is not None:
            if update_counts:
                wp.launch(
                    _copy_clamped_world_counts,
                    dim=self.num_worlds,
                    inputs=[self.limits.world_active_limits, self.world_limit_capacity],
                    outputs=[self.world_limit_count],
                    device=self.device,
                )
            wp.launch(
                _gather_limits,
                dim=self.limits.model_max_limits_host,
                inputs=[
                    self.limits.model_active_limits,
                    self.limits.model_max_limits_host,
                    self.limits.wid,
                    self.limits.lid,
                    self.limits.bids,
                    self.limits.r_q,
                    self.limits.reaction,
                    self.body_velocity_begin,
                    self.world_limit_capacity,
                    self.world_limit_offset,
                    self.model.info.bodies_offset,
                    self.model.info.num_body_dofs,
                    self.data.info.limit_cts_group_offset,
                    self._dense_jacobian_offsets,
                    self._dense_jacobian_data,
                    self.sparse_jacobian,
                    self._sparse_limit_offsets,
                    self._sparse_jacobian_data,
                    time_step,
                    limit_stabilization_fraction,
                    import_reactions,
                ],
                outputs=[
                    self.limit_body_first,
                    self.limit_body_second,
                    self.limit_jacobian_first,
                    self.limit_jacobian_second,
                    self.limit_bias,
                    self.limit_reaction,
                    self.limit_velocity,
                ],
                device=self.device,
            )
        elif update_counts:
            self.world_limit_count.zero_()

        if self.contact_capacity > 0 and self.contacts is not None:
            if update_counts:
                wp.launch(
                    _copy_clamped_world_counts,
                    dim=self.num_worlds,
                    inputs=[self.contacts.world_active_contacts, self.world_contact_capacity],
                    outputs=[self.world_contact_count],
                    device=self.device,
                )
            wp.launch(
                _gather_contacts,
                dim=self.contacts.model_max_contacts_host,
                inputs=[
                    self.contacts.model_active_contacts,
                    self.contacts.model_max_contacts_host,
                    self.contacts.wid,
                    self.contacts.cid,
                    self.contacts.bid_AB,
                    self.contacts.gapfunc,
                    self.contacts.material,
                    self.contacts.reaction,
                    self.body_velocity_begin,
                    self.world_contact_capacity,
                    self.world_contact_offset,
                    self.model.info.bodies_offset,
                    self.model.info.num_body_dofs,
                    self.data.info.contact_cts_group_offset,
                    self._dense_jacobian_offsets,
                    self._dense_jacobian_data,
                    self.sparse_jacobian,
                    self._sparse_contact_offsets,
                    self._sparse_jacobian_data,
                    time_step,
                    contact_stabilization_fraction,
                    contact_dead_zone,
                    impact_velocity_threshold,
                    import_reactions,
                ],
                outputs=[
                    self.contact_body_first,
                    self.contact_body_second,
                    self.contact_jacobian_first,
                    self.contact_jacobian_second,
                    self.contact_bias,
                    self.contact_friction,
                    self.contact_reaction,
                    self.contact_velocity,
                ],
                device=self.device,
            )
        elif update_counts:
            self.world_contact_count.zero_()

        if self.limit_capacity > 0:
            wp.launch(
                _clear_inactive_limits,
                dim=self.limit_capacity,
                inputs=[self.limit_world, self.limit_local, self.world_limit_count],
                outputs=[
                    self.limit_body_first,
                    self.limit_body_second,
                    self.limit_reaction,
                    self.limit_velocity,
                ],
                device=self.device,
            )
        if self.contact_capacity > 0:
            wp.launch(
                _clear_inactive_contacts,
                dim=self.contact_capacity,
                inputs=[self.contact_world, self.contact_local, self.world_contact_count],
                outputs=[
                    self.contact_body_first,
                    self.contact_body_second,
                    self.contact_reaction,
                    self.contact_velocity,
                ],
                device=self.device,
            )

        if update_counts:
            wp.launch(
                _mark_worlds_with_unilaterals,
                dim=self.num_worlds,
                inputs=[self.world_contact_count, self.world_limit_count, self.world_friction_count],
                outputs=[self.world_has_unilateral],
                device=self.device,
            )

    def unpack_system_solution(self, solution: wp.array[wp.float32] | None = None) -> None:
        """Convert a flat dense-system solution to packed body twists."""
        if solution is None:
            solution = self.system.solution
        elif solution.shape[0] != self.system.info.total_vec_size:
            raise ValueError("solution must match the packed body vector storage.")
        wp.launch(
            _unpack_body_vector,
            dim=self.model.size.sum_of_num_bodies,
            inputs=[self.system.body_vector_index, solution],
            outputs=[self.system_solution_twist],
            device=self.device,
        )

    def update_structural_multipliers_from_twist(
        self,
        time_step: float,
        structural_tolerance: float,
        linearization_twist: wp.array[vec6f],
        global_twist: wp.array[vec6f],
        projected_twist: wp.array[vec6f],
        world_active: wp.array[wp.bool],
        projected_fraction: float = 0.0,
    ) -> None:
        """Update the zero structural split and evaluate both body-twist copies."""
        if not time_step > 0.0 or not structural_tolerance > 0.0:
            raise ValueError("Time step and structural tolerance must be positive.")
        if linearization_twist.shape[0] != self.model.size.sum_of_num_bodies:
            raise ValueError("linearization_twist must contain one entry per body.")
        if global_twist.shape[0] != self.model.size.sum_of_num_bodies:
            raise ValueError("global_twist must contain one entry per body.")
        if projected_twist.shape[0] != self.model.size.sum_of_num_bodies:
            raise ValueError("projected_twist must contain one entry per body.")
        if world_active.shape[0] != self.num_worlds:
            raise ValueError("world_active must contain one entry per world.")
        if not 0.0 <= projected_fraction <= 1.0:
            raise ValueError("projected_fraction must be in [0, 1].")
        self.world_structural_residual.zero_()
        self.world_projected_structural_residual.zero_()
        if self.structural_row_count == 0:
            return
        if self._block_joint_metrics:
            wp.launch(
                _update_structural_multipliers_from_twist,
                dim=self.structural_block_count,
                inputs=[
                    time_step,
                    structural_tolerance,
                    self.structural_block_world,
                    world_active,
                    self.projection_status,
                    self.structural_block_row_offset,
                    self.structural_block_row_count,
                    self.structural_block_body_first_global,
                    self.structural_block_body_second_global,
                    self.structural_jacobian_first,
                    self.structural_jacobian_second,
                    self.structural_residual,
                    self.structural_penalty_metric,
                    self.structural_metric_status,
                    linearization_twist,
                    global_twist,
                    projected_twist,
                    projected_fraction,
                    self.system.body_vector_index,
                    self.structural_multiplier_index,
                ],
                outputs=[
                    self.structural_multiplier,
                    self.structural_candidate_residual,
                    self.structural_projected_residual,
                    self.world_structural_residual,
                    self.world_projected_structural_residual,
                    self.system.right_hand_side,
                    self.data.joints.lambda_j,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _update_structural_multipliers_from_twist_rows,
                dim=self.structural_row_count,
                inputs=[
                    time_step,
                    structural_tolerance,
                    self.structural_row_world,
                    world_active,
                    self.projection_status,
                    self.structural_body_first_global,
                    self.structural_body_second_global,
                    self.structural_jacobian_first,
                    self.structural_jacobian_second,
                    self.structural_residual,
                    self.structural_penalty,
                    linearization_twist,
                    global_twist,
                    projected_twist,
                    projected_fraction,
                    self.system.body_vector_index,
                    self.structural_multiplier_index,
                ],
                outputs=[
                    self.structural_multiplier,
                    self.structural_candidate_residual,
                    self.world_structural_residual,
                    self.system.right_hand_side,
                    self.data.joints.lambda_j,
                ],
                device=self.device,
            )
            wp.launch(
                _evaluate_structural_residual_from_twist,
                dim=self.structural_row_count,
                inputs=[
                    time_step,
                    structural_tolerance,
                    self.structural_row_world,
                    world_active,
                    self.projection_status,
                    self.structural_body_first_global,
                    self.structural_body_second_global,
                    self.structural_jacobian_first,
                    self.structural_jacobian_second,
                    self.structural_residual,
                    linearization_twist,
                    projected_twist,
                ],
                outputs=[self.structural_projected_residual, self.world_projected_structural_residual],
                device=self.device,
            )

    def evaluate_structural_residuals_from_twists(
        self,
        time_step: float,
        structural_tolerance: float,
        linearization_twist: wp.array[vec6f],
        global_twist: wp.array[vec6f],
        projected_twist: wp.array[vec6f],
        world_active: wp.array[wp.bool],
    ) -> None:
        """Evaluate structural residuals without changing joint multipliers."""
        if not time_step > 0.0 or not structural_tolerance > 0.0:
            raise ValueError("Time step and structural tolerance must be positive.")
        if any(
            twist.shape[0] != self.model.size.sum_of_num_bodies
            for twist in (linearization_twist, global_twist, projected_twist)
        ):
            raise ValueError("Twist arrays must contain one entry per body.")
        if world_active.shape[0] != self.num_worlds:
            raise ValueError("world_active must contain one entry per world.")
        self.world_structural_residual.zero_()
        self.world_projected_structural_residual.zero_()
        if self.structural_row_count == 0:
            return
        wp.launch(
            _evaluate_structural_residual_from_twist,
            dim=self.structural_row_count,
            inputs=[
                time_step,
                structural_tolerance,
                self.structural_row_world,
                world_active,
                self.projection_status,
                self.structural_body_first_global,
                self.structural_body_second_global,
                self.structural_jacobian_first,
                self.structural_jacobian_second,
                self.structural_residual,
                linearization_twist,
                global_twist,
            ],
            outputs=[self.structural_candidate_residual, self.world_structural_residual],
            device=self.device,
        )
        wp.launch(
            _evaluate_structural_residual_from_twist,
            dim=self.structural_row_count,
            inputs=[
                time_step,
                structural_tolerance,
                self.structural_row_world,
                world_active,
                self.projection_status,
                self.structural_body_first_global,
                self.structural_body_second_global,
                self.structural_jacobian_first,
                self.structural_jacobian_second,
                self.structural_residual,
                linearization_twist,
                projected_twist,
            ],
            outputs=[self.structural_projected_residual, self.world_projected_structural_residual],
            device=self.device,
        )

    def reset_structural_multipliers(self) -> None:
        """Clear persistent structural multiplier warm starts."""
        if self.structural_row_count == 0:
            return
        self.structural_multiplier.zero_()
        wp.launch(
            _write_structural_multipliers,
            dim=self.structural_row_count,
            inputs=[self.structural_multiplier_index, self.structural_multiplier],
            outputs=[self.data.joints.lambda_j],
            device=self.device,
        )

    def scale_structural_multipliers(self, scale: float) -> None:
        """Scale structural multipliers before using them as a warm start."""
        if not 0.0 <= scale <= 1.0:
            raise ValueError("scale must be in [0, 1].")
        if self.structural_row_count == 0:
            return
        wp.launch(
            _scale_structural_multipliers,
            dim=self.structural_row_count,
            inputs=[scale, self.structural_multiplier_index],
            outputs=[self.structural_multiplier, self.data.joints.lambda_j],
            device=self.device,
        )

    def gather_structural_candidate_residuals(self) -> wp.array[wp.float32]:
        """Gather structural residuals evaluated at the current body poses."""
        if self.structural_row_count > 0:
            wp.launch(
                _gather_structural_residuals,
                dim=self.structural_row_count,
                inputs=[self.structural_residual_index, self.data.joints.r_j],
                outputs=[self.structural_residual],
                device=self.device,
            )
        return self.structural_residual

    def write_outputs(self, time_step: float, body_velocity: wp.array[vec6f] | None = None) -> None:
        """Write body velocities and force-valued reactions to Kamino containers."""
        if time_step <= 0.0:
            raise ValueError("time_step must be positive.")
        if body_velocity is None:
            body_velocity = self.projected_twist
        if body_velocity.shape[0] != self.model.size.sum_of_num_bodies:
            raise ValueError("body_velocity must contain one entry per model body.")
        inverse_time_step = 1.0 / time_step
        wp.launch(
            _write_body_velocity,
            dim=self.model.size.sum_of_num_bodies,
            inputs=[body_velocity],
            outputs=[self.data.bodies.u_i],
            device=self.device,
        )
        if self.dynamic_row_count > 0:
            wp.launch(
                _write_dynamic_multipliers,
                dim=self.dynamic_row_count,
                inputs=[
                    inverse_time_step,
                    self.dynamic_multiplier_index,
                    self.dynamic_body_first_global,
                    self.dynamic_body_second_global,
                    self.dynamic_jacobian_first,
                    self.dynamic_jacobian_second,
                    self.dynamic_effective_inertia,
                    self.dynamic_free_velocity,
                    body_velocity,
                ],
                outputs=[self.data.joints.lambda_j],
                device=self.device,
            )
        if self.structural_row_count > 0:
            wp.launch(
                _write_structural_multipliers,
                dim=self.structural_row_count,
                inputs=[self.structural_multiplier_index, self.structural_multiplier],
                outputs=[self.data.joints.lambda_j],
                device=self.device,
            )
        if self.limit_capacity > 0 and self.limits is not None:
            wp.launch(
                _write_limit_outputs,
                dim=self.limits.model_max_limits_host,
                inputs=[
                    self.limits.model_active_limits,
                    self.limits.model_max_limits_host,
                    self.limits.wid,
                    self.limits.lid,
                    self.world_limit_capacity,
                    self.world_limit_offset,
                    inverse_time_step,
                    self.limit_reaction,
                    self.limit_velocity,
                ],
                outputs=[self.limits.reaction, self.limits.velocity],
                device=self.device,
            )
        if self.contact_capacity > 0 and self.contacts is not None:
            wp.launch(
                _write_contact_outputs,
                dim=self.contacts.model_max_contacts_host,
                inputs=[
                    self.contacts.model_active_contacts,
                    self.contacts.model_max_contacts_host,
                    self.contacts.wid,
                    self.contacts.cid,
                    self.world_contact_capacity,
                    self.world_contact_offset,
                    inverse_time_step,
                    self.contact_reaction,
                    self.contact_velocity,
                ],
                outputs=[self.contacts.reaction, self.contacts.velocity, self.contacts.mode],
                device=self.device,
            )

        self.write_constraint_wrenches(time_step)

    def write_constraint_wrenches(self, time_step: float) -> None:
        """Write joint, limit, and contact reaction wrenches to body data."""
        if time_step <= 0.0:
            raise ValueError("time_step must be positive.")
        data = self.data.bodies
        data.w_j_i.zero_()
        data.w_l_i.zero_()
        data.w_c_i.zero_()
        if self.dynamic_row_count > 0:
            wp.launch(
                _accumulate_joint_wrenches,
                dim=self.dynamic_row_count,
                inputs=[
                    self.dynamic_multiplier_index,
                    self.dynamic_body_first_global,
                    self.dynamic_body_second_global,
                    self.dynamic_jacobian_first,
                    self.dynamic_jacobian_second,
                    self.data.joints.lambda_j,
                ],
                outputs=[data.w_j_i],
                device=self.device,
            )
        if self.structural_row_count > 0:
            wp.launch(
                _accumulate_joint_wrenches,
                dim=self.structural_row_count,
                inputs=[
                    self.structural_multiplier_index,
                    self.structural_body_first_global,
                    self.structural_body_second_global,
                    self.structural_jacobian_first,
                    self.structural_jacobian_second,
                    self.data.joints.lambda_j,
                ],
                outputs=[data.w_j_i],
                device=self.device,
            )
        if self.friction_capacity > 0:
            wp.launch(
                _accumulate_joint_friction_wrenches,
                dim=self.friction_capacity,
                inputs=[
                    1.0 / time_step,
                    self.friction_body_first,
                    self.friction_body_second,
                    self.friction_jacobian_first,
                    self.friction_jacobian_second,
                    self.friction_reaction,
                ],
                outputs=[data.w_j_i],
                device=self.device,
            )
        if self.limit_capacity > 0:
            wp.launch(
                _accumulate_limit_wrenches,
                dim=self.limit_capacity,
                inputs=[
                    1.0 / time_step,
                    self.limit_world,
                    self.limit_local,
                    self.world_limit_count,
                    self.limit_body_first,
                    self.limit_body_second,
                    self.limit_jacobian_first,
                    self.limit_jacobian_second,
                    self.limit_reaction,
                ],
                outputs=[data.w_l_i],
                device=self.device,
            )
        if self.contact_capacity > 0:
            wp.launch(
                _accumulate_contact_wrenches,
                dim=self.contact_capacity,
                inputs=[
                    1.0 / time_step,
                    self.contact_world,
                    self.contact_local,
                    self.world_contact_count,
                    self.contact_body_first,
                    self.contact_body_second,
                    self.contact_jacobian_first,
                    self.contact_jacobian_second,
                    self.contact_reaction,
                ],
                outputs=[data.w_c_i],
                device=self.device,
            )
