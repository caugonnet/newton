# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block-diagonal primal body systems for the LOX backend.

This module owns only solver-internal, contact-free body-space assembly. Each
factor block is one connected dynamic-body component and each dynamic body
contributes six linear-first velocity unknowns. Prescribed bodies retain their
global packed body indices but map to ``-1`` in the matrix layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import warp as wp

from ...core.types import mat66f, vec6f
from ...linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from .linear import HybridLLTBlockedSolver
from .metric import METRIC_STATUS_VALID
from .problem import compute_augmented_joint_row, compute_body_inertial_system, compute_dynamic_joint_row
from .time import validate_world_time_step
from .weight import (
    BODY_WEIGHT_BETA_DEFAULT,
    BODY_WEIGHT_SIGMA_DEFAULT,
    BODY_WEIGHT_STATUS_VALID,
    compute_body_weight_anisotropic,
    compute_body_weight_mass_proportional,
)

__all__ = ["BatchedPrimalBodySystem"]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _mark_blocks_with_unilaterals(
    body_component: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
    block_has_unilateral: wp.array[wp.int32],
):
    body = wp.tid()
    component = body_component[body]
    if component >= 0 and body_has_unilateral[body] != 0:
        wp.atomic_max(block_has_unilateral, component, 1)


@wp.kernel
def _enable_bodies_in_weighted_blocks(
    body_component: wp.array[wp.int32],
    block_has_unilateral: wp.array[wp.int32],
    body_weight_enabled: wp.array[wp.int32],
):
    body = wp.tid()
    component = body_component[body]
    body_weight_enabled[body] = wp.where(component >= 0 and block_has_unilateral[component] != 0, 1, 0)


@wp.func
def _matrix_index(
    matrix_offset: wp.int32,
    dimension: wp.int32,
    row: wp.int32,
    col: wp.int32,
) -> wp.int32:
    return matrix_offset + dimension * row + col


@wp.func
def _atomic_add_body_vector(
    vector: wp.array[wp.float32],
    vector_offset: wp.int32,
    body: wp.int32,
    value: vec6f,
):
    body_offset = vector_offset + 6 * body
    for row in range(6):
        wp.atomic_add(vector, body_offset + row, value[row])


@wp.func
def _atomic_add_body_block(
    matrix: wp.array[wp.float32],
    matrix_offset: wp.int32,
    dimension: wp.int32,
    row_body: wp.int32,
    col_body: wp.int32,
    value: mat66f,
):
    row_offset = 6 * row_body
    col_offset = 6 * col_body
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, row_offset + row, col_offset + col)
            wp.atomic_add(matrix, index, value[row, col])


@wp.func
def _atomic_add_body_outer_product(
    matrix: wp.array[wp.float32],
    matrix_offset: wp.int32,
    dimension: wp.int32,
    row_body: wp.int32,
    col_body: wp.int32,
    row_jacobian: vec6f,
    col_jacobian: vec6f,
    scale: wp.float32,
):
    row_offset = 6 * row_body
    col_offset = 6 * col_body
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, row_offset + row, col_offset + col)
            wp.atomic_add(matrix, index, scale * row_jacobian[row] * col_jacobian[col])


@wp.kernel
def _assemble_body_inertial_systems(
    body_world: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    mass: wp.array[wp.float32],
    inertia_world: wp.array[wp.mat33f],
    velocity_previous: wp.array[vec6f],
    force_explicit: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    body = wp.tid()
    dt = time_step[body_world[body]]
    block = body_block[body]
    if block < 0:
        return
    local_body = body_local[body]
    dimension = dimensions[block]
    matrix_offset = matrix_offsets[block]
    vector_offset = vector_offsets[block]
    contribution = compute_body_inertial_system(
        mass[body], inertia_world[body], velocity_previous[body], force_explicit[body], dt
    )

    body_offset = 6 * local_body
    for row in range(6):
        right_hand_side[vector_offset + body_offset + row] = contribution.right_hand_side[row]
        for col in range(6):
            matrix[_matrix_index(matrix_offset, dimension, body_offset + row, body_offset + col)] = contribution.matrix[
                row, col
            ]


@wp.kernel
def _assemble_dynamic_joint_rows(
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    prescribed_twist: wp.array[vec6f],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    joint_row = wp.tid()
    first = body_first[joint_row]
    second = body_second[joint_row]
    body_count = body_block.shape[0]
    if effective_inertia[joint_row] <= 0.0 or (first < 0 and second < 0) or first >= body_count or second >= body_count:
        return

    first_local = body_local[first] if first >= 0 else -1
    second_local = body_local[second] if second >= 0 else -1
    if first_local < 0 and second_local < 0:
        return
    block = body_block[first] if first_local >= 0 else body_block[second]
    if block < 0 or block >= dimensions.shape[0] or (second_local >= 0 and body_block[second] != block):
        return
    dimension = dimensions[block]

    matrix_offset = matrix_offsets[block]
    vector_offset = vector_offsets[block]
    inertia = effective_inertia[joint_row]
    velocity = free_velocity[joint_row]
    first_jacobian = jacobian_first[joint_row]
    second_jacobian = jacobian_second[joint_row]
    if first >= 0 and first_local < 0:
        velocity -= wp.dot(first_jacobian, prescribed_twist[first])
    if second >= 0 and second_local < 0:
        velocity -= wp.dot(second_jacobian, prescribed_twist[second])

    if first_local >= 0:
        first_contribution = compute_dynamic_joint_row(first_jacobian, inertia, velocity)
        _atomic_add_body_block(matrix, matrix_offset, dimension, first_local, first_local, first_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, first_local, first_contribution.right_hand_side)
    if second_local >= 0:
        second_contribution = compute_dynamic_joint_row(second_jacobian, inertia, velocity)
        _atomic_add_body_block(matrix, matrix_offset, dimension, second_local, second_local, second_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, second_local, second_contribution.right_hand_side)
    if first_local >= 0 and second_local >= 0:
        _atomic_add_body_outer_product(
            matrix,
            matrix_offset,
            dimension,
            first_local,
            second_local,
            first_jacobian,
            second_jacobian,
            inertia,
        )
        _atomic_add_body_outer_product(
            matrix,
            matrix_offset,
            dimension,
            second_local,
            first_local,
            second_jacobian,
            first_jacobian,
            inertia,
        )


@wp.kernel
def _assemble_structural_joint_rows(
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    row_world: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    body_block: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    residual: wp.array[wp.float32],
    multiplier: wp.array[wp.float32],
    effective_mass: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    prescribed_twist: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    joint_penalty_scale: wp.array[wp.float32],
    penalty: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    joint_row = wp.tid()
    world = row_world[joint_row]
    dt = time_step[world]
    if world < 0 or world >= joint_penalty_scale.shape[0]:
        return
    first = body_first[joint_row]
    second = body_second[joint_row]
    body_count = body_block.shape[0]
    world_penalty_scale = joint_penalty_scale[world]
    if (
        dt <= 0.0
        or world_penalty_scale <= 0.0
        or effective_mass[joint_row] <= 0.0
        or (first < 0 and second < 0)
        or first >= body_count
        or second >= body_count
    ):
        penalty[joint_row] = 0.0
        return

    first_local = body_local[first] if first >= 0 else -1
    second_local = body_local[second] if second >= 0 else -1
    if first_local < 0 and second_local < 0:
        penalty[joint_row] = 0.0
        return
    block = body_block[first] if first_local >= 0 else body_block[second]
    if block < 0 or block >= dimensions.shape[0] or (second_local >= 0 and body_block[second] != block):
        penalty[joint_row] = 0.0
        return
    dimension = dimensions[block]

    row_penalty = world_penalty_scale * effective_mass[joint_row] / (dt * dt)
    penalty[joint_row] = row_penalty
    matrix_offset = matrix_offsets[block]
    vector_offset = vector_offsets[block]
    first_jacobian = jacobian_first[joint_row]
    second_jacobian = jacobian_second[joint_row]
    row_residual = residual[joint_row]
    row_multiplier = multiplier[joint_row]
    linearization_velocity = wp.float32(0.0)
    if first >= 0:
        linearization_velocity += wp.dot(first_jacobian, linearization_twist[first])
    if second >= 0:
        linearization_velocity += wp.dot(second_jacobian, linearization_twist[second])
    if first >= 0 and first_local < 0:
        linearization_velocity -= wp.dot(first_jacobian, prescribed_twist[first])
    if second >= 0 and second_local < 0:
        linearization_velocity -= wp.dot(second_jacobian, prescribed_twist[second])

    if first_local >= 0:
        first_contribution = compute_augmented_joint_row(
            first_jacobian, row_residual, row_multiplier, row_penalty, dt, linearization_velocity
        )
        _atomic_add_body_block(matrix, matrix_offset, dimension, first_local, first_local, first_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, first_local, first_contribution.right_hand_side)
    if second_local >= 0:
        second_contribution = compute_augmented_joint_row(
            second_jacobian, row_residual, row_multiplier, row_penalty, dt, linearization_velocity
        )
        _atomic_add_body_block(matrix, matrix_offset, dimension, second_local, second_local, second_contribution.matrix)
        _atomic_add_body_vector(right_hand_side, vector_offset, second_local, second_contribution.right_hand_side)
    if first_local >= 0 and second_local >= 0:
        cross_scale = dt * dt * row_penalty
        _atomic_add_body_outer_product(
            matrix,
            matrix_offset,
            dimension,
            first_local,
            second_local,
            first_jacobian,
            second_jacobian,
            cross_scale,
        )
        _atomic_add_body_outer_product(
            matrix,
            matrix_offset,
            dimension,
            second_local,
            first_local,
            second_jacobian,
            first_jacobian,
            cross_scale,
        )


@wp.func
def _structural_body_block(
    row_offset: wp.int32,
    row_count: wp.int32,
    row_jacobian: wp.array[vec6f],
    col_jacobian: wp.array[vec6f],
    penalty: mat66f,
    time_step_squared: wp.float32,
) -> mat66f:
    result = mat66f(0.0)
    for constraint_row in range(6):
        if constraint_row < row_count:
            first = row_jacobian[row_offset + constraint_row]
            for constraint_col in range(6):
                if constraint_col < row_count:
                    second = col_jacobian[row_offset + constraint_col]
                    scale = time_step_squared * penalty[constraint_row, constraint_col]
                    for row in range(6):
                        for col in range(6):
                            result[row, col] += scale * first[row] * second[col]
    return result


@wp.func
def _structural_body_right_hand_side(
    row_offset: wp.int32,
    row_count: wp.int32,
    jacobian: wp.array[vec6f],
    residual: wp.array[wp.float32],
    multiplier: wp.array[wp.float32],
    penalty: mat66f,
    linearization_velocity: vec6f,
    time_step: wp.float32,
) -> vec6f:
    result = vec6f(0.0)
    time_step_squared = time_step * time_step
    for row in range(6):
        if row < row_count:
            weighted_residual = wp.float32(0.0)
            weighted_velocity = wp.float32(0.0)
            for col in range(6):
                if col < row_count:
                    weighted_residual += penalty[row, col] * residual[row_offset + col]
                    weighted_velocity += penalty[row, col] * linearization_velocity[col]
            scale = -time_step * (multiplier[row_offset + row] + weighted_residual)
            scale += time_step_squared * weighted_velocity
            result += scale * jacobian[row_offset + row]
    return result


@wp.kernel
def _assemble_structural_joint_blocks(
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    block_world: wp.array[wp.int32],
    block_row_offset: wp.array[wp.int32],
    block_row_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    body_component: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    residual: wp.array[wp.float32],
    multiplier: wp.array[wp.float32],
    penalty: wp.array[mat66f],
    metric_status: wp.array[wp.int32],
    linearization_twist: wp.array[vec6f],
    prescribed_twist: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    block = wp.tid()
    world = block_world[block]
    dt = time_step[world]
    if world < 0 or metric_status[block] != METRIC_STATUS_VALID:
        return
    first = body_first[block]
    second = body_second[block]
    body_count = body_component.shape[0]
    if dt <= 0.0 or (first < 0 and second < 0) or first >= body_count or second >= body_count:
        return

    first_local = body_local[first] if first >= 0 else -1
    second_local = body_local[second] if second >= 0 else -1
    if first_local < 0 and second_local < 0:
        return
    component = body_component[first] if first_local >= 0 else body_component[second]
    if component < 0 or component >= dimensions.shape[0] or (second_local >= 0 and body_component[second] != component):
        return
    dimension = dimensions[component]

    offset = block_row_offset[block]
    count = block_row_count[block]
    matrix_offset = matrix_offsets[component]
    vector_offset = vector_offsets[component]
    linearization_velocity = vec6f(0.0)
    for row in range(6):
        if row < count:
            if first >= 0:
                linearization_velocity[row] += wp.dot(jacobian_first[offset + row], linearization_twist[first])
            if second >= 0:
                linearization_velocity[row] += wp.dot(jacobian_second[offset + row], linearization_twist[second])
            if first >= 0 and first_local < 0:
                linearization_velocity[row] -= wp.dot(jacobian_first[offset + row], prescribed_twist[first])
            if second >= 0 and second_local < 0:
                linearization_velocity[row] -= wp.dot(jacobian_second[offset + row], prescribed_twist[second])

    time_step_squared = dt * dt
    block_penalty = penalty[block]
    if first_local >= 0:
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            first_local,
            first_local,
            _structural_body_block(offset, count, jacobian_first, jacobian_first, block_penalty, time_step_squared),
        )
        _atomic_add_body_vector(
            right_hand_side,
            vector_offset,
            first_local,
            _structural_body_right_hand_side(
                offset,
                count,
                jacobian_first,
                residual,
                multiplier,
                block_penalty,
                linearization_velocity,
                dt,
            ),
        )
    if second_local >= 0:
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            second_local,
            second_local,
            _structural_body_block(offset, count, jacobian_second, jacobian_second, block_penalty, time_step_squared),
        )
        _atomic_add_body_vector(
            right_hand_side,
            vector_offset,
            second_local,
            _structural_body_right_hand_side(
                offset,
                count,
                jacobian_second,
                residual,
                multiplier,
                block_penalty,
                linearization_velocity,
                dt,
            ),
        )
    if first_local >= 0 and second_local >= 0:
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            first_local,
            second_local,
            _structural_body_block(offset, count, jacobian_first, jacobian_second, block_penalty, time_step_squared),
        )
        _atomic_add_body_block(
            matrix,
            matrix_offset,
            dimension,
            second_local,
            first_local,
            _structural_body_block(offset, count, jacobian_second, jacobian_first, block_penalty, time_step_squared),
        )


@wp.kernel
def _assemble_smooth_material_blocks(
    body_world: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    body_component: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    stress: wp.array[wp.float32],
    tangent_diagonal: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    prescribed_twist: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    material = wp.tid()
    first = body_first[material]
    second = body_second[material]
    body_count = body_component.shape[0]
    if (first < 0 and second < 0) or first >= body_count or second >= body_count:
        return
    world = body_world[first] if first >= 0 else body_world[second]
    dt = time_step[world]
    if dt <= 0.0:
        return

    first_local = body_local[first] if first >= 0 else -1
    second_local = body_local[second] if second >= 0 else -1
    if first_local < 0 and second_local < 0:
        return
    component = body_component[first] if first_local >= 0 else body_component[second]
    if component < 0 or component >= dimensions.shape[0] or (second_local >= 0 and body_component[second] != component):
        return
    dimension = dimensions[component]
    matrix_offset = matrix_offsets[component]
    vector_offset = vector_offsets[component]
    time_step_squared = dt * dt
    row_offset = 6 * material

    for material_row in range(6):
        row = row_offset + material_row
        first_jacobian = jacobian_first[row]
        second_jacobian = jacobian_second[row]
        tangent = tangent_diagonal[row]
        row_velocity = wp.float32(0.0)
        if first >= 0:
            row_velocity += wp.dot(first_jacobian, linearization_twist[first])
        if second >= 0:
            row_velocity += wp.dot(second_jacobian, linearization_twist[second])
        if first >= 0 and first_local < 0:
            row_velocity -= wp.dot(first_jacobian, prescribed_twist[first])
        if second >= 0 and second_local < 0:
            row_velocity -= wp.dot(second_jacobian, prescribed_twist[second])
        right_hand_side_scale = -dt * stress[row] + time_step_squared * tangent * row_velocity

        if first_local >= 0:
            _atomic_add_body_outer_product(
                matrix,
                matrix_offset,
                dimension,
                first_local,
                first_local,
                first_jacobian,
                first_jacobian,
                time_step_squared * tangent,
            )
            _atomic_add_body_vector(
                right_hand_side,
                vector_offset,
                first_local,
                right_hand_side_scale * first_jacobian,
            )
        if second_local >= 0:
            _atomic_add_body_outer_product(
                matrix,
                matrix_offset,
                dimension,
                second_local,
                second_local,
                second_jacobian,
                second_jacobian,
                time_step_squared * tangent,
            )
            _atomic_add_body_vector(
                right_hand_side,
                vector_offset,
                second_local,
                right_hand_side_scale * second_jacobian,
            )
        if first_local >= 0 and second_local >= 0:
            _atomic_add_body_outer_product(
                matrix,
                matrix_offset,
                dimension,
                first_local,
                second_local,
                first_jacobian,
                second_jacobian,
                time_step_squared * tangent,
            )
            _atomic_add_body_outer_product(
                matrix,
                matrix_offset,
                dimension,
                second_local,
                first_local,
                second_jacobian,
                first_jacobian,
                time_step_squared * tangent,
            )


@wp.kernel
def _add_simple_structural_metric_diagonal(
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    row_world: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    body_component: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_mass: wp.array[wp.float32],
    joint_metric_scale: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    if world < 0 or world >= joint_metric_scale.shape[0]:
        return
    first = body_first[row]
    second = body_second[row]
    body_count = body_component.shape[0]
    if (first < 0 and second < 0) or first >= body_count or second >= body_count:
        return
    first_local = body_local[first] if first >= 0 else -1
    second_local = body_local[second] if second >= 0 else -1
    if first_local < 0 and second_local < 0:
        return
    component = body_component[first] if first_local >= 0 else body_component[second]
    if component < 0 or component >= dimensions.shape[0] or (second_local >= 0 and body_component[second] != component):
        return
    dimension = dimensions[component]
    row_effective_mass = effective_mass[row]
    if not wp.isfinite(row_effective_mass) or row_effective_mass <= 0.0:
        return
    matrix_offset = matrix_offsets[component]
    scale = joint_metric_scale[world] * row_effective_mass
    if first_local >= 0:
        _atomic_add_body_outer_product(
            matrix,
            matrix_offset,
            dimension,
            first_local,
            first_local,
            jacobian_first[row],
            jacobian_first[row],
            scale,
        )
    if second_local >= 0:
        _atomic_add_body_outer_product(
            matrix,
            matrix_offset,
            dimension,
            second_local,
            second_local,
            jacobian_second[row],
            jacobian_second[row],
            scale,
        )


@wp.kernel
def _compute_body_weights_and_add(
    body_component: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    smooth_matrix: wp.array[wp.float32],
    mass: wp.array[wp.float32],
    inertia_world: wp.array[wp.mat33f],
    sigma: wp.float32,
    beta: wp.float32,
    mass_floor: wp.float32,
    inertia_floor: wp.float32,
    eta_floor: wp.float32,
    symmetry_tolerance: wp.float32,
    weight: wp.array[mat66f],
    inverse_weight: wp.array[mat66f],
    eta: wp.array[wp.float32],
    alpha: wp.array[wp.float32],
    status: wp.array[wp.int32],
    weighted_matrix: wp.array[wp.float32],
):
    body = wp.tid()
    component = body_component[body]
    if component < 0 or body_has_unilateral[body] == 0:
        weight[body] = mat66f(0.0)
        inverse_weight[body] = mat66f(0.0)
        eta[body] = 0.0
        alpha[body] = 0.0
        status[body] = BODY_WEIGHT_STATUS_VALID
        return
    local_body = body_local[body]
    dimension = dimensions[component]
    matrix_offset = matrix_offsets[component]
    body_offset = 6 * local_body
    smooth_diagonal = mat66f(0.0)
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, body_offset + row, body_offset + col)
            smooth_diagonal[row, col] = smooth_matrix[index]

    result = compute_body_weight_mass_proportional(
        smooth_diagonal,
        mass[body],
        inertia_world[body],
        sigma,
        beta,
        mass_floor,
        inertia_floor,
        eta_floor,
        symmetry_tolerance,
    )
    weight[body] = result.weight
    inverse_weight[body] = result.inverse_weight
    eta[body] = result.eta
    alpha[body] = result.alpha
    status[body] = result.status
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, body_offset + row, body_offset + col)
            weighted_matrix[index] += result.weight[row, col]


@wp.kernel
def _compute_anisotropic_body_weights_and_add(
    body_component: wp.array[wp.int32],
    body_local: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    smooth_matrix: wp.array[wp.float32],
    mass: wp.array[wp.float32],
    inertia_world: wp.array[wp.mat33f],
    sigma: wp.float32,
    beta: wp.float32,
    mass_floor: wp.float32,
    inertia_floor: wp.float32,
    eta_floor: wp.float32,
    symmetry_tolerance: wp.float32,
    weight: wp.array[mat66f],
    inverse_weight: wp.array[mat66f],
    eta: wp.array[wp.float32],
    eigenvalue_min: wp.array[wp.float32],
    eigenvalue_max: wp.array[wp.float32],
    status: wp.array[wp.int32],
    weighted_matrix: wp.array[wp.float32],
):
    body = wp.tid()
    component = body_component[body]
    if component < 0 or body_has_unilateral[body] == 0:
        weight[body] = mat66f(0.0)
        inverse_weight[body] = mat66f(0.0)
        eta[body] = 0.0
        eigenvalue_min[body] = 0.0
        eigenvalue_max[body] = 0.0
        status[body] = BODY_WEIGHT_STATUS_VALID
        return
    local_body = body_local[body]

    dimension = dimensions[component]
    matrix_offset = matrix_offsets[component]
    body_offset = 6 * local_body
    smooth_diagonal = mat66f(0.0)
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, body_offset + row, body_offset + col)
            smooth_diagonal[row, col] = smooth_matrix[index]

    result = compute_body_weight_anisotropic(
        smooth_diagonal,
        mass[body],
        inertia_world[body],
        sigma,
        beta,
        mass_floor,
        inertia_floor,
        eta_floor,
        symmetry_tolerance,
    )
    weight[body] = result.weight
    inverse_weight[body] = result.inverse_weight
    eta[body] = result.eta
    eigenvalue_min[body] = result.eigenvalue_min
    eigenvalue_max[body] = result.eigenvalue_max
    status[body] = result.status
    for row in range(6):
        for col in range(6):
            index = _matrix_index(matrix_offset, dimension, body_offset + row, body_offset + col)
            weighted_matrix[index] += result.weight[row, col]


@wp.kernel
def _build_candidate_right_hand_side(
    body_vector_index: wp.array[wp.int32],
    smooth_right_hand_side: wp.array[wp.float32],
    weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
    candidate_right_hand_side: wp.array[wp.float32],
):
    body = wp.tid()
    body_offset = body_vector_index[body]
    if body_offset < 0:
        return
    weighted_target = weight[body] @ (projected_twist[body] + splitting_dual[body])
    for row in range(6):
        candidate_right_hand_side[body_offset + row] = smooth_right_hand_side[body_offset + row] + weighted_target[row]


@wp.kernel
def _build_candidate_right_hand_side_with_effort(
    body_vector_index: wp.array[wp.int32],
    smooth_right_hand_side: wp.array[wp.float32],
    weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
    body_effort_offset: wp.array[wp.int32],
    body_effort_index: wp.array[wp.int32],
    body_effort_side: wp.array[wp.int32],
    effort_dynamic_row: wp.array[wp.int32],
    dynamic_jacobian_first: wp.array[vec6f],
    dynamic_jacobian_second: wp.array[vec6f],
    effort_counter_applied: wp.array[wp.float32],
    candidate_right_hand_side: wp.array[wp.float32],
):
    body = wp.tid()
    body_offset = body_vector_index[body]
    if body_offset < 0:
        return
    target = weight[body] @ (projected_twist[body] + splitting_dual[body])
    effort_start = body_effort_offset[body]
    effort_end = body_effort_offset[body + 1]
    for incidence in range(effort_start, effort_end):
        effort = body_effort_index[incidence]
        dynamic_row = effort_dynamic_row[effort]
        jacobian = dynamic_jacobian_first[dynamic_row]
        if body_effort_side[incidence] != 0:
            jacobian = dynamic_jacobian_second[dynamic_row]
        target += effort_counter_applied[effort] * jacobian
    for row in range(6):
        candidate_right_hand_side[body_offset + row] = smooth_right_hand_side[body_offset + row] + target[row]


class BatchedPrimalBodySystem:
    """Allocated dense contact-free systems over independent body components.

    World-level bookkeeping remains in model body order. The dense matrices and
    vectors are instead packed by ``body_components`` so disconnected
    articulations can be factorized independently. If no components are
    supplied, each world is retained as one factor block.
    """

    def __init__(
        self,
        body_counts: Sequence[int],
        device: wp.DeviceLike = None,
        *,
        body_components: Sequence[Sequence[int]] | None = None,
        dynamic_bodies: Sequence[int] | None = None,
    ):
        if len(body_counts) == 0:
            raise ValueError("At least one world is required.")
        if any(not isinstance(count, int) or count < 0 for count in body_counts) or sum(body_counts) == 0:
            raise ValueError("Body counts must be non-negative and include at least one active body.")

        self.device = wp.get_device(device)
        self.body_counts = tuple(body_counts)
        self.num_worlds = len(body_counts)
        self.num_bodies = sum(body_counts)
        world_body_offsets = [0]
        for count in body_counts:
            world_body_offsets.append(world_body_offsets[-1] + count)
        self.body_offsets = tuple(world_body_offsets)

        body_world_host = [0] * self.num_bodies
        for world, (start, end) in enumerate(pairwise(world_body_offsets)):
            for body in range(start, end):
                body_world_host[body] = world

        if dynamic_bodies is None:
            dynamic_body_indices = tuple(range(self.num_bodies))
        else:
            dynamic_body_indices = tuple(int(body) for body in dynamic_bodies)
            if len(set(dynamic_body_indices)) != len(dynamic_body_indices):
                raise ValueError("dynamic_bodies must not contain duplicates.")
            if any(body < 0 or body >= self.num_bodies for body in dynamic_body_indices):
                raise ValueError("dynamic_bodies must reference packed bodies.")
        dynamic_body_set = set(dynamic_body_indices)

        if body_components is None:
            components = [
                tuple(body for body in range(start, end) if body in dynamic_body_set)
                for start, end in pairwise(world_body_offsets)
            ]
            components = [component for component in components if component]
        else:
            components = [tuple(int(body) for body in component) for component in body_components]
            if (not components and dynamic_body_indices) or any(len(component) == 0 for component in components):
                raise ValueError("Each body component must contain at least one body.")
            flattened = [body for component in components for body in component]
            if sorted(flattened) != sorted(dynamic_body_indices):
                raise ValueError("body_components must contain every dynamic body exactly once.")
            for component in components:
                world = body_world_host[component[0]]
                if any(body_world_host[body] != world for body in component):
                    raise ValueError("A body component cannot span multiple worlds.")

        self.body_components = tuple(components)
        self.dynamic_bodies = dynamic_body_indices
        self.num_dynamic_bodies = len(dynamic_body_indices)
        if components:
            self.block_body_counts = tuple(len(component) for component in components)
            self.block_world_host = tuple(body_world_host[component[0]] for component in components)
            storage_dimensions = [6 * count for count in self.block_body_counts]
            active_dimensions = storage_dimensions
        else:
            # Dense multi-linear storage requires one positive allocation size,
            # while the active dimension remains zero.
            self.block_body_counts = (0,)
            self.block_world_host = (0,)
            storage_dimensions = [1]
            active_dimensions = [0]
        self.num_blocks = len(self.block_body_counts)

        self.info = DenseSquareMultiLinearInfo()
        self.info.finalize(dimensions=storage_dimensions, dtype=wp.float32, itype=wp.int32, device=self.device)
        if active_dimensions != storage_dimensions:
            self.info.dim = wp.array(active_dimensions, dtype=wp.int32, device=self.device)
        self.smooth_matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)
        self.weight_metric_matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)
        self.weighted_matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)
        self.right_hand_side = wp.zeros(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self.candidate_right_hand_side = wp.zeros(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self.solution = wp.zeros(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self.weight = wp.zeros(self.num_bodies, dtype=mat66f, device=self.device)
        self.inverse_weight = wp.zeros(self.num_bodies, dtype=mat66f, device=self.device)
        self.weight_eta = wp.zeros(self.num_bodies, dtype=wp.float32, device=self.device)
        self.weight_alpha = wp.zeros(self.num_bodies, dtype=wp.float32, device=self.device)
        self.weight_eigenvalue_min = wp.zeros(self.num_bodies, dtype=wp.float32, device=self.device)
        self.weight_eigenvalue_max = wp.zeros(self.num_bodies, dtype=wp.float32, device=self.device)
        self.weight_status = wp.zeros(self.num_bodies, dtype=wp.int32, device=self.device)
        self._zero_twist = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)

        body_block_host = [-1] * self.num_bodies
        body_local_host = [-1] * self.num_bodies
        block_vector_offsets = []
        offset = 0
        for count in self.block_body_counts:
            block_vector_offsets.append(offset)
            offset += 6 * count
        body_vector_index_host = [-1] * self.num_bodies
        for block, component in enumerate(components):
            for local, body in enumerate(component):
                body_block_host[body] = block
                body_local_host[body] = local
                body_vector_index_host[body] = block_vector_offsets[block] + 6 * local

        self.body_world_host = tuple(body_world_host)
        self.body_block_host = tuple(body_block_host)
        self.body_local_host = tuple(body_local_host)
        self.body_vector_index_host = tuple(body_vector_index_host)
        self.body_world = wp.array(body_world_host, dtype=wp.int32, device=self.device)
        self.body_block = wp.array(body_block_host, dtype=wp.int32, device=self.device)
        self.body_local = wp.array(body_local_host, dtype=wp.int32, device=self.device)
        self.body_vector_index = wp.array(body_vector_index_host, dtype=wp.int32, device=self.device)
        self.block_world = wp.array(self.block_world_host, dtype=wp.int32, device=self.device)
        self.body_weight_enabled = wp.ones(self.num_bodies, dtype=wp.int32, device=self.device)
        self.block_has_unilateral = wp.ones(self.num_blocks, dtype=wp.int32, device=self.device)
        self.selective_body_weights = False
        self.operator = DenseLinearOperatorData(info=self.info, mat=self.weighted_matrix)
        # Factorization benefits from fewer wide panels, while the repeated
        # single-RHS solves retain the smaller tile for better occupancy.
        self.linear_solver = HybridLLTBlockedSolver(
            operator=self.operator,
            factorize_block_size=64,
            solve_block_dim=256,
            dtype=wp.float32,
            device=self.device,
        )
        self._mass: wp.array[wp.float32] | None = None
        self._inertia_world: wp.array[wp.mat33f] | None = None

    def reset(self) -> None:
        """Clear assembled matrices, vectors, weights, and factorization state."""
        self.smooth_matrix.zero_()
        self.weight_metric_matrix.zero_()
        self.weighted_matrix.zero_()
        self.right_hand_side.zero_()
        self.candidate_right_hand_side.zero_()
        self.solution.zero_()
        self.weight.zero_()
        self.inverse_weight.zero_()
        self.weight_eta.zero_()
        self.weight_alpha.zero_()
        self.weight_eigenvalue_min.zero_()
        self.weight_eigenvalue_max.zero_()
        self.weight_status.zero_()
        self.body_weight_enabled.zero_()
        self.block_has_unilateral.zero_()
        self.linear_solver.reset()

    def assemble_bodies(
        self,
        mass: wp.array[wp.float32],
        inertia_world: wp.array[wp.mat33f],
        velocity_previous: wp.array[vec6f],
        force_explicit: wp.array[vec6f],
        time_step: wp.array[wp.float32],
    ) -> None:
        """Reset and assemble body inertia and explicit-force terms."""
        if any(array.shape[0] != self.num_bodies for array in (mass, inertia_world, velocity_previous, force_explicit)):
            raise ValueError("Body input arrays must contain one entry per packed active body.")
        validate_world_time_step(time_step, self.num_worlds, self.device)
        self.reset()
        self._mass = mass
        self._inertia_world = inertia_world
        wp.launch(
            _assemble_body_inertial_systems,
            dim=self.num_bodies,
            inputs=[
                self.body_world,
                self.body_block,
                self.body_local,
                self.info.dim,
                self.info.mio,
                self.info.vio,
                mass,
                inertia_world,
                velocity_previous,
                force_explicit,
                time_step,
            ],
            outputs=[self.smooth_matrix, self.right_hand_side],
            device=self.device,
        )

    def add_dynamic_rows(
        self,
        row_world: wp.array[wp.int32],
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        effective_inertia: wp.array[wp.float32],
        free_velocity: wp.array[wp.float32],
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Add implicit joint-dynamics rows to the smooth system."""
        row_count = row_world.shape[0]
        if row_count == 0:
            return
        if any(
            array.shape[0] != row_count
            for array in (
                body_first,
                body_second,
                jacobian_first,
                jacobian_second,
                effective_inertia,
                free_velocity,
            )
        ):
            raise ValueError("Dynamic-row arrays must have identical lengths.")
        if prescribed_twist is None:
            prescribed_twist = self._zero_twist
        elif prescribed_twist.shape[0] != self.num_bodies:
            raise ValueError("prescribed_twist must contain one entry per packed body.")
        wp.launch(
            _assemble_dynamic_joint_rows,
            dim=row_count,
            inputs=[
                self.info.dim,
                self.info.mio,
                self.info.vio,
                body_first,
                body_second,
                self.body_block,
                self.body_local,
                jacobian_first,
                jacobian_second,
                effective_inertia,
                free_velocity,
                prescribed_twist,
            ],
            outputs=[self.smooth_matrix, self.right_hand_side],
            device=self.device,
        )

    def add_structural_rows(
        self,
        row_world: wp.array[wp.int32],
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        residual: wp.array[wp.float32],
        multiplier: wp.array[wp.float32],
        effective_mass: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
        time_step: wp.array[wp.float32],
        joint_penalty_scale: wp.array[wp.float32],
        penalty: wp.array[wp.float32],
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Add augmented structural rows and write their derived penalties."""
        row_count = row_world.shape[0]
        if row_count == 0:
            return
        validate_world_time_step(time_step, self.num_worlds, self.device)
        if joint_penalty_scale.shape[0] != self.num_worlds:
            raise ValueError("joint_penalty_scale must contain one entry per world.")
        if linearization_twist.shape[0] != self.num_bodies:
            raise ValueError("linearization_twist must contain one entry per packed body.")
        if prescribed_twist is None:
            prescribed_twist = self._zero_twist
        elif prescribed_twist.shape[0] != self.num_bodies:
            raise ValueError("prescribed_twist must contain one entry per packed body.")
        if any(
            array.shape[0] != row_count
            for array in (
                body_first,
                body_second,
                jacobian_first,
                jacobian_second,
                residual,
                multiplier,
                effective_mass,
                penalty,
            )
        ):
            raise ValueError("Structural-row arrays must have identical lengths.")
        wp.launch(
            _assemble_structural_joint_rows,
            dim=row_count,
            inputs=[
                self.info.dim,
                self.info.mio,
                self.info.vio,
                row_world,
                body_first,
                body_second,
                self.body_block,
                self.body_local,
                jacobian_first,
                jacobian_second,
                residual,
                multiplier,
                effective_mass,
                linearization_twist,
                prescribed_twist,
                time_step,
                joint_penalty_scale,
            ],
            outputs=[penalty, self.smooth_matrix, self.right_hand_side],
            device=self.device,
        )

    def add_structural_blocks(
        self,
        block_world: wp.array[wp.int32],
        block_row_offset: wp.array[wp.int32],
        block_row_count: wp.array[wp.int32],
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        residual: wp.array[wp.float32],
        multiplier: wp.array[wp.float32],
        penalty: wp.array[mat66f],
        metric_status: wp.array[wp.int32],
        linearization_twist: wp.array[vec6f],
        time_step: wp.array[wp.float32],
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Add full structural joint blocks to the smooth system."""
        block_count = block_world.shape[0]
        if block_count == 0:
            return
        validate_world_time_step(time_step, self.num_worlds, self.device)
        if linearization_twist.shape[0] != self.num_bodies:
            raise ValueError("linearization_twist must contain one entry per packed body.")
        if prescribed_twist is None:
            prescribed_twist = self._zero_twist
        elif prescribed_twist.shape[0] != self.num_bodies:
            raise ValueError("prescribed_twist must contain one entry per packed body.")
        if any(
            array.shape[0] != block_count
            for array in (
                block_row_offset,
                block_row_count,
                body_first,
                body_second,
                penalty,
                metric_status,
            )
        ):
            raise ValueError("Structural-block arrays must have identical lengths.")
        row_count = residual.shape[0]
        if any(array.shape[0] != row_count for array in (jacobian_first, jacobian_second, multiplier)):
            raise ValueError("Structural-row arrays must have identical lengths.")
        wp.launch(
            _assemble_structural_joint_blocks,
            dim=block_count,
            inputs=[
                self.info.dim,
                self.info.mio,
                self.info.vio,
                block_world,
                block_row_offset,
                block_row_count,
                body_first,
                body_second,
                self.body_block,
                self.body_local,
                jacobian_first,
                jacobian_second,
                residual,
                multiplier,
                penalty,
                metric_status,
                linearization_twist,
                prescribed_twist,
                time_step,
            ],
            outputs=[self.smooth_matrix, self.right_hand_side],
            device=self.device,
        )

    def add_smooth_material_blocks(
        self,
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        stress: wp.array[wp.float32],
        tangent_diagonal: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
        time_step: wp.array[wp.float32],
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Add six-row diagonal-tangent material elements to the smooth system."""
        material_count = body_first.shape[0]
        if material_count == 0:
            return
        validate_world_time_step(time_step, self.num_worlds, self.device)
        if body_second.shape[0] != material_count:
            raise ValueError("Material endpoint arrays must have identical lengths.")
        if linearization_twist.shape[0] != self.num_bodies:
            raise ValueError("linearization_twist must contain one entry per packed body.")
        if prescribed_twist is None:
            prescribed_twist = self._zero_twist
        elif prescribed_twist.shape[0] != self.num_bodies:
            raise ValueError("prescribed_twist must contain one entry per packed body.")
        row_count = 6 * material_count
        if any(array.shape[0] != row_count for array in (jacobian_first, jacobian_second, stress, tangent_diagonal)):
            raise ValueError("Material row arrays must contain six entries per material.")
        wp.launch(
            _assemble_smooth_material_blocks,
            dim=material_count,
            inputs=[
                self.body_world,
                self.info.dim,
                self.info.mio,
                self.info.vio,
                body_first,
                body_second,
                self.body_block,
                self.body_local,
                jacobian_first,
                jacobian_second,
                stress,
                tangent_diagonal,
                linearization_twist,
                prescribed_twist,
                time_step,
            ],
            outputs=[self.smooth_matrix, self.right_hand_side],
            device=self.device,
        )

    def build_simple_joint_aware_weight_metric(
        self,
        row_world: wp.array[wp.int32],
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        effective_mass: wp.array[wp.float32],
        joint_metric_scale: wp.array[wp.float32],
    ) -> wp.array[wp.float32]:
        """Build body-local metric blocks from the simple structural row penalty."""
        row_count = row_world.shape[0]
        if joint_metric_scale.shape[0] != self.num_worlds:
            raise ValueError("joint_metric_scale must contain one entry per world.")
        if any(
            array.shape[0] != row_count
            for array in (
                body_first,
                body_second,
                jacobian_first,
                jacobian_second,
                effective_mass,
            )
        ):
            raise ValueError("Structural-row arrays must have identical lengths.")
        wp.copy(self.weight_metric_matrix, self.smooth_matrix)
        if row_count > 0:
            wp.launch(
                _add_simple_structural_metric_diagonal,
                dim=row_count,
                inputs=[
                    self.info.dim,
                    self.info.mio,
                    row_world,
                    body_first,
                    body_second,
                    self.body_block,
                    self.body_local,
                    jacobian_first,
                    jacobian_second,
                    effective_mass,
                    joint_metric_scale,
                ],
                outputs=[self.weight_metric_matrix],
                device=self.device,
            )
        return self.weight_metric_matrix

    def _mark_weighted_bodies(self, body_has_unilateral: wp.array[wp.int32] | None) -> None:
        if body_has_unilateral is None:
            self.body_weight_enabled.fill_(1)
            return
        if body_has_unilateral.shape[0] != self.num_bodies:
            raise ValueError("body_has_unilateral must contain one entry per body.")
        if self.selective_body_weights:
            wp.copy(self.body_weight_enabled, body_has_unilateral)
            return
        self.block_has_unilateral.zero_()
        wp.launch(
            _mark_blocks_with_unilaterals,
            dim=self.num_bodies,
            inputs=[self.body_block, body_has_unilateral],
            outputs=[self.block_has_unilateral],
            device=self.device,
        )
        wp.launch(
            _enable_bodies_in_weighted_blocks,
            dim=self.num_bodies,
            inputs=[self.body_block, self.block_has_unilateral],
            outputs=[self.body_weight_enabled],
            device=self.device,
        )

    def build_weighted_matrix(
        self,
        metric_matrix: wp.array[wp.float32] | None = None,
        body_has_unilateral: wp.array[wp.int32] | None = None,
        sigma: float = BODY_WEIGHT_SIGMA_DEFAULT,
        beta: float = BODY_WEIGHT_BETA_DEFAULT,
        mass_floor: float = 1.0e-8,
        inertia_floor: float = 1.0e-10,
        eta_floor: float = 1.0e-6,
        symmetry_tolerance: float = 1.0e-5,
    ) -> None:
        """Compute body weights and assemble ``A + W``."""
        if self._mass is None or self._inertia_world is None:
            raise ValueError("assemble_bodies() must be called before build_weighted_matrix().")
        if metric_matrix is None:
            metric_matrix = self.smooth_matrix
        elif metric_matrix.shape[0] != self.info.total_mat_size:
            raise ValueError("metric_matrix must match the packed body matrix storage.")
        self._mark_weighted_bodies(body_has_unilateral)
        wp.copy(self.weighted_matrix, self.smooth_matrix)
        wp.launch(
            _compute_body_weights_and_add,
            dim=self.num_bodies,
            inputs=[
                self.body_block,
                self.body_local,
                self.body_weight_enabled,
                self.info.dim,
                self.info.mio,
                metric_matrix,
                self._mass,
                self._inertia_world,
                sigma,
                beta,
                mass_floor,
                inertia_floor,
                eta_floor,
                symmetry_tolerance,
            ],
            outputs=[
                self.weight,
                self.inverse_weight,
                self.weight_eta,
                self.weight_alpha,
                self.weight_status,
                self.weighted_matrix,
            ],
            device=self.device,
        )

    def build_anisotropic_weighted_matrix(
        self,
        metric_matrix: wp.array[wp.float32] | None = None,
        body_has_unilateral: wp.array[wp.int32] | None = None,
        sigma: float = BODY_WEIGHT_SIGMA_DEFAULT,
        beta: float = BODY_WEIGHT_BETA_DEFAULT,
        mass_floor: float = 1.0e-8,
        inertia_floor: float = 1.0e-10,
        eta_floor: float = 1.0e-6,
        symmetry_tolerance: float = 1.0e-5,
    ) -> None:
        """Assemble and clamp the full body-space metric."""
        if self._mass is None or self._inertia_world is None:
            raise ValueError("assemble_bodies() must be called before building weights.")
        if metric_matrix is None:
            metric_matrix = self.smooth_matrix
        elif metric_matrix.shape[0] != self.info.total_mat_size:
            raise ValueError("metric_matrix must match the packed body matrix storage.")
        self._mark_weighted_bodies(body_has_unilateral)
        wp.copy(self.weighted_matrix, self.smooth_matrix)
        wp.launch(
            _compute_anisotropic_body_weights_and_add,
            dim=self.num_bodies,
            inputs=[
                self.body_block,
                self.body_local,
                self.body_weight_enabled,
                self.info.dim,
                self.info.mio,
                metric_matrix,
                self._mass,
                self._inertia_world,
                sigma,
                beta,
                mass_floor,
                inertia_floor,
                eta_floor,
                symmetry_tolerance,
            ],
            outputs=[
                self.weight,
                self.inverse_weight,
                self.weight_eta,
                self.weight_eigenvalue_min,
                self.weight_eigenvalue_max,
                self.weight_status,
                self.weighted_matrix,
            ],
            device=self.device,
        )

    def factorize(self) -> None:
        """Factorize the current weighted matrices with batched dense LLT."""
        self.linear_solver.compute(self.weighted_matrix)

    def build_candidate_right_hand_side(
        self,
        projected_twist: wp.array[vec6f],
        splitting_dual: wp.array[vec6f],
    ) -> None:
        """Build ``f + W (p + lambda)`` for one splitting iteration."""
        if projected_twist.shape[0] != self.num_bodies or splitting_dual.shape[0] != self.num_bodies:
            raise ValueError("Splitting vectors must contain one entry per packed active body.")
        wp.launch(
            _build_candidate_right_hand_side,
            dim=self.num_bodies,
            inputs=[
                self.body_vector_index,
                self.right_hand_side,
                self.weight,
                projected_twist,
                splitting_dual,
            ],
            outputs=[self.candidate_right_hand_side],
            device=self.device,
        )

    def solve_factorized(self) -> None:
        """Solve the current right-hand sides using the cached LLT factors."""
        self.linear_solver.solve(self.right_hand_side, self.solution)

    def build_candidate_right_hand_side_with_effort(
        self,
        projected_twist: wp.array[vec6f],
        splitting_dual: wp.array[vec6f],
        body_effort_offset: wp.array[wp.int32],
        body_effort_index: wp.array[wp.int32],
        body_effort_side: wp.array[wp.int32],
        effort_dynamic_row: wp.array[wp.int32],
        dynamic_jacobian_first: wp.array[vec6f],
        dynamic_jacobian_second: wp.array[vec6f],
        effort_counter_applied: wp.array[wp.float32],
    ) -> None:
        """Build a candidate right-hand side with finite-drive counter-impulses."""
        if projected_twist.shape[0] != self.num_bodies or splitting_dual.shape[0] != self.num_bodies:
            raise ValueError("Splitting vectors must contain one entry per packed active body.")
        if body_effort_offset.shape[0] != self.num_bodies + 1:
            raise ValueError("Effort offsets must contain one interval per packed active body.")
        wp.launch(
            _build_candidate_right_hand_side_with_effort,
            dim=self.num_bodies,
            inputs=[
                self.body_vector_index,
                self.right_hand_side,
                self.weight,
                projected_twist,
                splitting_dual,
                body_effort_offset,
                body_effort_index,
                body_effort_side,
                effort_dynamic_row,
                dynamic_jacobian_first,
                dynamic_jacobian_second,
                effort_counter_applied,
            ],
            outputs=[self.candidate_right_hand_side],
            device=self.device,
        )

    def solve_candidate(
        self,
        projected_twist: wp.array[vec6f],
        splitting_dual: wp.array[vec6f],
    ) -> None:
        """Build and solve the weighted splitting candidate system."""
        self.build_candidate_right_hand_side(projected_twist, splitting_dual)
        self.linear_solver.solve(self.candidate_right_hand_side, self.solution)

    def solve_candidate_with_effort(
        self,
        projected_twist: wp.array[vec6f],
        splitting_dual: wp.array[vec6f],
        body_effort_offset: wp.array[wp.int32],
        body_effort_index: wp.array[wp.int32],
        body_effort_side: wp.array[wp.int32],
        effort_dynamic_row: wp.array[wp.int32],
        dynamic_jacobian_first: wp.array[vec6f],
        dynamic_jacobian_second: wp.array[vec6f],
        effort_counter_applied: wp.array[wp.float32],
    ) -> None:
        """Build and solve a candidate system with finite-drive corrections."""
        self.build_candidate_right_hand_side_with_effort(
            projected_twist,
            splitting_dual,
            body_effort_offset,
            body_effort_index,
            body_effort_side,
            effort_dynamic_row,
            dynamic_jacobian_first,
            dynamic_jacobian_second,
            effort_counter_applied,
        )
        self.linear_solver.solve(self.candidate_right_hand_side, self.solution)

    def factorize_and_solve(self) -> None:
        """Factorize ``A + W`` and solve its contact-free right-hand side."""
        self.factorize()
        self.solve_factorized()
