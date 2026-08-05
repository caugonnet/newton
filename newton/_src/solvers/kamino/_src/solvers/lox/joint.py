# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural-joint solves for the body-space contact split."""

from __future__ import annotations

import warp as wp

from ...core.types import mat66f, vec6f
from ...linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, LLTBlockedRCMSolver
from .joint_factorize import make_batched_body_solve_kernel
from .linear import HybridLLTBlockedSolver
from .system import BatchedPrimalBodySystem
from .time import validate_world_time_step, validate_world_time_steps

__all__ = ["BatchedStructuralJointSolver"]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _permute_body_response_rows(
    body_dimensions: wp.array[wp.int32],
    body_vector_offsets: wp.array[wp.int32],
    joint_dimensions: wp.array[wp.int32],
    response_offsets: wp.array[wp.int32],
    response_leading_dimensions: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    component, reordered_row, column = wp.tid()
    if reordered_row >= body_dimensions[component] or column >= joint_dimensions[component]:
        return
    original_row = permutation[body_vector_offsets[component] + reordered_row]
    response_offset = response_offsets[component]
    leading_dimension = response_leading_dimensions[component]
    destination[response_offset + reordered_row * leading_dimension + column] = source[
        response_offset + original_row * leading_dimension + column
    ]


@wp.kernel
def _unpermute_body_response_rows(
    body_dimensions: wp.array[wp.int32],
    body_vector_offsets: wp.array[wp.int32],
    joint_dimensions: wp.array[wp.int32],
    response_offsets: wp.array[wp.int32],
    response_leading_dimensions: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    component, reordered_row, column = wp.tid()
    if reordered_row >= body_dimensions[component] or column >= joint_dimensions[component]:
        return
    original_row = permutation[body_vector_offsets[component] + reordered_row]
    response_offset = response_offsets[component]
    leading_dimension = response_leading_dimensions[component]
    destination[response_offset + original_row * leading_dimension + column] = source[
        response_offset + reordered_row * leading_dimension + column
    ]


@wp.func
def _load_body_vector(
    values: wp.array[wp.float32],
    body_vector_index: wp.array[wp.int32],
    prescribed_twist: wp.array[vec6f],
    body: wp.int32,
) -> vec6f:
    result = vec6f(0.0)
    if body >= 0:
        offset = body_vector_index[body]
        if offset >= 0:
            for axis in range(6):
                result[axis] = values[offset + axis]
        else:
            result = prescribed_twist[body]
    return result


@wp.kernel
def _build_joint_basis_body_right_hand_sides(
    body_dimensions: wp.array[wp.int32],
    joint_dimensions: wp.array[wp.int32],
    joint_vector_offsets: wp.array[wp.int32],
    response_offsets: wp.array[wp.int32],
    response_leading_dimensions: wp.array[wp.int32],
    vector_row: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    body_value: wp.array[wp.float32],
):
    component, row, column = wp.tid()
    if row >= body_dimensions[component] or column >= joint_dimensions[component]:
        return
    value = wp.float32(0.0)
    flat_row = vector_row[joint_vector_offsets[component] + column]
    body = row // 6
    axis = row - 6 * body
    if body_first[flat_row] == body:
        value += jacobian_first[flat_row][axis]
    if body_second[flat_row] == body:
        value += jacobian_second[flat_row][axis]
    body_value[response_offsets[component] + row * response_leading_dimensions[component] + column] = value


@wp.kernel
def _build_joint_right_hand_side(
    inverse_time_step: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    row_vector_index: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    residual: wp.array[wp.float32],
    body_vector_index: wp.array[wp.int32],
    prescribed_twist: wp.array[vec6f],
    linearization_twist: wp.array[vec6f],
    free_body_value: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    row = wp.tid()
    index = row_vector_index[row]
    if index < 0:
        return
    free_velocity = wp.float32(0.0)
    linearization_velocity = wp.float32(0.0)
    first_global = body_first_global[row]
    second_global = body_second_global[row]
    if first_global >= 0:
        free_velocity += wp.dot(
            jacobian_first[row],
            _load_body_vector(free_body_value, body_vector_index, prescribed_twist, first_global),
        )
        linearization_velocity += wp.dot(jacobian_first[row], linearization_twist[first_global])
    if second_global >= 0:
        free_velocity += wp.dot(
            jacobian_second[row],
            _load_body_vector(free_body_value, body_vector_index, prescribed_twist, second_global),
        )
        linearization_velocity += wp.dot(jacobian_second[row], linearization_twist[second_global])
    right_hand_side[index] = inverse_time_step[row_world[row]] * residual[row] + free_velocity - linearization_velocity


@wp.kernel
def _build_blended_joint_residual(
    inverse_time_step: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    row_vector_index: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    residual: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    projected_fraction: wp.float32,
    world_has_unilateral: wp.array[wp.bool],
    right_hand_side: wp.array[wp.float32],
):
    row = wp.tid()
    index = row_vector_index[row]
    if index < 0:
        return
    if not world_has_unilateral[row_world[row]]:
        right_hand_side[index] = 0.0
        return
    global_velocity = wp.float32(0.0)
    projected_velocity = wp.float32(0.0)
    linearization_velocity = wp.float32(0.0)
    first = body_first_global[row]
    second = body_second_global[row]
    if first >= 0:
        global_velocity += wp.dot(jacobian_first[row], global_twist[first])
        projected_velocity += wp.dot(jacobian_first[row], projected_twist[first])
        linearization_velocity += wp.dot(jacobian_first[row], linearization_twist[first])
    if second >= 0:
        global_velocity += wp.dot(jacobian_second[row], global_twist[second])
        projected_velocity += wp.dot(jacobian_second[row], projected_twist[second])
        linearization_velocity += wp.dot(jacobian_second[row], linearization_twist[second])
    blended_velocity = global_velocity + projected_fraction * (projected_velocity - global_velocity)
    right_hand_side[index] = (
        inverse_time_step[row_world[row]] * residual[row] + blended_velocity - linearization_velocity
    )


@wp.kernel
def _write_joint_multipliers(
    inverse_time_step: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    row_vector_index: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    impulse: wp.array[wp.float32],
    multiplier: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    row = wp.tid()
    if not world_active[row_world[row]]:
        return
    index = row_vector_index[row]
    if index < 0:
        multiplier[row] = 0.0
        destination[multiplier_index[row]] = 0.0
        return
    force = inverse_time_step[row_world[row]] * impulse[index]
    multiplier[row] = force
    destination[multiplier_index[row]] = -force


@wp.func
def _load_body_response(
    response: wp.array[wp.float32],
    response_offset: wp.int32,
    leading_dimension: wp.int32,
    body: wp.int32,
    column: wp.int32,
) -> vec6f:
    result = vec6f(0.0)
    if body >= 0:
        offset = response_offset + 6 * body * leading_dimension + column
        for axis in range(6):
            result[axis] = response[offset + axis * leading_dimension]
    return result


@wp.kernel
def _assemble_schur_from_body_response(
    joint_dimensions: wp.array[wp.int32],
    joint_matrix_offsets: wp.array[wp.int32],
    joint_vector_offsets: wp.array[wp.int32],
    response_offsets: wp.array[wp.int32],
    response_leading_dimensions: wp.array[wp.int32],
    vector_row: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    response: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
):
    component, row, column = wp.tid()
    joint_dimension = joint_dimensions[component]
    if row >= joint_dimension or column >= joint_dimension:
        return
    flat_row = vector_row[joint_vector_offsets[component] + row]
    value = wp.float32(0.0)
    first = body_first[flat_row]
    second = body_second[flat_row]
    response_offset = response_offsets[component]
    leading_dimension = response_leading_dimensions[component]
    if first >= 0:
        value += wp.dot(
            jacobian_first[flat_row],
            _load_body_response(response, response_offset, leading_dimension, first, column),
        )
    if second >= 0:
        value += wp.dot(
            jacobian_second[flat_row],
            _load_body_response(response, response_offset, leading_dimension, second, column),
        )
    matrix[joint_matrix_offsets[component] + row * joint_dimension + column] = value


@wp.kernel
def _factor_schur_block_diagonal(
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    block_component: wp.array[wp.int32],
    block_row_offset: wp.array[wp.int32],
    block_row_count: wp.array[wp.int32],
    row_vector_index: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    factor: wp.array[mat66f],
    status: wp.array[wp.int32],
):
    block = wp.tid()
    component = block_component[block]
    count = block_row_count[block]
    identity = mat66f(0.0)
    for row in range(6):
        if row < count:
            identity[row, row] = 1.0
    factor[block] = identity
    status[block] = 0
    if component < 0 or component >= dimensions.shape[0] or count < 1 or count > 6:
        return

    dimension = dimensions[component]
    matrix_offset = matrix_offsets[component]
    vector_offset = vector_offsets[component]
    row_offset = block_row_offset[block]
    diagonal_scale = wp.float32(0.0)
    for row in range(6):
        if row < count:
            index = row_vector_index[row_offset + row] - vector_offset
            diagonal = matrix[matrix_offset + index * dimension + index]
            if not wp.isfinite(diagonal) or diagonal <= 0.0:
                return
            diagonal_scale = wp.max(diagonal_scale, diagonal)
    pivot_tolerance = 1.0e-12 * wp.max(1.0, diagonal_scale)

    lower = mat66f(0.0)
    for row in range(6):
        if row < count:
            row_index = row_vector_index[row_offset + row] - vector_offset
            for col in range(6):
                if col <= row and col < count:
                    col_index = row_vector_index[row_offset + col] - vector_offset
                    value = matrix[matrix_offset + row_index * dimension + col_index]
                    if not wp.isfinite(value):
                        return
                    for inner in range(6):
                        if inner < col:
                            value -= lower[row, inner] * lower[col, inner]
                    if row == col:
                        if not wp.isfinite(value) or value <= pivot_tolerance:
                            return
                        lower[row, col] = wp.sqrt(value)
                    else:
                        lower[row, col] = value / lower[col, col]

    inverse_lower = mat66f(0.0)
    for col in range(6):
        if col < count:
            for row in range(6):
                if row >= col and row < count:
                    value = wp.float32(1.0) if row == col else wp.float32(0.0)
                    for inner in range(6):
                        if inner >= col and inner < row:
                            value -= lower[row, inner] * inverse_lower[inner, col]
                    inverse_lower[row, col] = value / lower[row, row]

    inverse_transpose = mat66f(0.0)
    for row in range(6):
        if row < count:
            for col in range(6):
                if col < count:
                    inverse_transpose[row, col] = inverse_lower[col, row]
    factor[block] = inverse_transpose
    status[block] = 1


@wp.kernel
def _transform_schur_with_joint_blocks(
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    vector_row: wp.array[wp.int32],
    row_vector_index: wp.array[wp.int32],
    row_block: wp.array[wp.int32],
    block_row_offset: wp.array[wp.int32],
    block_row_count: wp.array[wp.int32],
    factor: wp.array[mat66f],
    source: wp.array[wp.float32],
    regularization: wp.float32,
    destination: wp.array[wp.float32],
    regularized_destination: wp.array[wp.float32],
):
    component, row, col = wp.tid()
    dimension = dimensions[component]
    if row >= dimension or col >= dimension:
        return
    vector_offset = vector_offsets[component]
    flat_row = vector_row[vector_offset + row]
    flat_col = vector_row[vector_offset + col]
    row_joint_block = row_block[flat_row]
    col_joint_block = row_block[flat_col]
    row_block_offset = block_row_offset[row_joint_block]
    col_block_offset = block_row_offset[col_joint_block]
    row_local = flat_row - row_block_offset
    col_local = flat_col - col_block_offset
    row_factor = factor[row_joint_block]
    col_factor = factor[col_joint_block]
    matrix_offset = matrix_offsets[component]
    value = wp.float32(0.0)
    for first in range(6):
        if first < block_row_count[row_joint_block]:
            first_vector = row_vector_index[row_block_offset + first] - vector_offset
            left = row_factor[first, row_local]
            for second in range(6):
                if second < block_row_count[col_joint_block]:
                    second_vector = row_vector_index[col_block_offset + second] - vector_offset
                    first_index = matrix_offset + first_vector * dimension + second_vector
                    value += left * source[first_index] * col_factor[second, col_local]
    index = matrix_offset + row * dimension + col
    destination[index] = value
    regularized_destination[index] = value + (regularization if row == col else 0.0)


@wp.kernel
def _transform_joint_right_hand_side(
    block_row_offset: wp.array[wp.int32],
    block_row_count: wp.array[wp.int32],
    row_vector_index: wp.array[wp.int32],
    factor: wp.array[mat66f],
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    block = wp.tid()
    offset = block_row_offset[block]
    count = block_row_count[block]
    lower = factor[block]
    for row in range(6):
        if row < count:
            value = wp.float32(0.0)
            for inner in range(6):
                if inner < count:
                    value += lower[inner, row] * source[row_vector_index[offset + inner]]
            destination[row_vector_index[offset + row]] = value


@wp.kernel
def _warmstart_transformed_joint_impulse(
    time_step: wp.array[wp.float32],
    block_component: wp.array[wp.int32],
    component_world: wp.array[wp.int32],
    block_row_offset: wp.array[wp.int32],
    block_row_count: wp.array[wp.int32],
    row_vector_index: wp.array[wp.int32],
    factor: wp.array[mat66f],
    multiplier: wp.array[wp.float32],
    transformed: wp.array[wp.float32],
    impulse: wp.array[wp.float32],
):
    block = wp.tid()
    dt = time_step[component_world[block_component[block]]]
    offset = block_row_offset[block]
    count = block_row_count[block]
    scaling = factor[block]
    physical = vec6f(0.0)
    transformed_value = vec6f(0.0)
    for row in range(6):
        if row < count:
            physical[row] = dt * multiplier[offset + row]
    reverse = int(0)
    while reverse < 6:
        row = 5 - reverse
        if row < count:
            value = physical[row]
            for col in range(6):
                if col > row and col < count:
                    value -= scaling[row, col] * transformed_value[col]
            transformed_value[row] = value / scaling[row, row]
        reverse += 1
    for row in range(6):
        if row < count:
            index = row_vector_index[offset + row]
            transformed[index] = transformed_value[row]
            impulse[index] = physical[row]


@wp.kernel
def _accumulate_and_recover_joint_impulse(
    block_row_offset: wp.array[wp.int32],
    block_row_count: wp.array[wp.int32],
    row_vector_index: wp.array[wp.int32],
    factor: wp.array[mat66f],
    correction: wp.array[wp.float32],
    transformed: wp.array[wp.float32],
    impulse: wp.array[wp.float32],
):
    block = wp.tid()
    offset = block_row_offset[block]
    count = block_row_count[block]
    lower = factor[block]
    transformed_value = vec6f(0.0)
    for row in range(6):
        if row < count:
            index = row_vector_index[offset + row]
            transformed_value[row] = transformed[index] + correction[index]
            transformed[index] = transformed_value[row]
    for row in range(6):
        if row < count:
            value = wp.float32(0.0)
            for inner in range(6):
                if inner < count:
                    value += lower[row, inner] * transformed_value[inner]
            impulse[row_vector_index[offset + row]] = value


@wp.kernel
def _compute_progressive_residual(
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
    value: wp.array[wp.float32],
    tile_size: int,
    residual: wp.array[wp.float32],
):
    component, row, lane = wp.tid()
    dimension = dimensions[component]
    if row >= dimension:
        return
    matrix_offset = matrix_offsets[component]
    vector_offset = vector_offsets[component]
    partial = wp.float32(0.0)
    for col in range(lane, dimension, tile_size):
        partial += matrix[matrix_offset + row * dimension + col] * value[vector_offset + col]
    product = wp.tile_sum(wp.tile(partial))
    result = wp.tile_load(right_hand_side, shape=1, offset=vector_offset + row) - product
    wp.tile_store(residual, result, offset=vector_offset + row)


@wp.kernel
def _apply_body_response(
    body_dimensions: wp.array[wp.int32],
    body_vector_offsets: wp.array[wp.int32],
    joint_dimensions: wp.array[wp.int32],
    joint_vector_offsets: wp.array[wp.int32],
    response_offsets: wp.array[wp.int32],
    response_leading_dimensions: wp.array[wp.int32],
    response: wp.array[wp.float32],
    free_body_value: wp.array[wp.float32],
    impulse: wp.array[wp.float32],
    result: wp.array[wp.float32],
):
    component, row = wp.tid()
    if row >= body_dimensions[component]:
        return
    correction = wp.float32(0.0)
    response_offset = response_offsets[component]
    response_leading_dimension = response_leading_dimensions[component]
    joint_vector_offset = joint_vector_offsets[component]
    for col in range(joint_dimensions[component]):
        correction += (
            response[response_offset + row * response_leading_dimension + col] * impulse[joint_vector_offset + col]
        )
    body_index = body_vector_offsets[component] + row
    result[body_index] = free_body_value[body_index] - correction


class BatchedStructuralJointSolver:
    """Solve structural equalities through a progressively refined Schur factorization."""

    def __init__(
        self,
        body_system: BatchedPrimalBodySystem,
        row_world: wp.array[wp.int32],
        body_first_global: wp.array[wp.int32],
        body_second_global: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        block_row_offset: wp.array[wp.int32],
        block_row_count: wp.array[wp.int32],
        row_block: wp.array[wp.int32],
        prescribed_twist: wp.array[vec6f],
    ):
        self.body_system = body_system
        self.device = body_system.device
        self.row_count = row_world.shape[0]
        self.row_world = row_world
        self.body_first_global = body_first_global
        self.body_second_global = body_second_global
        self.jacobian_first = jacobian_first
        self.jacobian_second = jacobian_second
        self.prescribed_twist = prescribed_twist

        if any(
            array.shape[0] != self.row_count
            for array in (body_first_global, body_second_global, jacobian_first, jacobian_second, row_block)
        ):
            raise ValueError("Structural row arrays must have identical lengths.")
        if block_row_offset.shape[0] != block_row_count.shape[0]:
            raise ValueError("Structural block arrays must have identical lengths.")
        if prescribed_twist.shape[0] != body_system.num_bodies:
            raise ValueError("prescribed_twist must contain one entry per packed body.")

        first_global = body_first_global.numpy().astype(int).tolist()
        second_global = body_second_global.numpy().astype(int).tolist()
        row_components = [-1] * self.row_count
        body_first_local: list[int] = []
        body_second_local: list[int] = []
        component_row_counts = [0] * body_system.num_blocks
        component_row_local = [-1] * self.row_count
        for row, (first, second) in enumerate(zip(first_global, second_global, strict=True)):
            if first < 0 and second < 0:
                raise ValueError("Each structural row must reference at least one body.")
            if first >= body_system.num_bodies or second >= body_system.num_bodies:
                raise ValueError("Structural row body indices must reference packed bodies.")
            first_local = body_system.body_local_host[first] if first >= 0 else -1
            second_local = body_system.body_local_host[second] if second >= 0 else -1
            body_first_local.append(first_local)
            body_second_local.append(second_local)
            if first_local < 0 and second_local < 0:
                continue
            component = body_system.body_block_host[first] if first_local >= 0 else body_system.body_block_host[second]
            if second_local >= 0 and body_system.body_block_host[second] != component:
                raise ValueError("A structural row cannot connect independent body components.")
            row_components[row] = component
            component_row_local[row] = component_row_counts[component]
            component_row_counts[component] += 1

        storage_dimensions = [max(1, count) for count in component_row_counts]
        vector_offsets = [0]
        for dimension in storage_dimensions:
            vector_offsets.append(vector_offsets[-1] + dimension)
        row_vector_index = [-1] * self.row_count
        for row, (component, local) in enumerate(zip(row_components, component_row_local, strict=True)):
            if component >= 0:
                row_vector_index[row] = vector_offsets[component] + local

        block_offsets = block_row_offset.numpy().astype(int).tolist()
        block_counts = block_row_count.numpy().astype(int).tolist()
        active_block_offsets: list[int] = []
        active_block_counts: list[int] = []
        block_components: list[int] = []
        for offset, count in zip(block_offsets, block_counts, strict=True):
            if count < 1 or offset < 0 or offset + count > self.row_count:
                raise ValueError("Structural block row ranges must be nonempty and valid.")
            active_rows = [row_components[row] >= 0 for row in range(offset, offset + count)]
            if any(active_rows) and not all(active_rows):
                raise ValueError("A structural joint block cannot mix prescribed-only and dynamic rows.")
            if not any(active_rows):
                continue
            component = row_components[offset]
            if any(row_components[row] != component for row in range(offset, offset + count)):
                raise ValueError("A structural joint block cannot span body components.")
            active_block_offsets.append(offset)
            active_block_counts.append(count)
            block_components.append(component)

        active_row_block = [-1] * self.row_count
        for block, (offset, count) in enumerate(zip(active_block_offsets, active_block_counts, strict=True)):
            for row in range(offset, offset + count):
                active_row_block[row] = block

        self.component_row_counts = tuple(component_row_counts)
        self.component_world_host = body_system.block_world_host
        self.row_vector_index = wp.array(row_vector_index, dtype=wp.int32, device=self.device)
        self.body_first_local = wp.array(body_first_local, dtype=wp.int32, device=self.device)
        self.body_second_local = wp.array(body_second_local, dtype=wp.int32, device=self.device)
        self.block_row_offset = wp.array(active_block_offsets, dtype=wp.int32, device=self.device)
        self.block_row_count = wp.array(active_block_counts, dtype=wp.int32, device=self.device)
        self.row_block = wp.array(active_row_block, dtype=wp.int32, device=self.device)
        self.block_component = wp.array(block_components, dtype=wp.int32, device=self.device)

        self.info = DenseSquareMultiLinearInfo()
        self.info.finalize(dimensions=storage_dimensions, dtype=wp.float32, itype=wp.int32, device=self.device)
        self.info.dim = wp.array(component_row_counts, dtype=wp.int32, device=self.device)
        self.residual_tile_size = min(64, 1 << (self.info.max_dimension - 1).bit_length()) if self.device.is_cuda else 1
        self.body_solve_block_size = body_system.linear_solver.block_size
        self.body_solve_right_hand_side_tile_size = 4
        self.body_solve_right_hand_side_block_count = (
            self.info.max_dimension + self.body_solve_right_hand_side_tile_size - 1
        ) // self.body_solve_right_hand_side_tile_size
        self.batched_body_solve_kernel = make_batched_body_solve_kernel(
            self.body_solve_block_size,
            self.body_solve_right_hand_side_tile_size,
        )

        self.right_hand_side = wp.zeros(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self.impulse = wp.zeros_like(self.right_hand_side)
        self.free_body_solution = wp.zeros(body_system.info.total_vec_size, dtype=wp.float32, device=self.device)
        response_leading_dimensions = storage_dimensions
        response_sizes = [
            6 * body_count * joint_dimension
            for body_count, joint_dimension in zip(
                body_system.block_body_counts, response_leading_dimensions, strict=True
            )
        ]
        response_offsets = [0]
        for size in response_sizes:
            response_offsets.append(response_offsets[-1] + size)
        self.response_offsets = wp.array(response_offsets[:-1], dtype=wp.int32, device=self.device)
        self.response_leading_dimensions = wp.array(response_leading_dimensions, dtype=wp.int32, device=self.device)
        self.body_response = wp.zeros(response_offsets[-1], dtype=wp.float32, device=self.device)
        self.body_response_intermediate = wp.zeros_like(self.body_response)
        self.body_response_permuted = (
            wp.zeros_like(self.body_response)
            if isinstance(body_system.linear_solver, LLTBlockedRCMSolver)
            else self.body_response
        )
        vector_row = [-1] * self.info.total_vec_size
        for row, vector_index in enumerate(row_vector_index):
            if vector_index >= 0:
                vector_row[vector_index] = row
        self.vector_row = wp.array(vector_row, dtype=wp.int32, device=self.device)
        self.block_scaling_factor = wp.zeros(len(block_components), dtype=mat66f, device=self.device)
        self.block_scaling_status = wp.zeros(len(block_components), dtype=wp.int32, device=self.device)
        self.transformed_right_hand_side = wp.zeros_like(self.right_hand_side)
        self.transformed_impulse = wp.zeros_like(self.impulse)
        self.unscaled_schur_matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)
        self.transformed_schur_matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)
        self.schur_matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)
        schur_operator_data = DenseLinearOperatorData(info=self.info, mat=self.schur_matrix)
        self.schur_solver = HybridLLTBlockedSolver(
            operator=schur_operator_data,
            dtype=wp.float32,
            device=self.device,
        )
        self.progressive_residual = wp.zeros_like(self.right_hand_side)
        self.progressive_correction = wp.zeros_like(self.impulse)

    def factorize(self) -> None:
        """Assemble and factorize the structural Delassus."""
        self.transformed_impulse.zero_()
        self.assemble_delassus()
        wp.launch(
            _factor_schur_block_diagonal,
            dim=self.block_component.shape[0],
            inputs=[
                self.info.dim,
                self.info.mio,
                self.info.vio,
                self.block_component,
                self.block_row_offset,
                self.block_row_count,
                self.row_vector_index,
                self.unscaled_schur_matrix,
            ],
            outputs=[self.block_scaling_factor, self.block_scaling_status],
            device=self.device,
        )
        wp.launch(
            _transform_schur_with_joint_blocks,
            dim=(self.body_system.num_blocks, self.info.max_dimension, self.info.max_dimension),
            inputs=[
                self.info.dim,
                self.info.mio,
                self.info.vio,
                self.vector_row,
                self.row_vector_index,
                self.row_block,
                self.block_row_offset,
                self.block_row_count,
                self.block_scaling_factor,
                self.unscaled_schur_matrix,
                1.0e-5,
            ],
            outputs=[self.transformed_schur_matrix, self.schur_matrix],
            device=self.device,
        )
        self.schur_solver.compute(self.schur_matrix)

    def assemble_delassus(self) -> None:
        """Assemble the structural Delassus without factorizing it."""
        wp.launch(
            _build_joint_basis_body_right_hand_sides,
            dim=(
                self.body_system.num_blocks,
                self.body_system.info.max_dimension,
                self.info.max_dimension,
            ),
            inputs=[
                self.body_system.info.dim,
                self.info.dim,
                self.info.vio,
                self.response_offsets,
                self.response_leading_dimensions,
                self.vector_row,
                self.body_first_local,
                self.body_second_local,
                self.jacobian_first,
                self.jacobian_second,
            ],
            outputs=[self.body_response],
            device=self.device,
        )
        if self.body_response_permuted is not self.body_response:
            wp.launch(
                _permute_body_response_rows,
                dim=(
                    self.body_system.num_blocks,
                    self.body_system.info.max_dimension,
                    self.info.max_dimension,
                ),
                inputs=[
                    self.body_system.info.dim,
                    self.body_system.info.vio,
                    self.info.dim,
                    self.response_offsets,
                    self.response_leading_dimensions,
                    self.body_system.linear_solver.P,
                    self.body_response,
                ],
                outputs=[self.body_response_permuted],
                device=self.device,
            )
        wp.launch_tiled(
            self.batched_body_solve_kernel,
            dim=(self.body_system.num_blocks, self.body_solve_right_hand_side_block_count),
            block_dim=128,
            inputs=[
                self.body_system.info.dim,
                self.body_system.info.mio,
                self.info.dim,
                self.response_offsets,
                self.response_leading_dimensions,
                self.body_system.linear_solver.L,
                self.body_response_permuted,
            ],
            outputs=[self.body_response_intermediate],
            device=self.device,
        )
        if self.body_response_permuted is not self.body_response:
            wp.launch(
                _unpermute_body_response_rows,
                dim=(
                    self.body_system.num_blocks,
                    self.body_system.info.max_dimension,
                    self.info.max_dimension,
                ),
                inputs=[
                    self.body_system.info.dim,
                    self.body_system.info.vio,
                    self.info.dim,
                    self.response_offsets,
                    self.response_leading_dimensions,
                    self.body_system.linear_solver.P,
                    self.body_response_permuted,
                ],
                outputs=[self.body_response],
                device=self.device,
            )
        wp.launch(
            _assemble_schur_from_body_response,
            dim=(self.body_system.num_blocks, self.info.max_dimension, self.info.max_dimension),
            inputs=[
                self.info.dim,
                self.info.mio,
                self.info.vio,
                self.response_offsets,
                self.response_leading_dimensions,
                self.vector_row,
                self.body_first_local,
                self.body_second_local,
                self.jacobian_first,
                self.jacobian_second,
                self.body_response,
            ],
            outputs=[self.unscaled_schur_matrix],
            device=self.device,
        )

    def warmstart(self, time_step: wp.array[wp.float32], multiplier: wp.array[wp.float32]) -> None:
        """Transform persistent physical joint impulses into the current Schur coordinates."""
        validate_world_time_step(time_step, self.body_system.num_worlds, self.device)
        if multiplier.shape[0] != self.row_count:
            raise ValueError("multiplier must contain one value per structural row.")
        wp.launch(
            _warmstart_transformed_joint_impulse,
            dim=self.block_component.shape[0],
            inputs=[
                time_step,
                self.block_component,
                self.body_system.block_world,
                self.block_row_offset,
                self.block_row_count,
                self.row_vector_index,
                self.block_scaling_factor,
                multiplier,
            ],
            outputs=[self.transformed_impulse, self.impulse],
            device=self.device,
        )

    def solve_free_body(self) -> None:
        """Solve the current unconstrained body equation of motion."""
        self.body_system.linear_solver.solve(
            self.body_system.candidate_right_hand_side,
            self.free_body_solution,
        )

    def project(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
        world_active: wp.array[wp.bool],
        residual: wp.array[wp.float32],
        multiplier_index: wp.array[wp.int32],
        multiplier: wp.array[wp.float32],
        multiplier_destination: wp.array[wp.float32],
    ) -> None:
        """Project the free body solution onto the structural constraints."""
        validate_world_time_steps(time_step, inverse_time_step, self.body_system.num_worlds, self.device)
        if world_active.shape[0] != self.body_system.num_worlds:
            raise ValueError("world_active must contain one entry per world.")
        if residual.shape[0] != self.row_count or multiplier.shape[0] != self.row_count:
            raise ValueError("Structural row arrays must have identical lengths.")

        wp.launch(
            _build_joint_right_hand_side,
            dim=self.row_count,
            inputs=[
                inverse_time_step,
                self.row_world,
                self.row_vector_index,
                self.body_first_global,
                self.body_second_global,
                self.jacobian_first,
                self.jacobian_second,
                residual,
                self.body_system.body_vector_index,
                self.prescribed_twist,
                linearization_twist,
                self.free_body_solution,
            ],
            outputs=[self.right_hand_side],
            device=self.device,
        )
        self._apply_correction(
            inverse_time_step,
            world_active,
            multiplier_index,
            multiplier,
            multiplier_destination,
            subtract_current_impulse=True,
        )

    def refine_from_twists(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
        global_twist: wp.array[vec6f],
        projected_twist: wp.array[vec6f],
        projected_fraction: float,
        world_has_unilateral: wp.array[wp.bool],
        world_active: wp.array[wp.bool],
        residual: wp.array[wp.float32],
        multiplier_index: wp.array[wp.int32],
        multiplier: wp.array[wp.float32],
        multiplier_destination: wp.array[wp.float32],
    ) -> None:
        """Apply a Schur correction from the current blended structural residual."""
        validate_world_time_steps(time_step, inverse_time_step, self.body_system.num_worlds, self.device)
        if any(twist.shape[0] != self.body_system.num_bodies for twist in (global_twist, projected_twist)):
            raise ValueError("Twist arrays must contain one entry per body.")
        if not 0.0 <= projected_fraction <= 1.0:
            raise ValueError("projected_fraction must be in [0, 1].")
        if world_active.shape[0] != self.body_system.num_worlds:
            raise ValueError("world_active must contain one entry per world.")
        if world_has_unilateral.shape[0] != self.body_system.num_worlds:
            raise ValueError("world_has_unilateral must contain one entry per world.")
        if residual.shape[0] != self.row_count or multiplier.shape[0] != self.row_count:
            raise ValueError("Structural row arrays must have identical lengths.")
        wp.launch(
            _build_blended_joint_residual,
            dim=self.row_count,
            inputs=[
                inverse_time_step,
                self.row_world,
                self.row_vector_index,
                self.body_first_global,
                self.body_second_global,
                self.jacobian_first,
                self.jacobian_second,
                residual,
                linearization_twist,
                global_twist,
                projected_twist,
                projected_fraction,
                world_has_unilateral,
            ],
            outputs=[self.right_hand_side],
            device=self.device,
        )
        self._apply_correction(
            inverse_time_step,
            world_active,
            multiplier_index,
            multiplier,
            multiplier_destination,
            subtract_current_impulse=False,
        )

    def _apply_correction(
        self,
        inverse_time_step: wp.array[wp.float32],
        world_active: wp.array[wp.bool],
        multiplier_index: wp.array[wp.int32],
        multiplier: wp.array[wp.float32],
        multiplier_destination: wp.array[wp.float32],
        subtract_current_impulse: bool,
    ) -> None:
        wp.launch(
            _transform_joint_right_hand_side,
            dim=self.block_component.shape[0],
            inputs=[
                self.block_row_offset,
                self.block_row_count,
                self.row_vector_index,
                self.block_scaling_factor,
                self.right_hand_side,
            ],
            outputs=[self.transformed_right_hand_side],
            device=self.device,
        )
        correction_right_hand_side = self.transformed_right_hand_side
        if subtract_current_impulse:
            wp.launch(
                _compute_progressive_residual,
                dim=(self.body_system.num_blocks, self.info.max_dimension, self.residual_tile_size),
                inputs=[
                    self.info.dim,
                    self.info.mio,
                    self.info.vio,
                    self.transformed_schur_matrix,
                    self.transformed_right_hand_side,
                    self.transformed_impulse,
                    self.residual_tile_size,
                ],
                outputs=[self.progressive_residual],
                device=self.device,
                block_dim=self.residual_tile_size,
            )
            correction_right_hand_side = self.progressive_residual
        self.schur_solver.solve(correction_right_hand_side, self.progressive_correction)
        wp.launch(
            _accumulate_and_recover_joint_impulse,
            dim=self.block_component.shape[0],
            inputs=[
                self.block_row_offset,
                self.block_row_count,
                self.row_vector_index,
                self.block_scaling_factor,
                self.progressive_correction,
            ],
            outputs=[self.transformed_impulse, self.impulse],
            device=self.device,
        )
        wp.launch(
            _apply_body_response,
            dim=(self.body_system.num_blocks, self.body_system.info.max_dimension),
            inputs=[
                self.body_system.info.dim,
                self.body_system.info.vio,
                self.info.dim,
                self.info.vio,
                self.response_offsets,
                self.response_leading_dimensions,
                self.body_response,
                self.free_body_solution,
                self.impulse,
            ],
            outputs=[self.body_system.solution],
            device=self.device,
        )
        wp.launch(
            _write_joint_multipliers,
            dim=self.row_count,
            inputs=[
                inverse_time_step,
                self.row_world,
                self.row_vector_index,
                multiplier_index,
                world_active,
                self.impulse,
            ],
            outputs=[multiplier, multiplier_destination],
            device=self.device,
        )

    def solve(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
        world_active: wp.array[wp.bool],
        residual: wp.array[wp.float32],
        multiplier_index: wp.array[wp.int32],
        multiplier: wp.array[wp.float32],
        multiplier_destination: wp.array[wp.float32],
    ) -> None:
        """Solve the free body equation and apply one structural projection."""
        self.solve_free_body()
        self.project(
            time_step,
            inverse_time_step,
            linearization_twist,
            world_active,
            residual,
            multiplier_index,
            multiplier,
            multiplier_destination,
        )
