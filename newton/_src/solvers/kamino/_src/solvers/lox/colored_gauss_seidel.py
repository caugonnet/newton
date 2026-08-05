# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-batched Gauss--Seidel projection with final Jacobi smoothing."""

from __future__ import annotations

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact import _solve_contact_coulomb_newton_normal_last
from .deformable_contact import (
    DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS,
    DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
    DEFORMABLE_CONTACT_STATUS_VALID,
    _merge_rigid_prepared_status,
    project_deformable_contact_coulomb,
)
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    compute_limit_delassus,
    prepare_contact_coulomb,
    prepare_contact_coulomb_delassus,
)
from .sweep import (
    _initialize_jacobi_projection_status,
    _warmstart_contacts_jacobi,
    _warmstart_frictions_jacobi,
    _warmstart_limits_jacobi,
    prepare_jacobi_projection_data,
    project_constraints_jacobi,
)

wp.set_module_options({"enable_backward": False})

_COLOR_REPAIR_PASSES = 3
_COLOR_BLOCK_DIM = 128
_COLOR_BLOCKS_PER_SM = 2
# Avoid parallel-scan setup for the small color counts used by normal workloads.
_SERIAL_COLOR_PREFIX_LIMIT = 64
_NO_PROPOSAL = -1
_LOCK_FREE = 0x7FFFFFFF


def _bounded_worker_count(capacity: int, device) -> int:
    if capacity <= 0:
        return 0
    if device.is_cuda:
        return min(capacity, max(_COLOR_BLOCK_DIM, device.sm_count * _COLOR_BLOCKS_PER_SM * _COLOR_BLOCK_DIM))
    return capacity


@wp.func
def _mix_color_key(value: wp.uint32) -> wp.uint32:
    value = (value ^ (value >> wp.uint32(16))) * wp.uint32(0x7FEB352D)
    value = (value ^ (value >> wp.uint32(15))) * wp.uint32(0x846CA68B)
    return value ^ (value >> wp.uint32(16))


@wp.func
def _initial_color(world: int, local: int, first: int, second: int, family: int, color_count: int) -> int:
    key = wp.uint32(world + 1) * wp.uint32(0x9E3779B9)
    key = key ^ (wp.uint32(local + 1) * wp.uint32(0x85EBCA6B))
    key = key ^ (wp.uint32(first + 2) * wp.uint32(0xC2B2AE35))
    key = key ^ (wp.uint32(second + 2) * wp.uint32(0x27D4EB2F))
    key = key ^ wp.uint32(family * 0x165667B1)
    return int(_mix_color_key(key) % wp.uint32(color_count))


@wp.func
def _occupancy(occupancy: wp.array2d[wp.int32], endpoint: int, color: int) -> int:
    value = int(0)
    if endpoint >= 0:
        value = occupancy[endpoint, color]
    return value


@wp.func
def _choose_two_endpoint_color(
    first: int,
    second: int,
    current: int,
    color_count: int,
    occupancy: wp.array2d[wp.int32],
) -> int:
    current_sum = _occupancy(occupancy, first, current)
    endpoint_count = int(0)
    if first >= 0:
        endpoint_count += 1
    if second >= 0 and second != first:
        current_sum += _occupancy(occupancy, second, current)
        endpoint_count += 1
    best = current
    best_sum = current_sum
    best_max = wp.max(_occupancy(occupancy, first, current), _occupancy(occupancy, second, current))
    for color in range(color_count):
        candidate_sum = _occupancy(occupancy, first, color)
        if second >= 0 and second != first:
            candidate_sum += _occupancy(occupancy, second, color)
        candidate_max = wp.max(_occupancy(occupancy, first, color), _occupancy(occupancy, second, color))
        if candidate_sum < best_sum or (candidate_sum == best_sum and candidate_max < best_max):
            best = color
            best_sum = candidate_sum
            best_max = candidate_max
    # Moving one incidence adds one to the destination and removes one from
    # the source. This is exactly the strict-improvement condition for sum(m^2).
    if best != current and best_sum + endpoint_count < current_sum:
        return best
    return _NO_PROPOSAL


@wp.kernel
def _assign_two_endpoint_colors(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    family: int,
    color_count: int,
    colors: wp.array[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        colors[constraint] = _NO_PROPOSAL
        return
    colors[constraint] = _initial_color(
        world,
        constraint_local[constraint],
        endpoint_first[constraint],
        endpoint_second[constraint],
        family,
        color_count,
    )


@wp.kernel
def _count_two_endpoint_occupancy(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    colors: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        return
    color = colors[constraint]
    first = endpoint_first[constraint]
    second = endpoint_second[constraint]
    if first >= 0:
        wp.atomic_add(occupancy, first, color, 1)
    if second >= 0 and second != first:
        wp.atomic_add(occupancy, second, color, 1)


@wp.kernel
def _propose_two_endpoint_repairs(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    color_count: int,
    colors: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    proposals: wp.array[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        proposals[constraint] = _NO_PROPOSAL
        return
    proposals[constraint] = _choose_two_endpoint_color(
        endpoint_first[constraint],
        endpoint_second[constraint],
        colors[constraint],
        color_count,
        occupancy,
    )


@wp.kernel
def _claim_two_endpoint_repairs(
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    proposals: wp.array[wp.int32],
    key_offset: int,
    locks: wp.array[wp.int32],
):
    constraint = wp.tid()
    if proposals[constraint] < 0:
        return
    key = key_offset + constraint
    first = endpoint_first[constraint]
    second = endpoint_second[constraint]
    if first >= 0:
        wp.atomic_min(locks, first, key)
    if second >= 0 and second != first:
        wp.atomic_min(locks, second, key)


@wp.kernel
def _propose_and_claim_two_endpoint_repairs(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    color_count: int,
    key_offset: int,
    colors: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    proposals: wp.array[wp.int32],
    locks: wp.array[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        proposals[constraint] = _NO_PROPOSAL
        return
    proposal = _choose_two_endpoint_color(
        endpoint_first[constraint],
        endpoint_second[constraint],
        colors[constraint],
        color_count,
        occupancy,
    )
    proposals[constraint] = proposal
    if proposal < 0:
        return
    key = key_offset + constraint
    first = endpoint_first[constraint]
    second = endpoint_second[constraint]
    if first >= 0:
        wp.atomic_min(locks, first, key)
    if second >= 0 and second != first:
        wp.atomic_min(locks, second, key)


@wp.kernel
def _commit_two_endpoint_repairs(
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    key_offset: int,
    colors: wp.array[wp.int32],
    proposals: wp.array[wp.int32],
    locks: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
):
    constraint = wp.tid()
    candidate = proposals[constraint]
    if candidate < 0:
        return
    key = key_offset + constraint
    first = endpoint_first[constraint]
    second = endpoint_second[constraint]
    if (first >= 0 and locks[first] != key) or (second >= 0 and second != first and locks[second] != key):
        return
    current = colors[constraint]
    if first >= 0:
        wp.atomic_add(occupancy, first, current, -1)
        wp.atomic_add(occupancy, first, candidate, 1)
    if second >= 0 and second != first:
        wp.atomic_add(occupancy, second, current, -1)
        wp.atomic_add(occupancy, second, candidate, 1)
    colors[constraint] = candidate


@wp.func
def _deformable_endpoint_occupancy(
    particle_indices: wp.array2d[wp.int32],
    contact: int,
    body: int,
    color: int,
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
) -> int:
    score = _occupancy(body_occupancy, body, color)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            score += particle_occupancy[particle, color]
    return score


@wp.func
def _deformable_endpoint_maximum(
    particle_indices: wp.array2d[wp.int32],
    contact: int,
    body: int,
    color: int,
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
) -> int:
    maximum = _occupancy(body_occupancy, body, color)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            maximum = wp.max(maximum, particle_occupancy[particle, color])
    return maximum


@wp.kernel
def _assign_deformable_colors(
    particle_indices: wp.array2d[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    color_count: int,
    colors: wp.array[wp.int32],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        colors[contact] = _NO_PROPOSAL
        return
    first = particle_indices[contact, 0]
    second = particle_indices[contact, 1]
    colors[contact] = _initial_color(contact_world[contact], contact, first, second, 3, color_count)


@wp.kernel
def _count_deformable_occupancy(
    particle_indices: wp.array2d[wp.int32],
    contact_body: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    colors: wp.array[wp.int32],
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return
    color = colors[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            wp.atomic_add(particle_occupancy, particle, color, 1)
    body = contact_body[contact]
    if body >= 0:
        wp.atomic_add(body_occupancy, body, color, 1)


@wp.kernel
def _propose_deformable_repairs(
    particle_indices: wp.array2d[wp.int32],
    contact_body: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    color_count: int,
    colors: wp.array[wp.int32],
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
    proposals: wp.array[wp.int32],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        proposals[contact] = _NO_PROPOSAL
        return
    current = colors[contact]
    current_sum = _deformable_endpoint_occupancy(
        particle_indices, contact, contact_body[contact], current, particle_occupancy, body_occupancy
    )
    endpoint_count = int(contact_body[contact] >= 0)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            endpoint_count += 1
    best = current
    best_sum = current_sum
    best_max = _deformable_endpoint_maximum(
        particle_indices, contact, contact_body[contact], current, particle_occupancy, body_occupancy
    )
    for color in range(color_count):
        score = _deformable_endpoint_occupancy(
            particle_indices, contact, contact_body[contact], color, particle_occupancy, body_occupancy
        )
        maximum = _deformable_endpoint_maximum(
            particle_indices, contact, contact_body[contact], color, particle_occupancy, body_occupancy
        )
        if score < best_sum or (score == best_sum and maximum < best_max):
            best = color
            best_sum = score
            best_max = maximum
    proposal = _NO_PROPOSAL
    if best != current and best_sum + endpoint_count < current_sum:
        proposal = best
    proposals[contact] = proposal


@wp.kernel
def _claim_deformable_repairs(
    particle_indices: wp.array2d[wp.int32],
    contact_body: wp.array[wp.int32],
    proposals: wp.array[wp.int32],
    key_offset: int,
    particle_locks: wp.array[wp.int32],
    body_locks: wp.array[wp.int32],
):
    contact = wp.tid()
    if proposals[contact] < 0:
        return
    key = key_offset + contact
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            wp.atomic_min(particle_locks, particle, key)
    body = contact_body[contact]
    if body >= 0:
        wp.atomic_min(body_locks, body, key)


@wp.kernel
def _propose_and_claim_deformable_repairs(
    particle_indices: wp.array2d[wp.int32],
    contact_body: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    color_count: int,
    key_offset: int,
    colors: wp.array[wp.int32],
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
    proposals: wp.array[wp.int32],
    particle_locks: wp.array[wp.int32],
    body_locks: wp.array[wp.int32],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        proposals[contact] = _NO_PROPOSAL
        return
    current = colors[contact]
    current_sum = _deformable_endpoint_occupancy(
        particle_indices, contact, contact_body[contact], current, particle_occupancy, body_occupancy
    )
    endpoint_count = int(contact_body[contact] >= 0)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            endpoint_count += 1
    best = current
    best_sum = current_sum
    best_max = _deformable_endpoint_maximum(
        particle_indices, contact, contact_body[contact], current, particle_occupancy, body_occupancy
    )
    for color in range(color_count):
        score = _deformable_endpoint_occupancy(
            particle_indices, contact, contact_body[contact], color, particle_occupancy, body_occupancy
        )
        maximum = _deformable_endpoint_maximum(
            particle_indices, contact, contact_body[contact], color, particle_occupancy, body_occupancy
        )
        if score < best_sum or (score == best_sum and maximum < best_max):
            best = color
            best_sum = score
            best_max = maximum
    proposal = _NO_PROPOSAL
    if best != current and best_sum + endpoint_count < current_sum:
        proposal = best
    proposals[contact] = proposal
    if proposal < 0:
        return
    key = key_offset + contact
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            wp.atomic_min(particle_locks, particle, key)
    body = contact_body[contact]
    if body >= 0:
        wp.atomic_min(body_locks, body, key)


@wp.kernel
def _commit_deformable_repairs(
    particle_indices: wp.array2d[wp.int32],
    contact_body: wp.array[wp.int32],
    key_offset: int,
    colors: wp.array[wp.int32],
    proposals: wp.array[wp.int32],
    particle_locks: wp.array[wp.int32],
    body_locks: wp.array[wp.int32],
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
):
    contact = wp.tid()
    candidate = proposals[contact]
    if candidate < 0:
        return
    key = key_offset + contact
    accepted = True
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            accepted = accepted and particle_locks[particle] == key
    body = contact_body[contact]
    if body >= 0:
        accepted = accepted and body_locks[body] == key
    if not accepted:
        return
    current = colors[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            wp.atomic_add(particle_occupancy, particle, current, -1)
            wp.atomic_add(particle_occupancy, particle, candidate, 1)
    if body >= 0:
        wp.atomic_add(body_occupancy, body, current, -1)
        wp.atomic_add(body_occupancy, body, candidate, 1)
    colors[contact] = candidate


@wp.kernel
def _count_colors(colors: wp.array[wp.int32], color_count: int, counts: wp.array[wp.int32]):
    item = wp.tid()
    color = colors[item]
    if color >= 0 and color < color_count:
        wp.atomic_add(counts, color, 1)


@wp.kernel
def _prefix_color_counts(
    color_count: int, counts: wp.array[wp.int32], offsets: wp.array[wp.int32], cursors: wp.array[wp.int32]
):
    offset = int(0)
    for color in range(color_count):
        offsets[color] = offset
        cursors[color] = offset
        offset += counts[color]


@wp.kernel
def _scatter_color_order(
    colors: wp.array[wp.int32],
    color_count: int,
    cursors: wp.array[wp.int32],
    order: wp.array[wp.int32],
):
    item = wp.tid()
    color = colors[item]
    if color >= 0 and color < color_count:
        ordered = wp.atomic_add(cursors, color, 1)
        order[ordered] = item


@wp.kernel
def _prepare_frictions_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        constraint = order[ordered]
        first = endpoint_first[constraint]
        second = endpoint_second[constraint]
        if first < 0 and second < 0:
            delassus[constraint] = 0.0
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        value = compute_limit_delassus(
            jacobian_first[constraint], inverse_first, jacobian_second[constraint], inverse_second
        )
        delassus[constraint] = value
        if not wp.isfinite(value) or value <= 0.0:
            world_status[constraint_world[constraint]] = PROJECTION_STATUS_INVALID


@wp.kernel
def _prepare_contacts_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    jacobian_first: wp.array[mat36f],
    jacobian_second: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        constraint = order[ordered]
        first = endpoint_first[constraint]
        second = endpoint_second[constraint]
        if first < 0 and second < 0:
            delassus[constraint] = wp.mat33f(0.0)
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        data = prepare_contact_coulomb(
            jacobian_first[constraint],
            inverse_first,
            jacobian_second[constraint],
            inverse_second,
            bias[constraint],
            friction[constraint],
        )
        delassus[constraint] = data.delassus
        if data.status == PROJECTION_STATUS_INVALID:
            world_status[constraint_world[constraint]] = data.status


@wp.kernel
def _prepare_rigid_colored(
    launch_dim: int,
    target_color: int,
    friction_counts: wp.array[wp.int32],
    friction_offsets: wp.array[wp.int32],
    friction_order: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_first: wp.array[wp.int32],
    friction_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    contact_counts: wp.array[wp.int32],
    contact_offsets: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_first: wp.array[wp.int32],
    contact_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    limit_counts: wp.array[wp.int32],
    limit_offsets: wp.array[wp.int32],
    limit_order: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_first: wp.array[wp.int32],
    limit_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    friction_delassus: wp.array[wp.float32],
    contact_delassus: wp.array[wp.mat33f],
    limit_delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    friction_begin = friction_offsets[target_color] + lane
    friction_end = friction_offsets[target_color] + friction_counts[target_color]
    for ordered in range(friction_begin, friction_end, launch_dim):
        constraint = friction_order[ordered]
        first = friction_first[constraint]
        second = friction_second[constraint]
        if first < 0 and second < 0:
            friction_delassus[constraint] = 0.0
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        value = compute_limit_delassus(
            friction_jacobian_first[constraint],
            inverse_first,
            friction_jacobian_second[constraint],
            inverse_second,
        )
        friction_delassus[constraint] = value
        if not wp.isfinite(value) or value <= 0.0:
            world_status[friction_world[constraint]] = PROJECTION_STATUS_INVALID

    contact_begin = contact_offsets[target_color] + lane
    contact_end = contact_offsets[target_color] + contact_counts[target_color]
    for ordered in range(contact_begin, contact_end, launch_dim):
        constraint = contact_order[ordered]
        first = contact_first[constraint]
        second = contact_second[constraint]
        if first < 0 and second < 0:
            contact_delassus[constraint] = wp.mat33f(0.0)
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        data = prepare_contact_coulomb(
            contact_jacobian_first[constraint],
            inverse_first,
            contact_jacobian_second[constraint],
            inverse_second,
            contact_bias[constraint],
            contact_friction[constraint],
        )
        contact_delassus[constraint] = data.delassus
        if data.status == PROJECTION_STATUS_INVALID:
            world_status[contact_world[constraint]] = data.status

    limit_begin = limit_offsets[target_color] + lane
    limit_end = limit_offsets[target_color] + limit_counts[target_color]
    for ordered in range(limit_begin, limit_end, launch_dim):
        constraint = limit_order[ordered]
        first = limit_first[constraint]
        second = limit_second[constraint]
        if first < 0 and second < 0:
            limit_delassus[constraint] = 0.0
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        value = compute_limit_delassus(
            limit_jacobian_first[constraint], inverse_first, limit_jacobian_second[constraint], inverse_second
        )
        limit_delassus[constraint] = value
        if not wp.isfinite(value) or value <= 0.0:
            world_status[limit_world[constraint]] = PROJECTION_STATUS_INVALID


@wp.kernel
def _prepare_deformable_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    body_jacobian: wp.array[mat36f],
    rigid_bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    body_inverse_weight: wp.array[mat66f],
    include_rigid: bool,
    contact_status: wp.array[wp.int32],
    scalar_delassus: wp.array[wp.float32],
    delassus: wp.array[wp.mat33f],
    prepared_status: wp.array[wp.int32],
    contact_world_status: wp.array[wp.int32],
    invalid_count: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        contact = order[ordered]
        particle_value = wp.float32(0.0)
        for particle_slot in range(4):
            particle = particle_indices[contact, particle_slot]
            if particle >= 0:
                coefficient = coefficients[contact, particle_slot]
                multiplicity = wp.max(1, particle_occupancy[particle, target_color])
                particle_value += (
                    coefficient * coefficient * wp.float32(multiplicity) * particle_inverse_weight[particle]
                )
        value = particle_value * wp.identity(3, dtype=wp.float32)
        body = contact_body[contact]
        if include_rigid and body >= 0:
            multiplicity = wp.max(1, body_occupancy[body, target_color])
            jacobian = body_jacobian[contact]
            value += jacobian @ (wp.float32(multiplicity) * body_inverse_weight[body]) @ wp.transpose(jacobian)
        data = prepare_contact_coulomb_delassus(value, rigid_bias[contact], friction[contact])
        scalar_delassus[contact] = particle_value
        delassus[contact] = data.delassus
        if data.status == PROJECTION_STATUS_INVALID or (not include_rigid and particle_value <= 0.0):
            contact_status[contact] = DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS
            world = contact_world[contact]
            prepared_status[world] = PROJECTION_STATUS_INVALID
            wp.atomic_max(contact_world_status, world, DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS)
            wp.atomic_add(invalid_count, 0, 1)


@wp.func
def _finite_vec3(value: wp.vec3f) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _finite_twist(value: vec6f) -> bool:
    result = True
    for axis in range(6):
        result = result and wp.isfinite(value[axis])
    return result


@wp.func
def _accumulate_twist_colored(
    endpoint: int,
    color: int,
    correction: vec6f,
    occupancy: wp.array2d[wp.int32],
    twist_delta: wp.array[vec6f],
):
    if occupancy[endpoint, color] == 1:
        twist_delta[endpoint] += correction
    else:
        wp.atomic_add(twist_delta, endpoint, correction)


@wp.func
def _accumulate_particle_colored(
    particle: int,
    color: int,
    correction: wp.vec3f,
    occupancy: wp.array2d[wp.int32],
    particle_delta: wp.array[wp.vec3f],
):
    if occupancy[particle, color] == 1:
        particle_delta[particle] += correction
    else:
        wp.atomic_add(particle_delta, particle, correction)


@wp.kernel
def _project_frictions_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    impulse_bound: wp.array[wp.float32],
    delassus: wp.array[wp.float32],
    world_active: wp.array[wp.bool],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        constraint = order[ordered]
        world = constraint_world[constraint]
        if not world_active[world] or world_status[world] != PROJECTION_STATUS_VALID:
            continue
        first = endpoint_first[constraint]
        second = endpoint_second[constraint]
        if first < 0 and second < 0:
            reaction[constraint] = 0.0
            continue
        velocity = wp.float32(0.0)
        if first >= 0:
            velocity += wp.dot(jacobian_first[constraint], projected_twist[first])
        if second >= 0:
            velocity += wp.dot(jacobian_second[constraint], projected_twist[second])
        old = reaction[constraint]
        split = delassus[constraint]
        new = wp.clamp(-((velocity - split * old) / split), -impulse_bound[constraint], impulse_bound[constraint])
        delta = new - old
        if not wp.isfinite(new) or not wp.isfinite(delta):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        first_correction = vec6f(0.0)
        second_correction = vec6f(0.0)
        if first >= 0:
            first_correction = (inverse_weight[first] @ jacobian_first[constraint]) * delta
        if second >= 0:
            second_correction = (inverse_weight[second] @ jacobian_second[constraint]) * delta
        if not _finite_twist(first_correction) or not _finite_twist(second_correction):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        if first >= 0:
            if first == second:
                _accumulate_twist_colored(
                    first, target_color, first_correction + second_correction, occupancy, twist_delta
                )
            else:
                _accumulate_twist_colored(first, target_color, first_correction, occupancy, twist_delta)
        if second >= 0 and second != first:
            _accumulate_twist_colored(second, target_color, second_correction, occupancy, twist_delta)
        reaction[constraint] = new


@wp.kernel
def _project_limits_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    bias: wp.array[wp.float32],
    delassus: wp.array[wp.float32],
    world_active: wp.array[wp.bool],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        constraint = order[ordered]
        world = constraint_world[constraint]
        if not world_active[world] or world_status[world] != PROJECTION_STATUS_VALID:
            continue
        first = endpoint_first[constraint]
        second = endpoint_second[constraint]
        if first < 0 and second < 0:
            reaction[constraint] = 0.0
            continue
        velocity = bias[constraint]
        if first >= 0:
            velocity += wp.dot(jacobian_first[constraint], projected_twist[first])
        if second >= 0:
            velocity += wp.dot(jacobian_second[constraint], projected_twist[second])
        old = reaction[constraint]
        split = delassus[constraint]
        new = wp.max(-((velocity - split * old) / split), 0.0)
        delta = new - old
        if not wp.isfinite(new) or not wp.isfinite(delta):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        first_correction = vec6f(0.0)
        second_correction = vec6f(0.0)
        if first >= 0:
            first_correction = (inverse_weight[first] @ jacobian_first[constraint]) * delta
        if second >= 0:
            second_correction = (inverse_weight[second] @ jacobian_second[constraint]) * delta
        if not _finite_twist(first_correction) or not _finite_twist(second_correction):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        if first >= 0:
            if first == second:
                _accumulate_twist_colored(
                    first, target_color, first_correction + second_correction, occupancy, twist_delta
                )
            else:
                _accumulate_twist_colored(first, target_color, first_correction, occupancy, twist_delta)
        if second >= 0 and second != first:
            _accumulate_twist_colored(second, target_color, second_correction, occupancy, twist_delta)
        reaction[constraint] = new


@wp.kernel
def _project_contacts_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    jacobian_first: wp.array[mat36f],
    jacobian_second: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    delassus: wp.array[wp.mat33f],
    world_active: wp.array[wp.bool],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.vec3f],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        constraint = order[ordered]
        world = constraint_world[constraint]
        if not world_active[world] or world_status[world] != PROJECTION_STATUS_VALID:
            continue
        first = endpoint_first[constraint]
        second = endpoint_second[constraint]
        if first < 0 and second < 0:
            reaction[constraint] = wp.vec3f(0.0)
            continue
        twist_first = vec6f(0.0)
        twist_second = vec6f(0.0)
        if first >= 0:
            twist_first = projected_twist[first]
        if second >= 0:
            twist_second = projected_twist[second]
        old = reaction[constraint]
        velocity = (
            jacobian_first[constraint] @ twist_first + jacobian_second[constraint] @ twist_second + bias[constraint]
        )
        contact_block = delassus[constraint]
        free = velocity - contact_block @ old
        new = _solve_contact_coulomb_newton_normal_last(contact_block, free, friction[constraint])
        delta = new - old
        if not _finite_vec3(new) or not _finite_vec3(delta):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        first_correction = vec6f(0.0)
        second_correction = vec6f(0.0)
        if first >= 0:
            first_correction = inverse_weight[first] @ (wp.transpose(jacobian_first[constraint]) @ delta)
        if second >= 0:
            second_correction = inverse_weight[second] @ (wp.transpose(jacobian_second[constraint]) @ delta)
        if not _finite_twist(first_correction) or not _finite_twist(second_correction):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        if first >= 0:
            if first == second:
                _accumulate_twist_colored(
                    first, target_color, first_correction + second_correction, occupancy, twist_delta
                )
            else:
                _accumulate_twist_colored(first, target_color, first_correction, occupancy, twist_delta)
        if second >= 0 and second != first:
            _accumulate_twist_colored(second, target_color, second_correction, occupancy, twist_delta)
        reaction[constraint] = new


@wp.kernel
def _project_rigid_colored(
    launch_dim: int,
    target_color: int,
    friction_counts: wp.array[wp.int32],
    friction_offsets: wp.array[wp.int32],
    friction_order: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_first: wp.array[wp.int32],
    friction_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_bound: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    contact_counts: wp.array[wp.int32],
    contact_offsets: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_first: wp.array[wp.int32],
    contact_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_delassus: wp.array[wp.mat33f],
    limit_counts: wp.array[wp.int32],
    limit_offsets: wp.array[wp.int32],
    limit_order: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_first: wp.array[wp.int32],
    limit_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    world_active: wp.array[wp.bool],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    friction_begin = friction_offsets[target_color] + lane
    friction_end = friction_offsets[target_color] + friction_counts[target_color]
    for ordered in range(friction_begin, friction_end, launch_dim):
        constraint = friction_order[ordered]
        world = friction_world[constraint]
        if not world_active[world] or world_status[world] != PROJECTION_STATUS_VALID:
            continue
        first = friction_first[constraint]
        second = friction_second[constraint]
        if first < 0 and second < 0:
            friction_reaction[constraint] = 0.0
            continue
        friction_velocity = wp.float32(0.0)
        if first >= 0:
            friction_velocity += wp.dot(friction_jacobian_first[constraint], projected_twist[first])
        if second >= 0:
            friction_velocity += wp.dot(friction_jacobian_second[constraint], projected_twist[second])
        friction_old = friction_reaction[constraint]
        split = friction_delassus[constraint]
        friction_new = wp.clamp(
            -((friction_velocity - split * friction_old) / split),
            -friction_bound[constraint],
            friction_bound[constraint],
        )
        friction_delta = friction_new - friction_old
        if not wp.isfinite(friction_new) or not wp.isfinite(friction_delta):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        first_correction = vec6f(0.0)
        second_correction = vec6f(0.0)
        if first >= 0:
            first_correction = (inverse_weight[first] @ friction_jacobian_first[constraint]) * friction_delta
        if second >= 0:
            second_correction = (inverse_weight[second] @ friction_jacobian_second[constraint]) * friction_delta
        if not _finite_twist(first_correction) or not _finite_twist(second_correction):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        if first >= 0:
            if first == second:
                _accumulate_twist_colored(
                    first, target_color, first_correction + second_correction, occupancy, twist_delta
                )
            else:
                _accumulate_twist_colored(first, target_color, first_correction, occupancy, twist_delta)
        if second >= 0 and second != first:
            _accumulate_twist_colored(second, target_color, second_correction, occupancy, twist_delta)
        friction_reaction[constraint] = friction_new

    contact_begin = contact_offsets[target_color] + lane
    contact_end = contact_offsets[target_color] + contact_counts[target_color]
    for ordered in range(contact_begin, contact_end, launch_dim):
        constraint = contact_order[ordered]
        world = contact_world[constraint]
        if not world_active[world] or world_status[world] != PROJECTION_STATUS_VALID:
            continue
        first = contact_first[constraint]
        second = contact_second[constraint]
        if first < 0 and second < 0:
            contact_reaction[constraint] = wp.vec3f(0.0)
            continue
        twist_first = vec6f(0.0)
        twist_second = vec6f(0.0)
        if first >= 0:
            twist_first = projected_twist[first]
        if second >= 0:
            twist_second = projected_twist[second]
        contact_old = contact_reaction[constraint]
        contact_velocity = (
            contact_jacobian_first[constraint] @ twist_first
            + contact_jacobian_second[constraint] @ twist_second
            + contact_bias[constraint]
        )
        contact_block = contact_delassus[constraint]
        free = contact_velocity - contact_block @ contact_old
        contact_new = _solve_contact_coulomb_newton_normal_last(contact_block, free, contact_friction[constraint])
        contact_delta = contact_new - contact_old
        if not _finite_vec3(contact_new) or not _finite_vec3(contact_delta):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        first_correction = vec6f(0.0)
        second_correction = vec6f(0.0)
        if first >= 0:
            first_correction = inverse_weight[first] @ (
                wp.transpose(contact_jacobian_first[constraint]) @ contact_delta
            )
        if second >= 0:
            second_correction = inverse_weight[second] @ (
                wp.transpose(contact_jacobian_second[constraint]) @ contact_delta
            )
        if not _finite_twist(first_correction) or not _finite_twist(second_correction):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        if first >= 0:
            if first == second:
                _accumulate_twist_colored(
                    first, target_color, first_correction + second_correction, occupancy, twist_delta
                )
            else:
                _accumulate_twist_colored(first, target_color, first_correction, occupancy, twist_delta)
        if second >= 0 and second != first:
            _accumulate_twist_colored(second, target_color, second_correction, occupancy, twist_delta)
        contact_reaction[constraint] = contact_new

    limit_begin = limit_offsets[target_color] + lane
    limit_end = limit_offsets[target_color] + limit_counts[target_color]
    for ordered in range(limit_begin, limit_end, launch_dim):
        constraint = limit_order[ordered]
        world = limit_world[constraint]
        if not world_active[world] or world_status[world] != PROJECTION_STATUS_VALID:
            continue
        first = limit_first[constraint]
        second = limit_second[constraint]
        if first < 0 and second < 0:
            limit_reaction[constraint] = 0.0
            continue
        limit_velocity = limit_bias[constraint]
        if first >= 0:
            limit_velocity += wp.dot(limit_jacobian_first[constraint], projected_twist[first])
        if second >= 0:
            limit_velocity += wp.dot(limit_jacobian_second[constraint], projected_twist[second])
        limit_old = limit_reaction[constraint]
        split = limit_delassus[constraint]
        limit_new = wp.max(-((limit_velocity - split * limit_old) / split), 0.0)
        limit_delta = limit_new - limit_old
        if not wp.isfinite(limit_new) or not wp.isfinite(limit_delta):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        first_correction = vec6f(0.0)
        second_correction = vec6f(0.0)
        if first >= 0:
            first_correction = (inverse_weight[first] @ limit_jacobian_first[constraint]) * limit_delta
        if second >= 0:
            second_correction = (inverse_weight[second] @ limit_jacobian_second[constraint]) * limit_delta
        if not _finite_twist(first_correction) or not _finite_twist(second_correction):
            world_status[world] = PROJECTION_STATUS_INVALID
            continue
        if first >= 0:
            if first == second:
                _accumulate_twist_colored(
                    first, target_color, first_correction + second_correction, occupancy, twist_delta
                )
            else:
                _accumulate_twist_colored(first, target_color, first_correction, occupancy, twist_delta)
        if second >= 0 and second != first:
            _accumulate_twist_colored(second, target_color, second_correction, occupancy, twist_delta)
        limit_reaction[constraint] = limit_new


@wp.kernel
def _project_deformable_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    normal: wp.array[wp.vec3f],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    rigid_bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    scalar_delassus: wp.array[wp.float32],
    delassus: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    body_inverse_weight: wp.array[mat66f],
    include_rigid: bool,
    projected_velocity: wp.array[wp.vec3f],
    projected_twist: wp.array[vec6f],
    particle_reaction: wp.array[wp.vec3f],
    rigid_reaction: wp.array[wp.vec3f],
    particle_delta: wp.array[wp.vec3f],
    body_delta: wp.array[vec6f],
    projection_status: wp.array[wp.int32],
    contact_world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        contact = order[ordered]
        world = contact_world[contact]
        if (
            contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
            or not world_active[world]
            or projection_status[world] != PROJECTION_STATUS_VALID
        ):
            continue
        contact_frame = frame[contact]
        velocity = bias[contact]
        if include_rigid:
            velocity = rigid_bias[contact]
        for particle_slot in range(4):
            particle = particle_indices[contact, particle_slot]
            if particle >= 0:
                particle_velocity = projected_velocity[particle]
                if include_rigid:
                    particle_velocity = wp.transpose(contact_frame) @ particle_velocity
                velocity += coefficients[contact, particle_slot] * particle_velocity
        body = contact_body[contact]
        if include_rigid and body >= 0:
            velocity += body_jacobian[contact] @ projected_twist[body]
        old = particle_reaction[contact]
        new = wp.vec3f(0.0)
        if include_rigid:
            old = rigid_reaction[contact]
            contact_block = delassus[contact]
            free = velocity - contact_block @ old
            new = _solve_contact_coulomb_newton_normal_last(contact_block, free, friction[contact])
        else:
            value = scalar_delassus[contact]
            new = project_deformable_contact_coulomb(velocity - value * old, normal[contact], value, friction[contact])
        delta = new - old
        world_delta = delta
        if include_rigid:
            world_delta = contact_frame @ delta
        if not _finite_vec3(new) or not _finite_vec3(world_delta):
            contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            contact_world_status[world] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            projection_status[world] = PROJECTION_STATUS_INVALID
            continue
        correction_is_finite = True
        for particle_slot in range(4):
            particle = particle_indices[contact, particle_slot]
            if particle >= 0:
                coefficient = coefficients[contact, particle_slot]
                for later in range(particle_slot + 1, 4):
                    if particle_indices[contact, later] == particle:
                        coefficient += coefficients[contact, later]
                unique = True
                for previous in range(particle_slot):
                    unique = unique and particle_indices[contact, previous] != particle
                if unique:
                    correction = particle_inverse_weight[particle] * coefficient * world_delta
                    if not _finite_vec3(correction):
                        correction_is_finite = False
                    else:
                        _accumulate_particle_colored(
                            particle, target_color, correction, particle_occupancy, particle_delta
                        )
        body_correction = vec6f(0.0)
        if include_rigid and body >= 0:
            body_correction = body_inverse_weight[body] @ (wp.transpose(body_jacobian[contact]) @ delta)
            correction_is_finite = correction_is_finite and _finite_twist(body_correction)
        if not correction_is_finite:
            contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            contact_world_status[world] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            projection_status[world] = PROJECTION_STATUS_INVALID
            continue
        if include_rigid and body >= 0:
            _accumulate_twist_colored(body, target_color, body_correction, body_occupancy, body_delta)
        if include_rigid:
            rigid_reaction[contact] = new
        else:
            particle_reaction[contact] = new


@wp.kernel
def _apply_body_delta(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    delta: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
        projected_twist[body] += delta[body]
    delta[body] = vec6f(0.0)


@wp.kernel
def _apply_particle_delta_colored(
    particle_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    delta: wp.array[wp.vec3f],
    projected_velocity: wp.array[wp.vec3f],
):
    particle = wp.tid()
    world = particle_world[particle]
    if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
        projected_velocity[particle] += delta[particle]
    delta[particle] = wp.vec3f(0.0)


@wp.kernel
def _apply_body_particle_delta_colored(
    launch_dim: int,
    body_count: int,
    particle_count: int,
    body_world: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    body_delta: wp.array[vec6f],
    particle_delta: wp.array[wp.vec3f],
    projected_twist: wp.array[vec6f],
    projected_velocity: wp.array[wp.vec3f],
):
    lane = wp.tid()
    for body in range(lane, body_count, launch_dim):
        world = body_world[body]
        if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
            projected_twist[body] += body_delta[body]
        body_delta[body] = vec6f(0.0)
    for particle in range(lane, particle_count, launch_dim):
        world = particle_world[particle]
        if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
            projected_velocity[particle] += particle_delta[particle]
        particle_delta[particle] = wp.vec3f(0.0)


class _ColorFamily:
    def __init__(self, capacity: int, color_count: int, device):
        self.capacity = capacity
        self.worker_count = _bounded_worker_count(capacity, device)
        self.colors = wp.full(capacity, _NO_PROPOSAL, dtype=wp.int32, device=device)
        self.proposals = wp.full(capacity, _NO_PROPOSAL, dtype=wp.int32, device=device)
        self.counts = wp.zeros(color_count, dtype=wp.int32, device=device)
        self.offsets = wp.zeros(color_count, dtype=wp.int32, device=device)
        self.cursors = wp.zeros(color_count, dtype=wp.int32, device=device)
        self.order = wp.full(capacity, -1, dtype=wp.int32, device=device)

    def _prefix_counts(self, color_count: int, device) -> None:
        if color_count <= _SERIAL_COLOR_PREFIX_LIMIT:
            wp.launch(
                _prefix_color_counts,
                dim=1,
                inputs=[color_count, self.counts],
                outputs=[self.offsets, self.cursors],
                device=device,
            )
        else:
            wp.utils.array_scan(self.counts, self.offsets, inclusive=False)
            wp.copy(self.cursors, self.offsets)

    def compact(self, color_count: int, device) -> None:
        self.counts.zero_()
        if self.capacity == 0:
            return
        wp.launch(
            _count_colors, dim=self.capacity, inputs=[self.colors, color_count], outputs=[self.counts], device=device
        )
        self._prefix_counts(color_count, device)
        wp.launch(
            _scatter_color_order,
            dim=self.capacity,
            inputs=[self.colors, color_count],
            outputs=[self.cursors, self.order],
            device=device,
        )


class ColoredGaussSeidelProjection:
    """Own fixed-capacity coloring and projection scratch for one LOX solver.

    The effective color count is the requested maximum bounded by total
    allocated unilateral capacity, with one internal color retained for empty
    or single-constraint systems. The endpoint occupancy tables use int32
    atomics and occupy exactly ``4 * effective_color_count * (body_count +
    particle_count)`` bytes. They are only allocated for requested counts of
    at least two; the zero- and one-color endpoints use the existing sequential
    and Jacobi implementations directly.
    """

    def __init__(self, adapter, deformable_contacts, color_count: int):
        if color_count < 2:
            raise ValueError("Colored Gauss-Seidel requires at least two colors.")
        self.adapter = adapter
        self.deformable_contacts = deformable_contacts
        rigid_capacity = 0
        if adapter is not None:
            rigid_capacity = adapter.friction_capacity + adapter.contact_capacity + adapter.limit_capacity
        deformable_capacity = deformable_contacts.contact_capacity if deformable_contacts is not None else 0
        self.requested_color_count = color_count
        self.color_count = max(1, min(color_count, rigid_capacity + deformable_capacity))
        self.device = adapter.device if adapter is not None else deformable_contacts.device
        body_count = adapter.body_constraint_count.shape[0] if adapter is not None else 0
        particle_count = deformable_contacts.cloth_system.particle_count if deformable_contacts is not None else 0
        self.occupancy_storage_bytes = 4 * self.color_count * (body_count + particle_count)
        self.body_occupancy = wp.zeros((body_count, self.color_count), dtype=wp.int32, device=self.device)
        self.particle_occupancy = wp.zeros((particle_count, self.color_count), dtype=wp.int32, device=self.device)
        self.body_locks = wp.full(body_count, _LOCK_FREE, dtype=wp.int32, device=self.device)
        self.particle_locks = wp.full(particle_count, _LOCK_FREE, dtype=wp.int32, device=self.device)
        self.friction = _ColorFamily(
            adapter.friction_capacity if adapter is not None else 0, self.color_count, self.device
        )
        self.contact = _ColorFamily(
            adapter.contact_capacity if adapter is not None else 0, self.color_count, self.device
        )
        self.limit = _ColorFamily(adapter.limit_capacity if adapter is not None else 0, self.color_count, self.device)
        self.deformable = _ColorFamily(
            deformable_contacts.contact_capacity if deformable_contacts is not None else 0,
            self.color_count,
            self.device,
        )
        self.friction_delassus = wp.zeros(self.friction.capacity, dtype=wp.float32, device=self.device)
        self.contact_delassus = wp.zeros(self.contact.capacity, dtype=wp.mat33f, device=self.device)
        self.limit_delassus = wp.zeros(self.limit.capacity, dtype=wp.float32, device=self.device)
        self._families = (self.friction, self.contact, self.limit)
        self.rigid_worker_count = _bounded_worker_count(rigid_capacity, self.device)
        deformable_per_color_capacity = (self.deformable.capacity + self.color_count - 1) // self.color_count
        self.deformable_projection_worker_count = _bounded_worker_count(deformable_per_color_capacity, self.device)
        self._fuse_rigid_families = sum(family.capacity > 0 for family in self._families) > 1
        self.apply_worker_count = _bounded_worker_count(max(body_count, particle_count), self.device)
        # Split domain kernels preserve occupancy on older architectures; launch fusion wins on Blackwell and newer.
        self._combine_apply_domains = self.device.is_cuda and self.device.arch >= 100

    def matches(self, deformable_contacts) -> bool:
        return self.deformable_contacts is deformable_contacts

    def _launch_rigid_families(self, kernel, extra_inputs: tuple = ()) -> None:
        adapter = self.adapter
        if adapter is None:
            return
        entries = (
            (
                self.friction,
                adapter.friction_world,
                adapter.friction_local,
                adapter.world_friction_count,
                adapter.friction_body_first,
                adapter.friction_body_second,
                0,
            ),
            (
                self.contact,
                adapter.contact_world,
                adapter.contact_local,
                adapter.world_contact_count,
                adapter.contact_body_first,
                adapter.contact_body_second,
                1,
            ),
            (
                self.limit,
                adapter.limit_world,
                adapter.limit_local,
                adapter.world_limit_count,
                adapter.limit_body_first,
                adapter.limit_body_second,
                2,
            ),
        )
        key_offset = 0
        for family, worlds, local, counts, first, second, salt in entries:
            if family.capacity > 0:
                if kernel is _assign_two_endpoint_colors:
                    inputs = [worlds, local, counts, first, second, salt, self.color_count]
                    outputs = [family.colors]
                elif kernel is _count_two_endpoint_occupancy:
                    inputs = [worlds, local, counts, first, second, family.colors]
                    outputs = [self.body_occupancy]
                elif kernel is _propose_two_endpoint_repairs:
                    inputs = [
                        worlds,
                        local,
                        counts,
                        first,
                        second,
                        self.color_count,
                        family.colors,
                        self.body_occupancy,
                    ]
                    outputs = [family.proposals]
                elif kernel is _propose_and_claim_two_endpoint_repairs:
                    inputs = [
                        worlds,
                        local,
                        counts,
                        first,
                        second,
                        self.color_count,
                        key_offset,
                        family.colors,
                        self.body_occupancy,
                    ]
                    outputs = [family.proposals, self.body_locks]
                elif kernel is _claim_two_endpoint_repairs:
                    inputs = [first, second, family.proposals, key_offset]
                    outputs = [self.body_locks]
                else:
                    inputs = [first, second, key_offset, family.colors, family.proposals, self.body_locks]
                    outputs = [self.body_occupancy]
                wp.launch(kernel, dim=family.capacity, inputs=inputs, outputs=outputs, device=self.device)
            key_offset += family.capacity

    def build_colors(self) -> None:
        self.body_occupancy.zero_()
        self.particle_occupancy.zero_()
        self._launch_rigid_families(_assign_two_endpoint_colors)
        self._launch_rigid_families(_count_two_endpoint_occupancy)
        deformable = self.deformable_contacts
        deformable_key_offset = sum(family.capacity for family in self._families)
        if deformable is not None:
            wp.launch(
                _assign_deformable_colors,
                dim=deformable.contact_capacity,
                inputs=[
                    deformable.particle_indices,
                    deformable.contact_world,
                    deformable.body,
                    deformable.status,
                    self.color_count,
                ],
                outputs=[self.deformable.colors],
                device=self.device,
            )
            wp.launch(
                _count_deformable_occupancy,
                dim=deformable.contact_capacity,
                inputs=[deformable.particle_indices, deformable.body, deformable.status, self.deformable.colors],
                outputs=[self.particle_occupancy, self.body_occupancy],
                device=self.device,
            )
        for _repair in range(_COLOR_REPAIR_PASSES):
            self.body_locks.fill_(_LOCK_FREE)
            self.particle_locks.fill_(_LOCK_FREE)
            self._launch_rigid_families(_propose_and_claim_two_endpoint_repairs)
            if deformable is not None:
                wp.launch(
                    _propose_and_claim_deformable_repairs,
                    dim=deformable.contact_capacity,
                    inputs=[
                        deformable.particle_indices,
                        deformable.body,
                        deformable.status,
                        self.color_count,
                        deformable_key_offset,
                        self.deformable.colors,
                        self.particle_occupancy,
                        self.body_occupancy,
                    ],
                    outputs=[self.deformable.proposals, self.particle_locks, self.body_locks],
                    device=self.device,
                )
            self._launch_rigid_families(_commit_two_endpoint_repairs)
            if deformable is not None:
                wp.launch(
                    _commit_deformable_repairs,
                    dim=deformable.contact_capacity,
                    inputs=[
                        deformable.particle_indices,
                        deformable.body,
                        deformable_key_offset,
                        self.deformable.colors,
                        self.deformable.proposals,
                        self.particle_locks,
                        self.body_locks,
                    ],
                    outputs=[self.particle_occupancy, self.body_occupancy],
                    device=self.device,
                )
        for family in (*self._families, self.deformable):
            family.compact(self.color_count, self.device)

    def prepare(self, inverse_weight: wp.array[mat66f] | None, prepared_status: wp.array[wp.int32]) -> None:
        self.build_colors()
        adapter = self.adapter
        empty_body_weight = None
        if adapter is not None:
            prepare_jacobi_projection_data(
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
                adapter.contact_bias,
                adapter.contact_friction,
                adapter.limit_world,
                adapter.limit_local,
                adapter.world_limit_count,
                adapter.limit_body_first,
                adapter.limit_body_second,
                adapter.limit_jacobian_first,
                adapter.limit_jacobian_second,
                adapter.body_constraint_count,
                adapter.static_body_constraint_count,
                inverse_weight,
                adapter.friction_projection_delassus,
                adapter.contact_projection_delassus,
                adapter.limit_projection_delassus,
                prepared_status,
            )
            if self.deformable_contacts is not None:
                self.deformable_contacts.prepare_rigid_projection(
                    adapter.body_constraint_count,
                    adapter.static_body_constraint_count,
                    inverse_weight,
                    prepared_status,
                )
            empty_body_weight = inverse_weight
            if self.rigid_worker_count > 0:
                for color in range(self.color_count):
                    if self._fuse_rigid_families:
                        wp.launch(
                            _prepare_rigid_colored,
                            dim=self.rigid_worker_count,
                            inputs=[
                                self.rigid_worker_count,
                                color,
                                self.friction.counts,
                                self.friction.offsets,
                                self.friction.order,
                                adapter.friction_world,
                                adapter.friction_body_first,
                                adapter.friction_body_second,
                                adapter.friction_jacobian_first,
                                adapter.friction_jacobian_second,
                                self.contact.counts,
                                self.contact.offsets,
                                self.contact.order,
                                adapter.contact_world,
                                adapter.contact_body_first,
                                adapter.contact_body_second,
                                adapter.contact_jacobian_first,
                                adapter.contact_jacobian_second,
                                adapter.contact_bias,
                                adapter.contact_friction,
                                self.limit.counts,
                                self.limit.offsets,
                                self.limit.order,
                                adapter.limit_world,
                                adapter.limit_body_first,
                                adapter.limit_body_second,
                                adapter.limit_jacobian_first,
                                adapter.limit_jacobian_second,
                                self.body_occupancy,
                                inverse_weight,
                            ],
                            outputs=[
                                self.friction_delassus,
                                self.contact_delassus,
                                self.limit_delassus,
                                prepared_status,
                            ],
                            device=self.device,
                            block_dim=_COLOR_BLOCK_DIM,
                        )
                    else:
                        if self.friction.capacity > 0:
                            wp.launch(
                                _prepare_frictions_colored,
                                dim=self.friction.worker_count,
                                inputs=[
                                    self.friction.worker_count,
                                    color,
                                    self.friction.counts,
                                    self.friction.offsets,
                                    self.friction.order,
                                    adapter.friction_world,
                                    adapter.friction_body_first,
                                    adapter.friction_body_second,
                                    adapter.friction_jacobian_first,
                                    adapter.friction_jacobian_second,
                                    self.body_occupancy,
                                    inverse_weight,
                                ],
                                outputs=[self.friction_delassus, prepared_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        elif self.contact.capacity > 0:
                            wp.launch(
                                _prepare_contacts_colored,
                                dim=self.contact.worker_count,
                                inputs=[
                                    self.contact.worker_count,
                                    color,
                                    self.contact.counts,
                                    self.contact.offsets,
                                    self.contact.order,
                                    adapter.contact_world,
                                    adapter.contact_body_first,
                                    adapter.contact_body_second,
                                    adapter.contact_jacobian_first,
                                    adapter.contact_jacobian_second,
                                    adapter.contact_bias,
                                    adapter.contact_friction,
                                    self.body_occupancy,
                                    inverse_weight,
                                ],
                                outputs=[self.contact_delassus, prepared_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        else:
                            wp.launch(
                                _prepare_frictions_colored,
                                dim=self.limit.worker_count,
                                inputs=[
                                    self.limit.worker_count,
                                    color,
                                    self.limit.counts,
                                    self.limit.offsets,
                                    self.limit.order,
                                    adapter.limit_world,
                                    adapter.limit_body_first,
                                    adapter.limit_body_second,
                                    adapter.limit_jacobian_first,
                                    adapter.limit_jacobian_second,
                                    self.body_occupancy,
                                    inverse_weight,
                                ],
                                outputs=[self.limit_delassus, prepared_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
        else:
            prepared_status.fill_(PROJECTION_STATUS_VALID)
        deformable = self.deformable_contacts
        if deformable is not None:
            include_rigid = adapter is not None
            body_weight = empty_body_weight if include_rigid else deformable._empty_body_inverse_weight
            for color in range(self.color_count):
                wp.launch(
                    _prepare_deformable_colored,
                    dim=self.deformable.worker_count,
                    inputs=[
                        self.deformable.worker_count,
                        color,
                        self.deformable.counts,
                        self.deformable.offsets,
                        self.deformable.order,
                        deformable.particle_indices,
                        deformable.coefficients,
                        deformable.contact_world,
                        deformable.body,
                        deformable.body_jacobian,
                        deformable.rigid_bias,
                        deformable.friction,
                        self.particle_occupancy,
                        self.body_occupancy,
                        deformable.cloth_system.inverse_weight,
                        body_weight,
                        include_rigid,
                    ],
                    outputs=[
                        deformable.status,
                        deformable.gauss_seidel_scalar_delassus,
                        deformable.gauss_seidel_delassus,
                        prepared_status,
                        deformable.world_status,
                        deformable.invalid_count,
                    ],
                    device=self.device,
                    block_dim=_COLOR_BLOCK_DIM,
                )
            wp.launch(
                _merge_rigid_prepared_status,
                dim=prepared_status.shape[0],
                inputs=[deformable.world_status, deformable.global_status],
                outputs=[prepared_status],
                device=self.device,
            )

    def project(
        self,
        iterations: int,
        world_active: wp.array[wp.bool],
        body_world: wp.array[wp.int32] | None,
        inverse_weight: wp.array[mat66f] | None,
        projected_twist: wp.array[vec6f] | None,
        twist_delta: wp.array[vec6f] | None,
        projected_velocity: wp.array[wp.vec3f] | None,
        prepared_status: wp.array[wp.int32],
        projection_status: wp.array[wp.int32],
    ) -> None:
        adapter = self.adapter
        deformable = self.deformable_contacts
        if adapter is not None:
            wp.launch(
                _initialize_jacobi_projection_status,
                dim=world_active.shape[0],
                inputs=[world_active, prepared_status],
                outputs=[projection_status],
                device=self.device,
            )
            twist_delta.zero_()
            if deformable is not None:
                deformable.particle_delta.zero_()
            if self.friction.capacity > 0:
                wp.launch(
                    _warmstart_frictions_jacobi,
                    dim=self.friction.capacity,
                    inputs=[
                        adapter.friction_world,
                        adapter.friction_local,
                        world_active,
                        prepared_status,
                        adapter.world_friction_count,
                        adapter.friction_body_first,
                        adapter.friction_body_second,
                        adapter.friction_jacobian_first,
                        adapter.friction_jacobian_second,
                        inverse_weight,
                        adapter.friction_reaction,
                    ],
                    outputs=[twist_delta],
                    device=self.device,
                )
            if self.contact.capacity > 0:
                wp.launch(
                    _warmstart_contacts_jacobi,
                    dim=self.contact.capacity,
                    inputs=[
                        adapter.contact_world,
                        adapter.contact_local,
                        world_active,
                        prepared_status,
                        adapter.world_contact_count,
                        adapter.contact_body_first,
                        adapter.contact_body_second,
                        adapter.contact_jacobian_first,
                        adapter.contact_jacobian_second,
                        inverse_weight,
                        adapter.contact_reaction,
                    ],
                    outputs=[twist_delta],
                    device=self.device,
                )
            if self.limit.capacity > 0:
                wp.launch(
                    _warmstart_limits_jacobi,
                    dim=self.limit.capacity,
                    inputs=[
                        adapter.limit_world,
                        adapter.limit_local,
                        world_active,
                        prepared_status,
                        adapter.world_limit_count,
                        adapter.limit_body_first,
                        adapter.limit_body_second,
                        adapter.limit_jacobian_first,
                        adapter.limit_jacobian_second,
                        inverse_weight,
                        adapter.limit_reaction,
                    ],
                    outputs=[twist_delta],
                    device=self.device,
                )
            if deformable is not None:
                deformable.accumulate_rigid_reaction_warm_start(
                    world_active,
                    prepared_status,
                    deformable.cloth_system.inverse_weight,
                    inverse_weight,
                    projected_velocity,
                    projected_twist,
                    twist_delta,
                    projection_status,
                )
            if deformable is not None and self._combine_apply_domains:
                wp.launch(
                    _apply_body_particle_delta_colored,
                    dim=self.apply_worker_count,
                    inputs=[
                        self.apply_worker_count,
                        projected_twist.shape[0],
                        projected_velocity.shape[0],
                        body_world,
                        deformable.cloth_system.topology.packed_world,
                        world_active,
                        projection_status,
                    ],
                    outputs=[twist_delta, deformable.particle_delta, projected_twist, projected_velocity],
                    device=self.device,
                    block_dim=_COLOR_BLOCK_DIM,
                )
            else:
                wp.launch(
                    _apply_body_delta,
                    dim=projected_twist.shape[0],
                    inputs=[body_world, world_active, projection_status],
                    outputs=[twist_delta, projected_twist],
                    device=self.device,
                )
                if deformable is not None:
                    wp.launch(
                        _apply_particle_delta_colored,
                        dim=projected_velocity.shape[0],
                        inputs=[deformable.cloth_system.topology.packed_world, world_active, projection_status],
                        outputs=[deformable.particle_delta, projected_velocity],
                        device=self.device,
                    )
        else:
            wp.copy(projection_status, prepared_status)
            deformable.apply_reaction_warm_start(projected_velocity)
            deformable.particle_delta.zero_()
        for _iteration in range(iterations):
            for color in range(self.color_count):
                if adapter is not None:
                    if self.rigid_worker_count > 0:
                        if self._fuse_rigid_families:
                            wp.launch(
                                _project_rigid_colored,
                                dim=self.rigid_worker_count,
                                inputs=[
                                    self.rigid_worker_count,
                                    color,
                                    self.friction.counts,
                                    self.friction.offsets,
                                    self.friction.order,
                                    adapter.friction_world,
                                    adapter.friction_body_first,
                                    adapter.friction_body_second,
                                    adapter.friction_jacobian_first,
                                    adapter.friction_jacobian_second,
                                    adapter.friction_impulse_bound,
                                    self.friction_delassus,
                                    self.contact.counts,
                                    self.contact.offsets,
                                    self.contact.order,
                                    adapter.contact_world,
                                    adapter.contact_body_first,
                                    adapter.contact_body_second,
                                    adapter.contact_jacobian_first,
                                    adapter.contact_jacobian_second,
                                    adapter.contact_bias,
                                    adapter.contact_friction,
                                    self.contact_delassus,
                                    self.limit.counts,
                                    self.limit.offsets,
                                    self.limit.order,
                                    adapter.limit_world,
                                    adapter.limit_body_first,
                                    adapter.limit_body_second,
                                    adapter.limit_jacobian_first,
                                    adapter.limit_jacobian_second,
                                    adapter.limit_bias,
                                    self.limit_delassus,
                                    world_active,
                                    self.body_occupancy,
                                    inverse_weight,
                                    projected_twist,
                                ],
                                outputs=[
                                    adapter.friction_reaction,
                                    adapter.contact_reaction,
                                    adapter.limit_reaction,
                                    twist_delta,
                                    projection_status,
                                ],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        elif self.friction.capacity > 0:
                            wp.launch(
                                _project_frictions_colored,
                                dim=self.friction.worker_count,
                                inputs=[
                                    self.friction.worker_count,
                                    color,
                                    self.friction.counts,
                                    self.friction.offsets,
                                    self.friction.order,
                                    adapter.friction_world,
                                    adapter.friction_body_first,
                                    adapter.friction_body_second,
                                    adapter.friction_jacobian_first,
                                    adapter.friction_jacobian_second,
                                    adapter.friction_impulse_bound,
                                    self.friction_delassus,
                                    world_active,
                                    self.body_occupancy,
                                    inverse_weight,
                                    projected_twist,
                                ],
                                outputs=[adapter.friction_reaction, twist_delta, projection_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        elif self.contact.capacity > 0:
                            wp.launch(
                                _project_contacts_colored,
                                dim=self.contact.worker_count,
                                inputs=[
                                    self.contact.worker_count,
                                    color,
                                    self.contact.counts,
                                    self.contact.offsets,
                                    self.contact.order,
                                    adapter.contact_world,
                                    adapter.contact_body_first,
                                    adapter.contact_body_second,
                                    adapter.contact_jacobian_first,
                                    adapter.contact_jacobian_second,
                                    adapter.contact_bias,
                                    adapter.contact_friction,
                                    self.contact_delassus,
                                    world_active,
                                    self.body_occupancy,
                                    inverse_weight,
                                    projected_twist,
                                ],
                                outputs=[adapter.contact_reaction, twist_delta, projection_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        else:
                            wp.launch(
                                _project_limits_colored,
                                dim=self.limit.worker_count,
                                inputs=[
                                    self.limit.worker_count,
                                    color,
                                    self.limit.counts,
                                    self.limit.offsets,
                                    self.limit.order,
                                    adapter.limit_world,
                                    adapter.limit_body_first,
                                    adapter.limit_body_second,
                                    adapter.limit_jacobian_first,
                                    adapter.limit_jacobian_second,
                                    adapter.limit_bias,
                                    self.limit_delassus,
                                    world_active,
                                    self.body_occupancy,
                                    inverse_weight,
                                    projected_twist,
                                ],
                                outputs=[adapter.limit_reaction, twist_delta, projection_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                if deformable is not None:
                    include_rigid = adapter is not None
                    body_weight = inverse_weight if include_rigid else deformable._empty_body_inverse_weight
                    body_twist = projected_twist if include_rigid else deformable._empty_body_twist
                    body_delta = twist_delta if include_rigid else deformable._empty_body_twist
                    wp.launch(
                        _project_deformable_colored,
                        dim=self.deformable_projection_worker_count,
                        inputs=[
                            self.deformable_projection_worker_count,
                            color,
                            self.deformable.counts,
                            self.deformable.offsets,
                            self.deformable.order,
                            deformable.particle_indices,
                            deformable.coefficients,
                            deformable.contact_world,
                            deformable.body,
                            deformable.normal,
                            deformable.frame,
                            deformable.body_jacobian,
                            deformable.bias,
                            deformable.rigid_bias,
                            deformable.friction,
                            deformable.gauss_seidel_scalar_delassus,
                            deformable.gauss_seidel_delassus,
                            deformable.status,
                            world_active,
                            self.particle_occupancy,
                            self.body_occupancy,
                            deformable.cloth_system.inverse_weight,
                            body_weight,
                            include_rigid,
                            projected_velocity,
                            body_twist,
                        ],
                        outputs=[
                            deformable.reaction,
                            deformable.rigid_reaction,
                            deformable.particle_delta,
                            body_delta,
                            projection_status,
                            deformable.world_status,
                        ],
                        device=self.device,
                        block_dim=_COLOR_BLOCK_DIM,
                    )
                if adapter is not None and deformable is not None and self._combine_apply_domains:
                    wp.launch(
                        _apply_body_particle_delta_colored,
                        dim=self.apply_worker_count,
                        inputs=[
                            self.apply_worker_count,
                            projected_twist.shape[0],
                            projected_velocity.shape[0],
                            body_world,
                            deformable.cloth_system.topology.packed_world,
                            world_active,
                            projection_status,
                        ],
                        outputs=[twist_delta, deformable.particle_delta, projected_twist, projected_velocity],
                        device=self.device,
                        block_dim=_COLOR_BLOCK_DIM,
                    )
                else:
                    if adapter is not None:
                        wp.launch(
                            _apply_body_delta,
                            dim=projected_twist.shape[0],
                            inputs=[body_world, world_active, projection_status],
                            outputs=[twist_delta, projected_twist],
                            device=self.device,
                        )
                    if deformable is not None:
                        wp.launch(
                            _apply_particle_delta_colored,
                            dim=projected_velocity.shape[0],
                            inputs=[deformable.cloth_system.topology.packed_world, world_active, projection_status],
                            outputs=[deformable.particle_delta, projected_velocity],
                            device=self.device,
                        )

        if adapter is not None:
            project_constraints_jacobi(
                1,
                world_active,
                body_world,
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
                adapter.limit_world,
                adapter.limit_local,
                adapter.world_limit_count,
                adapter.limit_body_first,
                adapter.limit_body_second,
                adapter.limit_jacobian_first,
                adapter.limit_jacobian_second,
                adapter.limit_bias,
                adapter.limit_projection_delassus,
                inverse_weight,
                projected_twist,
                twist_delta,
                adapter.contact_reaction,
                adapter.limit_reaction,
                adapter.friction_reaction,
                prepared_status,
                projection_status,
                deformable_contacts=deformable,
                deformable_projected_velocity=projected_velocity,
                warm_start=False,
            )
        else:
            deformable.project_jacobi_smoothing_sweep(world_active, projected_velocity, projection_status)
