# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-space contact projection for LOX deformable particles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ......utils.mesh import MeshAdjacencyData
from ...core.types import mat36f, mat66f, vec6f
from ...geometry.contacts import make_contact_frame_znorm
from .bias import compute_contact_velocity_target
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
    convert_contact_matrix_normal_last_to_first,
    convert_contact_vector_normal_last_to_first,
    prepare_contact_coulomb_delassus,
    project_contact_coulomb_cone_orthogonal,
)
from .soft_contact_filter import (
    compute_soft_edge_normal_cones,
    soft_surface_contact_normal_cone_contains,
)
from .time import validate_world_time_step

if TYPE_CHECKING:
    from ......sim import Contacts, Model, State
    from .deformable_self_contact import DeformableSelfContactDetector
    from .deformable_system import DeformableClothSystem

__all__ = [
    "DEFORMABLE_CONTACT_STATUS_CROSS_WORLD",
    "DEFORMABLE_CONTACT_STATUS_DYNAMIC_RIGID",
    "DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS",
    "DEFORMABLE_CONTACT_STATUS_MALFORMED",
    "DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE",
    "DEFORMABLE_CONTACT_STATUS_UNUSED",
    "DEFORMABLE_CONTACT_STATUS_VALID",
    "DeformableContactSystem",
    "compute_deformable_contact_residual",
    "project_deformable_contact_coulomb",
]

DEFORMABLE_CONTACT_STATUS_UNUSED = 0
"""The contact-capacity slot does not contain an active source record."""

DEFORMABLE_CONTACT_STATUS_VALID = 1
"""The source record and its scalar Delassus coefficient are valid."""

DEFORMABLE_CONTACT_STATUS_MALFORMED = 2
"""The source record has invalid indices, coefficients, geometry, or shape data."""

DEFORMABLE_CONTACT_STATUS_CROSS_WORLD = 3
"""The contact feature or collider spans incompatible Newton worlds."""

DEFORMABLE_CONTACT_STATUS_DYNAMIC_RIGID = 4
"""Reserved legacy status for a rejected dynamic rigid collider."""

DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS = 5
"""The contact has no finite positive scalar Delassus coefficient."""

DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE = 6
"""The projection produced a non-finite reaction or particle correction."""

_BODY_FLAG_KINEMATIC = 1 << 1
_PARTICLE_FLAG_ACTIVE = 1
_COEFFICIENT_TOLERANCE = 1.0e-5
_NORMAL_EPSILON = 1.0e-12
_APGD_BLOCK_DIM = 256
_APGD_BLOCKS_PER_SM = 2
# The Coulomb solve's register footprint benefits from smaller projection blocks.
_RIGID_CONTACT_PROJECTION_BLOCK_DIM = 128


def _bounded_apgd_worker_count(capacity: int, device) -> int:
    if device.is_cuda:
        return min(capacity, max(_APGD_BLOCK_DIM, device.sm_count * _APGD_BLOCKS_PER_SM * _APGD_BLOCK_DIM))
    return capacity


wp.set_module_options({"enable_backward": False})


@wp.func
def _is_finite_vec3(value: wp.vec3) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _reject_contact(
    contact: int,
    world: int,
    status: int,
    contact_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    global_status: wp.array[wp.int32],
    invalid_count: wp.array[wp.int32],
):
    contact_status[contact] = status
    wp.atomic_add(invalid_count, 0, 1)
    if world >= 0 and world < world_status.shape[0]:
        wp.atomic_max(world_status, world, status)
    else:
        wp.atomic_max(global_status, 0, status)


@wp.kernel
def _check_contact_count(
    source_count: wp.array[wp.int32],
    capacity: int,
    global_status: wp.array[wp.int32],
    invalid_count: wp.array[wp.int32],
):
    count = source_count[0]
    if count < 0:
        global_status[0] = DEFORMABLE_CONTACT_STATUS_MALFORMED
        invalid_count[0] = 1
    elif count > capacity:
        global_status[0] = DEFORMABLE_CONTACT_STATUS_MALFORMED
        invalid_count[0] = count - capacity


@wp.kernel
def _adapt_soft_contacts(
    source_count: wp.array[wp.int32],
    source_indices: wp.array[wp.vec3i],
    source_barycentric: wp.array[wp.vec3],
    source_shape: wp.array[wp.int32],
    source_body_position: wp.array[wp.vec3],
    source_body_velocity: wp.array[wp.vec3],
    source_normal: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    cone_axis: wp.array[wp.vec3],
    cone_cosine: wp.array[float],
    filter_surface_contacts: bool,
    normal_cone_filtering_min_distance: float,
    particle_position: wp.array[wp.vec3],
    particle_velocity: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    newton_to_packed: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    inverse_weight: wp.array[float],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[float],
    body_pose: wp.array[wp.transform],
    body_velocity: wp.array[wp.spatial_vector],
    body_center_of_mass: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    stabilization_fraction: float,
    dead_zone: float,
    impact_velocity_threshold: float,
    recoverable_response: bool,
    friction: float,
    restitution: float,
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_shape: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    normal: wp.array[wp.vec3],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    gap: wp.array[float],
    bias: wp.array[wp.vec3],
    rigid_bias: wp.array[wp.vec3],
    contact_friction: wp.array[float],
    collider_velocity: wp.array[wp.vec3],
    surface_position_local: wp.array[wp.vec3],
    surface_velocity_local: wp.array[wp.vec3],
    contact_status: wp.array[wp.int32],
    particle_multiplicity: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    global_status: wp.array[wp.int32],
    invalid_count: wp.array[wp.int32],
):
    contact = wp.tid()
    for slot in range(4):
        particle_indices[contact, slot] = -1
        coefficients[contact, slot] = 0.0
    contact_world[contact] = -1
    contact_shape[contact] = -1
    contact_body[contact] = -1
    normal[contact] = wp.vec3(0.0)
    frame[contact] = wp.mat33f(0.0)
    body_jacobian[contact] = mat36f(0.0)
    gap[contact] = 0.0
    bias[contact] = wp.vec3(0.0)
    rigid_bias[contact] = wp.vec3(0.0)
    contact_friction[contact] = 0.0
    collider_velocity[contact] = wp.vec3(0.0)
    surface_position_local[contact] = wp.vec3(0.0)
    surface_velocity_local[contact] = wp.vec3(0.0)
    contact_status[contact] = DEFORMABLE_CONTACT_STATUS_UNUSED

    count = wp.min(source_count[0], source_indices.shape[0])
    if contact >= count:
        return

    source_particles = source_indices[contact]
    source_coefficients = source_barycentric[contact]
    shape = source_shape[contact]
    if (
        shape < 0
        or shape >= shape_body.shape[0]
        or shape >= shape_world.shape[0]
        or shape >= shape_margin.shape[0]
        or not _is_finite_vec3(source_body_position[contact])
        or not _is_finite_vec3(source_body_velocity[contact])
        or not _is_finite_vec3(source_normal[contact])
    ):
        _reject_contact(
            contact,
            -1,
            DEFORMABLE_CONTACT_STATUS_MALFORMED,
            contact_status,
            world_status,
            global_status,
            invalid_count,
        )
        return

    slot_count = int(0)
    coefficient_sum = float(0.0)
    found_padding = False
    malformed = False
    world = int(-1)
    feature_position = wp.vec3(0.0)
    feature_velocity = wp.vec3(0.0)
    prescribed_feature_velocity = wp.vec3(0.0)
    feature_radius = float(0.0)

    for slot in range(3):
        source_particle = source_particles[slot]
        coefficient = source_coefficients[slot]
        if source_particle < 0:
            found_padding = True
            malformed = malformed or source_particle != -1 or wp.abs(coefficient) > _COEFFICIENT_TOLERANCE
        else:
            malformed = (
                malformed
                or found_padding
                or source_particle >= newton_to_packed.shape[0]
                or not wp.isfinite(coefficient)
                or coefficient < -_COEFFICIENT_TOLERANCE
            )
            if not malformed:
                for previous_slot in range(3):
                    if previous_slot < slot:
                        malformed = malformed or source_particles[previous_slot] == source_particle
            if not malformed:
                packed_particle = newton_to_packed[source_particle]
                particle_world = packed_world[packed_particle]
                if world < 0:
                    world = particle_world
                elif particle_world != world:
                    _reject_contact(
                        contact,
                        world,
                        DEFORMABLE_CONTACT_STATUS_CROSS_WORLD,
                        contact_status,
                        world_status,
                        global_status,
                        invalid_count,
                    )
                    return
                particle_indices[contact, slot] = packed_particle
                coefficients[contact, slot] = coefficient
                feature_position += coefficient * particle_position[source_particle]
                if (particle_flags[source_particle] & _PARTICLE_FLAG_ACTIVE) != 0:
                    feature_velocity += coefficient * particle_velocity[source_particle]
                    if inverse_weight[packed_particle] <= 0.0:
                        prescribed_feature_velocity += coefficient * particle_velocity[source_particle]
                feature_radius = wp.max(feature_radius, particle_radius[source_particle])
                coefficient_sum += coefficient
                slot_count += 1

    if (
        malformed
        or slot_count == 0
        or world < 0
        or wp.abs(coefficient_sum - 1.0) > _COEFFICIENT_TOLERANCE
        or not _is_finite_vec3(feature_position)
        or not _is_finite_vec3(feature_velocity)
        or not wp.isfinite(feature_radius)
        or feature_radius < 0.0
    ):
        _reject_contact(
            contact,
            world,
            DEFORMABLE_CONTACT_STATUS_MALFORMED,
            contact_status,
            world_status,
            global_status,
            invalid_count,
        )
        return

    normal_length = wp.length(source_normal[contact])
    if not wp.isfinite(normal_length) or normal_length <= _NORMAL_EPSILON:
        _reject_contact(
            contact,
            world,
            DEFORMABLE_CONTACT_STATUS_MALFORMED,
            contact_status,
            world_status,
            global_status,
            invalid_count,
        )
        return
    contact_normal = source_normal[contact] / normal_length
    collider_world = shape_world[shape]
    if collider_world >= 0 and collider_world != world:
        _reject_contact(
            contact,
            world,
            DEFORMABLE_CONTACT_STATUS_CROSS_WORLD,
            contact_status,
            world_status,
            global_status,
            invalid_count,
        )
        return

    body = shape_body[shape]
    surface_position = source_body_position[contact]
    surface_velocity = source_body_velocity[contact]
    prescribed_surface_velocity = surface_velocity
    center_of_mass_world = wp.vec3(0.0)
    dynamic_body = int(-1)
    if body >= 0:
        if (
            body >= body_pose.shape[0]
            or body >= body_velocity.shape[0]
            or body >= body_center_of_mass.shape[0]
            or body >= body_flags.shape[0]
            or body >= body_world.shape[0]
        ):
            _reject_contact(
                contact,
                world,
                DEFORMABLE_CONTACT_STATUS_MALFORMED,
                contact_status,
                world_status,
                global_status,
                invalid_count,
            )
            return
        collider_body_world = body_world[body]
        if collider_body_world >= 0 and collider_body_world != world:
            _reject_contact(
                contact,
                world,
                DEFORMABLE_CONTACT_STATUS_CROSS_WORLD,
                contact_status,
                world_status,
                global_status,
                invalid_count,
            )
            return

        pose = body_pose[body]
        surface_position = wp.transform_point(pose, surface_position)
        center_of_mass_world = wp.transform_point(pose, body_center_of_mass[body])
        prescribed_surface_velocity = wp.transform_vector(pose, source_body_velocity[contact])
        spatial_velocity = body_velocity[body]
        linear_velocity = wp.spatial_top(spatial_velocity)
        angular_velocity = wp.spatial_bottom(spatial_velocity)
        surface_velocity = (
            linear_velocity
            + wp.cross(angular_velocity, surface_position - center_of_mass_world)
            + prescribed_surface_velocity
        )
        if (body_flags[body] & _BODY_FLAG_KINEMATIC) == 0:
            dynamic_body = body

    if (
        not _is_finite_vec3(surface_position)
        or not _is_finite_vec3(surface_velocity)
        or not wp.isfinite(shape_margin[shape])
        or shape_margin[shape] < 0.0
        or not wp.isfinite(friction)
        or friction < 0.0
        or not wp.isfinite(restitution)
        or restitution < 0.0
    ):
        _reject_contact(
            contact,
            world,
            DEFORMABLE_CONTACT_STATUS_MALFORMED,
            contact_status,
            world_status,
            global_status,
            invalid_count,
        )
        return

    surface_separation = wp.length(feature_position - surface_position)
    # At nearly coincident closest points, small positional errors can rotate
    # the derived normal arbitrarily, so retain the contact conservatively.
    if (
        filter_surface_contacts
        and surface_separation > normal_cone_filtering_min_distance
        and not soft_surface_contact_normal_cone_contains(
            source_particles,
            source_coefficients,
            -contact_normal,
            particle_position,
            edge_indices,
            adjacency,
            cone_axis,
            cone_cosine,
        )
    ):
        return

    contact_gap = wp.dot(contact_normal, feature_position - surface_position) - feature_radius - shape_margin[shape]
    previous_normal_velocity = wp.dot(contact_normal, feature_velocity - surface_velocity)
    dt = time_step[world]
    velocity_target = compute_contact_velocity_target(
        contact_gap,
        previous_normal_velocity,
        restitution,
        dt,
        stabilization_fraction,
        dead_zone,
        impact_velocity_threshold,
        recoverable_response,
    )
    contact_bias = prescribed_feature_velocity - surface_velocity - velocity_target * contact_normal
    contact_frame = make_contact_frame_znorm(contact_normal)
    contact_frame_transpose = wp.transpose(contact_frame)
    contact_rigid_bias = contact_frame_transpose @ contact_bias
    contact_body_jacobian = mat36f(0.0)
    if dynamic_body >= 0:
        contact_rigid_bias = contact_frame_transpose @ (
            prescribed_feature_velocity - prescribed_surface_velocity - velocity_target * contact_normal
        )
        angular_jacobian = contact_frame_transpose @ wp.skew(surface_position - center_of_mass_world)
        for row in range(3):
            for col in range(3):
                contact_body_jacobian[row, col] = -contact_frame_transpose[row, col]
                contact_body_jacobian[row, 3 + col] = angular_jacobian[row, col]
    if not wp.isfinite(contact_gap) or not _is_finite_vec3(contact_bias) or not _is_finite_vec3(contact_rigid_bias):
        _reject_contact(
            contact,
            world,
            DEFORMABLE_CONTACT_STATUS_MALFORMED,
            contact_status,
            world_status,
            global_status,
            invalid_count,
        )
        return

    contact_world[contact] = world
    contact_shape[contact] = shape
    contact_body[contact] = dynamic_body
    normal[contact] = contact_normal
    frame[contact] = contact_frame
    body_jacobian[contact] = contact_body_jacobian
    gap[contact] = contact_gap
    bias[contact] = contact_bias
    rigid_bias[contact] = contact_rigid_bias
    contact_friction[contact] = friction
    collider_velocity[contact] = surface_velocity
    surface_position_local[contact] = source_body_position[contact]
    surface_velocity_local[contact] = source_body_velocity[contact]
    contact_status[contact] = DEFORMABLE_CONTACT_STATUS_VALID
    wp.atomic_max(world_status, world, DEFORMABLE_CONTACT_STATUS_VALID)
    wp.atomic_add(world_contact_count, world, 1)
    for slot in range(3):
        packed_particle = particle_indices[contact, slot]
        if (
            packed_particle >= 0
            and coefficients[contact, slot] > _COEFFICIENT_TOLERANCE
            and inverse_weight[packed_particle] > 0.0
        ):
            wp.atomic_add(particle_multiplicity, packed_particle, 1)


@wp.kernel
def _capture_frozen_deformable_contact_geometry(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_shape: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    contact_gap: wp.array[float],
    contact_status: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_pose: wp.array[wp.transform],
    particle_position: wp.array[wp.vec3],
    surface_position_local: wp.array[wp.vec3],
    normal_local: wp.array[wp.vec3],
    gap_offset: wp.array[float],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return

    feature_position = wp.vec3(0.0)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            feature_position += coefficients[contact, slot] * particle_position[particle]

    shape = contact_shape[contact]
    surface_position = wp.vec3(0.0)
    normal = contact_normal[contact]
    normal_local[contact] = normal
    if shape >= 0:
        body = shape_body[shape]
        surface_position = surface_position_local[contact]
        if body >= 0:
            inverse_pose = wp.transform_inverse(body_pose[body])
            normal_local[contact] = wp.transform_vector(inverse_pose, normal)
            surface_position = wp.transform_point(body_pose[body], surface_position)
    gap_offset[contact] = contact_gap[contact] - wp.dot(normal, feature_position - surface_position)


@wp.kernel
def _refresh_frozen_deformable_contact_geometry(
    contact_world: wp.array[wp.int32],
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_shape: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    packed_to_newton: wp.array[wp.int32],
    particle_position: wp.array[wp.vec3],
    particle_velocity_begin: wp.array[wp.vec3],
    body_pose: wp.array[wp.transform],
    body_velocity_begin: wp.array[vec6f],
    body_center_of_mass: wp.array[wp.vec3],
    body_pose_is_center_of_mass: bool,
    surface_position_local: wp.array[wp.vec3],
    surface_velocity_local: wp.array[wp.vec3],
    normal_local: wp.array[wp.vec3],
    gap_offset: wp.array[float],
    time_step: wp.array[wp.float32],
    stabilization_fraction: float,
    dead_zone: float,
    impact_velocity_threshold: float,
    recoverable_response: bool,
    restitution: float,
    contact_normal: wp.array[wp.vec3],
    contact_frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    contact_gap: wp.array[float],
    bias: wp.array[wp.vec3],
    rigid_bias: wp.array[wp.vec3],
    collider_velocity: wp.array[wp.vec3],
    rigid_reaction: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return

    feature_position = wp.vec3(0.0)
    feature_velocity_begin = wp.vec3(0.0)
    prescribed_feature_velocity = wp.vec3(0.0)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            coefficient = coefficients[contact, slot]
            feature_position += coefficient * particle_position[particle]
            source_particle = packed_to_newton[particle]
            if (particle_flags[source_particle] & _PARTICLE_FLAG_ACTIVE) != 0:
                velocity = particle_velocity_begin[particle]
                feature_velocity_begin += coefficient * velocity
                if particle_mass[source_particle] <= 0.0:
                    prescribed_feature_velocity += coefficient * velocity

    shape = contact_shape[contact]
    surface_position = wp.vec3(0.0)
    surface_velocity_begin = wp.vec3(0.0)
    prescribed_surface_velocity = wp.vec3(0.0)
    normal = normal_local[contact]
    dynamic_body = contact_body[contact]
    if shape >= 0:
        body = shape_body[shape]
        surface_position = surface_position_local[contact]
        prescribed_surface_velocity = surface_velocity_local[contact]
        if body >= 0:
            pose = body_pose[body]
            if body_pose_is_center_of_mass:
                surface_position = wp.transform_point(pose, surface_position - body_center_of_mass[body])
                center_of_mass = wp.transform_get_translation(pose)
            else:
                surface_position = wp.transform_point(pose, surface_position)
                center_of_mass = wp.transform_point(pose, body_center_of_mass[body])
            normal = wp.transform_vector(pose, normal)
            prescribed_surface_velocity = wp.transform_vector(pose, prescribed_surface_velocity)
            twist = body_velocity_begin[body]
            surface_velocity_begin = (
                wp.vec3(twist[0], twist[1], twist[2])
                + wp.cross(wp.vec3(twist[3], twist[4], twist[5]), surface_position - center_of_mass)
                + prescribed_surface_velocity
            )
        else:
            surface_velocity_begin = prescribed_surface_velocity
    else:
        separation = wp.length(feature_position)
        if separation > _NORMAL_EPSILON:
            normal = feature_position / separation

    normal = wp.normalize(normal)
    gap = wp.dot(normal, feature_position - surface_position) + gap_offset[contact]
    previous_normal_velocity = wp.dot(normal, feature_velocity_begin - surface_velocity_begin)
    velocity_target = compute_contact_velocity_target(
        gap,
        previous_normal_velocity,
        restitution,
        time_step[contact_world[contact]],
        stabilization_fraction,
        dead_zone,
        impact_velocity_threshold,
        recoverable_response,
    )
    contact_bias = prescribed_feature_velocity - surface_velocity_begin - velocity_target * normal
    frame = make_contact_frame_znorm(normal)
    frame_transpose = wp.transpose(frame)
    local_rigid_bias = frame_transpose @ contact_bias
    jacobian = mat36f(0.0)
    if dynamic_body >= 0:
        if body_pose_is_center_of_mass:
            center_of_mass = wp.transform_get_translation(body_pose[dynamic_body])
        else:
            center_of_mass = wp.transform_point(body_pose[dynamic_body], body_center_of_mass[dynamic_body])
        local_rigid_bias = frame_transpose @ (
            prescribed_feature_velocity - prescribed_surface_velocity - velocity_target * normal
        )
        angular_jacobian = frame_transpose @ wp.skew(surface_position - center_of_mass)
        for row in range(3):
            for col in range(3):
                jacobian[row, col] = -frame_transpose[row, col]
                jacobian[row, 3 + col] = angular_jacobian[row, col]

    old_frame = contact_frame[contact]
    rigid_reaction[contact] = frame_transpose @ (old_frame @ rigid_reaction[contact])
    contact_normal[contact] = normal
    contact_frame[contact] = frame
    body_jacobian[contact] = jacobian
    contact_gap[contact] = gap
    bias[contact] = contact_bias
    rigid_bias[contact] = local_rigid_bias
    collider_velocity[contact] = surface_velocity_begin


@wp.kernel
def _finalize_split_inverse_weight(
    inverse_weight: wp.array[float],
    multiplicity: wp.array[wp.int32],
    split_inverse_weight: wp.array[float],
):
    particle = wp.tid()
    value = float(multiplicity[particle]) * inverse_weight[particle]
    if not wp.isfinite(value) or value < 0.0:
        value = 0.0
    split_inverse_weight[particle] = value


@wp.kernel
def _finalize_contact_delassus(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    split_inverse_weight: wp.array[float],
    contact_status: wp.array[wp.int32],
    delassus: wp.array[float],
    inverse_delassus: wp.array[float],
    world_contact_count: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    global_status: wp.array[wp.int32],
    invalid_count: wp.array[wp.int32],
):
    contact = wp.tid()
    delassus[contact] = 0.0
    inverse_delassus[contact] = 0.0
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return

    value = float(0.0)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        coefficient = coefficients[contact, slot]
        if particle >= 0:
            value += coefficient * coefficient * split_inverse_weight[particle]

    if not wp.isfinite(value) or (value <= 0.0 and contact_body[contact] < 0):
        world = contact_world[contact]
        if world >= 0:
            wp.atomic_add(world_contact_count, world, -1)
        _reject_contact(
            contact,
            world,
            DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS,
            contact_status,
            world_status,
            global_status,
            invalid_count,
        )
        return
    delassus[contact] = value
    if value > 0.0:
        inverse_delassus[contact] = 1.0 / value


@wp.kernel
def _scatter_contact_order(
    contact_world: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    world_cursor: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return
    world = contact_world[contact]
    if world >= 0:
        ordered = wp.atomic_add(world_cursor, world, 1)
        contact_order[ordered] = contact


@wp.kernel
def _prepare_gauss_seidel_contacts(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    body_jacobian: wp.array[mat36f],
    rigid_bias: wp.array[wp.vec3],
    friction: wp.array[float],
    particle_inverse_weight: wp.array[float],
    body_inverse_weight: wp.array[mat66f],
    include_rigid: bool,
    contact_status: wp.array[wp.int32],
    scalar_delassus: wp.array[float],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
    invalid_count: wp.array[wp.int32],
):
    contact = wp.tid()
    scalar_delassus[contact] = 0.0
    delassus[contact] = wp.mat33f(0.0)
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return

    particle_value = float(0.0)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            coefficient = coefficients[contact, slot]
            particle_value += coefficient * coefficient * particle_inverse_weight[particle]
    value = particle_value * wp.identity(3, dtype=wp.float32)
    body = contact_body[contact]
    if include_rigid and body >= 0:
        jacobian = body_jacobian[contact]
        value += jacobian @ body_inverse_weight[body] @ wp.transpose(jacobian)

    data = prepare_contact_coulomb_delassus(value, rigid_bias[contact], friction[contact])
    scalar_delassus[contact] = particle_value
    delassus[contact] = data.delassus
    if data.status == PROJECTION_STATUS_INVALID or (not include_rigid and particle_value <= 0.0):
        contact_status[contact] = DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS
        wp.atomic_max(world_status, contact_world[contact], DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS)
        wp.atomic_add(invalid_count, 0, 1)


@wp.func
def project_deformable_contact_coulomb(
    free_velocity: wp.vec3,
    normal: wp.vec3,
    delassus: float,
    friction: float,
) -> wp.vec3:
    """Solve one isotropic Coulomb contact in world space."""
    normal_velocity = wp.dot(normal, free_velocity)
    if normal_velocity >= 0.0:
        return wp.vec3(0.0)

    tangent_velocity = free_velocity - normal_velocity * normal
    tangent_speed = wp.length(tangent_velocity)
    normal_reaction = -normal_velocity / delassus
    if tangent_speed <= -friction * normal_velocity:
        return -free_velocity / delassus
    if tangent_speed > 0.0:
        return normal_reaction * normal - friction * normal_reaction * tangent_velocity / tangent_speed
    return normal_reaction * normal


@wp.func
def _project_world_coulomb_cone(value: wp.vec3, normal: wp.vec3, friction: float) -> wp.vec3:
    normal_value = wp.dot(normal, value)
    tangent_value = value - normal_value * normal
    tangent_norm = wp.length(tangent_value)
    if friction * tangent_norm <= -normal_value:
        return wp.vec3(0.0)
    if tangent_norm <= friction * normal_value:
        return value

    projected_normal = (friction * tangent_norm + normal_value) / (friction * friction + 1.0)
    if tangent_norm > 0.0:
        return projected_normal * normal + friction * projected_normal * tangent_value / tangent_norm
    return projected_normal * normal


@wp.func
def compute_deformable_contact_residual(
    delassus: float,
    reaction: wp.vec3,
    velocity: wp.vec3,
    normal: wp.vec3,
    friction: float,
) -> wp.vec3:
    """Compute the rotationally invariant scaled contact natural-map residual."""
    scale = wp.sqrt(delassus)
    scaled_reaction = scale * reaction
    scaled_velocity = velocity / scale
    normal_velocity = wp.dot(normal, scaled_velocity)
    tangent_velocity = scaled_velocity - normal_velocity * normal
    modified_velocity = scaled_velocity + friction * wp.length(tangent_velocity) * normal
    projected = _project_world_coulomb_cone(scaled_reaction - modified_velocity, normal, friction)
    return scaled_reaction - projected


@wp.kernel
def _warm_start_contacts(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    global_status: wp.array[wp.int32],
    inverse_weight: wp.array[float],
    reaction: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        global_status[0] != DEFORMABLE_CONTACT_STATUS_UNUSED
        or contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or contact_body[contact] >= 0
        or world < 0
        or world_active[world] == 0
        or world_status[world] != DEFORMABLE_CONTACT_STATUS_VALID
    ):
        return

    contact_reaction = reaction[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            correction = inverse_weight[particle] * coefficients[contact, slot] * contact_reaction
            wp.atomic_add(particle_delta, particle, correction)


@wp.kernel
def _project_contacts(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    normal: wp.array[wp.vec3],
    bias: wp.array[wp.vec3],
    friction: wp.array[float],
    delassus: wp.array[float],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    global_status: wp.array[wp.int32],
    inverse_weight: wp.array[float],
    projected_velocity: wp.array[wp.vec3],
    reaction: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        global_status[0] != DEFORMABLE_CONTACT_STATUS_UNUSED
        or contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or contact_body[contact] >= 0
        or world < 0
        or world_active[world] == 0
        or world_status[world] != DEFORMABLE_CONTACT_STATUS_VALID
    ):
        return

    particles = wp.vec4i(
        particle_indices[contact, 0],
        particle_indices[contact, 1],
        particle_indices[contact, 2],
        particle_indices[contact, 3],
    )
    weights = wp.vec4(
        coefficients[contact, 0],
        coefficients[contact, 1],
        coefficients[contact, 2],
        coefficients[contact, 3],
    )
    contact_velocity = bias[contact]
    for slot in range(4):
        particle = particles[slot]
        if particle >= 0:
            contact_velocity += weights[slot] * projected_velocity[particle]

    reaction_old = reaction[contact]
    free_velocity = contact_velocity - delassus[contact] * reaction_old
    reaction_new = project_deformable_contact_coulomb(
        free_velocity,
        normal[contact],
        delassus[contact],
        friction[contact],
    )
    reaction_delta = reaction_new - reaction_old
    if not _is_finite_vec3(reaction_new) or not _is_finite_vec3(reaction_delta):
        contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
        wp.atomic_max(world_status, world, DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE)
        return

    reaction[contact] = reaction_new
    for slot in range(4):
        particle = particles[slot]
        if particle >= 0:
            correction = inverse_weight[particle] * weights[slot] * reaction_delta
            if not _is_finite_vec3(correction):
                contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                wp.atomic_max(world_status, world, DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE)
                return
            wp.atomic_add(particle_delta, particle, correction)


@wp.kernel
def _apply_particle_delta(
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    particle_delta: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
):
    particle = wp.tid()
    if world_active[packed_world[particle]] != 0:
        projected_velocity[particle] += particle_delta[particle]


@wp.kernel
def _compute_contact_residuals(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    normal: wp.array[wp.vec3],
    bias: wp.array[wp.vec3],
    friction: wp.array[float],
    delassus: wp.array[float],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    projected_velocity: wp.array[wp.vec3],
    reaction: wp.array[wp.vec3],
    contact_velocity: wp.array[wp.vec3],
    contact_residual: wp.array[float],
    world_contact_residual: wp.array[float],
):
    contact = wp.tid()
    contact_velocity[contact] = wp.vec3(0.0)
    contact_residual[contact] = 0.0
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or contact_body[contact] >= 0
        or world < 0
        or world_active[world] == 0
    ):
        return

    velocity = bias[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            velocity += coefficients[contact, slot] * projected_velocity[particle]
    contact_velocity[contact] = velocity
    residual_vector = compute_deformable_contact_residual(
        delassus[contact],
        reaction[contact],
        velocity,
        normal[contact],
        friction[contact],
    )
    residual = wp.max(wp.abs(residual_vector[0]), wp.abs(residual_vector[1]))
    residual = wp.max(residual, wp.abs(residual_vector[2]))
    contact_residual[contact] = residual
    wp.atomic_max(world_contact_residual, world, residual)


@wp.func
def _is_finite_vec6(value: vec6f) -> bool:
    finite = True
    for index in range(6):
        finite = finite and wp.isfinite(value[index])
    return finite


@wp.kernel
def _accumulate_rigid_incidence(
    contact_body: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    body_constraint_count: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
    world_has_unilateral: wp.array[wp.bool],
):
    contact = wp.tid()
    body = contact_body[contact]
    if contact_status[contact] == DEFORMABLE_CONTACT_STATUS_VALID and body >= 0:
        wp.atomic_add(body_constraint_count, body, 1)
        wp.atomic_max(body_has_unilateral, body, 1)
        world_has_unilateral[contact_world[contact]] = True


@wp.kernel
def _prepare_rigid_contacts(
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    body_jacobian: wp.array[mat36f],
    rigid_bias: wp.array[wp.vec3],
    friction: wp.array[float],
    particle_delassus: wp.array[float],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    body_inverse_weight: wp.array[mat66f],
    contact_status: wp.array[wp.int32],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
    invalid_count: wp.array[wp.int32],
):
    contact = wp.tid()
    delassus[contact] = wp.mat33f(0.0)
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return

    value = particle_delassus[contact] * wp.identity(3, dtype=wp.float32)
    body = contact_body[contact]
    if body >= 0:
        multiplicity = wp.max(1, body_constraint_count[body] - static_body_constraint_count[body])
        split_inverse_weight = wp.float32(multiplicity) * body_inverse_weight[body]
        jacobian = body_jacobian[contact]
        value += jacobian @ split_inverse_weight @ wp.transpose(jacobian)

    data = prepare_contact_coulomb_delassus(
        value,
        rigid_bias[contact],
        friction[contact],
    )
    delassus[contact] = data.delassus
    if data.status == PROJECTION_STATUS_INVALID:
        contact_status[contact] = DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS
        wp.atomic_max(
            world_status,
            contact_world[contact],
            DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS,
        )
        wp.atomic_add(invalid_count, 0, 1)


@wp.kernel
def _merge_rigid_prepared_status(
    contact_world_status: wp.array[wp.int32],
    contact_global_status: wp.array[wp.int32],
    rigid_prepared_status: wp.array[wp.int32],
):
    world = wp.tid()
    if (
        contact_global_status[0] > DEFORMABLE_CONTACT_STATUS_VALID
        or contact_world_status[world] > DEFORMABLE_CONTACT_STATUS_VALID
    ):
        rigid_prepared_status[world] = 0


@wp.kernel
def _warm_start_rigid_contacts(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    particle_inverse_weight: wp.array[float],
    body_inverse_weight: wp.array[mat66f],
    apply_inverse_weight: wp.bool,
    reaction: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
    body_delta: wp.array[vec6f],
    projection_status: wp.array[wp.int32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or world < 0
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    impulse = reaction[contact]
    world_impulse = frame[contact] @ impulse
    if not _is_finite_vec3(world_impulse):
        projection_status[world] = 0
        return
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            particle_correction = particle_inverse_weight[particle] * coefficients[contact, slot] * world_impulse
            if not _is_finite_vec3(particle_correction):
                projection_status[world] = 0
                return
            wp.atomic_add(particle_delta, particle, particle_correction)

    body = contact_body[contact]
    if body >= 0:
        body_wrench = wp.transpose(body_jacobian[contact]) @ impulse
        if apply_inverse_weight:
            body_wrench = body_inverse_weight[body] @ body_wrench
        if not _is_finite_vec6(body_wrench):
            projection_status[world] = 0
            return
        wp.atomic_add(body_delta, body, body_wrench)


@wp.kernel
def _project_rigid_contacts(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    rigid_bias: wp.array[wp.vec3],
    friction: wp.array[float],
    delassus: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    particle_inverse_weight: wp.array[float],
    body_inverse_weight: wp.array[mat66f],
    projected_velocity: wp.array[wp.vec3],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
    body_delta: wp.array[vec6f],
    projection_status: wp.array[wp.int32],
    contact_world_status: wp.array[wp.int32],
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

    particles = wp.vec4i(
        particle_indices[contact, 0],
        particle_indices[contact, 1],
        particle_indices[contact, 2],
        particle_indices[contact, 3],
    )
    weights = wp.vec4(
        coefficients[contact, 0],
        coefficients[contact, 1],
        coefficients[contact, 2],
        coefficients[contact, 3],
    )
    contact_frame = frame[contact]
    velocity = rigid_bias[contact]
    for slot in range(4):
        particle = particles[slot]
        if particle >= 0:
            velocity += weights[slot] * (wp.transpose(contact_frame) @ projected_velocity[particle])
    body = contact_body[contact]
    if body >= 0:
        velocity += body_jacobian[contact] @ projected_twist[body]

    reaction_old = reaction[contact]
    contact_block = delassus[contact]
    free_velocity = velocity - contact_block @ reaction_old
    reaction_new = _solve_contact_coulomb_newton_normal_last(
        contact_block,
        free_velocity,
        friction[contact],
    )
    reaction_delta = reaction_new - reaction_old
    if not _is_finite_vec3(reaction_new) or not _is_finite_vec3(reaction_delta):
        contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
        wp.atomic_max(
            contact_world_status,
            world,
            DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
        )
        projection_status[world] = 0
        return

    world_impulse_delta = contact_frame @ reaction_delta
    for slot in range(4):
        particle = particles[slot]
        if particle >= 0:
            particle_correction = particle_inverse_weight[particle] * weights[slot] * world_impulse_delta
            if not _is_finite_vec3(particle_correction):
                contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                wp.atomic_max(
                    contact_world_status,
                    world,
                    DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
                )
                projection_status[world] = 0
                return
            wp.atomic_add(particle_delta, particle, particle_correction)

    if body >= 0:
        # Reload after the local solve instead of keeping 18 Jacobian scalars live across it.
        body_correction = body_inverse_weight[body] @ (wp.transpose(body_jacobian[contact]) @ reaction_delta)
        if not _is_finite_vec6(body_correction):
            contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            wp.atomic_max(
                contact_world_status,
                world,
                DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
            )
            projection_status[world] = 0
            return
        wp.atomic_add(body_delta, body, body_correction)
    reaction[contact] = reaction_new


@wp.kernel
def _project_rigid_contacts_instrumented(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    rigid_bias: wp.array[wp.vec3],
    friction: wp.array[float],
    delassus: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    particle_inverse_weight: wp.array[float],
    body_inverse_weight: wp.array[mat66f],
    projected_velocity: wp.array[wp.vec3],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
    body_delta: wp.array[vec6f],
    projection_status: wp.array[wp.int32],
    contact_world_status: wp.array[wp.int32],
    branch_histogram: wp.array2d[wp.int64],
    expansion_histogram: wp.array2d[wp.int64],
    root_histogram: wp.array2d[wp.int64],
    failure_counts: wp.array2d[wp.int64],
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
    velocity = rigid_bias[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            velocity += coefficients[contact, slot] * (wp.transpose(contact_frame) @ projected_velocity[particle])
    body = contact_body[contact]
    if body >= 0:
        velocity += body_jacobian[contact] @ projected_twist[body]

    reaction_old = reaction[contact]
    free_velocity = velocity - delassus[contact] @ reaction_old
    solve_result = _solve_contact_coulomb_newton_normal_last_instrumented(
        delassus[contact],
        free_velocity,
        friction[contact],
    )
    _record_coulomb_solve_statistics(
        solve_result,
        wp.int32(1),
        branch_histogram,
        expansion_histogram,
        root_histogram,
        failure_counts,
    )
    reaction_new = solve_result.reaction
    reaction_delta = reaction_new - reaction_old
    if not _is_finite_vec3(reaction_new) or not _is_finite_vec3(reaction_delta):
        contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
        wp.atomic_max(
            contact_world_status,
            world,
            DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
        )
        projection_status[world] = 0
        return

    world_impulse_delta = contact_frame @ reaction_delta
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            particle_correction = particle_inverse_weight[particle] * coefficients[contact, slot] * world_impulse_delta
            if not _is_finite_vec3(particle_correction):
                contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                wp.atomic_max(
                    contact_world_status,
                    world,
                    DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
                )
                projection_status[world] = 0
                return
            wp.atomic_add(particle_delta, particle, particle_correction)

    if body >= 0:
        body_correction = body_inverse_weight[body] @ (wp.transpose(body_jacobian[contact]) @ reaction_delta)
        if not _is_finite_vec6(body_correction):
            contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            wp.atomic_max(
                contact_world_status,
                world,
                DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
            )
            projection_status[world] = 0
            return
        wp.atomic_add(body_delta, body, body_correction)
    reaction[contact] = reaction_new


@wp.kernel
def _warm_start_contacts_sequential(
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    particle_inverse_weight: wp.array[float],
    body_inverse_weight: wp.array[mat66f],
    rigid_coordinates: bool,
    particle_reaction: wp.array[wp.vec3],
    rigid_reaction: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
    projected_twist: wp.array[vec6f],
    projection_status: wp.array[wp.int32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    if prepared_status[world] != PROJECTION_STATUS_VALID:
        projection_status[world] = prepared_status[world]
        return
    if projection_status[world] != PROJECTION_STATUS_VALID:
        return

    start = world_contact_offset[world]
    end = start + world_contact_count[world]
    for ordered in range(start, end):
        contact = contact_order[ordered]
        if contact < 0 or contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
            continue
        impulse = particle_reaction[contact]
        world_impulse = impulse
        if rigid_coordinates:
            impulse = rigid_reaction[contact]
            world_impulse = frame[contact] @ impulse
        if not _is_finite_vec3(world_impulse):
            projection_status[world] = PROJECTION_STATUS_INVALID
            return
        for slot in range(4):
            particle = particle_indices[contact, slot]
            if particle >= 0:
                particle_correction = particle_inverse_weight[particle] * coefficients[contact, slot] * world_impulse
                if not _is_finite_vec3(particle_correction):
                    projection_status[world] = PROJECTION_STATUS_INVALID
                    return
                projected_velocity[particle] += particle_correction
        body = contact_body[contact]
        if rigid_coordinates and body >= 0:
            body_correction = body_inverse_weight[body] @ (wp.transpose(body_jacobian[contact]) @ impulse)
            if not _is_finite_vec6(body_correction):
                projection_status[world] = PROJECTION_STATUS_INVALID
                return
            projected_twist[body] += body_correction


@wp.kernel
def _sweep_contacts_sequential(
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_body: wp.array[wp.int32],
    normal: wp.array[wp.vec3],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    bias: wp.array[wp.vec3],
    rigid_bias: wp.array[wp.vec3],
    friction: wp.array[float],
    scalar_delassus: wp.array[float],
    delassus: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    particle_inverse_weight: wp.array[float],
    body_inverse_weight: wp.array[mat66f],
    rigid_coordinates: bool,
    projected_velocity: wp.array[wp.vec3],
    projected_twist: wp.array[vec6f],
    particle_reaction: wp.array[wp.vec3],
    rigid_reaction: wp.array[wp.vec3],
    contact_velocity: wp.array[wp.vec3],
    projection_status: wp.array[wp.int32],
    contact_world_status: wp.array[wp.int32],
):
    world = wp.tid()
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return
    start = world_contact_offset[world]
    end = start + world_contact_count[world]
    for ordered in range(start, end):
        contact = contact_order[ordered]
        if contact < 0 or contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
            continue

        velocity = bias[contact]
        contact_frame = frame[contact]
        if rigid_coordinates:
            velocity = rigid_bias[contact]
        for slot in range(4):
            particle = particle_indices[contact, slot]
            if particle >= 0:
                particle_velocity = projected_velocity[particle]
                if rigid_coordinates:
                    particle_velocity = wp.transpose(contact_frame) @ particle_velocity
                velocity += coefficients[contact, slot] * particle_velocity
        body = contact_body[contact]
        if rigid_coordinates and body >= 0:
            velocity += body_jacobian[contact] @ projected_twist[body]

        reaction_old = particle_reaction[contact]
        reaction_new = wp.vec3(0.0)
        if rigid_coordinates:
            reaction_old = rigid_reaction[contact]
            free_velocity = velocity - delassus[contact] @ reaction_old
            reaction_new = _solve_contact_coulomb_newton_normal_last(
                delassus[contact],
                free_velocity,
                friction[contact],
            )
        else:
            value = scalar_delassus[contact]
            free_velocity = velocity - value * reaction_old
            reaction_new = project_deformable_contact_coulomb(
                free_velocity,
                normal[contact],
                value,
                friction[contact],
            )
        reaction_delta = reaction_new - reaction_old
        world_impulse_delta = reaction_delta
        if rigid_coordinates:
            world_impulse_delta = contact_frame @ reaction_delta
        if not _is_finite_vec3(reaction_new) or not _is_finite_vec3(world_impulse_delta):
            contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            contact_world_status[world] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            projection_status[world] = PROJECTION_STATUS_INVALID
            return

        for slot in range(4):
            particle = particle_indices[contact, slot]
            if particle >= 0:
                correction = particle_inverse_weight[particle] * coefficients[contact, slot] * world_impulse_delta
                if not _is_finite_vec3(correction):
                    contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                    contact_world_status[world] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                    projection_status[world] = PROJECTION_STATUS_INVALID
                    return
        body_correction = vec6f(0.0)
        if rigid_coordinates and body >= 0:
            body_correction = body_inverse_weight[body] @ (wp.transpose(body_jacobian[contact]) @ reaction_delta)
            if not _is_finite_vec6(body_correction):
                contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                contact_world_status[world] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                projection_status[world] = PROJECTION_STATUS_INVALID
                return

        if rigid_coordinates:
            rigid_reaction[contact] = reaction_new
            contact_velocity[contact] = contact_frame @ velocity
        else:
            particle_reaction[contact] = reaction_new
            contact_velocity[contact] = velocity
        for slot in range(4):
            particle = particle_indices[contact, slot]
            if particle >= 0:
                projected_velocity[particle] += (
                    particle_inverse_weight[particle] * coefficients[contact, slot] * world_impulse_delta
                )
        if rigid_coordinates and body >= 0:
            projected_twist[body] += body_correction


@wp.kernel
def _initialize_contacts_apgd(
    launch_dim: int,
    world_count: int,
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    normal: wp.array[wp.vec3],
    friction: wp.array[float],
    world_active: wp.array[wp.bool],
    rigid_coordinates: bool,
    particle_reaction: wp.array[wp.vec3],
    rigid_reaction: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
):
    lane = wp.tid()
    total = world_contact_offset[world_count - 1] + world_contact_count[world_count - 1]
    for ordered in range(lane, total, launch_dim):
        contact = contact_order[ordered]
        world = contact_world[contact]
        if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID or world < 0 or not world_active[world]:
            continue
        if rigid_coordinates:
            value = project_contact_coulomb_cone_orthogonal(rigid_reaction[contact], friction[contact])
            rigid_reaction[contact] = value
        else:
            value = particle_reaction[contact]
            value = _project_world_coulomb_cone(value, normal[contact], friction[contact])
            particle_reaction[contact] = value
        trial[contact] = value


@wp.func
def _scatter_contact_apgd(
    contact: int,
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    particle_inverse_weight: wp.array[float],
    body_inverse_weight: wp.array[mat66f],
    rigid_coordinates: bool,
    use_trial: bool,
    particle_reaction: wp.array[wp.vec3],
    rigid_reaction: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
    body_delta: wp.array[vec6f],
):
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or world < 0
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    if use_trial:
        impulse = trial[contact]
    elif rigid_coordinates:
        impulse = rigid_reaction[contact]
    else:
        impulse = particle_reaction[contact]
    world_impulse = impulse
    if rigid_coordinates:
        world_impulse = frame[contact] @ impulse
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            particle_correction = particle_inverse_weight[particle] * coefficients[contact, slot] * world_impulse
            wp.atomic_add(particle_delta, particle, particle_correction)
    body = contact_body[contact]
    if rigid_coordinates and body >= 0:
        body_correction = body_inverse_weight[body] @ (wp.transpose(body_jacobian[contact]) @ impulse)
        wp.atomic_add(body_delta, body, body_correction)


@wp.kernel
def _scatter_contacts_apgd(
    launch_dim: int,
    world_count: int,
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    particle_inverse_weight: wp.array[float],
    body_inverse_weight: wp.array[mat66f],
    rigid_coordinates: bool,
    use_trial: bool,
    particle_reaction: wp.array[wp.vec3],
    rigid_reaction: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
    body_delta: wp.array[vec6f],
):
    lane = wp.tid()
    total = world_contact_offset[world_count - 1] + world_contact_count[world_count - 1]
    for ordered in range(lane, total, launch_dim):
        contact = contact_order[ordered]
        _scatter_contact_apgd(
            contact,
            particle_indices,
            coefficients,
            contact_world,
            contact_body,
            frame,
            body_jacobian,
            contact_status,
            world_active,
            projection_status,
            particle_inverse_weight,
            body_inverse_weight,
            rigid_coordinates,
            use_trial,
            particle_reaction,
            rigid_reaction,
            trial,
            particle_delta,
            body_delta,
        )


@wp.kernel
def _project_contacts_apgd(
    launch_dim: int,
    world_count: int,
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    normal: wp.array[wp.vec3],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    bias: wp.array[wp.vec3],
    rigid_bias: wp.array[wp.vec3],
    friction: wp.array[float],
    scalar_delassus: wp.array[float],
    rigid_delassus: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    rigid_coordinates: bool,
    projected_velocity: wp.array[wp.vec3],
    projected_twist: wp.array[vec6f],
    particle_reaction: wp.array[wp.vec3],
    rigid_reaction: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    next_reaction: wp.array[wp.vec3],
    contact_velocity: wp.array[wp.vec3],
    restart_dot: wp.array[float],
    projection_status: wp.array[wp.int32],
    contact_world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    total = world_contact_offset[world_count - 1] + world_contact_count[world_count - 1]
    for ordered in range(lane, total, launch_dim):
        contact = contact_order[ordered]
        world = contact_world[contact]
        if (
            contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
            or world < 0
            or not world_active[world]
            or projection_status[world] != PROJECTION_STATUS_VALID
        ):
            continue
        contact_frame = frame[contact]
        if rigid_coordinates:
            velocity = rigid_bias[contact]
        else:
            velocity = bias[contact]
        for slot in range(4):
            particle = particle_indices[contact, slot]
            if particle >= 0:
                particle_velocity = projected_velocity[particle]
                if rigid_coordinates:
                    particle_velocity = wp.transpose(contact_frame) @ particle_velocity
                velocity += coefficients[contact, slot] * particle_velocity
        body = contact_body[contact]
        if rigid_coordinates and body >= 0:
            velocity += body_jacobian[contact] @ projected_twist[body]

        corrected = velocity
        if rigid_coordinates:
            contact_friction = friction[contact]
            tangent_speed = wp.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1])
            corrected[2] += contact_friction * tangent_speed
        else:
            contact_normal = normal[contact]
            contact_friction = friction[contact]
            normal_velocity = wp.dot(contact_normal, velocity)
            tangent_velocity = velocity - normal_velocity * contact_normal
            corrected += contact_friction * wp.length(tangent_velocity) * contact_normal
        if not _is_finite_vec3(corrected):
            contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            contact_world_status[world] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            projection_status[world] = PROJECTION_STATUS_INVALID
            continue

        local_trial = trial[contact]
        next_value = wp.vec3(0.0)
        if rigid_coordinates:
            contact_block = rigid_delassus[contact]
            free_velocity = velocity - contact_block @ local_trial
            next_value = _solve_contact_coulomb_newton_normal_last(
                contact_block,
                free_velocity,
                contact_friction,
            )
            current = rigid_reaction[contact]
        else:
            local_delassus = scalar_delassus[contact]
            current = particle_reaction[contact]
            if not wp.isfinite(local_delassus) or local_delassus <= 0.0:
                contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                contact_world_status[world] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                projection_status[world] = PROJECTION_STATUS_INVALID
                continue
            next_value = _project_world_coulomb_cone(
                local_trial - corrected / local_delassus,
                contact_normal,
                contact_friction,
            )
        if not _is_finite_vec3(next_value):
            contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            contact_world_status[world] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
            projection_status[world] = PROJECTION_STATUS_INVALID
            continue
        next_reaction[contact] = next_value
        if rigid_coordinates:
            contact_velocity[contact] = contact_frame @ velocity
        else:
            contact_velocity[contact] = velocity
        wp.atomic_add(restart_dot, world, wp.dot(next_value - current, -corrected))


@wp.kernel
def _extrapolate_contacts_apgd(
    launch_dim: int,
    world_count: int,
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    beta: wp.array[float],
    projection_status: wp.array[wp.int32],
    rigid_coordinates: bool,
    next_reaction: wp.array[wp.vec3],
    particle_reaction: wp.array[wp.vec3],
    rigid_reaction: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
):
    lane = wp.tid()
    total = world_contact_offset[world_count - 1] + world_contact_count[world_count - 1]
    for ordered in range(lane, total, launch_dim):
        contact = contact_order[ordered]
        world = contact_world[contact]
        if (
            contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
            or world < 0
            or not world_active[world]
            or projection_status[world] != PROJECTION_STATUS_VALID
        ):
            continue
        old = particle_reaction[contact]
        if rigid_coordinates:
            old = rigid_reaction[contact]
        value = next_reaction[contact]
        if rigid_coordinates:
            rigid_reaction[contact] = value
        else:
            particle_reaction[contact] = value
        trial[contact] = value + beta[world] * (value - old)


@wp.kernel
def _apply_rigid_particle_delta(
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    particle_delta: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
):
    particle = wp.tid()
    world = packed_world[particle]
    if world_active[world] and projection_status[world] == PROJECTION_STATUS_VALID:
        projected_velocity[particle] += particle_delta[particle]
    particle_delta[particle] = wp.vec3(0.0)


@wp.kernel
def _reconstruct_apgd_particle_velocity(
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    baseline_velocity: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
):
    particle = wp.tid()
    world = packed_world[particle]
    value = baseline_velocity[particle]
    if world_active[world] and projection_status[world] == PROJECTION_STATUS_VALID:
        value += particle_delta[particle]
    projected_velocity[particle] = value
    particle_delta[particle] = wp.vec3(0.0)


@wp.kernel
def _compute_rigid_contact_residuals(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    rigid_bias: wp.array[wp.vec3],
    friction: wp.array[float],
    delassus: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    projected_velocity: wp.array[wp.vec3],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.vec3],
    contact_velocity: wp.array[wp.vec3],
    contact_residual: wp.array[float],
    world_contact_residual: wp.array[float],
):
    contact = wp.tid()
    contact_velocity[contact] = wp.vec3(0.0)
    contact_residual[contact] = 0.0
    world = contact_world[contact]
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID or world < 0 or world_active[world] == 0:
        return

    contact_frame = frame[contact]
    velocity = rigid_bias[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            velocity += coefficients[contact, slot] * (wp.transpose(contact_frame) @ projected_velocity[particle])
    body = contact_body[contact]
    if body >= 0:
        velocity += body_jacobian[contact] @ projected_twist[body]
    contact_velocity[contact] = contact_frame @ velocity

    residual_vector = compute_contact_scaled_alart_curnier_residual(
        convert_contact_matrix_normal_last_to_first(delassus[contact]),
        convert_contact_vector_normal_last_to_first(reaction[contact]),
        convert_contact_vector_normal_last_to_first(velocity),
        friction[contact],
    )
    residual = wp.max(wp.abs(residual_vector[0]), wp.abs(residual_vector[1]))
    residual = wp.max(residual, wp.abs(residual_vector[2]))
    contact_residual[contact] = residual
    wp.atomic_max(world_contact_residual, world, residual)


@wp.kernel
def _compute_consensus_residuals(
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    global_velocity: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
    global_velocity_previous: wp.array[wp.vec3],
    projected_velocity_previous: wp.array[wp.vec3],
    time_step: wp.array[wp.float32],
    world_consensus_residual: wp.array[float],
    world_iterate_residual: wp.array[float],
    world_displacement_residual: wp.array[float],
):
    particle = wp.tid()
    world = packed_world[particle]
    if world_active[world] == 0:
        return

    consensus_delta = global_velocity[particle] - projected_velocity[particle]
    iterate_delta = projected_velocity[particle] - projected_velocity_previous[particle]
    global_delta = global_velocity[particle] - global_velocity_previous[particle]
    consensus = wp.max(wp.abs(consensus_delta[0]), wp.abs(consensus_delta[1]))
    consensus = wp.max(consensus, wp.abs(consensus_delta[2]))
    iterate = wp.max(wp.abs(iterate_delta[0]), wp.abs(iterate_delta[1]))
    iterate = wp.max(iterate, wp.abs(iterate_delta[2]))
    global_change = wp.max(wp.abs(global_delta[0]), wp.abs(global_delta[1]))
    global_change = wp.max(global_change, wp.abs(global_delta[2]))
    wp.atomic_max(world_consensus_residual, world, consensus)
    wp.atomic_max(world_iterate_residual, world, iterate)
    wp.atomic_max(world_displacement_residual, world, time_step[world] * (global_change + consensus))


class DeformableContactSystem:
    """Own packed soft contacts and their projection buffers."""

    def __init__(
        self,
        model: Model,
        cloth_system: DeformableClothSystem,
        contact_capacity: int,
        self_contact_capacity: int = 0,
        stabilization_fraction: float = 0.01,
        dead_zone: float = 1.0e-6,
        impact_velocity_threshold: float = 1.0e-3,
        recoverable_response: bool = False,
        enable_rigid_normal_cone_filtering: bool = False,
        normal_cone_filtering_min_distance: float = 1.0e-4,
    ):
        """Allocate one fixed-capacity deformable contact path.

        Args:
            model: Newton model shared with ``cloth_system``.
            cloth_system: Packed frozen cloth system supplying scalar weights.
            contact_capacity: Maximum number of rigid-soft contact records.
            self_contact_capacity: Maximum number of cloth self-contact candidate records.
            stabilization_fraction: Penetration recovery fraction.
            dead_zone: Symmetric contact distance dead zone [m].
            impact_velocity_threshold: Minimum approaching impact speed [m/s].
            recoverable_response: Whether to permit restitution-recoverable overlap.
            enable_rigid_normal_cone_filtering: Whether to prune rigid-soft contacts using soft normal cones.
            normal_cone_filtering_min_distance: Separation below which normal-cone filtering is bypassed [m].
        """
        if cloth_system.model is not model:
            raise ValueError("LOX deformable contacts require the same model as the cloth system.")
        if not isinstance(contact_capacity, int) or isinstance(contact_capacity, bool) or contact_capacity < 0:
            raise ValueError("LOX rigid-soft contact capacity must be a non-negative integer.")
        if (
            not isinstance(self_contact_capacity, int)
            or isinstance(self_contact_capacity, bool)
            or self_contact_capacity < 0
        ):
            raise ValueError("LOX deformable self-contact capacity must be a non-negative integer.")
        total_contact_capacity = contact_capacity + self_contact_capacity
        if total_contact_capacity < 1:
            raise ValueError("LOX deformable contact capacity must be a positive integer.")
        if not np.isfinite(stabilization_fraction) or not 0.0 <= stabilization_fraction <= 1.0:
            raise ValueError("LOX deformable contact stabilization fraction must be in [0, 1].")
        if not np.isfinite(dead_zone) or dead_zone < 0.0:
            raise ValueError("LOX deformable contact dead zone must be finite and non-negative.")
        if not np.isfinite(impact_velocity_threshold) or impact_velocity_threshold < 0.0:
            raise ValueError("LOX deformable impact velocity threshold must be finite and non-negative.")
        if not isinstance(recoverable_response, bool):
            raise ValueError("LOX deformable recoverable_response must be a boolean.")
        if not isinstance(enable_rigid_normal_cone_filtering, bool):
            raise ValueError("LOX rigid-deformable normal-cone filtering flag must be a boolean.")
        if not np.isfinite(normal_cone_filtering_min_distance) or normal_cone_filtering_min_distance < 0.0:
            raise ValueError("LOX normal-cone filtering minimum distance must be finite and non-negative.")
        if contact_capacity > 0 and model.shape_count < 1:
            raise ValueError("LOX deformable contacts require at least one collider shape.")

        self.model = model
        self.cloth_system = cloth_system
        self.device = model.device
        self.rigid_contact_capacity = contact_capacity
        self.self_contact_capacity = self_contact_capacity
        self.contact_capacity = total_contact_capacity
        self.apgd_worker_count = _bounded_apgd_worker_count(self.contact_capacity, self.device)
        self.stabilization_fraction = float(stabilization_fraction)
        self.dead_zone = float(dead_zone)
        self.impact_velocity_threshold = float(impact_velocity_threshold)
        self.recoverable_response = recoverable_response
        self.enable_rigid_normal_cone_filtering = enable_rigid_normal_cone_filtering
        self.normal_cone_filtering_min_distance = float(normal_cone_filtering_min_distance)
        self._source_count = None
        self._prepared = False
        self._empty_body_pose = wp.empty(0, dtype=wp.transform, device=self.device)
        self._empty_body_velocity = wp.empty(0, dtype=wp.spatial_vector, device=self.device)
        self._empty_body_vector = wp.empty(0, dtype=wp.vec3, device=self.device)
        self._empty_body_index = wp.empty(0, dtype=wp.int32, device=self.device)
        self._empty_body_inverse_weight = wp.empty(0, dtype=mat66f, device=self.device)
        self._empty_body_twist = wp.empty(0, dtype=vec6f, device=self.device)
        self._empty_edge_indices = wp.empty((0, 4), dtype=wp.int32, device=self.device)
        self.edge_cone_axis = wp.zeros(model.edge_count, dtype=wp.vec3, device=self.device)
        self.edge_cone_cosine = wp.full(model.edge_count, -1.0, dtype=wp.float32, device=self.device)

        self.particle_indices = wp.full(
            (self.contact_capacity, 4),
            -1,
            dtype=wp.int32,
            device=self.device,
        )
        self.coefficients = wp.zeros((self.contact_capacity, 4), dtype=wp.float32, device=self.device)
        self.contact_world = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.contact_shape = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.body = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.normal = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.frame = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.body_jacobian = wp.zeros(self.contact_capacity, dtype=mat36f, device=self.device)
        self.gap = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.bias = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.rigid_bias = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.friction = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.collider_velocity = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.surface_position_local = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.surface_velocity_local = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.normal_local = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.gap_offset = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.status = wp.zeros(self.contact_capacity, dtype=wp.int32, device=self.device)
        self.delassus = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.inverse_delassus = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.reaction = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.rigid_delassus = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.rigid_reaction = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.gauss_seidel_scalar_delassus = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.gauss_seidel_delassus = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.apgd_trial = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.apgd_next = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.avbd_delassus = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.avbd_inverse_delassus = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.avbd_augmented_reaction = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.contact_velocity = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.contact_residual = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)

        self.particle_multiplicity = wp.zeros(
            cloth_system.particle_count,
            dtype=wp.int32,
            device=self.device,
        )
        self.split_inverse_weight = wp.zeros(
            cloth_system.particle_count,
            dtype=wp.float32,
            device=self.device,
        )
        self.particle_delta = wp.zeros(
            cloth_system.particle_count,
            dtype=wp.vec3,
            device=self.device,
        )
        self.avbd_particle_force = wp.zeros(
            cloth_system.particle_count,
            dtype=wp.vec3,
            device=self.device,
        )
        self.avbd_particle_hessian = wp.zeros(
            cloth_system.particle_count,
            dtype=wp.mat33f,
            device=self.device,
        )
        self.avbd_particle_stationarity = wp.zeros(
            cloth_system.particle_count,
            dtype=wp.vec3,
            device=self.device,
        )

        world_count = model.world_count
        self.world_contact_count = wp.zeros(world_count, dtype=wp.int32, device=self.device)
        self.world_contact_offset = wp.zeros(world_count, dtype=wp.int32, device=self.device)
        self._world_contact_cursor = wp.zeros(world_count, dtype=wp.int32, device=self.device)
        self.contact_order = wp.empty(self.contact_capacity, dtype=wp.int32, device=self.device)
        self.sequential_projection_status = wp.zeros(world_count, dtype=wp.int32, device=self.device)
        self.world_status = wp.zeros(world_count, dtype=wp.int32, device=self.device)
        self.global_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.invalid_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.world_contact_residual = wp.zeros(world_count, dtype=wp.float32, device=self.device)
        self.world_consensus_residual = wp.zeros(world_count, dtype=wp.float32, device=self.device)
        self.world_iterate_residual = wp.zeros(world_count, dtype=wp.float32, device=self.device)
        self.world_displacement_residual = wp.zeros(world_count, dtype=wp.float32, device=self.device)
        self.world_avbd_stationarity_max = wp.zeros(world_count, dtype=wp.float32, device=self.device)
        self.world_avbd_update_max = wp.zeros(world_count, dtype=wp.float32, device=self.device)

    def _validate_source_arrays(
        self,
        contacts: Contacts | None,
        state: State,
        body_pose: wp.array[wp.transform] | None = None,
    ) -> None:
        if self.rigid_contact_capacity > 0 and contacts is None:
            raise ValueError("LOX rigid-soft contacts require a Newton contacts container.")
        if contacts is not None and contacts.soft_contact_max < self.rigid_contact_capacity:
            raise ValueError(
                f"LOX deformable contacts require source capacity at least {self.rigid_contact_capacity}, "
                f"found {contacts.soft_contact_max}."
            )
        arrays = {
            "particle_q": state.particle_q,
            "particle_qd": state.particle_qd,
        }
        if contacts is not None:
            arrays.update(
                {
                    "soft_contact_count": contacts.soft_contact_count,
                    "soft_contact_indices": contacts.soft_contact_indices,
                    "soft_contact_barycentric": contacts.soft_contact_barycentric,
                    "soft_contact_shape": contacts.soft_contact_shape,
                    "soft_contact_body_pos": contacts.soft_contact_body_pos,
                    "soft_contact_body_vel": contacts.soft_contact_body_vel,
                    "soft_contact_normal": contacts.soft_contact_normal,
                }
            )
        if self.model.body_count > 0:
            arrays["body_q"] = body_pose if body_pose is not None else state.body_q
            arrays["body_qd"] = state.body_qd
        for name, value in arrays.items():
            if value is None:
                raise ValueError(f"LOX deformable contacts require {name}.")
            if value.device != self.device:
                raise ValueError(f"LOX deformable contacts expected {name} on {self.device}, found {value.device}.")

    def prepare(
        self,
        contacts: Contacts | None,
        state: State,
        time_step: wp.array[wp.float32],
        self_contact_detector: DeformableSelfContactDetector | None = None,
        body_pose: wp.array[wp.transform] | None = None,
    ) -> None:
        """Adapt and validate frozen Newton soft-contact records for one step.

        Numeric record failures are reported through :attr:`status` and rejected
        by the projection without a host synchronization. Call
        :meth:`raise_if_invalid` for an explicit diagnostic exception.

        Args:
            contacts: Newton contacts containing soft particle/edge/face records.
            state: Beginning-of-step Newton state.
            time_step: Per-world simulation time steps [s].
            self_contact_detector: Optional frozen cloth self-contact detector.
            body_pose: Optional body-origin poses captured before adapting the state for another backend.
        """
        validate_world_time_step(time_step, self.model.world_count, self.device)
        self._prepared = False
        self._validate_source_arrays(contacts, state, body_pose)
        body_pose = body_pose if body_pose is not None else state.body_q
        friction = float(self.model.soft_contact_mu)
        restitution = float(self.model.soft_contact_restitution)
        if not np.isfinite(friction) or friction < 0.0:
            raise ValueError("LOX deformable contact friction must be finite and non-negative.")
        if not np.isfinite(restitution) or restitution < 0.0:
            raise ValueError("LOX deformable contact restitution must be finite and non-negative.")

        self.particle_multiplicity.zero_()
        self.world_contact_count.zero_()
        self.world_status.zero_()
        self.global_status.zero_()
        self.invalid_count.zero_()
        self.reaction.zero_()
        self.rigid_reaction.zero_()
        # Full-surface records carry enough topology to reduce boundary features before cone tests.
        filter_surface_contacts = bool(
            self.enable_rigid_normal_cone_filtering
            and contacts is not None
            and contacts._enable_rigid_soft_full_surface_contact
            and self.model.tri_count > 0
            and self.model.edge_count > 0
        )
        if filter_surface_contacts and self.model.edge_count > 0:
            wp.launch(
                compute_soft_edge_normal_cones,
                dim=self.model.edge_count,
                inputs=[state.particle_q, self.model.edge_indices],
                outputs=[self.edge_cone_axis, self.edge_cone_cosine],
                device=self.device,
            )
        if contacts is not None and self.rigid_contact_capacity > 0:
            wp.launch(
                _check_contact_count,
                dim=1,
                inputs=[contacts.soft_contact_count, self.rigid_contact_capacity],
                outputs=[self.global_status, self.invalid_count],
                device=self.device,
            )
            wp.launch(
                _adapt_soft_contacts,
                dim=self.rigid_contact_capacity,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self.model.edge_indices if self.model.edge_indices is not None else self._empty_edge_indices,
                    self.model.soft_mesh_adjacency_device,
                    self.edge_cone_axis,
                    self.edge_cone_cosine,
                    filter_surface_contacts,
                    self.normal_cone_filtering_min_distance,
                    state.particle_q,
                    state.particle_qd,
                    self.model.particle_radius,
                    self.model.particle_flags,
                    self.cloth_system.topology.newton_to_packed,
                    self.cloth_system.topology.packed_world,
                    self.cloth_system.full_inverse_weight,
                    self.model.shape_body,
                    self.model.shape_world,
                    self.model.shape_margin,
                    body_pose if body_pose is not None else self._empty_body_pose,
                    state.body_qd if state.body_qd is not None else self._empty_body_velocity,
                    self.model.body_com if self.model.body_com is not None else self._empty_body_vector,
                    self.model.body_flags if self.model.body_flags is not None else self._empty_body_index,
                    self.model.body_world if self.model.body_world is not None else self._empty_body_index,
                    time_step,
                    self.stabilization_fraction,
                    self.dead_zone,
                    self.impact_velocity_threshold,
                    self.recoverable_response,
                    friction,
                    restitution,
                ],
                outputs=[
                    self.particle_indices,
                    self.coefficients,
                    self.contact_world,
                    self.contact_shape,
                    self.body,
                    self.normal,
                    self.frame,
                    self.body_jacobian,
                    self.gap,
                    self.bias,
                    self.rigid_bias,
                    self.friction,
                    self.collider_velocity,
                    self.surface_position_local,
                    self.surface_velocity_local,
                    self.status,
                    self.particle_multiplicity,
                    self.world_contact_count,
                    self.world_status,
                    self.global_status,
                    self.invalid_count,
                ],
                device=self.device,
            )
        if self_contact_detector is not None:
            self_contact_detector.adapt(
                self,
                state.particle_q,
                state.particle_qd,
                time_step,
                friction,
                restitution,
            )
        wp.launch(
            _capture_frozen_deformable_contact_geometry,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_shape,
                self.normal,
                self.gap,
                self.status,
                self.model.shape_body,
                body_pose if body_pose is not None else self._empty_body_pose,
                self.cloth_system.position_start,
                self.surface_position_local,
            ],
            outputs=[self.normal_local, self.gap_offset],
            device=self.device,
        )
        wp.launch(
            _finalize_split_inverse_weight,
            dim=self.cloth_system.particle_count,
            inputs=[
                self.cloth_system.full_inverse_weight,
                self.particle_multiplicity,
            ],
            outputs=[self.split_inverse_weight],
            device=self.device,
        )
        wp.launch(
            _finalize_contact_delassus,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.split_inverse_weight,
            ],
            outputs=[
                self.status,
                self.delassus,
                self.inverse_delassus,
                self.world_contact_count,
                self.world_status,
                self.global_status,
                self.invalid_count,
            ],
            device=self.device,
        )
        self._source_count = contacts.soft_contact_count if contacts is not None else None
        self._prepared = True

    def refresh_geometry(
        self,
        time_step: wp.array[wp.float32],
        body_pose: wp.array[wp.transform] | None,
        body_velocity_begin: wp.array[vec6f] | None,
        body_pose_is_center_of_mass: bool = False,
    ) -> None:
        """Refresh numeric contact data while retaining the frozen support."""
        validate_world_time_step(time_step, self.model.world_count, self.device)
        if not self._prepared:
            raise RuntimeError("LOX deformable contacts must be prepared before refreshing geometry.")
        wp.launch(
            _refresh_frozen_deformable_contact_geometry,
            dim=self.contact_capacity,
            inputs=[
                self.contact_world,
                self.particle_indices,
                self.coefficients,
                self.contact_shape,
                self.body,
                self.status,
                self.model.shape_body,
                self.model.particle_mass,
                self.model.particle_flags,
                self.cloth_system.topology.packed_to_newton,
                self.cloth_system.position_linearized,
                self.cloth_system.velocity_start,
                body_pose if body_pose is not None else self._empty_body_pose,
                body_velocity_begin if body_velocity_begin is not None else self._empty_body_twist,
                self.model.body_com if self.model.body_com is not None else self._empty_body_vector,
                body_pose_is_center_of_mass,
                self.surface_position_local,
                self.surface_velocity_local,
                self.normal_local,
                self.gap_offset,
                time_step,
                self.stabilization_fraction,
                self.dead_zone,
                self.impact_velocity_threshold,
                self.recoverable_response,
                float(self.model.soft_contact_restitution),
            ],
            outputs=[
                self.normal,
                self.frame,
                self.body_jacobian,
                self.gap,
                self.bias,
                self.rigid_bias,
                self.collider_velocity,
                self.rigid_reaction,
            ],
            device=self.device,
        )

    def update_weight_metric(self) -> None:
        """Refresh particle-side Delassus values after cloth reassembly."""
        wp.launch(
            _finalize_split_inverse_weight,
            dim=self.cloth_system.particle_count,
            inputs=[
                self.cloth_system.inverse_weight,
                self.particle_multiplicity,
            ],
            outputs=[self.split_inverse_weight],
            device=self.device,
        )
        wp.launch(
            _finalize_contact_delassus,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.split_inverse_weight,
            ],
            outputs=[
                self.status,
                self.delassus,
                self.inverse_delassus,
                self.world_contact_count,
                self.world_status,
                self.global_status,
                self.invalid_count,
            ],
            device=self.device,
        )

    def _build_contact_order(self) -> None:
        """Compact valid contacts into device-resident per-world ranges."""
        wp.utils.array_scan(self.world_contact_count, self.world_contact_offset, inclusive=False)
        wp.copy(self._world_contact_cursor, self.world_contact_offset)
        wp.launch(
            _scatter_contact_order,
            dim=self.contact_capacity,
            inputs=[self.contact_world, self.status],
            outputs=[self._world_contact_cursor, self.contact_order],
            device=self.device,
        )

    def reset(self) -> None:
        """Clear adapted-contact status and within-step reaction warm starts."""
        self.particle_multiplicity.zero_()
        self.world_contact_count.zero_()
        self.world_status.zero_()
        self.global_status.zero_()
        self.invalid_count.zero_()
        self.reaction.zero_()
        self.rigid_reaction.zero_()
        self.particle_delta.zero_()
        self.sequential_projection_status.zero_()
        self._source_count = None
        self._prepared = False

    def raise_if_invalid(self) -> None:
        """Raise for a malformed, cross-world, degenerate, or failed record.

        This diagnostic performs a device-to-host synchronization and therefore
        must not be called inside CUDA graph capture.
        """
        if not self._prepared:
            raise RuntimeError("LOX deformable contacts have not been prepared.")
        source_count = int(self._source_count.numpy()[0]) if self._source_count is not None else 0
        if source_count > self.rigid_contact_capacity:
            raise ValueError(
                f"LOX deformable soft contact count {source_count} exceeds capacity {self.rigid_contact_capacity}."
            )
        if source_count < 0:
            raise ValueError("LOX deformable soft contact count must be non-negative.")
        statuses = self.status.numpy()
        invalid = np.flatnonzero(
            (statuses != DEFORMABLE_CONTACT_STATUS_UNUSED) & (statuses != DEFORMABLE_CONTACT_STATUS_VALID)
        )
        if invalid.size == 0:
            return
        contact = int(invalid[0])
        status = int(statuses[contact])
        descriptions = {
            DEFORMABLE_CONTACT_STATUS_MALFORMED: "malformed",
            DEFORMABLE_CONTACT_STATUS_CROSS_WORLD: "cross-world",
            DEFORMABLE_CONTACT_STATUS_DYNAMIC_RIGID: "dynamic-rigid",
            DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS: "non-positive scalar Delassus",
            DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE: "numerically failed",
        }
        description = descriptions.get(status, f"invalid status {status}")
        raise ValueError(f"LOX deformable contact {contact} is {description}.")

    def _validate_particle_velocity(self, value: wp.array[wp.vec3], name: str) -> None:
        if value.shape != (self.cloth_system.particle_count,) or value.dtype != wp.vec3:
            raise ValueError(f"LOX deformable {name} must be a vec3 array with one entry per packed particle.")
        if value.device != self.device:
            raise ValueError(f"LOX deformable contacts expected {name} on {self.device}, found {value.device}.")

    def accumulate_rigid_incidence(
        self,
        body_constraint_count: wp.array[wp.int32],
        body_has_unilateral: wp.array[wp.int32],
        world_has_unilateral: wp.array[wp.bool],
    ) -> None:
        """Merge dynamic soft contacts into rigid unilateral incidence."""
        body_count = int(self.model.body_count)
        if body_constraint_count.shape != (body_count,) or body_has_unilateral.shape != (body_count,):
            raise ValueError("LOX mixed contact body incidence must match the Newton body count.")
        if world_has_unilateral.shape != (int(self.model.world_count),):
            raise ValueError("LOX mixed contact world incidence must match the Newton world count.")
        wp.launch(
            _accumulate_rigid_incidence,
            dim=self.contact_capacity,
            inputs=[self.body, self.contact_world, self.status],
            outputs=[
                body_constraint_count,
                body_has_unilateral,
                world_has_unilateral,
            ],
            device=self.device,
        )

    def prepare_rigid_projection(
        self,
        body_constraint_count: wp.array[wp.int32],
        static_body_constraint_count: wp.array[wp.int32],
        body_inverse_weight: wp.array[mat66f],
        prepared_status: wp.array[wp.int32],
    ) -> None:
        """Prepare full anisotropic blocks for the shared rigid Jacobi sweep."""
        body_count = int(self.model.body_count)
        if (
            body_constraint_count.shape != (body_count,)
            or static_body_constraint_count.shape != (body_count,)
            or body_inverse_weight.shape != (body_count,)
        ):
            raise ValueError("LOX mixed contact rigid arrays must match the Newton body count.")
        if prepared_status.shape != (int(self.model.world_count),):
            raise ValueError("LOX mixed contact prepared status must match the Newton world count.")
        wp.launch(
            _prepare_rigid_contacts,
            dim=self.contact_capacity,
            inputs=[
                self.contact_world,
                self.body,
                self.body_jacobian,
                self.rigid_bias,
                self.friction,
                self.delassus,
                body_constraint_count,
                static_body_constraint_count,
                body_inverse_weight,
            ],
            outputs=[
                self.status,
                self.rigid_delassus,
                self.world_status,
                self.invalid_count,
            ],
            device=self.device,
        )
        wp.launch(
            _merge_rigid_prepared_status,
            dim=int(self.model.world_count),
            inputs=[self.world_status, self.global_status],
            outputs=[prepared_status],
            device=self.device,
        )

    def prepare_gauss_seidel_projection(
        self,
        body_inverse_weight: wp.array[mat66f] | None = None,
        prepared_status: wp.array[wp.int32] | None = None,
    ) -> wp.array[wp.int32]:
        """Prepare true local contact blocks for sequential projection."""
        include_rigid = body_inverse_weight is not None
        if (body_inverse_weight is None) != (prepared_status is None):
            raise ValueError("Mixed Gauss-Seidel projection requires both rigid weights and prepared status.")
        if include_rigid:
            if body_inverse_weight.shape != (int(self.model.body_count),):
                raise ValueError("LOX mixed contact rigid weights must match the Newton body count.")
            status = prepared_status
            inverse_weight = body_inverse_weight
        else:
            status = self.sequential_projection_status
            status.fill_(PROJECTION_STATUS_VALID)
            inverse_weight = self._empty_body_inverse_weight
        self._build_contact_order()
        wp.launch(
            _prepare_gauss_seidel_contacts,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.body_jacobian,
                self.rigid_bias,
                self.friction,
                self.cloth_system.inverse_weight,
                inverse_weight,
                include_rigid,
            ],
            outputs=[
                self.status,
                self.gauss_seidel_scalar_delassus,
                self.gauss_seidel_delassus,
                self.world_status,
                self.invalid_count,
            ],
            device=self.device,
        )
        wp.launch(
            _merge_rigid_prepared_status,
            dim=int(self.model.world_count),
            inputs=[self.world_status, self.global_status],
            outputs=[status],
            device=self.device,
        )
        return status

    def warm_start_gauss_seidel(
        self,
        world_active: wp.array[wp.bool],
        projected_velocity: wp.array[wp.vec3],
        body_inverse_weight: wp.array[mat66f] | None = None,
        projected_twist: wp.array[vec6f] | None = None,
        projection_status: wp.array[wp.int32] | None = None,
    ) -> None:
        """Apply deformable contact reaction warm starts once in world order."""
        self._validate_particle_velocity(projected_velocity, "projected velocity")
        rigid_coordinates = body_inverse_weight is not None
        if rigid_coordinates != (projected_twist is not None) or rigid_coordinates != (projection_status is not None):
            raise ValueError("Mixed Gauss-Seidel warm starts require all rigid arrays.")
        if rigid_coordinates:
            inverse_weight = body_inverse_weight
            twist = projected_twist
            status = projection_status
        else:
            inverse_weight = self._empty_body_inverse_weight
            twist = self._empty_body_twist
            status = self.sequential_projection_status
        wp.launch(
            _warm_start_contacts_sequential,
            dim=int(self.model.world_count),
            inputs=[
                self.world_contact_offset,
                self.world_contact_count,
                self.contact_order,
                self.particle_indices,
                self.coefficients,
                self.body,
                self.frame,
                self.body_jacobian,
                self.status,
                world_active,
                status,
                self.cloth_system.inverse_weight,
                inverse_weight,
                rigid_coordinates,
                self.reaction,
                self.rigid_reaction,
                projected_velocity,
                twist,
            ],
            outputs=[status],
            device=self.device,
        )

    def sweep_gauss_seidel(
        self,
        world_active: wp.array[wp.bool],
        projected_velocity: wp.array[wp.vec3],
        body_inverse_weight: wp.array[mat66f] | None = None,
        projected_twist: wp.array[vec6f] | None = None,
        projection_status: wp.array[wp.int32] | None = None,
    ) -> None:
        """Run one in-place sequential deformable contact sweep per world."""
        self._validate_particle_velocity(projected_velocity, "projected velocity")
        rigid_coordinates = body_inverse_weight is not None
        if rigid_coordinates != (projected_twist is not None) or rigid_coordinates != (projection_status is not None):
            raise ValueError("Mixed Gauss-Seidel sweeps require all rigid arrays.")
        inverse_weight = body_inverse_weight if rigid_coordinates else self._empty_body_inverse_weight
        twist = projected_twist if rigid_coordinates else self._empty_body_twist
        status = projection_status if rigid_coordinates else self.sequential_projection_status
        wp.launch(
            _sweep_contacts_sequential,
            dim=int(self.model.world_count),
            inputs=[
                self.world_contact_offset,
                self.world_contact_count,
                self.contact_order,
                self.particle_indices,
                self.coefficients,
                self.body,
                self.normal,
                self.frame,
                self.body_jacobian,
                self.bias,
                self.rigid_bias,
                self.friction,
                self.gauss_seidel_scalar_delassus,
                self.gauss_seidel_delassus,
                self.status,
                world_active,
                self.cloth_system.inverse_weight,
                inverse_weight,
                rigid_coordinates,
                projected_velocity,
                twist,
            ],
            outputs=[
                self.reaction,
                self.rigid_reaction,
                self.contact_velocity,
                status,
                self.world_status,
            ],
            device=self.device,
        )

    def prepare_apgd_projection(
        self,
        rigid_coordinates: bool,
        prepared_status: wp.array[wp.int32] | None = None,
    ) -> wp.array[wp.int32]:
        """Select the prepared status for APGD contact steps."""
        self._build_contact_order()
        if rigid_coordinates:
            if prepared_status is None:
                raise ValueError("Mixed APGD projection requires a prepared rigid status array.")
            return prepared_status
        status = self.sequential_projection_status
        status.fill_(PROJECTION_STATUS_VALID)
        wp.launch(
            _merge_rigid_prepared_status,
            dim=int(self.model.world_count),
            inputs=[self.world_status, self.global_status],
            outputs=[status],
            device=self.device,
        )
        return status

    def initialize_apgd(self, world_active: wp.array[wp.bool], rigid_coordinates: bool) -> None:
        """Project contact warm starts and initialize the extrapolated field."""
        wp.launch(
            _initialize_contacts_apgd,
            dim=self.apgd_worker_count,
            inputs=[
                self.apgd_worker_count,
                int(self.model.world_count),
                self.world_contact_offset,
                self.world_contact_count,
                self.contact_order,
                self.contact_world,
                self.status,
                self.normal,
                self.friction,
                world_active,
                rigid_coordinates,
            ],
            outputs=[self.reaction, self.rigid_reaction, self.apgd_trial],
            device=self.device,
            block_dim=_APGD_BLOCK_DIM,
        )

    def begin_apgd_scatter(self) -> None:
        """Clear the particle accumulator before a complete reaction scatter."""
        self.particle_delta.zero_()

    def scatter_apgd(
        self,
        world_active: wp.array[wp.bool],
        body_inverse_weight: wp.array[mat66f] | None,
        body_delta: wp.array[vec6f] | None,
        rigid_coordinates: bool,
        use_trial: bool,
        projection_status: wp.array[wp.int32] | None = None,
    ) -> None:
        """Scatter a complete feasible or extrapolated deformable reaction field."""
        if rigid_coordinates:
            if body_inverse_weight is None or body_delta is None or projection_status is None:
                raise ValueError("Mixed APGD scatter requires all rigid arrays.")
            inverse_weight = body_inverse_weight
            delta = body_delta
            status = projection_status
        else:
            inverse_weight = self._empty_body_inverse_weight
            delta = self._empty_body_twist
            status = self.sequential_projection_status
        wp.launch(
            _scatter_contacts_apgd,
            dim=self.apgd_worker_count,
            inputs=[
                self.apgd_worker_count,
                int(self.model.world_count),
                self.world_contact_offset,
                self.world_contact_count,
                self.contact_order,
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.frame,
                self.body_jacobian,
                self.status,
                world_active,
                status,
                self.cloth_system.inverse_weight,
                inverse_weight,
                rigid_coordinates,
                use_trial,
                self.reaction,
                self.rigid_reaction,
                self.apgd_trial,
            ],
            outputs=[self.particle_delta, delta],
            device=self.device,
            block_dim=_APGD_BLOCK_DIM,
        )

    def reconstruct_apgd_particle_velocity(
        self,
        world_active: wp.array[wp.bool],
        projection_status: wp.array[wp.int32],
        baseline_velocity: wp.array[wp.vec3],
        projected_velocity: wp.array[wp.vec3],
    ) -> None:
        """Reconstruct APGD particle velocities and clear their correction field."""
        wp.launch(
            _reconstruct_apgd_particle_velocity,
            dim=self.cloth_system.particle_count,
            inputs=[
                self.cloth_system.topology.packed_world,
                world_active,
                projection_status,
                baseline_velocity,
                self.particle_delta,
            ],
            outputs=[projected_velocity],
            device=self.device,
        )

    def project_apgd(
        self,
        world_active: wp.array[wp.bool],
        projected_velocity: wp.array[wp.vec3],
        projected_twist: wp.array[vec6f] | None,
        restart_dot: wp.array[float],
        projection_status: wp.array[wp.int32],
        rigid_coordinates: bool,
    ) -> None:
        """Project one APGD deformable contact step and reduce restart data."""
        twist = projected_twist if rigid_coordinates else self._empty_body_twist
        wp.launch(
            _project_contacts_apgd,
            dim=self.apgd_worker_count,
            inputs=[
                self.apgd_worker_count,
                int(self.model.world_count),
                self.world_contact_offset,
                self.world_contact_count,
                self.contact_order,
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.normal,
                self.frame,
                self.body_jacobian,
                self.bias,
                self.rigid_bias,
                self.friction,
                self.delassus,
                self.rigid_delassus,
                self.status,
                world_active,
                rigid_coordinates,
                projected_velocity,
                twist,
                self.reaction,
                self.rigid_reaction,
                self.apgd_trial,
            ],
            outputs=[
                self.apgd_next,
                self.contact_velocity,
                restart_dot,
                projection_status,
                self.world_status,
            ],
            device=self.device,
            block_dim=_APGD_BLOCK_DIM,
        )

    def extrapolate_apgd(
        self,
        world_active: wp.array[wp.bool],
        beta: wp.array[float],
        projection_status: wp.array[wp.int32],
        rigid_coordinates: bool,
    ) -> None:
        """Commit the feasible contact iterate and build its extrapolation."""
        wp.launch(
            _extrapolate_contacts_apgd,
            dim=self.apgd_worker_count,
            inputs=[
                self.apgd_worker_count,
                int(self.model.world_count),
                self.world_contact_offset,
                self.world_contact_count,
                self.contact_order,
                self.contact_world,
                self.status,
                world_active,
                beta,
                projection_status,
                rigid_coordinates,
                self.apgd_next,
            ],
            outputs=[self.reaction, self.rigid_reaction, self.apgd_trial],
            device=self.device,
            block_dim=_APGD_BLOCK_DIM,
        )

    def begin_rigid_jacobi_accumulation(self) -> None:
        """Clear the mixed-contact particle Jacobi accumulator."""
        self.particle_delta.zero_()

    def accumulate_rigid_reaction_warm_start(
        self,
        world_active: wp.array[wp.bool],
        prepared_status: wp.array[wp.int32],
        particle_inverse_weight: wp.array[float],
        body_inverse_weight: wp.array[mat66f],
        apply_inverse_weight: bool,
        particle_velocity: wp.array[wp.vec3],
        body_twist: wp.array[vec6f],
        body_delta: wp.array[vec6f],
        projection_status: wp.array[wp.int32],
    ) -> None:
        """Accumulate generalized soft-contact warm starts into both endpoints."""
        del particle_velocity, body_twist
        wp.launch(
            _warm_start_rigid_contacts,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.frame,
                self.body_jacobian,
                self.status,
                world_active,
                prepared_status,
                particle_inverse_weight,
                body_inverse_weight,
                apply_inverse_weight,
                self.rigid_reaction,
            ],
            outputs=[
                self.particle_delta,
                body_delta,
                projection_status,
            ],
            device=self.device,
        )

    def project_rigid_jacobi(
        self,
        world_active: wp.array[wp.bool],
        particle_inverse_weight: wp.array[float],
        body_inverse_weight: wp.array[mat66f],
        particle_velocity: wp.array[wp.vec3],
        body_twist: wp.array[vec6f],
        body_delta: wp.array[vec6f],
        projection_status: wp.array[wp.int32],
        coulomb_statistics: CoulombSolveStatistics | None = None,
    ) -> None:
        """Accumulate one anisotropic generalized-contact Jacobi sweep."""
        inputs = [
            self.particle_indices,
            self.coefficients,
            self.contact_world,
            self.body,
            self.frame,
            self.body_jacobian,
            self.rigid_bias,
            self.friction,
            self.rigid_delassus,
            self.status,
            world_active,
            particle_inverse_weight,
            body_inverse_weight,
            particle_velocity,
            body_twist,
        ]
        outputs = [
            self.rigid_reaction,
            self.particle_delta,
            body_delta,
            projection_status,
            self.world_status,
        ]
        if coulomb_statistics is None:
            wp.launch(
                _project_rigid_contacts,
                dim=self.contact_capacity,
                inputs=inputs,
                outputs=outputs,
                device=self.device,
                block_dim=_RIGID_CONTACT_PROJECTION_BLOCK_DIM,
            )
        else:
            wp.launch(
                _project_rigid_contacts_instrumented,
                dim=self.contact_capacity,
                inputs=inputs,
                outputs=[
                    *outputs,
                    coulomb_statistics.branch_histogram,
                    coulomb_statistics.expansion_histogram,
                    coulomb_statistics.root_histogram,
                    coulomb_statistics.failure_counts,
                ],
                device=self.device,
                block_dim=_RIGID_CONTACT_PROJECTION_BLOCK_DIM,
            )

    def apply_rigid_particle_delta(
        self,
        world_active: wp.array[wp.bool],
        projection_status: wp.array[wp.int32],
        projected_velocity: wp.array[wp.vec3],
    ) -> None:
        """Apply a shared-sweep particle delta only in valid active worlds."""
        wp.launch(
            _apply_rigid_particle_delta,
            dim=self.cloth_system.particle_count,
            inputs=[
                self.cloth_system.topology.packed_world,
                world_active,
                projection_status,
                self.particle_delta,
            ],
            outputs=[projected_velocity],
            device=self.device,
        )

    def apply_reaction_warm_start(self, projected_velocity: wp.array[wp.vec3]) -> None:
        """Apply stored within-step reactions to a fresh projected velocity."""
        self._validate_particle_velocity(projected_velocity, "projected velocity")
        self.particle_delta.zero_()
        wp.launch(
            _warm_start_contacts,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.status,
                self.cloth_system.world_active,
                self.world_status,
                self.global_status,
                self.cloth_system.inverse_weight,
                self.reaction,
            ],
            outputs=[self.particle_delta],
            device=self.device,
        )
        wp.launch(
            _apply_particle_delta,
            dim=self.cloth_system.particle_count,
            inputs=[
                self.cloth_system.topology.packed_world,
                self.cloth_system.world_active,
                self.particle_delta,
            ],
            outputs=[projected_velocity],
            device=self.device,
        )

    def project(self, projected_velocity: wp.array[wp.vec3], iterations: int = 1) -> None:
        """Run fixed-count mass-split Jacobi contact sweeps in place.

        Args:
            projected_velocity: Packed nodal velocity updated in place [m/s].
            iterations: Number of Jacobi sweeps.
        """
        self._validate_particle_velocity(projected_velocity, "projected velocity")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
            raise ValueError("LOX deformable contact iterations must be a positive integer.")
        for _iteration in range(iterations):
            self.particle_delta.zero_()
            wp.launch(
                _project_contacts,
                dim=self.contact_capacity,
                inputs=[
                    self.particle_indices,
                    self.coefficients,
                    self.contact_world,
                    self.body,
                    self.normal,
                    self.bias,
                    self.friction,
                    self.delassus,
                    self.status,
                    self.cloth_system.world_active,
                    self.world_status,
                    self.global_status,
                    self.cloth_system.inverse_weight,
                    projected_velocity,
                ],
                outputs=[self.reaction, self.particle_delta],
                device=self.device,
            )
            wp.launch(
                _apply_particle_delta,
                dim=self.cloth_system.particle_count,
                inputs=[
                    self.cloth_system.topology.packed_world,
                    self.cloth_system.world_active,
                    self.particle_delta,
                ],
                outputs=[projected_velocity],
                device=self.device,
            )

    def project_jacobi_smoothing_sweep(
        self,
        world_active: wp.array[wp.bool],
        projected_velocity: wp.array[wp.vec3],
        projection_status: wp.array[wp.int32],
    ) -> None:
        """Accumulate and apply one Jacobi sweep without a reaction warm start."""
        self._validate_particle_velocity(projected_velocity, "projected velocity")
        if world_active.shape != (int(self.model.world_count),) or projection_status.shape != world_active.shape:
            raise ValueError("LOX deformable smoothing status arrays must match the model world count.")
        self.particle_delta.zero_()
        wp.launch(
            _project_contacts,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.normal,
                self.bias,
                self.friction,
                self.delassus,
                self.status,
                self.cloth_system.world_active,
                self.world_status,
                self.global_status,
                self.cloth_system.inverse_weight,
                projected_velocity,
            ],
            outputs=[self.reaction, self.particle_delta],
            device=self.device,
        )
        wp.launch(
            _merge_rigid_prepared_status,
            dim=projection_status.shape[0],
            inputs=[self.world_status, self.global_status],
            outputs=[projection_status],
            device=self.device,
        )
        wp.launch(
            _apply_rigid_particle_delta,
            dim=self.cloth_system.particle_count,
            inputs=[
                self.cloth_system.topology.packed_world,
                world_active,
                projection_status,
                self.particle_delta,
            ],
            outputs=[projected_velocity],
            device=self.device,
        )

    def compute_contact_residuals(
        self,
        projected_velocity: wp.array[wp.vec3],
        projected_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Compute per-contact and per-world scaled natural-map residuals."""
        self._validate_particle_velocity(projected_velocity, "projected velocity")
        self.world_contact_residual.zero_()
        if projected_twist is not None:
            if projected_twist.shape != (int(self.model.body_count),) or projected_twist.dtype != vec6f:
                raise ValueError("LOX mixed contact projected twist must contain one vec6 per body.")
            wp.launch(
                _compute_rigid_contact_residuals,
                dim=self.contact_capacity,
                inputs=[
                    self.particle_indices,
                    self.coefficients,
                    self.contact_world,
                    self.body,
                    self.frame,
                    self.body_jacobian,
                    self.rigid_bias,
                    self.friction,
                    self.rigid_delassus,
                    self.status,
                    self.cloth_system.world_active,
                    projected_velocity,
                    projected_twist,
                    self.rigid_reaction,
                ],
                outputs=[
                    self.contact_velocity,
                    self.contact_residual,
                    self.world_contact_residual,
                ],
                device=self.device,
            )
            return
        wp.launch(
            _compute_contact_residuals,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.normal,
                self.bias,
                self.friction,
                self.delassus,
                self.status,
                self.cloth_system.world_active,
                projected_velocity,
                self.reaction,
            ],
            outputs=[
                self.contact_velocity,
                self.contact_residual,
                self.world_contact_residual,
            ],
            device=self.device,
        )

    def compute_consensus_residuals(
        self,
        global_velocity: wp.array[wp.vec3],
        projected_velocity: wp.array[wp.vec3],
        global_velocity_previous: wp.array[wp.vec3],
        projected_velocity_previous: wp.array[wp.vec3],
        time_step: wp.array[wp.float32],
    ) -> None:
        """Compute per-world consensus, iterate, and displacement residuals."""
        for name, value in (
            ("global velocity", global_velocity),
            ("projected velocity", projected_velocity),
            ("previous global velocity", global_velocity_previous),
            ("previous projected velocity", projected_velocity_previous),
        ):
            self._validate_particle_velocity(value, name)
        validate_world_time_step(time_step, self.model.world_count, self.device)

        self.world_consensus_residual.zero_()
        self.world_iterate_residual.zero_()
        self.world_displacement_residual.zero_()
        wp.launch(
            _compute_consensus_residuals,
            dim=self.cloth_system.particle_count,
            inputs=[
                self.cloth_system.topology.packed_world,
                self.cloth_system.world_active,
                global_velocity,
                projected_velocity,
                global_velocity_previous,
                projected_velocity_previous,
                time_step,
            ],
            outputs=[
                self.world_consensus_residual,
                self.world_iterate_residual,
                self.world_displacement_residual,
            ],
            device=self.device,
        )
