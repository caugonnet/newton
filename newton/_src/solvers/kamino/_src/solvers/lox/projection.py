# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free one-constraint projection primitives.

Contact-facing functions use Kamino's normal-last order
``(tangent_x, tangent_y, normal_z)``. Only the local Coulomb--Newton solve uses
normal-first vectors, through the explicit adapters in this module. Body
twists and Jacobian columns use Kamino's linear-first 6D convention.
"""

from __future__ import annotations

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact import solve_contact_coulomb_newton

__all__ = [
    "PROJECTION_STATUS_INVALID",
    "PROJECTION_STATUS_VALID",
    "ContactProjectionResult",
    "FrictionProjectionResult",
    "LimitProjectionResult",
    "compute_contact_delassus",
    "compute_limit_delassus",
    "convert_contact_matrix_normal_first_to_last",
    "convert_contact_matrix_normal_last_to_first",
    "convert_contact_vector_normal_first_to_last",
    "convert_contact_vector_normal_last_to_first",
    "project_contact_coulomb",
    "project_joint_friction",
    "project_limit_unilateral",
]

PROJECTION_STATUS_INVALID = 0
"""The local block or an input value was non-finite or non-positive."""

PROJECTION_STATUS_VALID = 1
"""The one-constraint update completed successfully."""

wp.set_module_options({"enable_backward": False})


@wp.struct
class ContactProjectionResult:
    """Output of one warm-started Coulomb contact update."""

    twist_first: vec6f
    """Updated linear-first twist of the first incident body."""
    twist_second: vec6f
    """Updated linear-first twist of the second incident body."""
    reaction: wp.vec3f
    """Updated normal-last contact impulse."""
    reaction_delta: wp.vec3f
    """Normal-last change from the warm-started impulse."""
    velocity: wp.vec3f
    """Resulting normal-last contact velocity, including the supplied bias."""
    delassus: wp.mat33f
    """Full normal-last local Delassus block."""
    status: wp.int32
    """One of the ``PROJECTION_STATUS_*`` values."""


@wp.struct
class LimitProjectionResult:
    """Output of one warm-started scalar unilateral-limit update."""

    twist_first: vec6f
    """Updated linear-first twist of the first incident body."""
    twist_second: vec6f
    """Updated linear-first twist of the second incident body."""
    reaction: wp.float32
    """Updated nonnegative limit impulse."""
    reaction_delta: wp.float32
    """Change from the warm-started limit impulse."""
    velocity: wp.float32
    """Resulting scalar limit velocity, including the supplied bias."""
    delassus: wp.float32
    """Positive scalar local Delassus coefficient."""
    status: wp.int32
    """One of the ``PROJECTION_STATUS_*`` values."""


@wp.struct
class FrictionProjectionResult:
    """Output of one warm-started scalar joint-friction update."""

    twist_first: vec6f
    twist_second: vec6f
    reaction: wp.float32
    """Updated signed friction impulse."""
    reaction_delta: wp.float32
    velocity: wp.float32
    delassus: wp.float32
    status: wp.int32


@wp.struct
class ContactProjectionData:
    """Contact data that is invariant throughout a body-space solve."""

    delassus: wp.mat33f
    delassus_normal_first: wp.mat33f
    status: wp.int32


@wp.struct
class ContactSweepResult:
    """Minimal result returned by the prepared contact sweep primitive."""

    twist_first: vec6f
    twist_second: vec6f
    reaction: wp.vec3f
    velocity: wp.vec3f
    status: wp.int32


@wp.func
def convert_contact_vector_normal_last_to_first(value: wp.vec3f) -> wp.vec3f:
    """Convert ``(tangent_x, tangent_y, normal_z)`` to normal-first order."""
    return wp.vec3f(value[2], value[0], value[1])


@wp.func
def convert_contact_vector_normal_first_to_last(value: wp.vec3f) -> wp.vec3f:
    """Convert ``(normal, tangent_x, tangent_y)`` to normal-last order."""
    return wp.vec3f(value[1], value[2], value[0])


@wp.func
def convert_contact_matrix_normal_last_to_first(value: wp.mat33f) -> wp.mat33f:
    """Permute both axes of a contact matrix from normal-last to normal-first."""
    result = wp.mat33f(0.0)
    result[0, 0] = value[2, 2]
    result[0, 1] = value[2, 0]
    result[0, 2] = value[2, 1]
    result[1, 0] = value[0, 2]
    result[1, 1] = value[0, 0]
    result[1, 2] = value[0, 1]
    result[2, 0] = value[1, 2]
    result[2, 1] = value[1, 0]
    result[2, 2] = value[1, 1]
    return result


@wp.func
def convert_contact_matrix_normal_first_to_last(value: wp.mat33f) -> wp.mat33f:
    """Permute both axes of a contact matrix from normal-first to normal-last."""
    result = wp.mat33f(0.0)
    result[0, 0] = value[1, 1]
    result[0, 1] = value[1, 2]
    result[0, 2] = value[1, 0]
    result[1, 0] = value[2, 1]
    result[1, 1] = value[2, 2]
    result[1, 2] = value[2, 0]
    result[2, 0] = value[0, 1]
    result[2, 1] = value[0, 2]
    result[2, 2] = value[0, 0]
    return result


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
        for col in range(3):
            finite = finite and wp.isfinite(value[row, col])
    return finite


@wp.func
def _is_finite_mat36(value: mat36f) -> wp.bool:
    finite = wp.bool(True)
    for row in range(3):
        for col in range(6):
            finite = finite and wp.isfinite(value[row, col])
    return finite


@wp.func
def _is_finite_mat66(value: mat66f) -> wp.bool:
    finite = wp.bool(True)
    for row in range(6):
        for col in range(6):
            finite = finite and wp.isfinite(value[row, col])
    return finite


@wp.func
def _is_symmetric_positive_definite_mat33(value: wp.mat33f) -> wp.bool:
    if not _is_finite_mat33(value):
        return False

    scale = wp.float32(1.0)
    asymmetry = wp.float32(0.0)
    for row in range(3):
        for col in range(3):
            scale = wp.max(scale, wp.abs(value[row, col]))
            asymmetry = wp.max(asymmetry, wp.abs(value[row, col] - value[col, row]))
    if asymmetry > 1.0e-5 * scale:
        return False

    pivot0 = value[0, 0]
    if pivot0 <= 0.0:
        return False
    lower10 = value[1, 0] / wp.sqrt(pivot0)
    lower20 = value[2, 0] / wp.sqrt(pivot0)
    pivot1 = value[1, 1] - lower10 * lower10
    if pivot1 <= 0.0:
        return False
    lower21 = (value[2, 1] - lower20 * lower10) / wp.sqrt(pivot1)
    pivot2 = value[2, 2] - lower20 * lower20 - lower21 * lower21
    return wp.isfinite(pivot2) and pivot2 > 0.0


@wp.func
def compute_contact_delassus(
    jacobian_first: mat36f,
    inverse_weight_first: mat66f,
    jacobian_second: mat36f,
    inverse_weight_second: mat66f,
) -> wp.mat33f:
    """Compute the full normal-last local contact Delassus block."""
    return jacobian_first @ inverse_weight_first @ wp.transpose(
        jacobian_first
    ) + jacobian_second @ inverse_weight_second @ wp.transpose(jacobian_second)


@wp.func
def prepare_contact_coulomb(
    jacobian_first: mat36f,
    inverse_weight_first: mat66f,
    jacobian_second: mat36f,
    inverse_weight_second: mat66f,
    velocity_bias: wp.vec3f,
    friction: wp.float32,
) -> ContactProjectionData:
    """Validate fixed inputs and prepare the normal-first contact block."""
    result = ContactProjectionData()
    delassus = compute_contact_delassus(
        jacobian_first,
        inverse_weight_first,
        jacobian_second,
        inverse_weight_second,
    )
    result.delassus = delassus
    result.delassus_normal_first = convert_contact_matrix_normal_last_to_first(delassus)
    result.status = PROJECTION_STATUS_INVALID
    if (
        _is_finite_mat36(jacobian_first)
        and _is_finite_mat66(inverse_weight_first)
        and _is_finite_mat36(jacobian_second)
        and _is_finite_mat66(inverse_weight_second)
        and _is_finite_vec3(velocity_bias)
        and wp.isfinite(friction)
        and friction >= 0.0
        and _is_symmetric_positive_definite_mat33(delassus)
    ):
        result.status = PROJECTION_STATUS_VALID
    return result


@wp.func
def compute_limit_delassus(
    jacobian_first: vec6f,
    inverse_weight_first: mat66f,
    jacobian_second: vec6f,
    inverse_weight_second: mat66f,
) -> wp.float32:
    """Compute the scalar local joint-limit Delassus coefficient."""
    return wp.dot(jacobian_first, inverse_weight_first @ jacobian_first) + wp.dot(
        jacobian_second, inverse_weight_second @ jacobian_second
    )


@wp.func
def _make_invalid_contact_projection_result(
    twist_first: vec6f,
    twist_second: vec6f,
    reaction: wp.vec3f,
    velocity: wp.vec3f,
    delassus: wp.mat33f,
) -> ContactProjectionResult:
    result = ContactProjectionResult()
    result.twist_first = twist_first
    result.twist_second = twist_second
    result.reaction = reaction
    result.reaction_delta = wp.vec3f(0.0)
    result.velocity = velocity
    result.delassus = delassus
    result.status = PROJECTION_STATUS_INVALID
    return result


@wp.func
def project_contact_coulomb(
    jacobian_first: mat36f,
    inverse_weight_first: mat66f,
    twist_first: vec6f,
    jacobian_second: mat36f,
    inverse_weight_second: mat66f,
    twist_second: vec6f,
    velocity_bias: wp.vec3f,
    reaction_old: wp.vec3f,
    friction: wp.float32,
) -> ContactProjectionResult:
    """Perform one warm-started matrix-free Coulomb contact update.

    The supplied body twists must already contain the velocity contribution of
    ``reaction_old``. The update subtracts that local contribution before
    solving, then applies only the impulse change to the incident body twists.
    A static side is represented by a zero Jacobian and zero inverse weight.

    Args:
        jacobian_first: Normal-last ``3 x 6`` Jacobian for the first body.
        inverse_weight_first: First body block ``W^-1``.
        twist_first: Current first body twist, including the warm start.
        jacobian_second: Normal-last ``3 x 6`` Jacobian for the second body.
        inverse_weight_second: Second body block ``W^-1``.
        twist_second: Current second body twist, including the warm start.
        velocity_bias: Frozen normal-last constraint velocity bias.
        reaction_old: Warm-started normal-last contact impulse.
        friction: Nonnegative isotropic friction coefficient.

    Returns:
        Updated twists, impulse, constraint velocity, local block, and status.
    """
    current_velocity = jacobian_first @ twist_first + jacobian_second @ twist_second + velocity_bias
    delassus = compute_contact_delassus(jacobian_first, inverse_weight_first, jacobian_second, inverse_weight_second)
    if (
        not _is_finite_mat36(jacobian_first)
        or not _is_finite_mat66(inverse_weight_first)
        or not _is_finite_vec6(twist_first)
        or not _is_finite_mat36(jacobian_second)
        or not _is_finite_mat66(inverse_weight_second)
        or not _is_finite_vec6(twist_second)
        or not _is_finite_vec3(velocity_bias)
        or not _is_finite_vec3(reaction_old)
        or not wp.isfinite(friction)
        or friction < 0.0
        or not _is_symmetric_positive_definite_mat33(delassus)
    ):
        return _make_invalid_contact_projection_result(
            twist_first, twist_second, reaction_old, current_velocity, delassus
        )

    free_velocity = current_velocity - delassus @ reaction_old
    reaction_new_normal_first = solve_contact_coulomb_newton(
        convert_contact_matrix_normal_last_to_first(delassus),
        convert_contact_vector_normal_last_to_first(free_velocity),
        friction,
    )
    reaction_new = convert_contact_vector_normal_first_to_last(reaction_new_normal_first)
    reaction_delta = reaction_new - reaction_old
    twist_first_new = twist_first + inverse_weight_first @ (wp.transpose(jacobian_first) @ reaction_delta)
    twist_second_new = twist_second + inverse_weight_second @ (wp.transpose(jacobian_second) @ reaction_delta)
    velocity_new = jacobian_first @ twist_first_new + jacobian_second @ twist_second_new + velocity_bias
    if (
        not _is_finite_vec3(reaction_new)
        or not _is_finite_vec3(reaction_delta)
        or not _is_finite_vec6(twist_first_new)
        or not _is_finite_vec6(twist_second_new)
        or not _is_finite_vec3(velocity_new)
    ):
        return _make_invalid_contact_projection_result(
            twist_first, twist_second, reaction_old, current_velocity, delassus
        )

    result = ContactProjectionResult()
    result.twist_first = twist_first_new
    result.twist_second = twist_second_new
    result.reaction = reaction_new
    result.reaction_delta = reaction_delta
    result.velocity = velocity_new
    result.delassus = delassus
    result.status = PROJECTION_STATUS_VALID
    return result


@wp.func
def project_contact_coulomb_prepared(
    jacobian_first: mat36f,
    inverse_weight_first: mat66f,
    twist_first: vec6f,
    jacobian_second: mat36f,
    inverse_weight_second: mat66f,
    twist_second: vec6f,
    velocity_bias: wp.vec3f,
    reaction_old: wp.vec3f,
    friction: wp.float32,
    delassus: wp.mat33f,
    delassus_normal_first: wp.mat33f,
) -> ContactSweepResult:
    """Perform the changing work of a prepared Coulomb contact update."""
    current_velocity = jacobian_first @ twist_first + jacobian_second @ twist_second + velocity_bias
    reaction_old_normal_first = convert_contact_vector_normal_last_to_first(reaction_old)
    free_velocity_normal_first = convert_contact_vector_normal_last_to_first(current_velocity - delassus @ reaction_old)

    result = ContactSweepResult()
    result.twist_first = twist_first
    result.twist_second = twist_second
    result.reaction = reaction_old
    result.velocity = current_velocity
    result.status = PROJECTION_STATUS_INVALID
    if not _is_finite_vec3(free_velocity_normal_first):
        return result

    reaction_new_normal_first = solve_contact_coulomb_newton(
        delassus_normal_first,
        free_velocity_normal_first,
        friction,
    )
    reaction_delta_normal_first = reaction_new_normal_first - reaction_old_normal_first
    reaction_new = convert_contact_vector_normal_first_to_last(reaction_new_normal_first)
    reaction_delta = convert_contact_vector_normal_first_to_last(reaction_delta_normal_first)
    twist_first_new = twist_first + inverse_weight_first @ (wp.transpose(jacobian_first) @ reaction_delta)
    twist_second_new = twist_second + inverse_weight_second @ (wp.transpose(jacobian_second) @ reaction_delta)
    velocity_new = jacobian_first @ twist_first_new + jacobian_second @ twist_second_new + velocity_bias
    if not _is_finite_vec3(reaction_new) or not _is_finite_vec3(velocity_new):
        return result

    result.twist_first = twist_first_new
    result.twist_second = twist_second_new
    result.reaction = reaction_new
    result.velocity = velocity_new
    result.status = PROJECTION_STATUS_VALID
    return result


@wp.func
def _make_invalid_limit_projection_result(
    twist_first: vec6f,
    twist_second: vec6f,
    reaction: wp.float32,
    velocity: wp.float32,
    delassus: wp.float32,
) -> LimitProjectionResult:
    result = LimitProjectionResult()
    result.twist_first = twist_first
    result.twist_second = twist_second
    result.reaction = reaction
    result.reaction_delta = 0.0
    result.velocity = velocity
    result.delassus = delassus
    result.status = PROJECTION_STATUS_INVALID
    return result


@wp.func
def project_limit_unilateral(
    jacobian_first: vec6f,
    inverse_weight_first: mat66f,
    twist_first: vec6f,
    jacobian_second: vec6f,
    inverse_weight_second: mat66f,
    twist_second: vec6f,
    velocity_bias: wp.float32,
    reaction_old: wp.float32,
) -> LimitProjectionResult:
    """Perform one warm-started scalar unilateral joint-limit update."""
    current_velocity = wp.dot(jacobian_first, twist_first) + wp.dot(jacobian_second, twist_second) + velocity_bias
    delassus = compute_limit_delassus(jacobian_first, inverse_weight_first, jacobian_second, inverse_weight_second)
    if (
        not _is_finite_vec6(jacobian_first)
        or not _is_finite_mat66(inverse_weight_first)
        or not _is_finite_vec6(twist_first)
        or not _is_finite_vec6(jacobian_second)
        or not _is_finite_mat66(inverse_weight_second)
        or not _is_finite_vec6(twist_second)
        or not wp.isfinite(velocity_bias)
        or not wp.isfinite(reaction_old)
        or not wp.isfinite(delassus)
        or delassus <= 0.0
    ):
        return _make_invalid_limit_projection_result(
            twist_first, twist_second, reaction_old, current_velocity, delassus
        )

    free_velocity = current_velocity - delassus * reaction_old
    reaction_new = wp.max(-free_velocity / delassus, 0.0)
    reaction_delta = reaction_new - reaction_old
    twist_first_new = twist_first + (inverse_weight_first @ jacobian_first) * reaction_delta
    twist_second_new = twist_second + (inverse_weight_second @ jacobian_second) * reaction_delta
    velocity_new = wp.dot(jacobian_first, twist_first_new) + wp.dot(jacobian_second, twist_second_new) + velocity_bias
    if (
        not wp.isfinite(reaction_new)
        or not wp.isfinite(reaction_delta)
        or not _is_finite_vec6(twist_first_new)
        or not _is_finite_vec6(twist_second_new)
        or not wp.isfinite(velocity_new)
    ):
        return _make_invalid_limit_projection_result(
            twist_first, twist_second, reaction_old, current_velocity, delassus
        )

    result = LimitProjectionResult()
    result.twist_first = twist_first_new
    result.twist_second = twist_second_new
    result.reaction = reaction_new
    result.reaction_delta = reaction_delta
    result.velocity = velocity_new
    result.delassus = delassus
    result.status = PROJECTION_STATUS_VALID
    return result


@wp.func
def project_joint_friction(
    jacobian_first: vec6f,
    inverse_weight_first: mat66f,
    twist_first: vec6f,
    jacobian_second: vec6f,
    inverse_weight_second: mat66f,
    twist_second: vec6f,
    reaction_old: wp.float32,
    impulse_bound: wp.float32,
) -> FrictionProjectionResult:
    """Perform one warm-started scalar dry-friction update.

    The signed reaction is projected onto the impulse interval
    ``[-impulse_bound, impulse_bound]``. This represents one constraint per
    frictional joint DOF, including both slip directions and stiction.
    """
    current_velocity = wp.dot(jacobian_first, twist_first) + wp.dot(jacobian_second, twist_second)
    delassus = compute_limit_delassus(jacobian_first, inverse_weight_first, jacobian_second, inverse_weight_second)
    result = FrictionProjectionResult()
    result.twist_first = twist_first
    result.twist_second = twist_second
    result.reaction = reaction_old
    result.reaction_delta = 0.0
    result.velocity = current_velocity
    result.delassus = delassus
    result.status = PROJECTION_STATUS_INVALID
    if (
        not _is_finite_vec6(jacobian_first)
        or not _is_finite_mat66(inverse_weight_first)
        or not _is_finite_vec6(twist_first)
        or not _is_finite_vec6(jacobian_second)
        or not _is_finite_mat66(inverse_weight_second)
        or not _is_finite_vec6(twist_second)
        or not wp.isfinite(reaction_old)
        or not wp.isfinite(impulse_bound)
        or impulse_bound < 0.0
        or not wp.isfinite(delassus)
        or delassus <= 0.0
    ):
        return result

    free_velocity = current_velocity - delassus * reaction_old
    reaction_new = wp.clamp(-free_velocity / delassus, -impulse_bound, impulse_bound)
    reaction_delta = reaction_new - reaction_old
    twist_first_new = twist_first + (inverse_weight_first @ jacobian_first) * reaction_delta
    twist_second_new = twist_second + (inverse_weight_second @ jacobian_second) * reaction_delta
    velocity_new = wp.dot(jacobian_first, twist_first_new) + wp.dot(jacobian_second, twist_second_new)
    if (
        not wp.isfinite(reaction_new)
        or not wp.isfinite(reaction_delta)
        or not _is_finite_vec6(twist_first_new)
        or not _is_finite_vec6(twist_second_new)
        or not wp.isfinite(velocity_new)
    ):
        return result

    result.twist_first = twist_first_new
    result.twist_second = twist_second_new
    result.reaction = reaction_new
    result.reaction_delta = reaction_delta
    result.velocity = velocity_new
    result.status = PROJECTION_STATUS_VALID
    return result
