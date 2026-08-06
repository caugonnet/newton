# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local isotropic Coulomb-contact numerical primitives.

This module is deliberately independent of Kamino containers. Its public
contact primitives use the normal-first convention ``(normal, tangent_0,
tangent_1)``. Internal normal-last wrappers let Kamino containers use their
native layout without materializing permuted vectors or Delassus blocks.

The local solve expects a symmetric positive-definite ``3 x 3`` Delassus
block. Degenerate blocks must be regularized during contact preprocessing.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "CoulombSolveStatistics",
    "compute_contact_scaled_alart_curnier_residual",
    "project_contact_coulomb_cone",
    "solve_contact_coulomb_newton",
]

wp.set_module_options({"enable_backward": False})


class CoulombSolveStatistics:
    """Aggregate debug statistics for LOX local Coulomb solves."""

    SOURCE_RIGID = 0
    SOURCE_RIGID_DEFORMABLE = 1
    SOURCE_NAMES = ("rigid", "rigid_deformable")
    BRANCH_NAMES = ("separating", "frictionless", "sticking", "sliding")

    def __init__(self, device: wp.DeviceLike):
        self.device = wp.get_device(device)
        self.branch_histogram = wp.zeros((2, 4), dtype=wp.int64, device=self.device)
        self.expansion_histogram = wp.zeros((2, 65), dtype=wp.int64, device=self.device)
        self.root_histogram = wp.zeros((2, 13), dtype=wp.int64, device=self.device)
        self.failure_counts = wp.zeros((2, 2), dtype=wp.int64, device=self.device)

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.branch_histogram.zero_()
        self.expansion_histogram.zero_()
        self.root_histogram.zero_()
        self.failure_counts.zero_()


@wp.struct
class _CoulombSolveResult:
    reaction: wp.vec3f
    branch: wp.int32
    expansion_iterations: wp.int32
    root_iterations: wp.int32
    bracketed: wp.int32
    converged: wp.int32


@wp.func
def _solve_symmetric_mat22(
    a00: wp.float32,
    a01: wp.float32,
    a11: wp.float32,
    b0: wp.float32,
    b1: wp.float32,
) -> wp.vec2f:
    det = a00 * a11 - a01 * a01
    inv_det = 1.0 / det
    return wp.vec2f((a11 * b0 - a01 * b1) * inv_det, (a00 * b1 - a01 * b0) * inv_det)


@wp.func
def _compute_sliding_root_value(
    tangent00: wp.float32,
    tangent01: wp.float32,
    tangent11: wp.float32,
    tangent_rhs: wp.vec2f,
    friction_normal_tangent: wp.vec2f,
    friction_normal_rhs: wp.float32,
    alpha: wp.float32,
) -> wp.float32:
    shifted00 = tangent00 + alpha
    shifted11 = tangent11 + alpha
    s = _solve_symmetric_mat22(
        shifted00,
        tangent01,
        shifted11,
        tangent_rhs[0],
        tangent_rhs[1],
    )
    return wp.length(s) - wp.dot(friction_normal_tangent, s) + friction_normal_rhs


@wp.func
def _compute_sliding_root_data(
    tangent00: wp.float32,
    tangent01: wp.float32,
    tangent11: wp.float32,
    tangent_rhs: wp.vec2f,
    friction_normal_tangent: wp.vec2f,
    friction_normal_rhs: wp.float32,
    alpha: wp.float32,
) -> wp.vec4f:
    shifted00 = tangent00 + alpha
    shifted11 = tangent11 + alpha
    determinant = shifted00 * shifted11 - tangent01 * tangent01
    inverse_determinant = 1.0 / determinant
    s = wp.vec2f(
        (shifted11 * tangent_rhs[0] - tangent01 * tangent_rhs[1]) * inverse_determinant,
        (shifted00 * tangent_rhs[1] - tangent01 * tangent_rhs[0]) * inverse_determinant,
    )
    t = wp.vec2f(
        (shifted11 * s[0] - tangent01 * s[1]) * inverse_determinant,
        (shifted00 * s[1] - tangent01 * s[0]) * inverse_determinant,
    )
    s_norm = wp.length(s)
    value = s_norm - wp.dot(friction_normal_tangent, s) + friction_normal_rhs
    derivative = wp.float32(0.0)
    if s_norm > 1.0e-30:
        derivative = -(wp.dot(s, t)) / s_norm + wp.dot(friction_normal_tangent, t)
    return wp.vec4f(value, derivative, s[0], s[1])


@wp.func
def _solve_contact_coulomb_newton_components(
    normal_delassus: wp.float32,
    normal_tangent: wp.vec2f,
    tangent_delassus00: wp.float32,
    tangent_delassus01: wp.float32,
    tangent_delassus11: wp.float32,
    normal_rhs: wp.float32,
    tangent_rhs_raw: wp.vec2f,
    friction: wp.float32,
) -> wp.vec3f:
    """Solve from layout-independent normal and tangential components."""
    # These branches also avoid touching an unused, potentially ill-conditioned
    # tangential block for separating or frictionless contacts.
    if normal_rhs >= 0.0:
        return wp.vec3f(0.0, 0.0, 0.0)
    if friction <= 0.0:
        return wp.vec3f(-normal_rhs / normal_delassus, 0.0, 0.0)

    inverse_normal_delassus = 1.0 / normal_delassus
    tangent00 = tangent_delassus00 - normal_tangent[0] * normal_tangent[0] * inverse_normal_delassus
    tangent01 = tangent_delassus01 - normal_tangent[0] * normal_tangent[1] * inverse_normal_delassus
    tangent11 = tangent_delassus11 - normal_tangent[1] * normal_tangent[1] * inverse_normal_delassus
    tangent_rhs = tangent_rhs_raw - (normal_rhs * inverse_normal_delassus) * normal_tangent
    friction_over_normal = friction * inverse_normal_delassus
    friction_normal_tangent = friction_over_normal * normal_tangent
    friction_normal_rhs = friction_over_normal * normal_rhs

    unshifted = _solve_symmetric_mat22(
        tangent00,
        tangent01,
        tangent11,
        tangent_rhs[0],
        tangent_rhs[1],
    )
    value_at_zero = wp.length(unshifted) - wp.dot(friction_normal_tangent, unshifted) + friction_normal_rhs
    if value_at_zero <= 1.0e-7:
        tangent_reaction = -unshifted
        normal_reaction = -(wp.dot(normal_tangent, tangent_reaction) + normal_rhs) * inverse_normal_delassus
        return wp.vec3f(normal_reaction, tangent_reaction[0], tangent_reaction[1])

    # Normalizing alpha by the Schur-complement scale keeps the fixed root
    # tolerances useful across contact blocks with different magnitudes.
    alpha_scale = wp.max(wp.abs(tangent00), wp.abs(tangent01))
    alpha_scale = wp.max(alpha_scale, wp.abs(tangent11))
    alpha_scale = wp.max(alpha_scale, 1.0e-20)

    lower = wp.float32(0.0)
    upper = wp.float32(1.0)
    last_s = unshifted
    for _ in range(64):
        upper_value = _compute_sliding_root_value(
            tangent00,
            tangent01,
            tangent11,
            tangent_rhs,
            friction_normal_tangent,
            friction_normal_rhs,
            upper * alpha_scale,
        )
        if upper_value <= 0.0:
            break
        upper = upper * 2.0

    alpha = 0.5 * (lower + upper)
    for _ in range(12):
        root_data = _compute_sliding_root_data(
            tangent00,
            tangent01,
            tangent11,
            tangent_rhs,
            friction_normal_tangent,
            friction_normal_rhs,
            alpha * alpha_scale,
        )
        value = root_data[0]
        derivative = root_data[1] * alpha_scale
        last_s = wp.vec2f(root_data[2], root_data[3])

        if wp.abs(value) <= 1.0e-7 or wp.abs(upper - lower) <= 1.0e-7 * (1.0 + upper):
            break

        if value > 0.0:
            lower = alpha
        else:
            upper = alpha

        width = upper - lower
        next_alpha = 0.5 * (lower + upper)
        if derivative != 0.0 and wp.isfinite(derivative):
            newton_alpha = alpha - value / derivative
            if (
                newton_alpha > lower
                and newton_alpha < upper
                and wp.abs(newton_alpha - alpha) > 1.0e-7 * wp.max(1.0, width)
            ):
                next_alpha = newton_alpha
        alpha = next_alpha

    tangent_reaction = -last_s
    normal_reaction = -(wp.dot(normal_tangent, tangent_reaction) + normal_rhs) * inverse_normal_delassus
    return wp.vec3f(normal_reaction, tangent_reaction[0], tangent_reaction[1])


@wp.func
def solve_contact_coulomb_newton(
    delassus: wp.mat33f,
    free_velocity: wp.vec3f,
    friction: wp.float32,
) -> wp.vec3f:
    """Solve one normal-first isotropic Coulomb contact.

    The returned impulse ``reaction`` satisfies the contact law for
    ``velocity = delassus @ reaction + free_velocity``. Sliding contacts are
    reduced to a scalar root and solved by bracketed Newton. Every rejected or
    unusable Newton step falls back to bisection of the current bracket.

    Args:
        delassus: Symmetric positive-definite local Delassus block.
        free_velocity: Contact velocity before applying the local impulse.
        friction: Nonnegative isotropic Coulomb friction coefficient.

    Returns:
        The normal-first contact impulse.
    """
    return _solve_contact_coulomb_newton_components(
        delassus[0, 0],
        wp.vec2f(delassus[1, 0], delassus[2, 0]),
        delassus[1, 1],
        delassus[1, 2],
        delassus[2, 2],
        free_velocity[0],
        wp.vec2f(free_velocity[1], free_velocity[2]),
        friction,
    )


@wp.func
def _solve_contact_coulomb_newton_normal_last(
    delassus: wp.mat33f,
    free_velocity: wp.vec3f,
    friction: wp.float32,
) -> wp.vec3f:
    reaction = _solve_contact_coulomb_newton_components(
        delassus[2, 2],
        wp.vec2f(delassus[0, 2], delassus[1, 2]),
        delassus[0, 0],
        delassus[0, 1],
        delassus[1, 1],
        free_velocity[2],
        wp.vec2f(free_velocity[0], free_velocity[1]),
        friction,
    )
    return wp.vec3f(reaction[1], reaction[2], reaction[0])


@wp.func
def _solve_contact_coulomb_newton_instrumented_components(
    normal_delassus: wp.float32,
    normal_tangent: wp.vec2f,
    tangent_delassus00: wp.float32,
    tangent_delassus01: wp.float32,
    tangent_delassus11: wp.float32,
    normal_rhs: wp.float32,
    tangent_rhs_raw: wp.vec2f,
    friction: wp.float32,
) -> _CoulombSolveResult:
    result = _CoulombSolveResult()
    result.reaction = wp.vec3f(0.0)
    result.branch = wp.int32(0)
    result.expansion_iterations = wp.int32(0)
    result.root_iterations = wp.int32(0)
    result.bracketed = wp.int32(1)
    result.converged = wp.int32(1)

    if normal_rhs >= 0.0:
        return result
    if friction <= 0.0:
        result.branch = wp.int32(1)
        result.reaction = wp.vec3f(-normal_rhs / normal_delassus, 0.0, 0.0)
        return result

    inverse_normal_delassus = 1.0 / normal_delassus
    tangent00 = tangent_delassus00 - normal_tangent[0] * normal_tangent[0] * inverse_normal_delassus
    tangent01 = tangent_delassus01 - normal_tangent[0] * normal_tangent[1] * inverse_normal_delassus
    tangent11 = tangent_delassus11 - normal_tangent[1] * normal_tangent[1] * inverse_normal_delassus
    tangent_rhs = tangent_rhs_raw - (normal_rhs * inverse_normal_delassus) * normal_tangent
    friction_over_normal = friction * inverse_normal_delassus
    friction_normal_tangent = friction_over_normal * normal_tangent
    friction_normal_rhs = friction_over_normal * normal_rhs

    unshifted = _solve_symmetric_mat22(
        tangent00,
        tangent01,
        tangent11,
        tangent_rhs[0],
        tangent_rhs[1],
    )
    value_at_zero = wp.length(unshifted) - wp.dot(friction_normal_tangent, unshifted) + friction_normal_rhs
    if value_at_zero <= 1.0e-7:
        tangent_reaction = -unshifted
        normal_reaction = -(wp.dot(normal_tangent, tangent_reaction) + normal_rhs) * inverse_normal_delassus
        result.branch = wp.int32(2)
        result.reaction = wp.vec3f(normal_reaction, tangent_reaction[0], tangent_reaction[1])
        return result

    result.branch = wp.int32(3)
    result.bracketed = wp.int32(0)
    result.converged = wp.int32(0)
    alpha_scale = wp.max(wp.abs(tangent00), wp.abs(tangent01))
    alpha_scale = wp.max(alpha_scale, wp.abs(tangent11))
    alpha_scale = wp.max(alpha_scale, 1.0e-20)

    lower = wp.float32(0.0)
    upper = wp.float32(1.0)
    last_s = unshifted
    for _ in range(64):
        result.expansion_iterations += 1
        upper_value = _compute_sliding_root_value(
            tangent00,
            tangent01,
            tangent11,
            tangent_rhs,
            friction_normal_tangent,
            friction_normal_rhs,
            upper * alpha_scale,
        )
        if upper_value <= 0.0:
            result.bracketed = wp.int32(1)
            break
        upper = upper * 2.0

    alpha = 0.5 * (lower + upper)
    for _ in range(12):
        result.root_iterations += 1
        root_data = _compute_sliding_root_data(
            tangent00,
            tangent01,
            tangent11,
            tangent_rhs,
            friction_normal_tangent,
            friction_normal_rhs,
            alpha * alpha_scale,
        )
        value = root_data[0]
        derivative = root_data[1] * alpha_scale
        last_s = wp.vec2f(root_data[2], root_data[3])

        if wp.abs(value) <= 1.0e-7 or wp.abs(upper - lower) <= 1.0e-7 * (1.0 + upper):
            result.converged = wp.int32(1)
            break

        if value > 0.0:
            lower = alpha
        else:
            upper = alpha

        width = upper - lower
        next_alpha = 0.5 * (lower + upper)
        if derivative != 0.0 and wp.isfinite(derivative):
            newton_alpha = alpha - value / derivative
            if (
                newton_alpha > lower
                and newton_alpha < upper
                and wp.abs(newton_alpha - alpha) > 1.0e-7 * wp.max(1.0, width)
            ):
                next_alpha = newton_alpha
        alpha = next_alpha

    tangent_reaction = -last_s
    normal_reaction = -(wp.dot(normal_tangent, tangent_reaction) + normal_rhs) * inverse_normal_delassus
    result.reaction = wp.vec3f(normal_reaction, tangent_reaction[0], tangent_reaction[1])
    return result


@wp.func
def _solve_contact_coulomb_newton_instrumented(
    delassus: wp.mat33f,
    free_velocity: wp.vec3f,
    friction: wp.float32,
) -> _CoulombSolveResult:
    return _solve_contact_coulomb_newton_instrumented_components(
        delassus[0, 0],
        wp.vec2f(delassus[1, 0], delassus[2, 0]),
        delassus[1, 1],
        delassus[1, 2],
        delassus[2, 2],
        free_velocity[0],
        wp.vec2f(free_velocity[1], free_velocity[2]),
        friction,
    )


@wp.func
def _solve_contact_coulomb_newton_normal_last_instrumented(
    delassus: wp.mat33f,
    free_velocity: wp.vec3f,
    friction: wp.float32,
) -> _CoulombSolveResult:
    result = _solve_contact_coulomb_newton_instrumented_components(
        delassus[2, 2],
        wp.vec2f(delassus[0, 2], delassus[1, 2]),
        delassus[0, 0],
        delassus[0, 1],
        delassus[1, 1],
        free_velocity[2],
        wp.vec2f(free_velocity[0], free_velocity[1]),
        friction,
    )
    reaction = result.reaction
    result.reaction = wp.vec3f(reaction[1], reaction[2], reaction[0])
    return result


@wp.func
def _record_coulomb_solve_statistics(
    result: _CoulombSolveResult,
    source: wp.int32,
    branch_histogram: wp.array2d[wp.int64],
    expansion_histogram: wp.array2d[wp.int64],
    root_histogram: wp.array2d[wp.int64],
    failure_counts: wp.array2d[wp.int64],
):
    wp.atomic_add(branch_histogram, source, result.branch, wp.int64(1))
    wp.atomic_add(expansion_histogram, source, result.expansion_iterations, wp.int64(1))
    wp.atomic_add(root_histogram, source, result.root_iterations, wp.int64(1))
    if result.bracketed == 0:
        wp.atomic_add(failure_counts, source, 0, wp.int64(1))
    if result.converged == 0:
        wp.atomic_add(failure_counts, source, 1, wp.int64(1))


@wp.func
def project_contact_coulomb_cone(value: wp.vec3f, friction: wp.float32) -> wp.vec3f:
    """Project a normal-first vector onto an isotropic Coulomb cone.

    Args:
        value: Normal-first vector to project.
        friction: Nonnegative isotropic Coulomb friction coefficient.

    Returns:
        The Euclidean projection of ``value``.
    """
    tangent_norm = wp.sqrt(value[1] * value[1] + value[2] * value[2])
    if friction * tangent_norm <= -value[0]:
        return wp.vec3f(0.0, 0.0, 0.0)
    if tangent_norm <= friction * value[0]:
        return value

    normal = (friction * tangent_norm + value[0]) / (friction * friction + 1.0)
    if tangent_norm > 0.0:
        return wp.vec3f(
            normal,
            friction * normal * value[1] / tangent_norm,
            friction * normal * value[2] / tangent_norm,
        )
    return wp.vec3f(normal, 0.0, 0.0)


@wp.func
def _compute_contact_delassus_scale(delassus: wp.mat33f) -> wp.float32:
    trace_scale = (wp.abs(delassus[0, 0]) + wp.abs(delassus[1, 1]) + wp.abs(delassus[2, 2])) / 3.0
    return wp.sqrt(wp.max(trace_scale, 1.0e-30))


@wp.func
def compute_contact_scaled_alart_curnier_residual(
    delassus: wp.mat33f,
    reaction: wp.vec3f,
    velocity: wp.vec3f,
    friction: wp.float32,
) -> wp.vec3f:
    """Compute the Delassus-scaled Alart--Curnier contact residual.

    Args:
        delassus: Symmetric positive-definite local Delassus block.
        reaction: Normal-first contact impulse.
        velocity: Resulting normal-first contact velocity.
        friction: Nonnegative isotropic Coulomb friction coefficient.

    Returns:
        The normal-first scaled natural-map residual.
    """
    scale = _compute_contact_delassus_scale(delassus)
    scaled_reaction = scale * reaction
    scaled_velocity = velocity / scale
    modified_velocity = wp.vec3f(
        scaled_velocity[0]
        + friction * wp.sqrt(scaled_velocity[1] * scaled_velocity[1] + scaled_velocity[2] * scaled_velocity[2]),
        scaled_velocity[1],
        scaled_velocity[2],
    )
    projected = project_contact_coulomb_cone(scaled_reaction - modified_velocity, friction)
    return scaled_reaction - projected
