# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local isotropic Coulomb-contact numerical primitives.

This module is deliberately independent of Kamino containers. Its contact
vectors use the normal-first convention ``(normal, tangent_0, tangent_1)`` and
the rows and columns of every Delassus block follow the same order. Kamino's
contact containers use a normal-last convention, so the backend must perform
an explicit permutation at this module's boundary.

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
def _solve_mat22(
    a00: wp.float32,
    a01: wp.float32,
    a10: wp.float32,
    a11: wp.float32,
    b0: wp.float32,
    b1: wp.float32,
) -> wp.vec2f:
    det = a00 * a11 - a01 * a10
    inv_det = 1.0 / det
    return wp.vec2f((a11 * b0 - a01 * b1) * inv_det, (a00 * b1 - a10 * b0) * inv_det)


@wp.func
def _compute_sliding_root_data(
    tangent00: wp.float32,
    tangent01: wp.float32,
    tangent10: wp.float32,
    tangent11: wp.float32,
    tangent_rhs: wp.vec2f,
    normal_tangent: wp.vec2f,
    normal_rhs: wp.float32,
    normal_delassus: wp.float32,
    friction: wp.float32,
    alpha: wp.float32,
) -> wp.vec4f:
    shifted00 = tangent00 + alpha
    shifted11 = tangent11 + alpha
    s = _solve_mat22(
        shifted00,
        tangent01,
        tangent10,
        shifted11,
        tangent_rhs[0],
        tangent_rhs[1],
    )
    t = _solve_mat22(shifted00, tangent01, tangent10, shifted11, s[0], s[1])
    s_norm = wp.length(s)
    value = s_norm - friction * (wp.dot(normal_tangent, s) - normal_rhs) / normal_delassus
    derivative = wp.float32(0.0)
    if s_norm > 1.0e-30:
        derivative = -(wp.dot(s, t)) / s_norm + friction * wp.dot(normal_tangent, t) / normal_delassus
    return wp.vec4f(value, derivative, s[0], s[1])


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
    normal_delassus = delassus[0, 0]
    normal_rhs = free_velocity[0]

    # These branches also avoid touching an unused, potentially ill-conditioned
    # tangential block for separating or frictionless contacts.
    if normal_rhs >= 0.0:
        return wp.vec3f(0.0, 0.0, 0.0)
    if friction <= 0.0:
        return wp.vec3f(-normal_rhs / normal_delassus, 0.0, 0.0)

    normal_tangent = wp.vec2f(delassus[1, 0], delassus[2, 0])
    tangent00 = delassus[1, 1] - normal_tangent[0] * normal_tangent[0] / normal_delassus
    tangent01 = delassus[1, 2] - normal_tangent[0] * normal_tangent[1] / normal_delassus
    tangent10 = delassus[2, 1] - normal_tangent[1] * normal_tangent[0] / normal_delassus
    tangent11 = delassus[2, 2] - normal_tangent[1] * normal_tangent[1] / normal_delassus
    tangent_rhs = wp.vec2f(free_velocity[1], free_velocity[2]) - (normal_rhs / normal_delassus) * normal_tangent

    unshifted = _solve_mat22(
        tangent00,
        tangent01,
        tangent10,
        tangent11,
        tangent_rhs[0],
        tangent_rhs[1],
    )
    value_at_zero = wp.length(unshifted) - friction * (wp.dot(normal_tangent, unshifted) - normal_rhs) / normal_delassus
    if value_at_zero <= 1.0e-7:
        tangent_reaction = -unshifted
        normal_reaction = -(wp.dot(normal_tangent, tangent_reaction) + normal_rhs) / normal_delassus
        return wp.vec3f(normal_reaction, tangent_reaction[0], tangent_reaction[1])

    # Normalizing alpha by the Schur-complement scale keeps the fixed root
    # tolerances useful across contact blocks with different magnitudes.
    alpha_scale = wp.max(wp.abs(tangent00), wp.abs(tangent01))
    alpha_scale = wp.max(alpha_scale, wp.abs(tangent10))
    alpha_scale = wp.max(alpha_scale, wp.abs(tangent11))
    alpha_scale = wp.max(alpha_scale, 1.0e-20)

    lower = wp.float32(0.0)
    upper = wp.float32(1.0)
    last_s = unshifted
    for _ in range(64):
        upper_data = _compute_sliding_root_data(
            tangent00,
            tangent01,
            tangent10,
            tangent11,
            tangent_rhs,
            normal_tangent,
            normal_rhs,
            normal_delassus,
            friction,
            upper * alpha_scale,
        )
        if upper_data[0] <= 0.0:
            break
        upper = upper * 2.0

    alpha = 0.5 * (lower + upper)
    for _ in range(12):
        root_data = _compute_sliding_root_data(
            tangent00,
            tangent01,
            tangent10,
            tangent11,
            tangent_rhs,
            normal_tangent,
            normal_rhs,
            normal_delassus,
            friction,
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
    normal_reaction = -(wp.dot(normal_tangent, tangent_reaction) + normal_rhs) / normal_delassus
    return wp.vec3f(normal_reaction, tangent_reaction[0], tangent_reaction[1])


@wp.func
def _solve_contact_coulomb_newton_instrumented(
    delassus: wp.mat33f,
    free_velocity: wp.vec3f,
    friction: wp.float32,
) -> _CoulombSolveResult:
    result = _CoulombSolveResult()
    result.reaction = wp.vec3f(0.0)
    result.branch = wp.int32(0)
    result.expansion_iterations = wp.int32(0)
    result.root_iterations = wp.int32(0)
    result.bracketed = wp.int32(1)
    result.converged = wp.int32(1)

    normal_delassus = delassus[0, 0]
    normal_rhs = free_velocity[0]
    if normal_rhs >= 0.0:
        return result
    if friction <= 0.0:
        result.branch = wp.int32(1)
        result.reaction = wp.vec3f(-normal_rhs / normal_delassus, 0.0, 0.0)
        return result

    normal_tangent = wp.vec2f(delassus[1, 0], delassus[2, 0])
    tangent00 = delassus[1, 1] - normal_tangent[0] * normal_tangent[0] / normal_delassus
    tangent01 = delassus[1, 2] - normal_tangent[0] * normal_tangent[1] / normal_delassus
    tangent10 = delassus[2, 1] - normal_tangent[1] * normal_tangent[0] / normal_delassus
    tangent11 = delassus[2, 2] - normal_tangent[1] * normal_tangent[1] / normal_delassus
    tangent_rhs = wp.vec2f(free_velocity[1], free_velocity[2]) - (normal_rhs / normal_delassus) * normal_tangent

    unshifted = _solve_mat22(
        tangent00,
        tangent01,
        tangent10,
        tangent11,
        tangent_rhs[0],
        tangent_rhs[1],
    )
    value_at_zero = wp.length(unshifted) - friction * (wp.dot(normal_tangent, unshifted) - normal_rhs) / normal_delassus
    if value_at_zero <= 1.0e-7:
        tangent_reaction = -unshifted
        normal_reaction = -(wp.dot(normal_tangent, tangent_reaction) + normal_rhs) / normal_delassus
        result.branch = wp.int32(2)
        result.reaction = wp.vec3f(normal_reaction, tangent_reaction[0], tangent_reaction[1])
        return result

    result.branch = wp.int32(3)
    result.bracketed = wp.int32(0)
    result.converged = wp.int32(0)
    alpha_scale = wp.max(wp.abs(tangent00), wp.abs(tangent01))
    alpha_scale = wp.max(alpha_scale, wp.abs(tangent10))
    alpha_scale = wp.max(alpha_scale, wp.abs(tangent11))
    alpha_scale = wp.max(alpha_scale, 1.0e-20)

    lower = wp.float32(0.0)
    upper = wp.float32(1.0)
    last_s = unshifted
    for _ in range(64):
        result.expansion_iterations += 1
        upper_data = _compute_sliding_root_data(
            tangent00,
            tangent01,
            tangent10,
            tangent11,
            tangent_rhs,
            normal_tangent,
            normal_rhs,
            normal_delassus,
            friction,
            upper * alpha_scale,
        )
        if upper_data[0] <= 0.0:
            result.bracketed = wp.int32(1)
            break
        upper = upper * 2.0

    alpha = 0.5 * (lower + upper)
    for _ in range(12):
        result.root_iterations += 1
        root_data = _compute_sliding_root_data(
            tangent00,
            tangent01,
            tangent10,
            tangent11,
            tangent_rhs,
            normal_tangent,
            normal_rhs,
            normal_delassus,
            friction,
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
    normal_reaction = -(wp.dot(normal_tangent, tangent_reaction) + normal_rhs) / normal_delassus
    result.reaction = wp.vec3f(normal_reaction, tangent_reaction[0], tangent_reaction[1])
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
