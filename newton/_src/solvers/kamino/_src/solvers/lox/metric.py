# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small mass-split constraint metrics for the LOX backend."""

from __future__ import annotations

import warp as wp

from ...core.types import mat66f, vec6f

__all__ = [
    "METRIC_STATUS_INVALID",
    "METRIC_STATUS_VALID",
    "ConstraintMetricResult",
    "compute_mass_split_metric",
]

METRIC_STATUS_INVALID = 0
"""The active block is non-finite, nonpositive, or singular."""

METRIC_STATUS_VALID = 1
"""The active block was inverted without regularization."""

wp.set_module_options({"enable_backward": False})


@wp.struct
class ConstraintMetricResult:
    """Physical and mass-split metric matrices for one constraint entity."""

    delassus: mat66f
    """Symmetric physical Delassus block."""
    inverse: mat66f
    """Inverse of the multiplicity-scaled Delassus block."""
    penalty: mat66f
    """Scaled inverse block used by the augmented system."""
    status: wp.int32
    """One of the ``METRIC_STATUS_*`` values."""


@wp.func
def _invalid_metric(delassus: mat66f) -> ConstraintMetricResult:
    result = ConstraintMetricResult()
    result.delassus = delassus
    result.inverse = mat66f(0.0)
    result.penalty = mat66f(0.0)
    result.status = METRIC_STATUS_INVALID
    return result


@wp.func
def compute_mass_split_metric(
    physical_delassus: mat66f,
    active_dimension: wp.int32,
    multiplicity: wp.int32,
    penalty_scale: wp.float32,
    pivot_tolerance: wp.float32 = 1.0e-12,
) -> ConstraintMetricResult:
    """Invert one leading Delassus block after a uniform scale.

    The inverse is computed with a fixed-size Cholesky factorization. Padding
    outside the leading active block is always zero. Keeping the failure
    policy here localizes the deferred near-singular-block decision. For
    per-body mass splitting, the caller first assembles
    ``sum_b n_b J_b M_b^-1 J_b^T`` and passes ``multiplicity=1``.

    Args:
        physical_delassus: Padded physical Delassus block.
        active_dimension: Leading active dimension in ``[1, 6]``.
        multiplicity: Additional uniform scale applied to the complete block.
        penalty_scale: Explicit scale applied to the inverse block.
        pivot_tolerance: Smallest accepted Cholesky pivot.

    Returns:
        The symmetric Delassus, its mass-split inverse, scaled penalty, and
        validation status.
    """
    symmetric = mat66f(0.0)
    if (
        active_dimension < 1
        or active_dimension > 6
        or multiplicity < 1
        or not wp.isfinite(penalty_scale)
        or penalty_scale <= 0.0
        or not wp.isfinite(pivot_tolerance)
        or pivot_tolerance <= 0.0
    ):
        return _invalid_metric(symmetric)

    for row in range(6):
        for col in range(6):
            if row < active_dimension and col < active_dimension:
                first = physical_delassus[row, col]
                second = physical_delassus[col, row]
                if not wp.isfinite(first) or not wp.isfinite(second):
                    return _invalid_metric(symmetric)
                symmetric[row, col] = 0.5 * (first + second)

    lower = mat66f(0.0)
    for row in range(6):
        if row < active_dimension:
            for col in range(6):
                if col <= row and col < active_dimension:
                    value = wp.float32(multiplicity) * symmetric[row, col]
                    for inner in range(6):
                        if inner < col:
                            value -= lower[row, inner] * lower[col, inner]
                    if row == col:
                        if not wp.isfinite(value) or value <= pivot_tolerance:
                            return _invalid_metric(symmetric)
                        lower[row, col] = wp.sqrt(value)
                    else:
                        lower[row, col] = value / lower[col, col]

    inverse = mat66f(0.0)
    for column in range(6):
        if column < active_dimension:
            forward = vec6f(0.0)
            for row in range(6):
                if row < active_dimension:
                    value = wp.float32(1.0) if row == column else wp.float32(0.0)
                    for inner in range(6):
                        if inner < row:
                            value -= lower[row, inner] * forward[inner]
                    forward[row] = value / lower[row, row]

            backward = vec6f(0.0)
            reverse = int(0)
            while reverse < 6:
                row = 5 - reverse
                if row < active_dimension:
                    value = forward[row]
                    for inner in range(6):
                        if inner > row and inner < active_dimension:
                            value -= lower[inner, row] * backward[inner]
                    backward[row] = value / lower[row, row]
                reverse += 1
            for row in range(6):
                if row < active_dimension:
                    inverse[row, column] = backward[row]

    inverse = 0.5 * (inverse + wp.transpose(inverse))
    result = ConstraintMetricResult()
    result.delassus = symmetric
    result.inverse = inverse
    result.penalty = penalty_scale * inverse
    result.status = METRIC_STATUS_VALID
    return result
