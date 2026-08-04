# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical primitives for the LOX rigid-contact backend."""

from .adapter import LOXKaminoAdapter
from .bias import compute_contact_velocity_target, compute_limit_velocity_target
from .contact import (
    compute_contact_scaled_alart_curnier_residual,
    project_contact_coulomb_cone,
    solve_contact_coulomb_newton,
)
from .integration import accept_projected_body_state, integrate_projected_body_poses
from .iteration import SplittingState
from .metric import (
    METRIC_STATUS_INVALID,
    METRIC_STATUS_VALID,
    ConstraintMetricResult,
    compute_mass_split_metric,
)
from .problem import (
    PrimalRowContribution,
    compute_augmented_joint_multiplier,
    compute_augmented_joint_row,
    compute_body_explicit_wrench,
    compute_body_inertial_system,
    compute_dynamic_joint_row,
    compute_velocity_distance,
    make_spatial_mass_matrix,
)
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    ContactProjectionResult,
    FrictionProjectionResult,
    LimitProjectionResult,
    compute_contact_delassus,
    compute_limit_delassus,
    convert_contact_matrix_normal_first_to_last,
    convert_contact_matrix_normal_last_to_first,
    convert_contact_vector_normal_first_to_last,
    convert_contact_vector_normal_last_to_first,
    project_contact_coulomb,
    project_joint_friction,
    project_limit_unilateral,
)
from .solver import (
    LOX_STATUS_ACTIVE,
    LOX_STATUS_CONVERGED,
    LOX_STATUS_FAILED,
    LOX_STATUS_ITERATION_LIMIT,
    LOXSolver,
)
from .sweep import (
    compute_projection_residuals,
    prepare_jacobi_projection_data,
    project_constraints_jacobi,
    project_constraints_sequential,
)
from .system import BatchedPrimalBodySystem
from .weight import (
    BODY_WEIGHT_BETA_DEFAULT,
    BODY_WEIGHT_SIGMA_DEFAULT,
    BODY_WEIGHT_STATUS_INVALID,
    BODY_WEIGHT_STATUS_REGULARIZED,
    BODY_WEIGHT_STATUS_VALID,
    BodyWeightAnisotropicResult,
    BodyWeightResult,
    compute_body_weight_anisotropic,
    compute_body_weight_mass_proportional,
)

__all__ = [
    "BODY_WEIGHT_BETA_DEFAULT",
    "BODY_WEIGHT_SIGMA_DEFAULT",
    "BODY_WEIGHT_STATUS_INVALID",
    "BODY_WEIGHT_STATUS_REGULARIZED",
    "BODY_WEIGHT_STATUS_VALID",
    "LOX_STATUS_ACTIVE",
    "LOX_STATUS_CONVERGED",
    "LOX_STATUS_FAILED",
    "LOX_STATUS_ITERATION_LIMIT",
    "METRIC_STATUS_INVALID",
    "METRIC_STATUS_VALID",
    "PROJECTION_STATUS_INVALID",
    "PROJECTION_STATUS_VALID",
    "BatchedPrimalBodySystem",
    "BodyWeightAnisotropicResult",
    "BodyWeightResult",
    "ConstraintMetricResult",
    "ContactProjectionResult",
    "FrictionProjectionResult",
    "LOXKaminoAdapter",
    "LOXSolver",
    "LimitProjectionResult",
    "PrimalRowContribution",
    "SplittingState",
    "accept_projected_body_state",
    "compute_augmented_joint_multiplier",
    "compute_augmented_joint_row",
    "compute_body_explicit_wrench",
    "compute_body_inertial_system",
    "compute_body_weight_anisotropic",
    "compute_body_weight_mass_proportional",
    "compute_contact_delassus",
    "compute_contact_scaled_alart_curnier_residual",
    "compute_contact_velocity_target",
    "compute_dynamic_joint_row",
    "compute_limit_delassus",
    "compute_limit_velocity_target",
    "compute_mass_split_metric",
    "compute_projection_residuals",
    "compute_velocity_distance",
    "convert_contact_matrix_normal_first_to_last",
    "convert_contact_matrix_normal_last_to_first",
    "convert_contact_vector_normal_first_to_last",
    "convert_contact_vector_normal_last_to_first",
    "integrate_projected_body_poses",
    "make_spatial_mass_matrix",
    "prepare_jacobi_projection_data",
    "project_constraints_jacobi",
    "project_constraints_sequential",
    "project_contact_coulomb",
    "project_contact_coulomb_cone",
    "project_joint_friction",
    "project_limit_unilateral",
    "solve_contact_coulomb_newton",
]
