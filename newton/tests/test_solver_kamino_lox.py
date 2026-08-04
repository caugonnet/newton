# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform quality gates for the Kamino LOX solver."""

import unittest

from newton._src.solvers.kamino.tests.test_solver_kamino_lox import TestSolverKaminoLOX

_LOX_QUALITY_TESTS = (
    "test_binary_joint_friction_sparse_matches_dense_and_counts_once_per_body",
    "test_binary_cable_balances_wrenches_and_respects_enabled",
    "test_box_on_plane_projects_detected_contact",
    "test_cable_accepts_implicit_single_world",
    "test_cable_bend_and_twist_restore_rotation",
    "test_cable_world_parent_stretch_and_damping",
    "test_cartpole_projects_detected_joint_limit",
    "test_cartpole_sustained_joint_force_remains_bounded",
    "test_contact_metrics_use_projected_twist",
    "test_direct_structural_config_builds_solver",
    "test_free_fall_advances_projected_velocity_and_pose",
    "test_free_fall_produces_solution_metrics",
    "test_joint_damping_is_implicit_in_the_smooth_row",
    "test_joint_friction_slips_then_sticks_with_one_bounded_row",
    "test_product_space_structural_split_hinged_contact",
    "test_revolute_joint_step_updates_structural_multipliers",
    "test_sparse_jacobian_direct_structural_step",
    "test_sparse_jacobian_matches_dense_step",
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load a focused cross-platform subset into the main test suite."""
    del loader, tests, pattern
    return unittest.TestSuite(TestSolverKaminoLOX(name) for name in _LOX_QUALITY_TESTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
