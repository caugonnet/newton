# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform quality gates for the Kamino LOX cloth smooth system."""

import unittest

from newton._src.solvers.kamino.tests.test_solvers_lox_deformable_contact import (
    TestLOXDeformableContact,
)
from newton._src.solvers.kamino.tests.test_solvers_lox_deformable_integration import (
    TestLOXDeformableIntegration,
)
from newton._src.solvers.kamino.tests.test_solvers_lox_deformable_linear import (
    TestLOXDeformableLinearSolve,
)
from newton._src.solvers.kamino.tests.test_solvers_lox_deformable_system import TestLOXDeformableSystem

_LOX_CLOTH_QUALITY_TESTS = (
    "test_accept_pure_tetrahedral_deformable",
    "test_assemble_tetrahedron_against_dense_reference",
    "test_apply_triangle_drag_and_lift_explicitly",
    "test_assemble_triangle_against_dense_reference",
    "test_bending_force_matches_energy_gradient",
    "test_build_static_batched_topology",
    "test_deformable_dual_impulse_round_trips_state_order",
    "test_membrane_force_matches_energy_gradient",
    "test_pack_model_coloring_to_bound_ic0_depth",
    "test_pin_rows_and_columns",
    "test_replace_masked_triplet_values_between_steps",
    "test_replace_invalid_model_coloring_with_topology_coloring",
    "test_step_mixed_cloth_and_tetrahedron",
    "test_step_soft_grid_in_rigid_free_fall",
    "test_validate_tetrahedron_data_and_coloring",
    "test_validate_supported_and_unsupported_cloth",
)

_LOX_CLOTH_LINEAR_QUALITY_TESTS = (
    "test_apply_block_jacobi_against_dense_diagonal",
    "test_apply_incomplete_ldlt_against_dense_factor",
    "test_batch_small_components_with_body_cholesky",
    "test_bound_persistent_apply_by_world_size_and_block_width",
    "test_build_symbolic_ic_levels_and_match_filled_dense_factor",
    "test_capture_persistent_eligible_apply_as_levels",
    "test_capture_block_jacobi_and_batched_cr",
    "test_capture_mixed_direct_and_iterative_components",
    "test_form_scalar_consensus_weight_and_system_matrix",
    "test_match_persistent_and_level_incomplete_ldlt_apply",
    "test_reduce_one_step_cr_residual_with_ic1_fill",
    "test_reduce_residual_more_than_jacobi_preconditioning",
    "test_report_assembly_factorization_and_cr_counters",
    "test_report_preconditioner_regularization_and_failure",
    "test_reuse_factorization_and_warm_start",
    "test_select_persistent_apply_for_small_eager_cuda_system",
    "test_select_bounded_tiled_dot_for_large_single_batch",
    "test_solve_batched_system_and_mask_inactive_world",
    "test_split_small_direct_and_large_iterative_components",
)

_LOX_CLOTH_CONTACT_QUALITY_TESTS = (
    "test_accumulate_multiple_contacts_with_mass_split",
    "test_adapt_dynamic_rigid_frame_jacobian_and_bias",
    "test_adapt_particle_edge_and_face_coefficients",
    "test_compute_scalar_delassus_with_mass_split_and_pins",
    "test_filter_boundary_edge_with_outward_half_plane",
    "test_filter_full_surface_contacts_by_soft_normal_cones",
    "test_include_kinematic_collider_velocity_in_bias",
    "test_keep_close_contact_before_normal_cone_filtering",
    "test_keep_tetrahedral_particle_contact_without_surface_topology",
    "test_project_all_coulomb_branches_and_rotate",
    "test_reject_contact_with_only_pinned_nodes",
    "test_reject_malformed_cross_world_and_adapt_dynamic_rigid_records",
    "test_warm_start_and_compute_residuals",
)

_LOX_CLOTH_INTEGRATION_QUALITY_TESTS = (
    "test_accept_iteration_limit_output_and_persist_dual_impulse",
    "test_bypass_normal_cone_filter_for_close_self_contact",
    "test_capture_public_dynamic_rigid_cloth_contact_step",
    "test_capture_public_first_dynamic_rigid_cloth_contact_step",
    "test_capture_public_pure_cloth_step",
    "test_capture_step_in_place_matches_ping_pong_with_nonzero_body_com",
    "test_keep_boundary_corner_self_contact",
    "test_penetration_free_contact_derives_tetrahedral_surface_edges",
    "test_penetration_free_contact_truncates_edge_crossing",
    "test_penetration_free_contact_truncates_frozen_candidates",
    "test_penetration_free_contact_uses_cone_pruned_candidate",
    "test_register_dual_impulse_state_attributes",
    "test_reset_pure_cloth_state_and_warm_start",
    "test_reset_selected_cloth_arrays_in_masked_world",
    "test_step_applies_penetration_free_isotropic_bound",
    "test_step_dynamic_rigid_contact_updates_both_endpoints",
    "test_step_dynamic_rigid_contact_against_pinned_particle",
    "test_step_combined_rigid_and_cloth_self_contact",
    "test_step_hanging_stiff_cloth_preserves_pins",
    "test_step_ignores_fully_prescribed_rigid_contacts",
    "test_step_kinematic_contact_uses_body_origin_pose",
    "test_step_mixed_rigid_and_cloth_without_coupling",
    "test_step_multiworld_cloth_independently",
    "test_step_pure_cloth_free_fall_and_factor_once",
    "test_step_pure_cloth_self_contact",
    "test_step_pure_cloth_with_collision_pipeline_contacts",
    "test_step_static_contact_applies_isotropic_friction",
    "test_validate_cloth_backend_and_config",
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load the cross-platform cloth smooth-system and linear-solve tests."""
    del loader, tests, pattern
    return unittest.TestSuite(
        [
            *(TestLOXDeformableSystem(name) for name in _LOX_CLOTH_QUALITY_TESTS),
            *(TestLOXDeformableLinearSolve(name) for name in _LOX_CLOTH_LINEAR_QUALITY_TESTS),
            *(TestLOXDeformableContact(name) for name in _LOX_CLOTH_CONTACT_QUALITY_TESTS),
            *(TestLOXDeformableIntegration(name) for name in _LOX_CLOTH_INTEGRATION_QUALITY_TESTS),
        ]
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
