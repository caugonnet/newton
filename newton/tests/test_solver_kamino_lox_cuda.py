# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph-capture quality gate for the Kamino LOX solver."""

import unittest

from newton._src.solvers.kamino.tests.test_solver_kamino_lox import TestSolverKaminoLOX
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


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load the focused CUDA graph-capture check into the main test suite."""
    del loader, tests, pattern
    return unittest.TestSuite(
        (
            TestSolverKaminoLOX("test_cuda_graph_capture_uses_conditional_loop"),
            TestLOXDeformableSystem("test_capture_frozen_assembly"),
            TestLOXDeformableLinearSolve("test_capture_factorization_and_batched_cr"),
            TestLOXDeformableLinearSolve("test_capture_mixed_direct_and_iterative_components"),
            TestLOXDeformableLinearSolve("test_capture_two_level_preconditioner_setup_and_apply"),
            TestLOXDeformableLinearSolve("test_capture_two_level_with_mixed_direct_and_iterative_components"),
            TestLOXDeformableContact("test_capture_contact_prepare_projection_and_residuals"),
            TestLOXDeformableIntegration("test_capture_public_pure_cloth_step"),
            TestLOXDeformableIntegration("test_capture_step_in_place_matches_ping_pong_with_nonzero_body_com"),
        )
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
