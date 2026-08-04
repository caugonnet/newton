# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the `kamino.core.joints` module"""

import unittest

import numpy as np
import warp as wp

from newton import JointType
from newton._src.solvers.kamino._src.core.joints import JointDescriptor, JointDoFType
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestCoreJoints(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_joint_dof_type_enum(self):
        doftype = JointDoFType.REVOLUTE

        # Optional verbose output
        msg.info(f"doftype: {doftype}")
        msg.info(f"doftype.value: {doftype.value}")
        msg.info(f"doftype.name: {doftype.name}")
        msg.info(f"doftype.num_cts: {doftype.num_cts}")
        msg.info(f"doftype.num_dofs: {doftype.num_dofs}")
        msg.info(f"doftype.cts_axes: {doftype.cts_axes}")
        msg.info(f"doftype.dofs_axes: {doftype.dofs_axes}")

        # Check the enum values
        self.assertEqual(doftype.value, JointDoFType.REVOLUTE)
        self.assertEqual(doftype.name, "REVOLUTE")
        self.assertEqual(doftype.num_cts, 5)
        self.assertEqual(doftype.num_dofs, 1)
        self.assertEqual(doftype.cts_axes, (0, 1, 2, 4, 5))
        self.assertEqual(doftype.dofs_axes, (3,))

    def test_cable_dof_type_retains_material_storage_without_joint_rows(self):
        """Retain cable material slots without ordinary joint constraints."""
        dof_type = JointDoFType.CABLE

        self.assertEqual(dof_type.num_coords, 4)
        self.assertEqual(dof_type.num_dofs, 4)
        self.assertEqual(dof_type.num_cts, 0)
        self.assertEqual(dof_type.cts_axes, [])
        self.assertEqual(dof_type.dofs_axes, [])
        self.assertEqual(
            JointDoFType.from_newton(
                JointType.CABLE,
                q_count=4,
                qd_count=4,
                dof_dim=(2, 2),
                limit_lower=np.full(4, -1.0e10),
                limit_upper=np.full(4, 1.0e10),
            ),
            JointDoFType.CABLE,
        )

    def test_cable_dof_type_rejects_malformed_storage_layout(self):
        """Reject cable layouts that do not match the builder representation."""
        with self.assertRaisesRegex(ValueError, "expected q_count=4"):
            JointDoFType.from_newton(
                JointType.CABLE,
                q_count=2,
                qd_count=2,
                dof_dim=(1, 1),
                limit_lower=np.full(2, -1.0e10),
                limit_upper=np.full(2, 1.0e10),
            )

    def test_joint_effort_limit_accepts_zero_and_positive_infinity(self):
        """Accept nonnegative finite and positively infinite effort limits."""
        self.assertEqual(JointDescriptor(name="joint", dof_type=JointDoFType.REVOLUTE, tau_j_max=0.0).tau_j_max, [0.0])
        self.assertEqual(
            JointDescriptor(name="joint", dof_type=JointDoFType.REVOLUTE, tau_j_max=float("inf")).tau_j_max,
            [float("inf")],
        )

    def test_joint_effort_limit_rejects_negative_and_nan(self):
        """Reject negative and NaN effort limits."""
        for value in (-1.0, float("-inf"), float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "joint effort limit"):
                JointDescriptor(name="joint", dof_type=JointDoFType.REVOLUTE, tau_j_max=value)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
