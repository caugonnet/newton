# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX unilateral velocity targets."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.lox import (
    compute_contact_velocity_target,
    compute_limit_velocity_target,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


@wp.kernel
def _compute_targets(
    distance: wp.array[wp.float32],
    previous_velocity: wp.array[wp.float32],
    restitution: wp.array[wp.float32],
    time_step: wp.float32,
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
    impact_velocity_threshold: wp.float32,
    limit_violation: wp.array[wp.float32],
    contact_target: wp.array[wp.float32],
    limit_target: wp.array[wp.float32],
):
    index = wp.tid()
    contact_target[index] = compute_contact_velocity_target(
        distance[index],
        previous_velocity[index],
        restitution[index],
        time_step,
        stabilization_fraction,
        dead_zone,
        impact_velocity_threshold,
    )
    limit_target[index] = compute_limit_velocity_target(limit_violation[index], time_step, stabilization_fraction)


class TestLOXBias(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_contact_and_limit_velocity_targets(self):
        distance = np.asarray([-0.011, 0.011, 0.0005, -0.011, 0.011, 0.0], dtype=np.float32)
        previous_velocity = np.asarray([0.0, 0.0, -2.0, -0.2, -2.0, -0.0005], dtype=np.float32)
        restitution = np.asarray([0.5] * 6, dtype=np.float32)
        limit_violation = np.asarray([-0.01, 0.01, 0.0, -0.02, 0.02, -0.005], dtype=np.float32)
        contact_target = wp.empty(len(distance), dtype=wp.float32, device=self.device)
        limit_target = wp.empty(len(distance), dtype=wp.float32, device=self.device)

        wp.launch(
            _compute_targets,
            dim=len(distance),
            inputs=[
                wp.array(distance, dtype=wp.float32, device=self.device),
                wp.array(previous_velocity, dtype=wp.float32, device=self.device),
                wp.array(restitution, dtype=wp.float32, device=self.device),
                0.01,
                0.2,
                0.001,
                0.001,
                wp.array(limit_violation, dtype=wp.float32, device=self.device),
            ],
            outputs=[contact_target, limit_target],
            device=self.device,
        )

        # Penetration recovery, speculative approach, restitution, max(recovery, bounce),
        # no speculative restitution, and impact-threshold suppression.
        expected_contact = np.asarray([0.2, -1.0, 1.0, 0.2, -1.0, 0.0], dtype=np.float32)
        expected_limit = np.asarray([0.2, 0.0, 0.0, 0.4, 0.0, 0.1], dtype=np.float32)
        np.testing.assert_allclose(contact_target.numpy(), expected_contact, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(limit_target.numpy(), expected_limit, rtol=0.0, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
