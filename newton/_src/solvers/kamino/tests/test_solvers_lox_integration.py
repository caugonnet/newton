# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for projected LOX pose integration."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.solvers.lox.integration import (
    accept_projected_body_state,
    integrate_projected_body_poses,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


class TestLOXIntegration(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_integrates_heterogeneous_world_timesteps_from_fixed_pose(self):
        body_world = wp.array([0, 1], dtype=wp.int32, device=self.device)
        world_time_step = wp.array([0.1, 0.25], dtype=wp.float32, device=self.device)
        pose_previous = wp.array(
            [
                wp.transformf(wp.vec3f(1.0, 2.0, 3.0), wp.quat_identity()),
                wp.transformf(wp.vec3f(-1.0, 0.0, 2.0), wp.quat_identity()),
            ],
            dtype=wp.transformf,
            device=self.device,
        )
        projected_twist = wp.array(
            [[2.0, -1.0, 0.5, 0.0, 0.0, 1.0], [-0.4, 0.8, 0.0, 0.0, 0.0, 0.0]],
            dtype=vec6f,
            device=self.device,
        )
        pose_candidate = wp.empty(2, dtype=wp.transformf, device=self.device)

        integrate_projected_body_poses(body_world, world_time_step, pose_previous, projected_twist, pose_candidate)
        result = pose_candidate.numpy()
        np.testing.assert_allclose(result[0, :3], [1.2, 1.9, 3.05], rtol=0.0, atol=2.0e-7)
        np.testing.assert_allclose(result[1, :3], [-1.1, 0.2, 2.0], rtol=0.0, atol=2.0e-7)
        expected_rotation = np.asarray([0.0, 0.0, np.sin(0.05), np.cos(0.05)], dtype=np.float32)
        np.testing.assert_allclose(result[0, 3:], expected_rotation, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(result[1, 3:], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=2.0e-7)

    def test_accepts_candidate_state_only_for_successful_worlds(self):
        body_world = wp.array([0, 1], dtype=wp.int32, device=self.device)
        world_accepted = wp.array([True, False], dtype=wp.bool, device=self.device)
        pose_candidate = wp.array(
            [wp.transformf(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)] * 2,
            dtype=wp.transformf,
            device=self.device,
        )
        twist_candidate = wp.array([[1.0] * 6, [2.0] * 6], dtype=vec6f, device=self.device)
        pose_accepted = wp.array(
            [wp.transformf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)] * 2,
            dtype=wp.transformf,
            device=self.device,
        )
        twist_accepted = wp.zeros(2, dtype=vec6f, device=self.device)

        accept_projected_body_state(
            body_world,
            world_accepted,
            pose_candidate,
            twist_candidate,
            pose_accepted,
            twist_accepted,
        )

        np.testing.assert_allclose(pose_accepted.numpy()[0], pose_candidate.numpy()[0], atol=0.0)
        np.testing.assert_allclose(pose_accepted.numpy()[1, :3], 0.0, atol=0.0)
        np.testing.assert_allclose(twist_accepted.numpy()[0], 1.0, atol=0.0)
        np.testing.assert_allclose(twist_accepted.numpy()[1], 0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
