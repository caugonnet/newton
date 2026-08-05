# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX local Coulomb-contact primitives."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.lox import (
    compute_contact_scaled_alart_curnier_residual,
    solve_contact_coulomb_newton,
)
from newton._src.solvers.kamino._src.solvers.lox.contact import (
    _solve_contact_coulomb_newton_normal_last,
    _solve_contact_coulomb_newton_normal_last_instrumented,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


@wp.kernel
def _solve_contacts(
    delassus: wp.array[wp.mat33f],
    free_velocity: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
    residual: wp.array[wp.vec3f],
):
    contact = wp.tid()
    reaction_i = solve_contact_coulomb_newton(delassus[contact], free_velocity[contact], friction[contact])
    velocity_i = delassus[contact] @ reaction_i + free_velocity[contact]
    reaction[contact] = reaction_i
    velocity[contact] = velocity_i
    residual[contact] = compute_contact_scaled_alart_curnier_residual(
        delassus[contact], reaction_i, velocity_i, friction[contact]
    )


@wp.kernel
def _solve_contacts_normal_last(
    delassus: wp.array[wp.mat33f],
    free_velocity: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    reaction: wp.array[wp.vec3f],
):
    contact = wp.tid()
    reaction[contact] = _solve_contact_coulomb_newton_normal_last(
        delassus[contact], free_velocity[contact], friction[contact]
    )


@wp.kernel
def _solve_contacts_normal_last_instrumented(
    delassus: wp.array[wp.mat33f],
    free_velocity: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    reaction: wp.array[wp.vec3f],
    branch: wp.array[wp.int32],
):
    contact = wp.tid()
    result = _solve_contact_coulomb_newton_normal_last_instrumented(
        delassus[contact], free_velocity[contact], friction[contact]
    )
    reaction[contact] = result.reaction
    branch[contact] = result.branch


class TestLOXContact(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def _solve(
        self,
        delassus: np.ndarray,
        free_velocity: np.ndarray,
        friction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delassus_wp = wp.array(delassus, dtype=wp.mat33f, device=self.device)
        free_velocity_wp = wp.array(free_velocity, dtype=wp.vec3f, device=self.device)
        friction_wp = wp.array(friction, dtype=wp.float32, device=self.device)
        reaction_wp = wp.empty(len(friction), dtype=wp.vec3f, device=self.device)
        velocity_wp = wp.empty(len(friction), dtype=wp.vec3f, device=self.device)
        residual_wp = wp.empty(len(friction), dtype=wp.vec3f, device=self.device)

        wp.launch(
            _solve_contacts,
            dim=len(friction),
            inputs=[delassus_wp, free_velocity_wp, friction_wp],
            outputs=[reaction_wp, velocity_wp, residual_wp],
            device=self.device,
        )
        return reaction_wp.numpy(), velocity_wp.numpy(), residual_wp.numpy()

    def _solve_normal_last(
        self,
        delassus: np.ndarray,
        free_velocity: np.ndarray,
        friction: np.ndarray,
        *,
        instrumented: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        delassus_wp = wp.array(delassus, dtype=wp.mat33f, device=self.device)
        free_velocity_wp = wp.array(free_velocity, dtype=wp.vec3f, device=self.device)
        friction_wp = wp.array(friction, dtype=wp.float32, device=self.device)
        reaction_wp = wp.empty(len(friction), dtype=wp.vec3f, device=self.device)
        if not instrumented:
            wp.launch(
                _solve_contacts_normal_last,
                dim=len(friction),
                inputs=[delassus_wp, free_velocity_wp, friction_wp],
                outputs=[reaction_wp],
                device=self.device,
            )
            return reaction_wp.numpy(), None

        branch_wp = wp.empty(len(friction), dtype=wp.int32, device=self.device)
        wp.launch(
            _solve_contacts_normal_last_instrumented,
            dim=len(friction),
            inputs=[delassus_wp, free_velocity_wp, friction_wp],
            outputs=[reaction_wp, branch_wp],
            device=self.device,
        )
        return reaction_wp.numpy(), branch_wp.numpy()

    def test_separating_contact(self):
        delassus = np.asarray([np.diag([2.0, 3.0, 4.0])], dtype=np.float32)
        free_velocity = np.asarray([[0.25, -4.0, 2.0]], dtype=np.float32)
        friction = np.asarray([0.7], dtype=np.float32)

        reaction, velocity, residual = self._solve(delassus, free_velocity, friction)

        np.testing.assert_array_equal(reaction[0], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(velocity[0], free_velocity[0], rtol=0.0, atol=1.0e-7)
        self.assertLess(np.linalg.norm(residual[0]), 1.0e-6)

    def test_frictionless_contact_skips_degenerate_tangent_block(self):
        # Only W_N is used in the frictionless branch, which protects the
        # prototype while preprocessing regularizes active frictional blocks.
        delassus = np.asarray([np.diag([2.0, 0.0, 0.0])], dtype=np.float32)
        free_velocity = np.asarray([[-1.0, 3.0, -2.0]], dtype=np.float32)
        friction = np.asarray([0.0], dtype=np.float32)

        reaction, velocity, residual = self._solve(delassus, free_velocity, friction)

        np.testing.assert_allclose(reaction[0], [0.5, 0.0, 0.0], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(velocity[0], [0.0, 3.0, -2.0], rtol=0.0, atol=1.0e-7)
        self.assertTrue(np.all(np.isfinite(residual[0])))
        self.assertLess(np.linalg.norm(residual[0]), 1.0e-6)

    def test_sticking_contact(self):
        delassus = np.asarray([np.diag([2.0, 3.0, 4.0])], dtype=np.float32)
        free_velocity = np.asarray([[-1.0, 0.2, -0.1]], dtype=np.float32)
        friction = np.asarray([0.8], dtype=np.float32)
        expected_reaction = np.asarray([0.5, -0.2 / 3.0, 0.025], dtype=np.float32)

        reaction, velocity, residual = self._solve(delassus, free_velocity, friction)

        np.testing.assert_allclose(reaction[0], expected_reaction, rtol=2.0e-6, atol=2.0e-7)
        np.testing.assert_allclose(velocity[0], np.zeros(3), rtol=0.0, atol=2.0e-7)
        self.assertLess(np.linalg.norm(reaction[0, 1:]), friction[0] * reaction[0, 0])
        self.assertLess(np.linalg.norm(residual[0]), 1.0e-6)

    def test_sliding_contact(self):
        delassus = np.asarray([np.diag([2.0, 1.0, 1.0])], dtype=np.float32)
        free_velocity = np.asarray([[-1.0, 2.0, 0.0]], dtype=np.float32)
        friction = np.asarray([0.5], dtype=np.float32)
        expected_reaction = np.asarray([0.5, -0.25, 0.0], dtype=np.float32)

        reaction, velocity, residual = self._solve(delassus, free_velocity, friction)

        np.testing.assert_allclose(reaction[0], expected_reaction, rtol=2.0e-6, atol=2.0e-7)
        np.testing.assert_allclose(velocity[0], [0.0, 1.75, 0.0], rtol=2.0e-6, atol=2.0e-7)
        self.assertAlmostEqual(np.linalg.norm(reaction[0, 1:]), friction[0] * reaction[0, 0], delta=2.0e-7)
        self.assertLess(np.dot(reaction[0, 1:], velocity[0, 1:]), 0.0)
        self.assertLess(np.linalg.norm(residual[0]), 1.0e-6)

    def test_strongly_coupled_full_block(self):
        delassus_i = np.asarray(
            [
                [2.0, 0.35, -0.15],
                [0.35, 1.7, 0.25],
                [-0.15, 0.25, 0.8],
            ],
            dtype=np.float32,
        )
        friction_i = np.float32(0.5)
        expected_reaction = np.asarray([0.45, -0.18, -0.135], dtype=np.float32)
        expected_velocity = np.asarray([0.0, 0.28, 0.21], dtype=np.float32)
        free_velocity_i = expected_velocity - delassus_i @ expected_reaction

        reaction, velocity, residual = self._solve(
            delassus_i[None, ...], free_velocity_i[None, ...], friction_i[None, ...]
        )

        np.testing.assert_allclose(reaction[0], expected_reaction, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(velocity[0], expected_velocity, rtol=2.0e-5, atol=2.0e-6)
        self.assertLess(np.linalg.norm(residual[0]), 2.0e-6)

    def test_bracketed_bisection_fallback(self):
        """Keep the bounded bisection fallback stable for a difficult contact."""
        # This coupled problem rejects multiple pure Newton steps and does not
        # reach the root tolerance within the bounded local solve.
        delassus_i = np.asarray(
            [
                [3.05015024, -2.21409240, -0.05816527],
                [-2.21409240, 7.38549845, -0.55674596],
                [-0.05816527, -0.55674596, 0.53603802],
            ],
            dtype=np.float32,
        )
        free_velocity_i = np.asarray([-0.97200092, -3.11677977, -4.67238632], dtype=np.float32)
        friction_i = np.float32(5.40898023)

        reaction, velocity, residual = self._solve(
            delassus_i[None, ...], free_velocity_i[None, ...], friction_i[None, ...]
        )

        self.assertTrue(np.all(np.isfinite(reaction[0])))
        self.assertAlmostEqual(velocity[0, 0], 0.0, delta=1.0e-5)
        self.assertLess(np.dot(reaction[0, 1:], velocity[0, 1:]), 0.0)
        self.assertLess(np.linalg.norm(residual[0]), 3.0e-2)

    def test_nearly_singular_regularized_block(self):
        delassus_i = np.diag([1.0, 1.0e-6, 2.0e-6]).astype(np.float32)
        expected_reaction = np.asarray([1.0, -0.1, 0.1], dtype=np.float32)
        free_velocity_i = -(delassus_i @ expected_reaction)
        friction_i = np.float32(0.5)

        reaction, velocity, residual = self._solve(
            delassus_i[None, ...], free_velocity_i[None, ...], friction_i[None, ...]
        )

        np.testing.assert_allclose(reaction[0], expected_reaction, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(velocity[0], np.zeros(3), rtol=0.0, atol=2.0e-7)
        self.assertTrue(np.all(np.isfinite(reaction[0])))
        self.assertTrue(np.all(np.isfinite(residual[0])))
        self.assertLess(np.linalg.norm(residual[0]), 1.0e-6)

    def test_normal_last_wrapper_matches_all_solver_branches(self):
        """Match normal-first results and diagnostics across every solver branch."""
        delassus = np.asarray(
            [
                np.diag([2.0, 3.0, 4.0]),
                np.diag([2.0, 0.0, 0.0]),
                np.diag([2.0, 3.0, 4.0]),
                np.diag([2.0, 1.0, 1.0]),
            ],
            dtype=np.float32,
        )
        free_velocity = np.asarray(
            [
                [0.25, -4.0, 2.0],
                [-1.0, 3.0, -2.0],
                [-1.0, 0.2, -0.1],
                [-1.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        friction = np.asarray([0.7, 0.0, 0.8, 0.5], dtype=np.float32)
        permutation = (1, 2, 0)
        delassus_normal_last = delassus[:, permutation][:, :, permutation]
        velocity_normal_last = free_velocity[:, permutation]

        expected, _, _ = self._solve(delassus, free_velocity, friction)
        actual, _ = self._solve_normal_last(delassus_normal_last, velocity_normal_last, friction)
        instrumented, branches = self._solve_normal_last(
            delassus_normal_last,
            velocity_normal_last,
            friction,
            instrumented=True,
        )

        np.testing.assert_array_equal(actual, expected[:, permutation])
        np.testing.assert_array_equal(instrumented, actual)
        np.testing.assert_array_equal(branches, [0, 1, 2, 3])

    def test_normal_last_wrapper_matches_random_spd_blocks(self):
        """Match normal-first results for randomized coupled SPD contact blocks."""
        rng = np.random.default_rng(20260805)
        count = 512
        factors = rng.normal(size=(count, 3, 3)).astype(np.float32)
        delassus = factors @ np.transpose(factors, (0, 2, 1))
        delassus += 0.2 * np.identity(3, dtype=np.float32)[None, :, :]
        free_velocity = rng.normal(size=(count, 3)).astype(np.float32)
        free_velocity[:, 0] -= 0.35
        friction = rng.uniform(0.0, 2.0, size=count).astype(np.float32)
        friction[::11] = 0.0
        free_velocity[::13, 0] = np.abs(free_velocity[::13, 0])
        permutation = (1, 2, 0)

        expected, _, _ = self._solve(delassus, free_velocity, friction)
        actual, _ = self._solve_normal_last(
            delassus[:, permutation][:, :, permutation],
            free_velocity[:, permutation],
            friction,
        )

        np.testing.assert_array_equal(actual, expected[:, permutation])


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
