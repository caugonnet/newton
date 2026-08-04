# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX rigid-body weight primitives."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat66f
from newton._src.solvers.kamino._src.solvers.lox import (
    BODY_WEIGHT_STATUS_INVALID,
    BODY_WEIGHT_STATUS_REGULARIZED,
    BODY_WEIGHT_STATUS_VALID,
    compute_body_weight_anisotropic,
    compute_body_weight_mass_proportional,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


@wp.kernel
def _compute_body_weights_default(
    smooth_diagonal: wp.array[mat66f],
    mass: wp.array[wp.float32],
    inertia_world: wp.array[wp.mat33f],
    weight: wp.array[mat66f],
    inverse_weight: wp.array[mat66f],
    eta: wp.array[wp.float32],
    alpha: wp.array[wp.float32],
    status: wp.array[wp.int32],
):
    body = wp.tid()
    result = compute_body_weight_mass_proportional(smooth_diagonal[body], mass[body], inertia_world[body])
    weight[body] = result.weight
    inverse_weight[body] = result.inverse_weight
    eta[body] = result.eta
    alpha[body] = result.alpha
    status[body] = result.status


@wp.kernel
def _compute_body_weights_configured(
    smooth_diagonal: wp.array[mat66f],
    mass: wp.array[wp.float32],
    inertia_world: wp.array[wp.mat33f],
    sigma: wp.float32,
    beta: wp.float32,
    mass_floor: wp.float32,
    inertia_floor: wp.float32,
    eta_floor: wp.float32,
    symmetry_tolerance: wp.float32,
    weight: wp.array[mat66f],
    inverse_weight: wp.array[mat66f],
    eta: wp.array[wp.float32],
    alpha: wp.array[wp.float32],
    status: wp.array[wp.int32],
):
    body = wp.tid()
    result = compute_body_weight_mass_proportional(
        smooth_diagonal[body],
        mass[body],
        inertia_world[body],
        sigma,
        beta,
        mass_floor,
        inertia_floor,
        eta_floor,
        symmetry_tolerance,
    )
    weight[body] = result.weight
    inverse_weight[body] = result.inverse_weight
    eta[body] = result.eta
    alpha[body] = result.alpha
    status[body] = result.status


@wp.kernel
def _compute_anisotropic_body_weights(
    smooth_diagonal: wp.array[mat66f],
    mass: wp.array[wp.float32],
    inertia_world: wp.array[wp.mat33f],
    sigma: wp.float32,
    beta: wp.float32,
    weight: wp.array[mat66f],
    inverse_weight: wp.array[mat66f],
    eta: wp.array[wp.float32],
    eigenvalue_min: wp.array[wp.float32],
    eigenvalue_max: wp.array[wp.float32],
    status: wp.array[wp.int32],
):
    body = wp.tid()
    result = compute_body_weight_anisotropic(smooth_diagonal[body], mass[body], inertia_world[body], sigma, beta)
    weight[body] = result.weight
    inverse_weight[body] = result.inverse_weight
    eta[body] = result.eta
    eigenvalue_min[body] = result.eigenvalue_min
    eigenvalue_max[body] = result.eigenvalue_max
    status[body] = result.status


def _spatial_mass(mass: float, inertia: np.ndarray) -> np.ndarray:
    spatial_mass = np.zeros((6, 6), dtype=np.float64)
    spatial_mass[:3, :3] = mass * np.eye(3)
    spatial_mass[3:, 3:] = inertia
    return spatial_mass


def _reference_weight(
    smooth_diagonal: np.ndarray,
    mass: float,
    inertia: np.ndarray,
    sigma: float = 1.0e-3,
    beta: float = 25.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    inertia_eigenvalues, inertia_axes = np.linalg.eigh(inertia)
    inverse_sqrt_inertia = (inertia_axes * np.reciprocal(np.sqrt(inertia_eigenvalues))[None, :]) @ inertia_axes.T
    inverse_sqrt_mass = np.zeros((6, 6), dtype=np.float64)
    inverse_sqrt_mass[:3, :3] = np.eye(3) / np.sqrt(mass)
    inverse_sqrt_mass[3:, 3:] = inverse_sqrt_inertia
    normalized = inverse_sqrt_mass @ smooth_diagonal @ inverse_sqrt_mass
    eta = float(np.linalg.eigvalsh(normalized)[0])
    alpha = max(sigma * eta, min(beta, eta))
    spatial_mass = _spatial_mass(mass, inertia)
    return eta, alpha, alpha * spatial_mass, np.linalg.inv(alpha * spatial_mass)


class TestLOXWeight(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def _compute(
        self,
        smooth_diagonal: np.ndarray,
        mass: np.ndarray,
        inertia_world: np.ndarray,
        parameters: tuple[float, float, float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        body_count = len(mass)
        smooth_wp = wp.array(smooth_diagonal, dtype=mat66f, device=self.device)
        mass_wp = wp.array(mass, dtype=wp.float32, device=self.device)
        inertia_wp = wp.array(inertia_world, dtype=wp.mat33f, device=self.device)
        weight_wp = wp.empty(body_count, dtype=mat66f, device=self.device)
        inverse_weight_wp = wp.empty(body_count, dtype=mat66f, device=self.device)
        eta_wp = wp.empty(body_count, dtype=wp.float32, device=self.device)
        alpha_wp = wp.empty(body_count, dtype=wp.float32, device=self.device)
        status_wp = wp.empty(body_count, dtype=wp.int32, device=self.device)

        outputs = [weight_wp, inverse_weight_wp, eta_wp, alpha_wp, status_wp]
        if parameters is None:
            wp.launch(
                _compute_body_weights_default,
                dim=body_count,
                inputs=[smooth_wp, mass_wp, inertia_wp],
                outputs=outputs,
                device=self.device,
            )
        else:
            wp.launch(
                _compute_body_weights_configured,
                dim=body_count,
                inputs=[smooth_wp, mass_wp, inertia_wp, *parameters],
                outputs=outputs,
                device=self.device,
            )

        return (
            weight_wp.numpy(),
            inverse_weight_wp.numpy(),
            eta_wp.numpy(),
            alpha_wp.numpy(),
            status_wp.numpy(),
        )

    def test_generalized_eigenvalue_with_coupling_and_anisotropic_inertia(self):
        mass = 2.5
        angle = 0.63
        cosine = np.cos(angle)
        sine = np.sin(angle)
        inertia_axes = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        inertia = inertia_axes @ np.diag([0.2, 0.7, 1.9]) @ inertia_axes.T
        spatial_mass = _spatial_mass(mass, inertia)
        spatial_eigenvalues, spatial_axes = np.linalg.eigh(spatial_mass)
        sqrt_spatial_mass = (spatial_axes * np.sqrt(spatial_eigenvalues)[None, :]) @ spatial_axes.T

        normalized_axes, _ = np.linalg.qr(
            np.asarray(
                [
                    [1.0, 0.2, -0.3, 0.4, 0.1, -0.2],
                    [0.3, 1.0, 0.1, -0.2, 0.5, 0.4],
                    [-0.1, 0.4, 1.0, 0.3, -0.2, 0.5],
                    [0.5, -0.2, 0.4, 1.0, 0.3, -0.1],
                    [0.2, 0.6, -0.1, 0.4, 1.0, 0.2],
                    [-0.4, 0.1, 0.5, -0.2, 0.3, 1.0],
                ],
                dtype=np.float64,
            )
        )
        normalized = normalized_axes @ np.diag([0.4, 0.8, 1.2, 2.0, 4.0, 7.0]) @ normalized_axes.T
        smooth_diagonal = sqrt_spatial_mass @ normalized @ sqrt_spatial_mass
        eta_ref, alpha_ref, weight_ref, inverse_weight_ref = _reference_weight(smooth_diagonal, mass, inertia)

        weight, inverse_weight, eta, alpha, status = self._compute(
            smooth_diagonal[None, ...].astype(np.float32),
            np.asarray([mass], dtype=np.float32),
            inertia[None, ...].astype(np.float32),
        )

        self.assertEqual(status[0], BODY_WEIGHT_STATUS_VALID)
        self.assertAlmostEqual(eta[0], eta_ref, delta=2.0e-5)
        self.assertAlmostEqual(alpha[0], alpha_ref, delta=2.0e-5)
        np.testing.assert_allclose(weight[0], weight_ref, rtol=3.0e-5, atol=3.0e-6)
        np.testing.assert_allclose(inverse_weight[0], inverse_weight_ref, rtol=3.0e-5, atol=3.0e-6)
        np.testing.assert_allclose(weight[0] @ inverse_weight[0], np.eye(6), rtol=3.0e-5, atol=3.0e-5)
        self.assertGreater(np.linalg.norm(smooth_diagonal[:3, 3:]), 0.1)

    def test_paper_clamp_branches_and_defaults(self):
        normalized_eigenvalues = np.asarray([0.5, 100.0, 50000.0], dtype=np.float32)
        masses = np.asarray([1.5, 1.5, 1.5], dtype=np.float32)
        inertias = np.repeat(np.diag([0.3, 0.7, 1.1])[None, ...], 3, axis=0).astype(np.float32)
        smooth = np.asarray(
            [
                value * _spatial_mass(float(mass), inertia)
                for value, mass, inertia in zip(normalized_eigenvalues, masses, inertias, strict=True)
            ],
            dtype=np.float32,
        )

        weight, inverse_weight, eta, alpha, status = self._compute(smooth, masses, inertias)

        np.testing.assert_allclose(eta, normalized_eigenvalues, rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(alpha, [0.5, 4.0, 50.0], rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_array_equal(status, np.full(3, BODY_WEIGHT_STATUS_VALID, dtype=np.int32))
        for body in range(3):
            np.testing.assert_allclose(weight[body] @ inverse_weight[body], np.eye(6), rtol=3.0e-5, atol=3.0e-5)

    def test_finite_degenerate_inputs_are_regularized(self):
        mass_floor = 1.0e-4
        inertia_floor = 2.0e-4
        eta_floor = 3.0e-4
        mass = np.asarray([1.0e-8], dtype=np.float32)
        inertia = np.asarray([np.diag([-1.0, 1.0e-8, 0.5])], dtype=np.float32)
        smooth = np.asarray([np.diag([-2.0, 1.0, 2.0, 3.0, 4.0, 5.0])], dtype=np.float32)

        weight, inverse_weight, eta, alpha, status = self._compute(
            smooth,
            mass,
            inertia,
            parameters=(1.0e-3, 25.0, mass_floor, inertia_floor, eta_floor, 1.0e-5),
        )

        self.assertEqual(status[0], BODY_WEIGHT_STATUS_REGULARIZED)
        self.assertAlmostEqual(eta[0], eta_floor, delta=1.0e-8)
        self.assertAlmostEqual(alpha[0], eta_floor, delta=1.0e-8)
        self.assertTrue(np.all(np.isfinite(weight[0])))
        self.assertTrue(np.all(np.isfinite(inverse_weight[0])))
        np.testing.assert_allclose(weight[0] @ inverse_weight[0], np.eye(6), rtol=3.0e-5, atol=3.0e-5)

    def test_asymmetric_finite_inputs_are_symmetrized(self):
        mass = np.asarray([2.0], dtype=np.float32)
        inertia = np.asarray([[[0.5, 0.2, 0.0], [0.1, 0.8, 0.0], [0.0, 0.0, 1.1]]], dtype=np.float32)
        smooth = np.asarray([2.0 * _spatial_mass(2.0, np.eye(3))], dtype=np.float32)
        smooth[0, 0, 3] = 0.3
        smooth[0, 3, 0] = 0.1

        weight, inverse_weight, eta, alpha, status = self._compute(smooth, mass, inertia)

        self.assertEqual(status[0], BODY_WEIGHT_STATUS_REGULARIZED)
        self.assertTrue(np.all(np.isfinite([eta[0], alpha[0]])))
        np.testing.assert_allclose(weight[0], weight[0].T, rtol=0.0, atol=2.0e-7)
        np.testing.assert_allclose(inverse_weight[0], inverse_weight[0].T, rtol=0.0, atol=2.0e-7)
        np.testing.assert_allclose(weight[0] @ inverse_weight[0], np.eye(6), rtol=3.0e-5, atol=3.0e-5)

    def test_nonfinite_and_nonpositive_inputs_are_invalid(self):
        smooth = np.repeat(np.eye(6, dtype=np.float32)[None, ...], 3, axis=0)
        smooth[0, 2, 2] = np.nan
        mass = np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
        inertia = np.repeat(np.eye(3, dtype=np.float32)[None, ...], 3, axis=0)
        inertia[2, 0, 0] = np.inf

        weight, inverse_weight, eta, alpha, status = self._compute(smooth, mass, inertia)

        np.testing.assert_array_equal(status, np.full(3, BODY_WEIGHT_STATUS_INVALID, dtype=np.int32))
        np.testing.assert_array_equal(weight, np.zeros_like(weight))
        np.testing.assert_array_equal(inverse_weight, np.zeros_like(inverse_weight))
        np.testing.assert_array_equal(eta, np.zeros_like(eta))
        np.testing.assert_array_equal(alpha, np.zeros_like(alpha))

    def test_anisotropic_weight_preserves_coupling_and_clamps_spectrum(self):
        mass = 2.0
        inertia = np.asarray([[0.8, 0.2, 0.0], [0.2, 1.1, -0.1], [0.0, -0.1, 1.5]], dtype=np.float64)
        spatial_mass = _spatial_mass(mass, inertia)
        spatial_values, spatial_axes = np.linalg.eigh(spatial_mass)
        sqrt_mass = (spatial_axes * np.sqrt(spatial_values)[None, :]) @ spatial_axes.T
        inverse_sqrt_mass = (spatial_axes * np.reciprocal(np.sqrt(spatial_values))[None, :]) @ spatial_axes.T
        normalized_hessian = np.asarray(
            [
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.7, 0.25, -0.1],
                [0.0, 0.0, 0.0, 0.25, 3.0, 0.4],
                [0.0, 0.0, 0.0, -0.1, 0.4, 9.0],
            ],
            dtype=np.float64,
        )
        smooth = sqrt_mass @ normalized_hessian @ sqrt_mass
        weight = wp.zeros(1, dtype=mat66f, device=self.device)
        inverse_weight = wp.zeros_like(weight)
        eta = wp.zeros(1, dtype=wp.float32, device=self.device)
        eigenvalue_min = wp.zeros_like(eta)
        eigenvalue_max = wp.zeros_like(eta)
        status = wp.zeros(1, dtype=wp.int32, device=self.device)
        sigma = 0.1
        beta = 4.0
        wp.launch(
            _compute_anisotropic_body_weights,
            dim=1,
            inputs=[
                wp.array(smooth[None, ...].astype(np.float32), dtype=mat66f, device=self.device),
                wp.array([mass], dtype=wp.float32, device=self.device),
                wp.array(inertia[None, ...].astype(np.float32), dtype=wp.mat33f, device=self.device),
                sigma,
                beta,
            ],
            outputs=[weight, inverse_weight, eta, eigenvalue_min, eigenvalue_max, status],
            device=self.device,
        )

        actual = weight.numpy()[0]
        actual_inverse = inverse_weight.numpy()[0]
        generalized = inverse_sqrt_mass @ actual @ inverse_sqrt_mass
        generalized_values = np.linalg.eigvalsh(generalized)
        np.testing.assert_allclose(actual, actual.T, atol=3.0e-6)
        np.testing.assert_allclose(actual @ actual_inverse, np.eye(6), rtol=2.0e-4, atol=2.0e-4)
        self.assertGreater(abs(actual[3, 4]), 0.05)
        normalized_values, normalized_axes = np.linalg.eigh(normalized_hessian)
        eta_expected = normalized_values[0]
        expected_values = np.maximum(sigma * normalized_values, np.minimum(beta, normalized_values))
        expected_generalized = (normalized_axes * expected_values[None, :]) @ normalized_axes.T
        np.testing.assert_allclose(generalized, expected_generalized, rtol=3.0e-4, atol=3.0e-4)
        np.testing.assert_allclose(generalized_values, expected_values, rtol=3.0e-4, atol=3.0e-4)
        self.assertLessEqual(generalized_values[-1], beta + 3.0e-4)
        self.assertAlmostEqual(float(eta.numpy()[0]), eta_expected, delta=3.0e-4)
        self.assertAlmostEqual(float(eigenvalue_min.numpy()[0]), generalized_values[0], delta=3.0e-4)
        self.assertAlmostEqual(float(eigenvalue_max.numpy()[0]), generalized_values[-1], delta=3.0e-4)
        self.assertEqual(int(status.numpy()[0]), BODY_WEIGHT_STATUS_REGULARIZED)


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
