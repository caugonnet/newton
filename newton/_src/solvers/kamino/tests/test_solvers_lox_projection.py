# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX one-constraint projection primitives."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat36f, mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    convert_contact_matrix_normal_first_to_last,
    convert_contact_matrix_normal_last_to_first,
    convert_contact_vector_normal_first_to_last,
    convert_contact_vector_normal_last_to_first,
    project_contact_coulomb,
    project_joint_friction,
    project_limit_unilateral,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


@wp.kernel
def _convert_contact_order(
    vector_normal_last: wp.array[wp.vec3f],
    matrix_normal_last: wp.array[wp.mat33f],
    vector_normal_first: wp.array[wp.vec3f],
    vector_round_trip: wp.array[wp.vec3f],
    matrix_normal_first: wp.array[wp.mat33f],
    matrix_round_trip: wp.array[wp.mat33f],
):
    index = wp.tid()
    vector_converted = convert_contact_vector_normal_last_to_first(vector_normal_last[index])
    matrix_converted = convert_contact_matrix_normal_last_to_first(matrix_normal_last[index])
    vector_normal_first[index] = vector_converted
    vector_round_trip[index] = convert_contact_vector_normal_first_to_last(vector_converted)
    matrix_normal_first[index] = matrix_converted
    matrix_round_trip[index] = convert_contact_matrix_normal_first_to_last(matrix_converted)


@wp.kernel
def _project_contacts(
    jacobian_first: wp.array[mat36f],
    inverse_weight_first: wp.array[mat66f],
    twist_first: wp.array[vec6f],
    jacobian_second: wp.array[mat36f],
    inverse_weight_second: wp.array[mat66f],
    twist_second: wp.array[vec6f],
    velocity_bias: wp.array[wp.vec3f],
    reaction_old: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    twist_first_new: wp.array[vec6f],
    twist_second_new: wp.array[vec6f],
    reaction_new: wp.array[wp.vec3f],
    reaction_delta: wp.array[wp.vec3f],
    velocity_new: wp.array[wp.vec3f],
    delassus: wp.array[wp.mat33f],
    status: wp.array[wp.int32],
):
    contact = wp.tid()
    result = project_contact_coulomb(
        jacobian_first[contact],
        inverse_weight_first[contact],
        twist_first[contact],
        jacobian_second[contact],
        inverse_weight_second[contact],
        twist_second[contact],
        velocity_bias[contact],
        reaction_old[contact],
        friction[contact],
    )
    twist_first_new[contact] = result.twist_first
    twist_second_new[contact] = result.twist_second
    reaction_new[contact] = result.reaction
    reaction_delta[contact] = result.reaction_delta
    velocity_new[contact] = result.velocity
    delassus[contact] = result.delassus
    status[contact] = result.status


@wp.kernel
def _project_limits(
    jacobian_first: wp.array[vec6f],
    inverse_weight_first: wp.array[mat66f],
    twist_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    inverse_weight_second: wp.array[mat66f],
    twist_second: wp.array[vec6f],
    velocity_bias: wp.array[wp.float32],
    reaction_old: wp.array[wp.float32],
    twist_first_new: wp.array[vec6f],
    twist_second_new: wp.array[vec6f],
    reaction_new: wp.array[wp.float32],
    reaction_delta: wp.array[wp.float32],
    velocity_new: wp.array[wp.float32],
    delassus: wp.array[wp.float32],
    status: wp.array[wp.int32],
):
    limit = wp.tid()
    result = project_limit_unilateral(
        jacobian_first[limit],
        inverse_weight_first[limit],
        twist_first[limit],
        jacobian_second[limit],
        inverse_weight_second[limit],
        twist_second[limit],
        velocity_bias[limit],
        reaction_old[limit],
    )
    twist_first_new[limit] = result.twist_first
    twist_second_new[limit] = result.twist_second
    reaction_new[limit] = result.reaction
    reaction_delta[limit] = result.reaction_delta
    velocity_new[limit] = result.velocity
    delassus[limit] = result.delassus
    status[limit] = result.status


@wp.kernel
def _project_frictions(
    jacobian_first: wp.array[vec6f],
    inverse_weight_first: wp.array[mat66f],
    twist_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    inverse_weight_second: wp.array[mat66f],
    twist_second: wp.array[vec6f],
    reaction_old: wp.array[wp.float32],
    impulse_bound: wp.array[wp.float32],
    reaction_new: wp.array[wp.float32],
    velocity_new: wp.array[wp.float32],
    status: wp.array[wp.int32],
):
    friction = wp.tid()
    result = project_joint_friction(
        jacobian_first[friction],
        inverse_weight_first[friction],
        twist_first[friction],
        jacobian_second[friction],
        inverse_weight_second[friction],
        twist_second[friction],
        reaction_old[friction],
        impulse_bound[friction],
    )
    reaction_new[friction] = result.reaction
    velocity_new[friction] = result.velocity
    status[friction] = result.status


def _skew(value: np.ndarray) -> np.ndarray:
    return np.asarray([[0.0, -value[2], value[1]], [value[2], 0.0, -value[0]], [-value[1], value[0], 0.0]])


def _point_jacobian(frame: np.ndarray, lever_arm: np.ndarray, sign: float) -> np.ndarray:
    return sign * frame.T @ np.concatenate((np.eye(3), -_skew(lever_arm)), axis=1)


class TestLOXProjection(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_joint_friction_has_one_signed_box_constraint_per_dof(self):
        identity = wp.array([np.eye(6, dtype=np.float32)] * 2, dtype=mat66f, device=self.device)
        jacobian = wp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2, dtype=vec6f, device=self.device)
        zero_jacobian = wp.zeros(2, dtype=vec6f, device=self.device)
        twist = wp.array(
            [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=vec6f, device=self.device
        )
        zero_twist = wp.zeros(2, dtype=vec6f, device=self.device)
        old_reaction = wp.zeros(2, dtype=wp.float32, device=self.device)
        bound = wp.array([0.5, 0.5], dtype=wp.float32, device=self.device)
        reaction = wp.zeros(2, dtype=wp.float32, device=self.device)
        velocity = wp.zeros(2, dtype=wp.float32, device=self.device)
        status = wp.zeros(2, dtype=wp.int32, device=self.device)

        wp.launch(
            _project_frictions,
            dim=2,
            inputs=[jacobian, identity, twist, zero_jacobian, identity, zero_twist, old_reaction, bound],
            outputs=[reaction, velocity, status],
            device=self.device,
        )

        # The first row slips at the negative bound; the second sticks exactly.
        np.testing.assert_allclose(reaction.numpy(), [-0.5, -0.25], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(velocity.numpy(), [1.5, 0.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(status.numpy(), [PROJECTION_STATUS_VALID] * 2)

    def _project_contact(
        self,
        jacobian_first: np.ndarray,
        inverse_weight_first: np.ndarray,
        twist_first: np.ndarray,
        jacobian_second: np.ndarray,
        inverse_weight_second: np.ndarray,
        twist_second: np.ndarray,
        velocity_bias: np.ndarray,
        reaction_old: np.ndarray,
        friction: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        count = len(friction)
        inputs = [
            wp.array(jacobian_first, dtype=mat36f, device=self.device),
            wp.array(inverse_weight_first, dtype=mat66f, device=self.device),
            wp.array(twist_first, dtype=vec6f, device=self.device),
            wp.array(jacobian_second, dtype=mat36f, device=self.device),
            wp.array(inverse_weight_second, dtype=mat66f, device=self.device),
            wp.array(twist_second, dtype=vec6f, device=self.device),
            wp.array(velocity_bias, dtype=wp.vec3f, device=self.device),
            wp.array(reaction_old, dtype=wp.vec3f, device=self.device),
            wp.array(friction, dtype=wp.float32, device=self.device),
        ]
        outputs = [
            wp.empty(count, dtype=vec6f, device=self.device),
            wp.empty(count, dtype=vec6f, device=self.device),
            wp.empty(count, dtype=wp.vec3f, device=self.device),
            wp.empty(count, dtype=wp.vec3f, device=self.device),
            wp.empty(count, dtype=wp.vec3f, device=self.device),
            wp.empty(count, dtype=wp.mat33f, device=self.device),
            wp.empty(count, dtype=wp.int32, device=self.device),
        ]
        wp.launch(_project_contacts, dim=count, inputs=inputs, outputs=outputs, device=self.device)
        return tuple(output.numpy() for output in outputs)

    def _project_limit(
        self,
        jacobian_first: np.ndarray,
        inverse_weight_first: np.ndarray,
        twist_first: np.ndarray,
        jacobian_second: np.ndarray,
        inverse_weight_second: np.ndarray,
        twist_second: np.ndarray,
        velocity_bias: np.ndarray,
        reaction_old: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        count = len(reaction_old)
        inputs = [
            wp.array(jacobian_first, dtype=vec6f, device=self.device),
            wp.array(inverse_weight_first, dtype=mat66f, device=self.device),
            wp.array(twist_first, dtype=vec6f, device=self.device),
            wp.array(jacobian_second, dtype=vec6f, device=self.device),
            wp.array(inverse_weight_second, dtype=mat66f, device=self.device),
            wp.array(twist_second, dtype=vec6f, device=self.device),
            wp.array(velocity_bias, dtype=wp.float32, device=self.device),
            wp.array(reaction_old, dtype=wp.float32, device=self.device),
        ]
        outputs = [
            wp.empty(count, dtype=vec6f, device=self.device),
            wp.empty(count, dtype=vec6f, device=self.device),
            wp.empty(count, dtype=wp.float32, device=self.device),
            wp.empty(count, dtype=wp.float32, device=self.device),
            wp.empty(count, dtype=wp.float32, device=self.device),
            wp.empty(count, dtype=wp.float32, device=self.device),
            wp.empty(count, dtype=wp.int32, device=self.device),
        ]
        wp.launch(_project_limits, dim=count, inputs=inputs, outputs=outputs, device=self.device)
        return tuple(output.numpy() for output in outputs)

    def test_normal_order_conversion(self):
        vector = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        matrix = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32)
        outputs = [
            wp.empty(1, dtype=wp.vec3f, device=self.device),
            wp.empty(1, dtype=wp.vec3f, device=self.device),
            wp.empty(1, dtype=wp.mat33f, device=self.device),
            wp.empty(1, dtype=wp.mat33f, device=self.device),
        ]
        wp.launch(
            _convert_contact_order,
            dim=1,
            inputs=[
                wp.array(vector, dtype=wp.vec3f, device=self.device),
                wp.array(matrix, dtype=wp.mat33f, device=self.device),
            ],
            outputs=outputs,
            device=self.device,
        )

        np.testing.assert_array_equal(outputs[0].numpy()[0], [3.0, 1.0, 2.0])
        np.testing.assert_array_equal(outputs[2].numpy()[0], [[9.0, 7.0, 8.0], [3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        np.testing.assert_array_equal(outputs[1].numpy(), vector)
        np.testing.assert_array_equal(outputs[3].numpy(), matrix)

    def test_two_body_lever_arm_full_block_and_warm_start(self):
        frame, _ = np.linalg.qr(np.asarray([[1.0, 0.2, -0.1], [0.3, 1.0, 0.4], [-0.2, 0.1, 1.0]], dtype=np.float64))
        jacobian_first = _point_jacobian(frame, np.asarray([0.4, -0.2, 0.3]), 1.0)
        jacobian_second = _point_jacobian(frame, np.asarray([-0.3, 0.5, -0.1]), -1.0)
        inverse_weight_first = np.diag([0.5, 0.5, 0.5, 1.4, 0.8, 0.6])
        inverse_weight_first[3, 4] = inverse_weight_first[4, 3] = 0.18
        inverse_weight_second = np.diag([0.8, 0.8, 0.8, 0.7, 1.2, 0.9])
        inverse_weight_second[4, 5] = inverse_weight_second[5, 4] = -0.16
        delassus = (
            jacobian_first @ inverse_weight_first @ jacobian_first.T
            + jacobian_second @ inverse_weight_second @ jacobian_second.T
        )
        self.assertGreater(np.linalg.norm(delassus - np.diag(np.diag(delassus))), 0.1)

        friction = 0.6
        reaction_expected = np.asarray([-0.336, -0.252, 0.7])
        velocity_expected = np.asarray([0.256, 0.192, 0.0])
        free_velocity = velocity_expected - delassus @ reaction_expected
        reaction_old = np.asarray([0.02, -0.01, 0.2])
        velocity_current = free_velocity + delassus @ reaction_old
        velocity_bias = np.asarray([0.03, -0.02, 0.01])
        combined_jacobian = np.concatenate((jacobian_first, jacobian_second), axis=1)
        combined_twist = np.linalg.lstsq(combined_jacobian, velocity_current - velocity_bias, rcond=None)[0]
        twist_first = combined_twist[:6]
        twist_second = combined_twist[6:]
        reaction_delta_expected = reaction_expected - reaction_old
        twist_first_expected = twist_first + inverse_weight_first @ jacobian_first.T @ reaction_delta_expected
        twist_second_expected = twist_second + inverse_weight_second @ jacobian_second.T @ reaction_delta_expected

        result = self._project_contact(
            jacobian_first[None, ...].astype(np.float32),
            inverse_weight_first[None, ...].astype(np.float32),
            twist_first[None, ...].astype(np.float32),
            jacobian_second[None, ...].astype(np.float32),
            inverse_weight_second[None, ...].astype(np.float32),
            twist_second[None, ...].astype(np.float32),
            velocity_bias[None, ...].astype(np.float32),
            reaction_old[None, ...].astype(np.float32),
            np.asarray([friction], dtype=np.float32),
        )
        twist_first_new, twist_second_new, reaction_new, reaction_delta, velocity_new, delassus_new, status = result

        self.assertEqual(status[0], PROJECTION_STATUS_VALID)
        np.testing.assert_allclose(delassus_new[0], delassus, rtol=3.0e-6, atol=3.0e-6)
        np.testing.assert_allclose(reaction_new[0], reaction_expected, rtol=2.0e-5, atol=3.0e-6)
        np.testing.assert_allclose(reaction_delta[0], reaction_delta_expected, rtol=2.0e-5, atol=3.0e-6)
        np.testing.assert_allclose(twist_first_new[0], twist_first_expected, rtol=2.0e-5, atol=3.0e-6)
        np.testing.assert_allclose(twist_second_new[0], twist_second_expected, rtol=2.0e-5, atol=3.0e-6)
        np.testing.assert_allclose(velocity_new[0], velocity_expected, rtol=3.0e-5, atol=5.0e-6)

    def test_static_second_side_uses_zero_jacobian(self):
        jacobian_first = np.zeros((3, 6), dtype=np.float32)
        jacobian_first[:, :3] = np.eye(3)
        inverse_weight_first = np.diag([1.0, 1.5, 2.0, 0.5, 0.5, 0.5]).astype(np.float32)
        jacobian_second = np.zeros((3, 6), dtype=np.float32)
        inverse_weight_second = np.zeros((6, 6), dtype=np.float32)
        twist_first = np.asarray([1.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        twist_second = np.asarray([4.0, -3.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32)

        result = self._project_contact(
            jacobian_first[None, ...],
            inverse_weight_first[None, ...],
            twist_first[None, ...],
            jacobian_second[None, ...],
            inverse_weight_second[None, ...],
            twist_second[None, ...],
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.asarray([0.5], dtype=np.float32),
        )
        twist_first_new, twist_second_new, reaction_new, _, velocity_new, delassus, status = result

        self.assertEqual(status[0], PROJECTION_STATUS_VALID)
        np.testing.assert_allclose(delassus[0], np.diag([1.0, 1.5, 2.0]), rtol=0.0, atol=1.0e-7)
        np.testing.assert_array_equal(twist_second_new[0], twist_second)
        self.assertGreater(reaction_new[0, 2], 0.0)
        self.assertAlmostEqual(velocity_new[0, 2], 0.0, delta=2.0e-6)
        self.assertTrue(np.all(np.isfinite(twist_first_new[0])))

    def test_scalar_unilateral_limit_update(self):
        jacobian_first = np.asarray([0.8, -0.3, 0.2, 0.5, -0.1, 0.4])
        jacobian_second = np.asarray([-0.2, 0.6, -0.4, 0.1, 0.3, -0.5])
        inverse_weight_first = np.diag([0.5, 0.8, 1.1, 0.7, 1.3, 0.9])
        inverse_weight_second = np.diag([1.0, 0.6, 0.9, 1.2, 0.8, 0.5])
        delassus = (
            jacobian_first @ inverse_weight_first @ jacobian_first
            + jacobian_second @ inverse_weight_second @ jacobian_second
        )
        reaction_old = 0.4
        free_velocity = -0.6
        velocity_current = free_velocity + delassus * reaction_old
        velocity_bias = 0.05
        twist_first = jacobian_first * (velocity_current - velocity_bias) / np.dot(jacobian_first, jacobian_first)
        twist_second = np.zeros(6)
        reaction_expected = -free_velocity / delassus
        reaction_delta_expected = reaction_expected - reaction_old
        twist_first_expected = twist_first + inverse_weight_first @ jacobian_first * reaction_delta_expected
        twist_second_expected = twist_second + inverse_weight_second @ jacobian_second * reaction_delta_expected

        result = self._project_limit(
            jacobian_first[None, ...].astype(np.float32),
            inverse_weight_first[None, ...].astype(np.float32),
            twist_first[None, ...].astype(np.float32),
            jacobian_second[None, ...].astype(np.float32),
            inverse_weight_second[None, ...].astype(np.float32),
            twist_second[None, ...].astype(np.float32),
            np.asarray([velocity_bias], dtype=np.float32),
            np.asarray([reaction_old], dtype=np.float32),
        )
        twist_first_new, twist_second_new, reaction_new, reaction_delta, velocity_new, delassus_new, status = result

        self.assertEqual(status[0], PROJECTION_STATUS_VALID)
        self.assertAlmostEqual(delassus_new[0], delassus, delta=2.0e-6)
        self.assertAlmostEqual(reaction_new[0], reaction_expected, delta=2.0e-6)
        self.assertAlmostEqual(reaction_delta[0], reaction_delta_expected, delta=2.0e-6)
        self.assertAlmostEqual(velocity_new[0], 0.0, delta=2.0e-6)
        np.testing.assert_allclose(twist_first_new[0], twist_first_expected, rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(twist_second_new[0], twist_second_expected, rtol=2.0e-6, atol=2.0e-6)

    def test_nonpositive_local_blocks_are_invalid(self):
        zero_jacobian_contact = np.zeros((1, 3, 6), dtype=np.float32)
        zero_jacobian_limit = np.zeros((1, 6), dtype=np.float32)
        zero_inverse_weight = np.zeros((1, 6, 6), dtype=np.float32)
        twist = np.ones((1, 6), dtype=np.float32)
        reaction = np.asarray([[0.1, -0.2, 0.3]], dtype=np.float32)

        contact_result = self._project_contact(
            zero_jacobian_contact,
            zero_inverse_weight,
            twist,
            zero_jacobian_contact,
            zero_inverse_weight,
            twist,
            np.zeros((1, 3), dtype=np.float32),
            reaction,
            np.asarray([0.5], dtype=np.float32),
        )
        self.assertEqual(contact_result[-1][0], PROJECTION_STATUS_INVALID)
        np.testing.assert_array_equal(contact_result[0][0], twist[0])
        np.testing.assert_array_equal(contact_result[2][0], reaction[0])

        limit_result = self._project_limit(
            zero_jacobian_limit,
            zero_inverse_weight,
            twist,
            zero_jacobian_limit,
            zero_inverse_weight,
            twist,
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.4], dtype=np.float32),
        )
        self.assertEqual(limit_result[-1][0], PROJECTION_STATUS_INVALID)
        np.testing.assert_array_equal(limit_result[0][0], twist[0])
        self.assertEqual(limit_result[2][0], np.float32(0.4))


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
