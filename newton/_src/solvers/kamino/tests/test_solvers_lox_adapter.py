# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Kamino LOX container adapter."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.bodies import update_body_inertias
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.kinematics.constraints import (
    make_unilateral_constraints_info,
    update_constraints_info,
)
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from newton._src.solvers.kamino._src.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.models.builders.basics import (
    build_box_on_plane,
    build_boxes_hinged,
    build_cartpole,
)
from newton._src.solvers.kamino._src.solvers.lox.adapter import LOXKaminoAdapter
from newton._src.solvers.kamino._src.solvers.lox.projection import PROJECTION_STATUS_VALID
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _extract_world_matrices(adapter: LOXKaminoAdapter) -> list[np.ndarray]:
    flat = adapter.system.smooth_matrix.numpy()
    result = []
    offset = 0
    for body_count in adapter.system.body_counts:
        dimension = 6 * body_count
        result.append(flat[offset : offset + dimension * dimension].reshape(dimension, dimension))
        offset += dimension * dimension
    return result


def _extract_world_vectors(adapter: LOXKaminoAdapter) -> list[np.ndarray]:
    flat = adapter.system.right_hand_side.numpy()
    result = []
    offset = 0
    for body_count in adapter.system.body_counts:
        dimension = 6 * body_count
        result.append(flat[offset : offset + dimension])
        offset += dimension
    return result


def _spatial_mass(mass: float, inertia: np.ndarray) -> np.ndarray:
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = mass * np.eye(3)
    result[3:, 3:] = inertia
    return result


def _spatial_inverse_mass(inverse_mass: float, inverse_inertia: np.ndarray) -> np.ndarray:
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = inverse_mass * np.eye(3)
    result[3:, 3:] = inverse_inertia
    return result


class TestLOXAdapter(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_heterogeneous_body_joint_contact_mapping_and_outputs(self):
        builder = ModelBuilderKamino(default_world=False)
        builder.add_builder(build_box_on_plane(ground=False))
        builder.add_builder(build_boxes_hinged(ground=False, dynamic_joints=True))
        model = builder.finalize(device=self.device)
        data = model.data()
        contacts = ContactsKamino(capacity=[1, 2], device=self.device)
        make_unilateral_constraints_info(model, data, contacts=contacts)

        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))

        velocity = np.asarray(
            [
                [0.2, -0.1, 0.3, 0.1, -0.2, 0.4],
                [0.0, 0.1, 0.0, 0.2, 0.3, -0.1],
                [0.0, -0.2, -2.0, -0.3, 0.1, 0.2],
            ],
            dtype=np.float32,
        )
        external = np.asarray(
            [[0.5, 0.0, -0.1, 0.0, 0.2, 0.0], [0.0, 0.3, 0.1, -0.2, 0.0, 0.1], [0.2, 0.0, 0.4, 0.0, -0.1, 0.3]],
            dtype=np.float32,
        )
        actuation = np.asarray(
            [[0.0, 0.1, 0.0, 0.2, 0.0, -0.1], [0.1, 0.0, -0.2, 0.0, 0.2, 0.0], [0.0, 0.2, 0.0, -0.1, 0.0, 0.1]],
            dtype=np.float32,
        )
        data.bodies.u_i.assign(velocity)
        data.bodies.w_e_i.assign(external)
        data.bodies.w_a_i.assign(actuation)
        data.joints.m_j.assign(np.asarray([2.5], dtype=np.float32))
        data.joints.dq_b_j.assign(np.asarray([-0.35], dtype=np.float32))
        model.joints.a_j.fill_(0.75)
        residual = np.asarray([-0.03, 0.02, -0.01, 0.04, -0.02], dtype=np.float32)
        multiplier = np.asarray([90.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float32)
        data.joints.r_j.assign(residual)
        data.joints.lambda_j.assign(multiplier)

        body_pose = data.bodies.q_i.numpy()
        positions_first = np.zeros((3, 3), dtype=np.float32)
        positions_second = np.zeros((3, 3), dtype=np.float32)
        positions_first[0] = body_pose[1, :3]
        positions_second[0] = body_pose[2, :3]
        contacts.model_active_contacts.assign(np.asarray([1], dtype=np.int32))
        contacts.world_active_contacts.assign(np.asarray([0, 1], dtype=np.int32))
        contacts.wid.assign(np.asarray([1, -1, -1], dtype=np.int32))
        contacts.cid.assign(np.asarray([0, -1, -1], dtype=np.int32))
        contacts.bid_AB.assign(np.asarray([[1, 2], [-1, -1], [-1, -1]], dtype=np.int32))
        contacts.position_A.assign(positions_first)
        contacts.position_B.assign(positions_second)
        contacts.gapfunc.assign(
            np.asarray([[0.0, 0.0, 1.0, -0.02], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        )
        contacts.frame.assign(
            np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        )
        contacts.material.assign(np.asarray([[0.6, 0.4], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32))
        contacts.reaction.assign(np.asarray([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
        update_constraints_info(model, data)

        jacobians = DenseSystemJacobians(model=model, contacts=contacts)
        jacobians.build(model=model, data=data, contacts=contacts)
        sparse_jacobians = SparseSystemJacobians(model=model, contacts=contacts)
        sparse_jacobians.build(model=model, data=data, contacts=contacts)
        adapter = LOXKaminoAdapter(model, data, jacobians, contacts=contacts)
        sparse_adapter = LOXKaminoAdapter(model, data, sparse_jacobians, contacts=contacts)
        time_step = 0.1
        penalty_scale = 3.0
        adapter.begin_time_step(
            time_step,
            contact_stabilization_fraction=0.25,
            contact_dead_zone=0.01,
            impact_velocity_threshold=0.1,
        )
        sparse_adapter.begin_time_step(
            time_step,
            contact_stabilization_fraction=0.25,
            contact_dead_zone=0.01,
            impact_velocity_threshold=0.1,
        )
        data.joints.dq_j.fill_(2.0)
        evaluation_velocity = velocity.copy()
        evaluation_velocity[1, 3:] = np.asarray([-0.4, 0.2, 0.7], dtype=np.float32)
        evaluation_velocity[2, 2] = 4.0
        evaluation_velocity[2, 3:] = np.asarray([0.5, -0.6, 0.3], dtype=np.float32)
        data.bodies.u_i.assign(evaluation_velocity)
        contacts.reaction.assign(np.asarray([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
        linearization_twist = evaluation_velocity.copy()
        linearization_twist[1] += np.asarray([0.1, 0.2, -0.3, 0.4, -0.2, 0.1], dtype=np.float32)
        adapter.update(
            time_step,
            joint_penalty_scale=penalty_scale,
            linearization_twist=wp.array(linearization_twist, dtype=vec6f, device=self.device),
        )
        sparse_adapter.update(
            time_step,
            joint_penalty_scale=penalty_scale,
            linearization_twist=wp.array(linearization_twist, dtype=vec6f, device=self.device),
        )

        equivalent_arrays = (
            "dynamic_jacobian_first",
            "dynamic_jacobian_second",
            "dynamic_effective_inertia",
            "dynamic_free_velocity",
            "structural_jacobian_first",
            "structural_jacobian_second",
            "structural_residual",
            "structural_multiplier",
            "structural_delassus",
            "structural_inverse_metric",
            "structural_penalty_metric",
            "limit_jacobian_first",
            "limit_jacobian_second",
            "contact_jacobian_first",
            "contact_jacobian_second",
            "contact_bias",
            "contact_reaction",
        )
        for name in equivalent_arrays:
            np.testing.assert_allclose(
                getattr(sparse_adapter, name).numpy(),
                getattr(adapter, name).numpy(),
                rtol=0.0,
                atol=1.0e-7,
                err_msg=name,
            )
        np.testing.assert_allclose(
            sparse_adapter.system.smooth_matrix.numpy(), adapter.system.smooth_matrix.numpy(), rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            sparse_adapter.system.right_hand_side.numpy(),
            adapter.system.right_hand_side.numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )

        np.testing.assert_array_equal(adapter.system.body_counts, (1, 2))
        np.testing.assert_array_equal(adapter.dynamic_row_world.numpy(), [1])
        np.testing.assert_array_equal(adapter.dynamic_body_first.numpy(), [0])
        np.testing.assert_array_equal(adapter.dynamic_body_second.numpy(), [1])
        np.testing.assert_array_equal(adapter.dynamic_body_first_global.numpy(), [1])
        np.testing.assert_array_equal(adapter.dynamic_body_second_global.numpy(), [2])
        np.testing.assert_array_equal(adapter.dynamic_dof_index.numpy(), [0])
        np.testing.assert_array_equal(adapter.dynamic_velocity_begin.numpy(), [0.0])
        np.testing.assert_array_equal(adapter.structural_multiplier.numpy(), -multiplier[1:])
        np.testing.assert_array_equal(adapter.structural_residual.numpy(), residual)
        np.testing.assert_array_equal(adapter.world_contact_offset.numpy(), [0, 1])
        np.testing.assert_array_equal(adapter.world_contact_count.numpy(), [0, 1])
        np.testing.assert_array_equal(adapter.world_has_unilateral.numpy(), [False, True])
        np.testing.assert_array_equal(adapter.body_has_unilateral.numpy(), [0, 1, 1])
        np.testing.assert_array_equal(adapter.contact_body_first.numpy(), [-1, 1, -1])
        np.testing.assert_array_equal(adapter.contact_body_second.numpy(), [-1, 2, -1])
        np.testing.assert_allclose(adapter.contact_reaction.numpy()[1], time_step * np.asarray([1.0, -2.0, 3.0]))
        self.assertAlmostEqual(float(adapter.contact_friction.numpy()[1]), 0.6, places=6)

        body_dofs = int(model.info.num_body_dofs.numpy()[1])
        jacobian_offset = int(jacobians.data.J_cts_offsets.numpy()[1])
        dense = jacobians.data.J_cts_data.numpy()[jacobian_offset:].reshape(-1, body_dofs)
        np.testing.assert_allclose(adapter.dynamic_jacobian_first.numpy()[0], dense[0, :6], atol=1.0e-7)
        np.testing.assert_allclose(adapter.dynamic_jacobian_second.numpy()[0], dense[0, 6:12], atol=1.0e-7)
        np.testing.assert_allclose(adapter.structural_jacobian_first.numpy(), dense[1:6, :6], atol=1.0e-7)
        np.testing.assert_allclose(adapter.structural_jacobian_second.numpy(), dense[1:6, 6:12], atol=1.0e-7)
        np.testing.assert_allclose(adapter.contact_jacobian_first.numpy()[1], dense[6:9, :6], atol=1.0e-7)
        np.testing.assert_allclose(adapter.contact_jacobian_second.numpy()[1], dense[6:9, 6:12], atol=1.0e-7)

        contact_previous_velocity = dense[6:9] @ velocity[1:].reshape(-1)
        self.assertAlmostEqual(float(contact_previous_velocity[2]), -2.0, places=5)
        np.testing.assert_allclose(adapter.contact_bias.numpy()[1], [0.0, 0.0, -0.8], atol=1.0e-6)

        mass = model.bodies.m_i.numpy()
        inverse_mass = model.bodies.inv_m_i.numpy()
        inertia = data.bodies.I_i.numpy()
        inverse_inertia = data.bodies.inv_I_i.numpy()
        structural_jacobian = dense[1:6]
        inverse_mass_matrix = np.zeros((12, 12), dtype=np.float64)
        inverse_mass_matrix[:6, :6] = _spatial_inverse_mass(float(inverse_mass[1]), inverse_inertia[1])
        inverse_mass_matrix[6:, 6:] = _spatial_inverse_mass(float(inverse_mass[2]), inverse_inertia[2])
        delassus = structural_jacobian @ inverse_mass_matrix @ structural_jacobian.T
        inverse_metric = np.linalg.inv(delassus)
        np.testing.assert_array_equal(adapter.body_constraint_count.numpy()[1:], [2, 2])
        np.testing.assert_array_equal(adapter.structural_block_body_first_multiplicity.numpy(), [1])
        np.testing.assert_array_equal(adapter.structural_block_body_second_multiplicity.numpy(), [1])
        np.testing.assert_array_equal(adapter.structural_block_multiplicity.numpy(), [1])
        np.testing.assert_allclose(adapter.structural_delassus.numpy()[0, :5, :5], delassus, rtol=2.0e-6, atol=2.0e-7)
        np.testing.assert_allclose(
            adapter.structural_inverse_metric.numpy()[0, :5, :5], inverse_metric, rtol=2.0e-5, atol=2.0e-6
        )
        np.testing.assert_allclose(
            adapter.structural_effective_mass.numpy(), np.diag(inverse_metric), rtol=2.0e-5, atol=2.0e-6
        )

        expected_matrix = np.zeros((12, 12), dtype=np.float64)
        expected_matrix[:6, :6] = _spatial_mass(float(mass[1]), inertia[1])
        expected_matrix[6:, 6:] = _spatial_mass(float(mass[2]), inertia[2])
        expected_matrix += 2.5 * np.outer(dense[0], dense[0])
        expected_matrix += penalty_scale * structural_jacobian.T @ inverse_metric @ structural_jacobian

        gravity_data = model.gravity.vector.numpy()
        expected_rhs = np.zeros(12, dtype=np.float64)
        for local, body in enumerate((1, 2)):
            body_mass = _spatial_mass(float(mass[body]), inertia[body])
            omega = evaluation_velocity[body, 3:]
            gravity = gravity_data[1]
            explicit = external[body] + actuation[body]
            explicit = explicit.copy()
            explicit[:3] += mass[body] * gravity
            explicit[3:] -= np.cross(omega, inertia[body] @ omega)
            expected_rhs[6 * local : 6 * (local + 1)] = body_mass @ velocity[body] + time_step * explicit
        expected_dynamic_free_velocity = -0.35 + 0.75 * (0.0 - 2.0) / 2.5
        self.assertAlmostEqual(float(adapter.dynamic_free_velocity.numpy()[0]), expected_dynamic_free_velocity)
        expected_rhs += 2.5 * expected_dynamic_free_velocity * dense[0]
        penalty = penalty_scale * inverse_metric / (time_step * time_step)
        internal_multiplier = -multiplier[1:]
        expected_rhs += structural_jacobian.T @ (-time_step * (internal_multiplier + penalty @ residual))
        row_linearization_velocity = structural_jacobian @ linearization_twist[1:].reshape(-1)
        expected_rhs += structural_jacobian.T @ (time_step**2 * penalty @ row_linearization_velocity)
        matrices = _extract_world_matrices(adapter)
        vectors = _extract_world_vectors(adapter)
        np.testing.assert_allclose(matrices[1], expected_matrix, rtol=5.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(vectors[1], expected_rhs, rtol=5.0e-6, atol=3.0e-5)

        final_body_velocity = np.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]],
            dtype=np.float32,
        )
        adapter.projected_twist.assign(final_body_velocity)
        contact_impulses = adapter.contact_reaction.numpy()
        contact_impulses[1] = np.asarray([0.2, 0.4, 0.6], dtype=np.float32)
        adapter.contact_reaction.assign(contact_impulses)
        contact_velocity = adapter.contact_velocity.numpy()
        contact_velocity[1] = np.asarray([0.7, -0.2, 0.3], dtype=np.float32)
        adapter.contact_velocity.assign(contact_velocity)
        structural_output = adapter.structural_multiplier.numpy()
        adapter.write_outputs(time_step)
        np.testing.assert_allclose(data.bodies.u_i.numpy(), final_body_velocity, atol=1.0e-7)
        np.testing.assert_allclose(contacts.reaction.numpy()[0], [2.0, 4.0, 6.0], atol=1.0e-6)
        np.testing.assert_allclose(contacts.velocity.numpy()[0], [0.7, -0.2, 0.3], atol=1.0e-7)
        np.testing.assert_array_equal(contacts.mode.numpy()[0], 0)
        expected_dynamic_reaction = (
            2.5 * (expected_dynamic_free_velocity - dense[0] @ final_body_velocity[1:].reshape(-1)) / time_step
        )
        np.testing.assert_allclose(data.joints.lambda_j.numpy()[0], expected_dynamic_reaction, rtol=1.0e-6, atol=1.0e-5)
        np.testing.assert_allclose(data.joints.lambda_j.numpy()[1:], -structural_output, atol=1.0e-7)
        expected_joint_wrench = expected_dynamic_reaction * dense[0]
        expected_joint_wrench -= structural_output @ dense[1:6]
        np.testing.assert_allclose(data.bodies.w_j_i.numpy()[1:].reshape(-1), expected_joint_wrench, atol=2.0e-5)
        expected_contact_wrench = dense[6:9].T @ np.asarray([2.0, 4.0, 6.0])
        np.testing.assert_allclose(data.bodies.w_c_i.numpy()[1:].reshape(-1), expected_contact_wrench, atol=1.0e-6)

    def test_structural_mass_split_scales_each_body_by_joint_incidence(self):
        model = build_cartpole(ground=False, limits=False).finalize(device=self.device)
        data = model.data()
        make_unilateral_constraints_info(model, data)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)

        time_step = 0.01
        adapter.begin_time_step(time_step)
        adapter.update(
            time_step,
            block_joint_metrics=True,
            mass_split_joint_metrics=True,
        )

        np.testing.assert_array_equal(adapter.static_body_constraint_count.numpy(), [2, 1])
        np.testing.assert_array_equal(adapter.structural_block_body_first_multiplicity.numpy(), [0, 2])
        np.testing.assert_array_equal(adapter.structural_block_body_second_multiplicity.numpy(), [2, 1])

        inverse_mass = model.bodies.inv_m_i.numpy()
        inverse_inertia = data.bodies.inv_I_i.numpy()
        jacobian_first = adapter.structural_jacobian_first.numpy()
        jacobian_second = adapter.structural_jacobian_second.numpy()
        offsets = adapter.structural_block_row_offset.numpy()
        counts = adapter.structural_block_row_count.numpy()
        bodies_first = adapter.structural_block_body_first_global.numpy()
        bodies_second = adapter.structural_block_body_second_global.numpy()
        multiplicities_first = adapter.structural_block_body_first_multiplicity.numpy()
        multiplicities_second = adapter.structural_block_body_second_multiplicity.numpy()
        physical_blocks = adapter.structural_delassus.numpy()
        inverse_blocks = adapter.structural_inverse_metric.numpy()

        for block in range(adapter.structural_block_count):
            offset = int(offsets[block])
            count = int(counts[block])
            first = int(bodies_first[block])
            second = int(bodies_second[block])
            first_contribution = np.zeros((count, count), dtype=np.float64)
            second_contribution = np.zeros((count, count), dtype=np.float64)
            if first >= 0:
                first_rows = jacobian_first[offset : offset + count]
                first_contribution = (
                    first_rows
                    @ _spatial_inverse_mass(float(inverse_mass[first]), inverse_inertia[first])
                    @ first_rows.T
                )
            if second >= 0:
                second_rows = jacobian_second[offset : offset + count]
                second_contribution = (
                    second_rows
                    @ _spatial_inverse_mass(float(inverse_mass[second]), inverse_inertia[second])
                    @ second_rows.T
                )

            physical = first_contribution + second_contribution
            split = (
                int(multiplicities_first[block]) * first_contribution
                + int(multiplicities_second[block]) * second_contribution
            )
            np.testing.assert_allclose(physical_blocks[block, :count, :count], physical, rtol=2.0e-6, atol=2.0e-7)
            np.testing.assert_allclose(
                inverse_blocks[block, :count, :count], np.linalg.inv(split), rtol=2.0e-5, atol=2.0e-6
            )
            if first >= 0 and second >= 0:
                max_scaled = max(int(multiplicities_first[block]), int(multiplicities_second[block])) * physical
                self.assertGreater(float(np.linalg.norm(split - max_scaled)), 1.0e-4)

        adapter.update(time_step, block_joint_metrics=True, mass_split_joint_metrics=False)
        np.testing.assert_array_equal(adapter.structural_block_body_first_multiplicity.numpy(), [0, 1])
        np.testing.assert_array_equal(adapter.structural_block_body_second_multiplicity.numpy(), [1, 1])
        for block in range(adapter.structural_block_count):
            count = int(counts[block])
            np.testing.assert_allclose(
                adapter.structural_inverse_metric.numpy()[block, :count, :count],
                np.linalg.inv(physical_blocks[block, :count, :count]),
                rtol=2.0e-5,
                atol=2.0e-6,
            )

    def test_scale_one_structural_update_matches_block_jacobi(self):
        model = build_cartpole(ground=False, limits=False).finalize(device=self.device)
        data = model.data()
        make_unilateral_constraints_info(model, data)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)

        time_step = 0.01
        linearization_twist = np.asarray(
            [[0.1, -0.2, 0.3, -0.4, 0.2, 0.1], [-0.3, 0.1, -0.2, 0.2, -0.1, 0.4]],
            dtype=np.float32,
        )
        candidate_twist = np.asarray(
            [[-0.2, 0.4, 0.1, 0.3, -0.5, 0.2], [0.5, -0.3, 0.2, -0.1, 0.4, -0.2]],
            dtype=np.float32,
        )
        adapter.begin_time_step(time_step)
        adapter.update(
            time_step,
            joint_penalty_scale=1.0,
            linearization_twist=wp.array(linearization_twist, dtype=vec6f, device=self.device),
            block_joint_metrics=True,
            mass_split_joint_metrics=True,
        )
        adapter.projection_status.fill_(PROJECTION_STATUS_VALID)

        offsets = adapter.structural_block_row_offset.numpy()
        counts = adapter.structural_block_row_count.numpy()
        bodies_first = adapter.structural_block_body_first_global.numpy()
        bodies_second = adapter.structural_block_body_second_global.numpy()
        jacobian_first = adapter.structural_jacobian_first.numpy()
        jacobian_second = adapter.structural_jacobian_second.numpy()
        residual = adapter.structural_residual.numpy()
        inverse_mass = model.bodies.inv_m_i.numpy()
        inverse_inertia = data.bodies.inv_I_i.numpy()

        incidence = np.zeros(model.size.sum_of_num_bodies, dtype=np.int32)
        for first, second in zip(bodies_first, bodies_second, strict=True):
            if first >= 0:
                incidence[first] += 1
            if second >= 0 and second != first:
                incidence[second] += 1

        expected_impulse_delta = np.zeros(adapter.structural_row_count, dtype=np.float64)
        for block in range(adapter.structural_block_count):
            offset = int(offsets[block])
            count = int(counts[block])
            first = int(bodies_first[block])
            second = int(bodies_second[block])
            first_rows = jacobian_first[offset : offset + count]
            second_rows = jacobian_second[offset : offset + count]
            split_delassus = np.zeros((count, count), dtype=np.float64)
            velocity_residual = residual[offset : offset + count].astype(np.float64) / time_step
            if first >= 0:
                split_delassus += (
                    int(incidence[first])
                    * first_rows
                    @ _spatial_inverse_mass(float(inverse_mass[first]), inverse_inertia[first])
                    @ first_rows.T
                )
                velocity_residual += first_rows @ (candidate_twist[first] - linearization_twist[first])
            if second >= 0:
                split_delassus += (
                    int(incidence[second])
                    * second_rows
                    @ _spatial_inverse_mass(float(inverse_mass[second]), inverse_inertia[second])
                    @ second_rows.T
                )
                velocity_residual += second_rows @ (candidate_twist[second] - linearization_twist[second])
            expected_impulse_delta[offset : offset + count] = np.linalg.solve(split_delassus, velocity_residual)

        multiplier_before = adapter.structural_multiplier.numpy().copy()
        candidate_twist_device = wp.array(candidate_twist, dtype=vec6f, device=self.device)
        adapter.update_structural_multipliers_from_twist(
            time_step,
            1.0e-5,
            wp.array(linearization_twist, dtype=vec6f, device=self.device),
            candidate_twist_device,
            candidate_twist_device,
            wp.ones(model.info.num_worlds, dtype=wp.bool, device=self.device),
            projected_fraction=0.0,
        )
        actual_impulse_delta = time_step * (adapter.structural_multiplier.numpy() - multiplier_before)

        np.testing.assert_allclose(actual_impulse_delta, expected_impulse_delta, rtol=3.0e-5, atol=3.0e-6)

    def test_structural_penalty_scale_is_per_world(self):
        builder = ModelBuilderKamino(default_world=False)
        builder.add_builder(build_boxes_hinged(ground=False))
        builder.add_builder(build_boxes_hinged(ground=False))
        model = builder.finalize(device=self.device)
        data = model.data()
        make_unilateral_constraints_info(model, data)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)

        time_step = 0.01
        adapter.begin_time_step(time_step)
        adapter.update(
            time_step,
            joint_penalty_scale=wp.array([2.0, 5.0], dtype=wp.float32, device=self.device),
            block_joint_metrics=True,
            mass_split_joint_metrics=False,
        )

        block_world = adapter.structural_block_world.numpy()
        block_count = adapter.structural_block_row_count.numpy()
        penalty = adapter.structural_penalty_metric.numpy()
        first = int(np.flatnonzero(block_world == 0)[0])
        second = int(np.flatnonzero(block_world == 1)[0])
        count = int(block_count[first])
        self.assertEqual(count, int(block_count[second]))
        np.testing.assert_allclose(
            penalty[second, :count, :count],
            2.5 * penalty[first, :count, :count],
            rtol=2.0e-5,
            atol=2.0e-6,
        )

    def test_structural_multiplier_update_modifies_smooth_rhs(self):
        model = build_boxes_hinged(ground=False, dynamic_joints=True).finalize(device=self.device)
        data = model.data()
        make_unilateral_constraints_info(model, data)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))
        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)

        time_step = 0.01
        tolerance = 2.0e-4
        projected_fraction = 0.5
        linearization_twist = np.zeros((model.size.sum_of_num_bodies, 6), dtype=np.float32)
        global_twist = np.asarray(
            [[0.2, -0.1, 0.3, 0.4, -0.2, 0.1], [-0.1, 0.3, -0.2, 0.2, 0.1, -0.4]],
            dtype=np.float32,
        )
        projected_twist = np.asarray(
            [[-0.1, 0.2, 0.1, -0.3, 0.4, 0.2], [0.3, -0.2, 0.4, -0.1, 0.2, 0.5]],
            dtype=np.float32,
        )
        adapter.begin_time_step(time_step)
        adapter.update(
            time_step,
            joint_penalty_scale=3.0,
            linearization_twist=wp.array(linearization_twist, dtype=vec6f, device=self.device),
        )
        adapter.projection_status.fill_(PROJECTION_STATUS_VALID)

        residual = adapter.structural_residual.numpy().copy()
        penalty = adapter.structural_penalty_metric.numpy()[
            0, : adapter.structural_row_count, : adapter.structural_row_count
        ]
        multiplier = adapter.structural_multiplier.numpy().copy()
        jacobian_first = adapter.structural_jacobian_first.numpy()
        jacobian_second = adapter.structural_jacobian_second.numpy()
        first_global = adapter.structural_body_first_global.numpy()
        second_global = adapter.structural_body_second_global.numpy()
        global_row_velocity = np.zeros(adapter.structural_row_count, dtype=np.float32)
        projected_row_velocity = np.zeros(adapter.structural_row_count, dtype=np.float32)
        for row in range(adapter.structural_row_count):
            if first_global[row] >= 0:
                global_row_velocity[row] += jacobian_first[row] @ global_twist[first_global[row]]
                projected_row_velocity[row] += jacobian_first[row] @ projected_twist[first_global[row]]
            if second_global[row] >= 0:
                global_row_velocity[row] += jacobian_second[row] @ global_twist[second_global[row]]
                projected_row_velocity[row] += jacobian_second[row] @ projected_twist[second_global[row]]
        global_residual = residual + time_step * global_row_velocity
        projected_residual = residual + time_step * projected_row_velocity
        update_residual = global_residual + projected_fraction * (projected_residual - global_residual)
        multiplier_delta = penalty @ update_residual
        expected_rhs = adapter.system.right_hand_side.numpy().copy()
        vector_offset = int(adapter.system.info.vio.numpy()[0])
        first_local = adapter.structural_body_first.numpy()
        second_local = adapter.structural_body_second.numpy()
        for row in range(adapter.structural_row_count):
            if first_local[row] >= 0:
                begin = vector_offset + 6 * first_local[row]
                expected_rhs[begin : begin + 6] -= time_step * multiplier_delta[row] * jacobian_first[row]
            if second_local[row] >= 0:
                begin = vector_offset + 6 * second_local[row]
                expected_rhs[begin : begin + 6] -= time_step * multiplier_delta[row] * jacobian_second[row]

        adapter.update_structural_multipliers_from_twist(
            time_step,
            tolerance,
            wp.array(linearization_twist, dtype=vec6f, device=self.device),
            wp.array(global_twist, dtype=vec6f, device=self.device),
            wp.array(projected_twist, dtype=vec6f, device=self.device),
            wp.ones(model.info.num_worlds, dtype=wp.bool, device=self.device),
            projected_fraction=projected_fraction,
        )

        np.testing.assert_allclose(adapter.structural_candidate_residual.numpy(), global_residual, atol=1.0e-7)
        np.testing.assert_allclose(adapter.structural_projected_residual.numpy(), projected_residual, atol=1.0e-7)
        np.testing.assert_allclose(adapter.structural_multiplier.numpy(), multiplier + multiplier_delta, atol=1.0e-5)
        np.testing.assert_allclose(data.joints.lambda_j.numpy()[1:], -(multiplier + multiplier_delta), atol=1.0e-5)
        np.testing.assert_allclose(adapter.system.right_hand_side.numpy(), expected_rhs, atol=2.0e-5)
        self.assertAlmostEqual(
            float(adapter.world_structural_residual.numpy()[0]),
            float(np.max(np.abs(global_residual)) / tolerance),
            places=4,
        )
        self.assertAlmostEqual(
            float(adapter.world_projected_structural_residual.numpy()[0]),
            float(np.max(np.abs(projected_residual)) / tolerance),
            places=4,
        )

    def test_limit_mapping_bias_and_force_impulse_conversion(self):
        model = build_cartpole(ground=False, limits=True).finalize(device=self.device)
        data = model.data()
        limits = LimitsKamino(model)
        make_unilateral_constraints_info(model, data, limits=limits)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))

        limits.model_active_limits.assign(np.asarray([1], dtype=np.int32))
        limits.world_active_limits.assign(np.asarray([1], dtype=np.int32))
        limits.wid.assign(np.asarray([0], dtype=np.int32))
        limits.lid.assign(np.asarray([0], dtype=np.int32))
        limits.jid.assign(np.asarray([0], dtype=np.int32))
        limits.bids.assign(np.asarray([[-1, 0]], dtype=np.int32))
        limits.dof.assign(np.asarray([0], dtype=np.int32))
        limits.side.assign(np.asarray([1.0], dtype=np.float32))
        limits.r_q.assign(np.asarray([-0.4], dtype=np.float32))
        limits.reaction.assign(np.asarray([5.0], dtype=np.float32))
        update_constraints_info(model, data)

        jacobians = DenseSystemJacobians(model=model, limits=limits)
        jacobians.build(model=model, data=data, limits=limits)
        sparse_jacobians = SparseSystemJacobians(model=model, limits=limits)
        sparse_jacobians.build(model=model, data=data, limits=limits)
        adapter = LOXKaminoAdapter(model, data, jacobians, limits=limits)
        sparse_adapter = LOXKaminoAdapter(model, data, sparse_jacobians, limits=limits)
        time_step = 0.2
        adapter.begin_time_step(time_step, limit_stabilization_fraction=0.25)
        sparse_adapter.begin_time_step(time_step, limit_stabilization_fraction=0.25)
        limits.reaction.assign(np.asarray([50.0], dtype=np.float32))
        adapter.update(time_step)
        sparse_adapter.update(time_step)

        for name in (
            "limit_jacobian_first",
            "limit_jacobian_second",
            "limit_bias",
            "limit_reaction",
        ):
            np.testing.assert_allclose(
                getattr(sparse_adapter, name).numpy(),
                getattr(adapter, name).numpy(),
                rtol=0.0,
                atol=1.0e-7,
                err_msg=name,
            )

        np.testing.assert_array_equal(adapter.world_limit_count.numpy(), [1])
        np.testing.assert_array_equal(adapter.limit_body_first.numpy(), [-1])
        np.testing.assert_array_equal(adapter.limit_body_second.numpy(), [0])
        np.testing.assert_array_equal(adapter.body_constraint_count.numpy(), [3, 1])
        self.assertAlmostEqual(float(adapter.limit_bias.numpy()[0]), -0.5, places=6)
        self.assertAlmostEqual(float(adapter.limit_reaction.numpy()[0]), 1.0, places=6)

        body_dofs = int(model.info.num_body_dofs.numpy()[0])
        dense = jacobians.data.J_cts_data.numpy().reshape(-1, body_dofs)
        limit_row = int(data.info.limit_cts_group_offset.numpy()[0])
        np.testing.assert_allclose(adapter.limit_jacobian_second.numpy()[0], dense[limit_row, :6], atol=1.0e-7)

        adapter.limit_reaction.assign(np.asarray([1.4], dtype=np.float32))
        adapter.limit_velocity.assign(np.asarray([0.35], dtype=np.float32))
        adapter.write_outputs(
            time_step, body_velocity=wp.zeros(model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        )
        self.assertAlmostEqual(float(limits.reaction.numpy()[0]), 7.0, places=6)
        self.assertAlmostEqual(float(limits.velocity.numpy()[0]), 0.35, places=6)

    def test_dynamic_position_drive_relinearization_rhs(self):
        model = build_boxes_hinged(ground=False, dynamic_joints=True, implicit_pd=True).finalize(device=self.device)
        data = model.data()
        make_unilateral_constraints_info(model, data)
        update_body_inertias(model.bodies, data.bodies)
        compute_joints_data(model, data, q_j_p=wp.zeros_like(data.joints.q_j))

        jacobians = DenseSystemJacobians(model=model)
        jacobians.build(model=model, data=data)
        adapter = LOXKaminoAdapter(model, data, jacobians)
        time_step = 0.1
        adapter.begin_time_step(time_step)

        model.joints.a_j.fill_(0.75)
        model.joints.k_p_j.fill_(100.0)
        data.joints.m_j.assign(np.asarray([2.5], dtype=np.float32))
        data.joints.dq_b_j.assign(np.asarray([-0.35], dtype=np.float32))
        data.joints.dq_j.assign(np.asarray([2.0], dtype=np.float32))

        linearization_twist = np.zeros((2, 6), dtype=np.float32)
        dynamic_jacobian = jacobians.data.J_cts_data.numpy().reshape(-1, 12)[0]
        linearization_twist[0] = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32)
        linearization_twist[1] = np.asarray([-0.2, 0.1, -0.4, 0.3, -0.5, 0.7], dtype=np.float32)
        linearization_velocity = float(dynamic_jacobian @ linearization_twist.reshape(-1))

        adapter.update(
            time_step,
            linearization_twist=wp.array(linearization_twist, dtype=vec6f, device=self.device),
        )

        expected = -0.35 + 0.75 * (0.0 - 2.0) / 2.5
        expected += time_step * time_step * 100.0 * linearization_velocity / 2.5
        self.assertAlmostEqual(float(adapter.dynamic_free_velocity.numpy()[0]), expected, places=6)


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
