# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX colored Gauss--Seidel projection."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat36f, mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox.colored_gauss_seidel import (
    ColoredGaussSeidelProjection,
    _assign_two_endpoint_colors,
    _ColorFamily,
    _count_two_endpoint_occupancy,
    _project_deformable_colored,
)
from newton._src.solvers.kamino._src.solvers.lox.deformable_contact import (
    DEFORMABLE_CONTACT_STATUS_CROSS_WORLD,
    DEFORMABLE_CONTACT_STATUS_MALFORMED,
    DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
    DEFORMABLE_CONTACT_STATUS_VALID,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _empty(device, dtype):
    return wp.empty(0, dtype=dtype, device=device)


def _make_adapter(device, endpoints):
    count = len(endpoints)
    empty_int = _empty(device, wp.int32)
    empty_vec6 = _empty(device, vec6f)
    contact_jacobian = np.zeros((count, 3, 6), dtype=np.float32)
    contact_jacobian[:, 0, 1] = 1.0
    contact_jacobian[:, 1, 2] = 1.0
    contact_jacobian[:, 2, 0] = 1.0
    contact_jacobian_first = wp.array(contact_jacobian, dtype=mat36f, device=device)
    contact_jacobian_second = wp.zeros(count, dtype=mat36f, device=device)
    body_count = max((max(pair) for pair in endpoints), default=-1) + 1
    return SimpleNamespace(
        device=device,
        body_constraint_count=wp.zeros(body_count, dtype=wp.int32, device=device),
        static_body_constraint_count=wp.zeros(body_count, dtype=wp.int32, device=device),
        friction_capacity=0,
        friction_world=empty_int,
        friction_local=empty_int,
        world_friction_count=wp.zeros(1, dtype=wp.int32, device=device),
        friction_body_first=empty_int,
        friction_body_second=empty_int,
        friction_jacobian_first=empty_vec6,
        friction_jacobian_second=empty_vec6,
        friction_impulse_bound=_empty(device, wp.float32),
        friction_projection_delassus=_empty(device, wp.float32),
        friction_reaction=_empty(device, wp.float32),
        contact_capacity=count,
        contact_world=wp.zeros(count, dtype=wp.int32, device=device),
        contact_local=wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=device),
        world_contact_count=wp.array([count], dtype=wp.int32, device=device),
        contact_body_first=wp.array([pair[0] for pair in endpoints], dtype=wp.int32, device=device),
        contact_body_second=wp.array([pair[1] for pair in endpoints], dtype=wp.int32, device=device),
        contact_jacobian_first=contact_jacobian_first,
        contact_jacobian_second=contact_jacobian_second,
        contact_bias=wp.zeros(count, dtype=wp.vec3f, device=device),
        contact_friction=wp.zeros(count, dtype=wp.float32, device=device),
        contact_projection_delassus=wp.zeros(count, dtype=wp.mat33f, device=device),
        contact_projection_delassus_normal_first=wp.zeros(count, dtype=wp.mat33f, device=device),
        contact_reaction=wp.zeros(count, dtype=wp.vec3f, device=device),
        limit_capacity=0,
        limit_world=empty_int,
        limit_local=empty_int,
        world_limit_count=wp.zeros(1, dtype=wp.int32, device=device),
        limit_body_first=empty_int,
        limit_body_second=empty_int,
        limit_jacobian_first=empty_vec6,
        limit_jacobian_second=empty_vec6,
        limit_bias=_empty(device, wp.float32),
        limit_projection_delassus=_empty(device, wp.float32),
        limit_reaction=_empty(device, wp.float32),
    )


def _make_all_rigid_family_adapter(device):
    endpoints = [(0, -1), (0, 1), (0, -1), (2, -1), (3, 4), (5, -1), (6, 0), (7, -1)]
    adapter = _make_adapter(device, endpoints)
    count = len(endpoints)
    scalar_jacobian_first = np.zeros((count, 6), dtype=np.float32)
    scalar_jacobian_first[:, 0] = 1.0
    scalar_jacobian_second = np.zeros((count, 6), dtype=np.float32)
    scalar_jacobian_second[:, 0] = -0.25
    contact_jacobian_second = -0.25 * adapter.contact_jacobian_first.numpy()

    adapter.contact_jacobian_second.assign(contact_jacobian_second)
    adapter.contact_bias.assign([[-0.15, 0.1, -1.0 - 0.05 * constraint] for constraint in range(count)])
    adapter.contact_friction.fill_(0.4)
    for family, biases in (("friction", None), ("limit", -0.6 - 0.03 * np.arange(count))):
        setattr(adapter, f"{family}_capacity", count)
        setattr(adapter, f"{family}_world", wp.zeros(count, dtype=wp.int32, device=device))
        setattr(
            adapter,
            f"{family}_local",
            wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=device),
        )
        setattr(adapter, f"world_{family}_count", wp.array([count], dtype=wp.int32, device=device))
        setattr(
            adapter,
            f"{family}_body_first",
            wp.array([pair[0] for pair in endpoints], dtype=wp.int32, device=device),
        )
        setattr(
            adapter,
            f"{family}_body_second",
            wp.array([pair[1] for pair in endpoints], dtype=wp.int32, device=device),
        )
        setattr(
            adapter,
            f"{family}_jacobian_first",
            wp.array(scalar_jacobian_first, dtype=vec6f, device=device),
        )
        setattr(
            adapter,
            f"{family}_jacobian_second",
            wp.array(scalar_jacobian_second, dtype=vec6f, device=device),
        )
        setattr(
            adapter,
            f"{family}_projection_delassus",
            wp.zeros(count, dtype=wp.float32, device=device),
        )
        setattr(adapter, f"{family}_reaction", wp.zeros(count, dtype=wp.float32, device=device))
        if biases is not None:
            setattr(adapter, f"{family}_bias", wp.array(biases, dtype=wp.float32, device=device))
    adapter.friction_impulse_bound = wp.full(count, 10.0, dtype=wp.float32, device=device)
    return adapter


def _multiplicity_objective(occupancy):
    return int(np.sum(np.asarray(occupancy, dtype=np.int64) ** 2))


def _make_deformable(device, *, body, status, world_status, global_status):
    particle_indices = wp.array([[0, -1, -1, -1]], dtype=wp.int32, device=device)
    deformable = SimpleNamespace(
        device=device,
        contact_capacity=1,
        cloth_system=SimpleNamespace(
            particle_count=1,
            inverse_weight=wp.ones(1, dtype=wp.float32, device=device),
        ),
        particle_indices=particle_indices,
        coefficients=wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32, device=device),
        contact_world=wp.zeros(1, dtype=wp.int32, device=device),
        body=wp.array([body], dtype=wp.int32, device=device),
        body_jacobian=wp.zeros(1, dtype=mat36f, device=device),
        normal=wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3f, device=device),
        frame=wp.array([np.eye(3, dtype=np.float32)], dtype=wp.mat33f, device=device),
        bias=wp.zeros(1, dtype=wp.vec3f, device=device),
        rigid_bias=wp.zeros(1, dtype=wp.vec3f, device=device),
        friction=wp.zeros(1, dtype=wp.float32, device=device),
        status=wp.full(1, status, dtype=wp.int32, device=device),
        gauss_seidel_scalar_delassus=wp.zeros(1, dtype=wp.float32, device=device),
        gauss_seidel_delassus=wp.zeros(1, dtype=wp.mat33f, device=device),
        gauss_seidel_delassus_normal_first=wp.zeros(1, dtype=wp.mat33f, device=device),
        rigid_delassus=wp.zeros(1, dtype=wp.mat33f, device=device),
        rigid_delassus_normal_first=wp.zeros(1, dtype=wp.mat33f, device=device),
        reaction=wp.zeros(1, dtype=wp.vec3f, device=device),
        rigid_reaction=wp.zeros(1, dtype=wp.vec3f, device=device),
        particle_delta=wp.zeros(1, dtype=wp.vec3f, device=device),
        world_status=wp.array([world_status], dtype=wp.int32, device=device),
        global_status=wp.array([global_status], dtype=wp.int32, device=device),
        invalid_count=wp.full(
            1,
            int(status != DEFORMABLE_CONTACT_STATUS_VALID),
            dtype=wp.int32,
            device=device,
        ),
        _empty_body_inverse_weight=wp.empty(0, dtype=mat66f, device=device),
    )

    def prepare_rigid_projection(*_args):
        return None

    deformable.prepare_rigid_projection = prepare_rigid_projection
    return deformable


class TestLOXColoredGaussSeidel(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_repair_is_deterministic_and_balances_incidence(self):
        """Reduce endpoint multiplicity deterministically with bounded repair passes."""
        endpoints = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)]
        first = ColoredGaussSeidelProjection(_make_adapter(self.device, endpoints), None, 4)
        first.build_colors()
        first_colors = first.contact.colors.numpy()
        first_occupancy = first.body_occupancy.numpy()

        second = ColoredGaussSeidelProjection(_make_adapter(self.device, endpoints), None, 4)
        second.build_colors()
        np.testing.assert_array_equal(first_colors, second.contact.colors.numpy())
        np.testing.assert_array_equal(first_occupancy, second.body_occupancy.numpy())
        self.assertTrue(np.all((first_colors >= 0) & (first_colors < 4)))
        np.testing.assert_array_equal(np.sort(first.contact.order.numpy()), np.arange(len(endpoints)))
        self.assertLessEqual(np.max(first_occupancy[0]), 3)

    def test_repair_does_not_increase_multiplicity_objective(self):
        """Never increase sum of squared endpoint/color multiplicities."""
        endpoints = [(0, 1)] * 12 + [(0, 2)] * 7 + [(0, 3)] * 5
        projection = ColoredGaussSeidelProjection(_make_adapter(self.device, endpoints), None, 5)
        projection.body_occupancy.zero_()
        projection._launch_rigid_families(_assign_two_endpoint_colors)
        projection._launch_rigid_families(_count_two_endpoint_occupancy)
        initial = _multiplicity_objective(projection.body_occupancy.numpy())
        projection.build_colors()
        repaired = _multiplicity_objective(projection.body_occupancy.numpy())

        colors = projection.contact.colors.numpy()
        self.assertLessEqual(repaired, initial)
        self.assertEqual(len(colors), len(endpoints))

    def test_colors_propagate_updates_sequentially(self):
        """Expose each completed color's velocity update to the following color."""
        adapter = _make_adapter(self.device, [(0, -1), (0, -1)])
        adapter.contact_bias.assign([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
        projection = ColoredGaussSeidelProjection(adapter, None, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device)
        prepared_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        projection.prepare(inverse_weight, prepared_status)
        self.assertEqual(len(np.unique(projection.contact.colors.numpy())), 2)

        projected_twist = wp.zeros(1, dtype=vec6f, device=self.device)
        projection_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        projection.project(
            1,
            wp.ones(1, dtype=wp.bool, device=self.device),
            wp.zeros(1, dtype=wp.int32, device=self.device),
            inverse_weight,
            projected_twist,
            wp.zeros(1, dtype=vec6f, device=self.device),
            None,
            prepared_status,
            projection_status,
        )
        np.testing.assert_allclose(projected_twist.numpy()[0, 0], 2.0, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(adapter.contact_reaction.numpy()[:, 2], [0.0, 2.0], rtol=0.0, atol=2.0e-6)

    def test_projection_clears_scratch_across_consecutive_calls(self):
        """Clear stale body deltas after valid and rejected projection calls."""
        adapter = _make_adapter(self.device, [(0, -1)])
        adapter.contact_bias.assign([[0.0, 0.0, -1.0]])
        projection = ColoredGaussSeidelProjection(adapter, None, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device)
        prepared_status = wp.ones(1, dtype=wp.int32, device=self.device)
        projection.prepare(inverse_weight, prepared_status)
        twist_delta = wp.full(1, vec6f(3.0), dtype=vec6f, device=self.device)
        projected_twist = wp.zeros(1, dtype=vec6f, device=self.device)
        projection_status = wp.ones(1, dtype=wp.int32, device=self.device)
        arguments = (
            wp.ones(1, dtype=wp.bool, device=self.device),
            wp.zeros(1, dtype=wp.int32, device=self.device),
            inverse_weight,
            projected_twist,
            twist_delta,
            None,
            prepared_status,
            projection_status,
        )

        projection.project(1, *arguments)
        np.testing.assert_array_equal(twist_delta.numpy(), np.zeros((1, 6), dtype=np.float32))

        twist_delta.fill_(vec6f(5.0))
        prepared_status.zero_()
        projection.project(1, *arguments)
        np.testing.assert_array_equal(projection_status.numpy(), [0])
        np.testing.assert_array_equal(twist_delta.numpy(), np.zeros((1, 6), dtype=np.float32))

    def test_metrics_use_exact_per_color_multiplicity(self):
        """Scale every rigid contact block by its endpoint occupancy in that color."""
        adapter = _make_adapter(self.device, [(0, -1), (0, -1), (0, -1)])
        projection = ColoredGaussSeidelProjection(adapter, None, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device)
        projection.prepare(inverse_weight, wp.zeros(1, dtype=wp.int32, device=self.device))

        occupancy = projection.body_occupancy.numpy()
        colors = projection.contact.colors.numpy()
        blocks = projection.contact_delassus.numpy()
        for contact, color in enumerate(colors):
            np.testing.assert_allclose(
                np.diag(blocks[contact]),
                np.full(3, occupancy[0, color], dtype=np.float32),
                rtol=0.0,
                atol=0.0,
            )

    def test_invalid_world_discards_only_its_color_delta(self):
        """Keep a malformed world's partial update out of every color barrier."""
        adapter = _make_adapter(self.device, [(0, -1), (1, -1)])
        adapter.contact_world.assign([0, 1])
        adapter.contact_local.assign([0, 0])
        adapter.world_friction_count = wp.zeros(2, dtype=wp.int32, device=self.device)
        adapter.world_contact_count = wp.ones(2, dtype=wp.int32, device=self.device)
        adapter.world_limit_count = wp.zeros(2, dtype=wp.int32, device=self.device)
        jacobians = adapter.contact_jacobian_first.numpy()
        jacobians[1] = 0.0
        adapter.contact_jacobian_first.assign(jacobians)
        adapter.contact_bias.assign([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
        projection = ColoredGaussSeidelProjection(adapter, None, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)] * 2, dtype=mat66f, device=self.device)
        prepared_status = wp.zeros(2, dtype=wp.int32, device=self.device)
        projection.prepare(inverse_weight, prepared_status)
        np.testing.assert_array_equal(prepared_status.numpy(), [1, 0])

        projected_twist = wp.zeros(2, dtype=vec6f, device=self.device)
        projection_status = wp.zeros(2, dtype=wp.int32, device=self.device)
        twist_delta = wp.zeros(2, dtype=vec6f, device=self.device)
        projection.project(
            1,
            wp.ones(2, dtype=wp.bool, device=self.device),
            wp.array([0, 1], dtype=wp.int32, device=self.device),
            inverse_weight,
            projected_twist,
            twist_delta,
            None,
            prepared_status,
            projection_status,
        )
        np.testing.assert_allclose(projected_twist.numpy()[:, 0], [1.0, 0.0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(projection_status.numpy(), [1, 0])
        np.testing.assert_array_equal(twist_delta.numpy(), np.zeros((2, 6), dtype=np.float32))

    def test_pure_deformable_preserves_global_preparation_failure(self):
        """Reject a pure deformable world when contact adaptation failed globally."""
        deformable = _make_deformable(
            self.device,
            body=-1,
            status=DEFORMABLE_CONTACT_STATUS_MALFORMED,
            world_status=0,
            global_status=DEFORMABLE_CONTACT_STATUS_MALFORMED,
        )
        projection = ColoredGaussSeidelProjection(None, deformable, 2)
        prepared_status = wp.ones(1, dtype=wp.int32, device=self.device)

        projection.prepare(None, prepared_status)

        np.testing.assert_array_equal(prepared_status.numpy(), [0])

    def test_mixed_deformable_preserves_world_preparation_failure(self):
        """Reject only the mixed-contact world whose adapted record crossed worlds."""
        adapter = _make_adapter(self.device, [(0, -1)])
        adapter.world_contact_count.zero_()
        deformable = _make_deformable(
            self.device,
            body=0,
            status=DEFORMABLE_CONTACT_STATUS_MALFORMED,
            world_status=DEFORMABLE_CONTACT_STATUS_CROSS_WORLD,
            global_status=0,
        )
        projection = ColoredGaussSeidelProjection(adapter, deformable, 2)
        prepared_status = wp.ones(1, dtype=wp.int32, device=self.device)

        projection.prepare(
            wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device),
            prepared_status,
        )

        np.testing.assert_array_equal(prepared_status.numpy(), [0])

    def test_cross_family_metrics_share_body_color_occupancy(self):
        """Share one body's color incidence across every rigid and mixed family."""
        adapter = _make_adapter(self.device, [(0, -1)])
        scalar_jacobian = np.zeros((1, 6), dtype=np.float32)
        scalar_jacobian[0, 0] = 1.0
        adapter.friction_capacity = 1
        adapter.friction_world = wp.zeros(1, dtype=wp.int32, device=self.device)
        adapter.friction_local = wp.zeros(1, dtype=wp.int32, device=self.device)
        adapter.world_friction_count.assign([1])
        adapter.friction_body_first = wp.zeros(1, dtype=wp.int32, device=self.device)
        adapter.friction_body_second = wp.full(1, -1, dtype=wp.int32, device=self.device)
        adapter.friction_jacobian_first = wp.array(scalar_jacobian, dtype=vec6f, device=self.device)
        adapter.friction_jacobian_second = wp.zeros(1, dtype=vec6f, device=self.device)
        adapter.friction_projection_delassus = wp.zeros(1, dtype=wp.float32, device=self.device)
        adapter.limit_capacity = 1
        adapter.limit_world = wp.zeros(1, dtype=wp.int32, device=self.device)
        adapter.limit_local = wp.zeros(1, dtype=wp.int32, device=self.device)
        adapter.world_limit_count.assign([1])
        adapter.limit_body_first = wp.zeros(1, dtype=wp.int32, device=self.device)
        adapter.limit_body_second = wp.full(1, -1, dtype=wp.int32, device=self.device)
        adapter.limit_jacobian_first = wp.array(scalar_jacobian, dtype=vec6f, device=self.device)
        adapter.limit_jacobian_second = wp.zeros(1, dtype=vec6f, device=self.device)
        adapter.limit_projection_delassus = wp.zeros(1, dtype=wp.float32, device=self.device)
        deformable = _make_deformable(
            self.device,
            body=0,
            status=DEFORMABLE_CONTACT_STATUS_VALID,
            world_status=DEFORMABLE_CONTACT_STATUS_VALID,
            global_status=0,
        )
        body_jacobian = np.zeros((1, 3, 6), dtype=np.float32)
        body_jacobian[0, 0, 0] = 1.0
        body_jacobian[0, 1, 1] = 1.0
        body_jacobian[0, 2, 2] = 1.0
        deformable.body_jacobian.assign(body_jacobian)
        projection = ColoredGaussSeidelProjection(adapter, deformable, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device)

        projection.prepare(inverse_weight, wp.ones(1, dtype=wp.int32, device=self.device))

        occupancy = projection.body_occupancy.numpy()[0]
        self.assertEqual(int(np.sum(occupancy)), 4)
        self.assertGreaterEqual(int(np.max(occupancy)), 2)
        friction_color = int(projection.friction.colors.numpy()[0])
        contact_color = int(projection.contact.colors.numpy()[0])
        limit_color = int(projection.limit.colors.numpy()[0])
        deformable_color = int(projection.deformable.colors.numpy()[0])
        self.assertEqual(float(projection.friction_delassus.numpy()[0]), float(occupancy[friction_color]))
        np.testing.assert_allclose(
            np.diag(projection.contact_delassus.numpy()[0]),
            occupancy[contact_color],
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(float(projection.limit_delassus.numpy()[0]), float(occupancy[limit_color]))
        np.testing.assert_allclose(
            np.diag(deformable.gauss_seidel_delassus.numpy()[0]),
            1.0 + occupancy[deformable_color],
            rtol=0.0,
            atol=0.0,
        )

    def test_nonfinite_mixed_body_correction_marks_contact_failure(self):
        """Record deformable diagnostics when a mixed body correction becomes nonfinite."""
        adapter = _make_adapter(self.device, [(0, -1)])
        adapter.world_contact_count.zero_()
        deformable = _make_deformable(
            self.device,
            body=0,
            status=DEFORMABLE_CONTACT_STATUS_VALID,
            world_status=DEFORMABLE_CONTACT_STATUS_VALID,
            global_status=0,
        )
        projection = ColoredGaussSeidelProjection(adapter, deformable, 2)
        valid_inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device)
        prepared_status = wp.ones(1, dtype=wp.int32, device=self.device)
        projection.prepare(valid_inverse_weight, prepared_status)
        invalid_inverse_weight = wp.array(
            [np.full((6, 6), np.nan, dtype=np.float32)],
            dtype=mat66f,
            device=self.device,
        )
        projection_status = wp.ones(1, dtype=wp.int32, device=self.device)
        body_delta = wp.zeros(1, dtype=vec6f, device=self.device)
        color = int(projection.deformable.colors.numpy()[0])

        wp.launch(
            _project_deformable_colored,
            dim=1,
            inputs=[
                1,
                color,
                projection.deformable.counts,
                projection.deformable.offsets,
                projection.deformable.order,
                deformable.particle_indices,
                deformable.coefficients,
                deformable.contact_world,
                deformable.body,
                deformable.normal,
                deformable.frame,
                deformable.body_jacobian,
                deformable.bias,
                deformable.rigid_bias,
                deformable.friction,
                deformable.gauss_seidel_scalar_delassus,
                deformable.gauss_seidel_delassus,
                deformable.gauss_seidel_delassus_normal_first,
                deformable.status,
                wp.ones(1, dtype=wp.bool, device=self.device),
                projection.particle_occupancy,
                projection.body_occupancy,
                deformable.cloth_system.inverse_weight,
                invalid_inverse_weight,
                True,
                wp.zeros(1, dtype=wp.vec3f, device=self.device),
                wp.zeros(1, dtype=vec6f, device=self.device),
            ],
            outputs=[
                deformable.reaction,
                deformable.rigid_reaction,
                deformable.particle_delta,
                body_delta,
                projection_status,
                deformable.world_status,
            ],
            device=self.device,
        )

        np.testing.assert_array_equal(deformable.status.numpy(), [DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE])
        np.testing.assert_array_equal(
            deformable.world_status.numpy(),
            [DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE],
        )
        np.testing.assert_array_equal(projection_status.numpy(), [0])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for graph capture")
    def test_cuda_graph_capture_replays_large_color_compaction(self):
        """Capture the parallel color-prefix scan and cursor copy path."""
        device = wp.get_device("cuda:0")
        color_count = 96
        capacity = 2 * color_count + 11
        colors = np.arange(capacity, dtype=np.int32) % color_count
        colors[::17] = -1
        family = _ColorFamily(capacity, color_count, device)
        family.colors.assign(colors)
        family.compact(color_count, device)

        with wp.ScopedCapture(device=device) as capture:
            family.compact(color_count, device)
        wp.capture_launch(capture.graph)

        valid = colors >= 0
        expected_counts = np.bincount(colors[valid], minlength=color_count).astype(np.int32)
        expected_offsets = np.zeros(color_count, dtype=np.int32)
        expected_offsets[1:] = np.cumsum(expected_counts[:-1], dtype=np.int32)
        counts = family.counts.numpy()
        offsets = family.offsets.numpy()
        order = family.order.numpy()[: int(np.sum(expected_counts))]
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(offsets, expected_offsets)
        np.testing.assert_array_equal(np.sort(order), np.flatnonzero(valid))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for graph capture")
    def test_cuda_graph_capture_replays_coloring_and_projection(self):
        """Capture and replay device coloring, metric preparation, and projection."""
        device = wp.get_device("cuda:0")
        adapter = _make_adapter(device, [(0, -1), (0, -1)])
        adapter.contact_bias.assign([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
        projection = ColoredGaussSeidelProjection(adapter, None, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=device)
        prepared_status = wp.zeros(1, dtype=wp.int32, device=device)
        projection_status = wp.zeros(1, dtype=wp.int32, device=device)
        projected_twist = wp.zeros(1, dtype=vec6f, device=device)
        twist_delta = wp.zeros(1, dtype=vec6f, device=device)
        world_active = wp.ones(1, dtype=wp.bool, device=device)
        body_world = wp.zeros(1, dtype=wp.int32, device=device)

        def run():
            projection.prepare(inverse_weight, prepared_status)
            projection.project(
                1,
                world_active,
                body_world,
                inverse_weight,
                projected_twist,
                twist_delta,
                None,
                prepared_status,
                projection_status,
            )

        run()
        projected_twist.zero_()
        adapter.contact_reaction.zero_()
        with wp.ScopedCapture(device=device) as capture:
            run()
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(projected_twist.numpy()[0, 0], 2.0, rtol=0.0, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
