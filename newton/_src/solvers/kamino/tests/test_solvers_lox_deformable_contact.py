# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LOX scalar deformable contact path."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.solvers.lox import (
    DEFORMABLE_CONTACT_STATUS_CROSS_WORLD,
    DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS,
    DEFORMABLE_CONTACT_STATUS_MALFORMED,
    DEFORMABLE_CONTACT_STATUS_UNUSED,
    DEFORMABLE_CONTACT_STATUS_VALID,
    PROJECTION_STATUS_VALID,
    DeformableClothSystem,
    DeformableContactSystem,
)
from newton._src.solvers.kamino._src.solvers.lox.avbd import project_deformable_constraints_avbd
from newton._src.solvers.kamino._src.solvers.lox.soft_contact_filter import (
    compute_soft_edge_normal_cones,
    edge_normal_cone_contains,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _world_dt(model: newton.Model, value: float, device: wp.DeviceLike) -> wp.array[wp.float32]:
    """Construct an explicit uniform per-world time-step array."""
    return wp.full(model.world_count, value, dtype=wp.float32, device=device)


@wp.kernel
def _evaluate_edge_normal_cone(
    direction: wp.array[wp.vec3],
    cone_axis: wp.array[wp.vec3],
    cone_cosine: wp.array[float],
    accepted: wp.array[wp.int32],
):
    candidate = wp.tid()
    if edge_normal_cone_contains(0, direction[candidate], cone_axis, cone_cosine):
        accepted[candidate] = 1
    else:
        accepted[candidate] = 0


def _build_contact_model(
    *,
    device: wp.DeviceLike,
    world_count: int = 1,
    fix_left: bool = False,
    collider: str = "static",
    shape_margin: float = 0.02,
    body_com: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[newton.Model, list[int], list[int]]:
    """Build cloth grids and one static, kinematic, or dynamic plane per world."""
    builder = newton.ModelBuilder()
    shape_indices: list[int] = []
    body_indices: list[int] = []
    for world in range(world_count):
        builder.begin_world()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.5 + 2.0 * world),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
            fix_left=fix_left,
            tri_ke=100.0,
            tri_ka=80.0,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=2.0,
            edge_kd=0.0,
        )
        body = -1
        if collider != "static":
            body = builder.add_body(
                xform=wp.transform_identity(),
                com=body_com,
                mass=1.0,
                inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                lock_inertia=body_com != (0.0, 0.0, 0.0),
                is_kinematic=collider == "kinematic",
            )
            body_indices.append(body)
        shape_indices.append(
            builder.add_shape_plane(
                body=body,
                cfg=newton.ModelBuilder.ShapeConfig(margin=shape_margin),
            )
        )
        builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.soft_contact_mu = 0.5
    model.soft_contact_restitution = 0.0
    return model, shape_indices, body_indices


def _build_tetrahedral_contact_model(device: wp.DeviceLike) -> tuple[newton.Model, int]:
    """Build one tetrahedral deformable without triangle-surface topology and a plane."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    for position in (
        wp.vec3(0.0, 0.0, 0.5),
        wp.vec3(1.0, 0.0, 0.5),
        wp.vec3(0.0, 1.0, 0.5),
        wp.vec3(0.0, 0.0, 1.5),
    ):
        builder.add_particle(position, wp.vec3(0.0), mass=1.0, radius=0.0)
    builder.add_tetrahedron(0, 1, 2, 3, k_mu=100.0, k_lambda=80.0, k_damp=0.0)
    shape = builder.add_shape_plane()
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model, shape


def _make_contacts(
    *,
    device: wp.DeviceLike,
    capacity: int,
    records: list[dict],
) -> newton.Contacts:
    """Create manually specified Newton soft-contact records."""
    contacts = newton.Contacts(0, capacity, device=device)
    indices = np.full((capacity, 3), -1, dtype=np.int32)
    barycentric = np.zeros((capacity, 3), dtype=np.float32)
    shapes = np.full(capacity, -1, dtype=np.int32)
    body_positions = np.zeros((capacity, 3), dtype=np.float32)
    body_velocities = np.zeros((capacity, 3), dtype=np.float32)
    normals = np.zeros((capacity, 3), dtype=np.float32)
    for contact, record in enumerate(records):
        indices[contact] = record["indices"]
        barycentric[contact] = record["barycentric"]
        shapes[contact] = record["shape"]
        body_positions[contact] = record["body_position"]
        body_velocities[contact] = record.get("body_velocity", (0.0, 0.0, 0.0))
        normals[contact] = record.get("normal", (0.0, 0.0, 1.0))

    contacts.soft_contact_count.assign(np.array([len(records)], dtype=np.int32))
    contacts.soft_contact_indices.assign(indices)
    contacts.soft_contact_barycentric.assign(barycentric)
    contacts.soft_contact_shape.assign(shapes)
    contacts.soft_contact_body_pos.assign(body_positions)
    contacts.soft_contact_body_vel.assign(body_velocities)
    contacts.soft_contact_normal.assign(normals)
    return contacts


def _feature_position(positions: np.ndarray, indices: tuple[int, int, int], coefficients: tuple[float, ...]):
    """Interpolate one source feature position."""
    value = np.zeros(3, dtype=np.float64)
    for particle, coefficient in zip(indices, coefficients, strict=True):
        if particle >= 0:
            value += coefficient * positions[particle]
    return value


def _surface_position_for_gap(
    model: newton.Model,
    positions: np.ndarray,
    indices: tuple[int, int, int],
    coefficients: tuple[float, float, float],
    shape: int,
    gap: float,
    normal: np.ndarray,
) -> np.ndarray:
    """Place a raw collider point at a requested effective gap."""
    feature = _feature_position(positions, indices, coefficients)
    radii = model.particle_radius.numpy()
    radius = max(float(radii[particle]) for particle in indices if particle >= 0)
    margin = float(model.shape_margin.numpy()[shape])
    return feature - (gap + radius + margin) * normal


class TestLOXDeformableContact(unittest.TestCase):
    """Test packed soft contacts and scalar world-space Coulomb projection."""

    def setUp(self):
        """Select the configured Newton test device."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def _make_system(
        self,
        model: newton.Model,
        capacity: int,
        *,
        stabilization_fraction: float = 0.5,
        dead_zone: float = 0.0,
        enable_rigid_normal_cone_filtering: bool = False,
        normal_cone_filtering_min_distance: float = 1.0e-4,
    ) -> tuple[newton.State, DeformableClothSystem, DeformableContactSystem]:
        """Assemble the cloth system and allocate its contact path."""
        state = model.state()
        cloth = DeformableClothSystem(model)
        cloth.assemble(state, _world_dt(model, 0.1, self.device))
        contact = DeformableContactSystem(
            model,
            cloth,
            capacity,
            stabilization_fraction=stabilization_fraction,
            dead_zone=dead_zone,
            enable_rigid_normal_cone_filtering=enable_rigid_normal_cone_filtering,
            normal_cone_filtering_min_distance=normal_cone_filtering_min_distance,
        )
        return state, cloth, contact

    def test_adapt_particle_edge_and_face_coefficients(self):
        """Pack particle, edge, and face records and form signed gap biases."""
        model, shapes, _ = _build_contact_model(device=self.device)
        model.soft_contact_mu = 0.6
        state, cloth, contact = self._make_system(model, 3)
        positions = state.particle_q.numpy()
        normal = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        definitions = (
            ((0, -1, -1), (1.0, 0.0, 0.0), -0.02),
            ((0, 1, -1), (0.25, 0.75, 0.0), 0.03),
            ((0, 1, 3), (0.2, 0.3, 0.5), 0.0),
        )
        body_velocities = (
            (0.1, -0.2, 0.0),
            (-0.3, 0.05, 0.0),
            (0.0, 0.0, 0.0),
        )
        records = []
        for (indices, coefficients, desired_gap), body_velocity in zip(
            definitions,
            body_velocities,
            strict=True,
        ):
            records.append(
                {
                    "indices": indices,
                    "barycentric": coefficients,
                    "shape": shapes[0],
                    "body_position": _surface_position_for_gap(
                        model,
                        positions,
                        indices,
                        coefficients,
                        shapes[0],
                        desired_gap,
                        normal,
                    ),
                    "body_velocity": body_velocity,
                    "normal": normal,
                }
            )
        contacts = _make_contacts(device=self.device, capacity=3, records=records)
        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        np.testing.assert_array_equal(
            contact.particle_indices.numpy(),
            np.array(((0, -1, -1, -1), (0, 1, -1, -1), (0, 1, 3, -1)), dtype=np.int32),
        )
        np.testing.assert_allclose(
            contact.coefficients.numpy(),
            np.array(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.25, 0.75, 0.0, 0.0),
                    (0.2, 0.3, 0.5, 0.0),
                ),
                dtype=np.float32,
            ),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(contact.gap.numpy(), (-0.02, 0.03, 0.0), atol=2.0e-6)
        expected_targets = (0.1, -0.3, 0.0)
        expected_bias = np.array(
            [
                -np.asarray(body_velocity) - target * normal
                for body_velocity, target in zip(body_velocities, expected_targets, strict=True)
            ]
        )
        np.testing.assert_allclose(contact.bias.numpy(), expected_bias, atol=2.0e-6)
        np.testing.assert_allclose(contact.friction.numpy(), 0.6, atol=1.0e-7)
        np.testing.assert_array_equal(contact.particle_multiplicity.numpy(), (3, 2, 0, 1))
        np.testing.assert_array_equal(contact.world_contact_count.numpy(), (3,))
        np.testing.assert_array_equal(contact.status.numpy(), DEFORMABLE_CONTACT_STATUS_VALID)
        np.testing.assert_array_equal(cloth.topology.packed_to_newton.numpy(), (0, 1, 2, 3))

    def test_filter_full_surface_contacts_by_soft_normal_cones(self):
        """Keep valid contacts after reducing boundary records to edges or vertices."""
        model, shapes, _ = _build_contact_model(device=self.device)
        within_tolerance = np.array((np.sin(np.deg2rad(4.0)), 0.0, np.cos(np.deg2rad(4.0))))
        outside_tolerance = np.array((np.sin(np.deg2rad(6.0)), 0.0, np.cos(np.deg2rad(6.0))))
        within_tolerance_inward = np.array((-np.sin(np.deg2rad(4.0)), 0.0, np.cos(np.deg2rad(4.0))))
        outside_tolerance_inward = np.array((-np.sin(np.deg2rad(6.0)), 0.0, np.cos(np.deg2rad(6.0))))
        definitions = (
            ((0, -1, -1), (1.0, 0.0, 0.0), np.array((0.0, 0.0, 1.0)), True),
            ((0, -1, -1), (1.0, 0.0, 0.0), within_tolerance, True),
            ((0, -1, -1), (1.0, 0.0, 0.0), outside_tolerance, True),
            ((0, -1, -1), (1.0, 0.0, 0.0), within_tolerance_inward, True),
            ((0, -1, -1), (1.0, 0.0, 0.0), outside_tolerance_inward, False),
            ((0, -1, -1), (1.0, 0.0, 0.0), np.array((1.0, 0.0, 0.0)), True),
            ((0, -1, -1), (1.0, 0.0, 0.0), np.array((-1.0, 0.0, 0.0)), False),
            ((0, 1, -1), (0.5, 0.5, 0.0), np.array((1.0, 0.0, 0.0)), True),
            ((0, 1, -1), (0.5, 0.5, 0.0), np.array((0.0, 1.0, 0.0)), True),
            ((0, 1, -1), (0.5, 0.5, 0.0), np.array((0.0, -1.0, 0.0)), False),
            ((0, 1, -1), (1.0, 0.0, 0.0), np.array((0.0, 0.0, 1.0)), True),
            ((0, 1, -1), (1.0, 0.0, 0.0), np.array((1.0, 0.0, 0.0)), True),
            ((0, 1, -1), (1.0, 0.0, 0.0), np.array((-1.0, 0.0, 0.0)), False),
            ((1, 2, -1), (0.5, 0.5, 0.0), np.array((0.0, 0.0, 1.0)), True),
            ((1, 2, -1), (0.5, 0.5, 0.0), np.array((1.0, 0.0, 0.0)), False),
            ((0, 1, 2), (0.2, 0.3, 0.5), np.array((0.0, 0.0, 1.0)), True),
            ((0, 1, 2), (0.2, 0.3, 0.5), within_tolerance, True),
            ((0, 1, 2), (0.2, 0.3, 0.5), outside_tolerance, False),
            ((0, 1, 2), (0.5, 0.5, 0.0), np.array((0.0, 0.0, 1.0)), True),
            ((0, 1, 2), (1.0, 0.0, 0.0), np.array((0.0, 0.0, 1.0)), True),
            ((0, 1, 2), (1.0, 0.0, 0.0), np.array((1.0, 0.0, 0.0)), True),
            ((0, 1, 2), (1.0, 0.0, 0.0), np.array((-1.0, 0.0, 0.0)), False),
        )
        state, cloth, contact = self._make_system(
            model,
            len(definitions),
            enable_rigid_normal_cone_filtering=True,
        )
        positions = state.particle_q.numpy()
        records = []
        for indices, barycentric, normal, _expected_valid in definitions:
            records.append(
                {
                    "indices": indices,
                    "barycentric": barycentric,
                    "shape": shapes[0],
                    "body_position": _surface_position_for_gap(
                        model,
                        positions,
                        indices,
                        barycentric,
                        shapes[0],
                        0.0,
                        normal,
                    ),
                    "normal": normal,
                }
            )
        contacts = _make_contacts(device=self.device, capacity=len(records), records=records)
        contacts._enable_rigid_soft_full_surface_contact = True

        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        expected = np.array(
            [
                DEFORMABLE_CONTACT_STATUS_VALID if expected_valid else DEFORMABLE_CONTACT_STATUS_UNUSED
                for *_definition, expected_valid in definitions
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(contact.status.numpy(), expected)
        np.testing.assert_array_equal(
            contact.world_contact_count.numpy(),
            (sum(expected_valid for *_definition, expected_valid in definitions),),
        )

        unfiltered = DeformableContactSystem(
            model,
            cloth,
            len(records),
            enable_rigid_normal_cone_filtering=False,
        )
        unfiltered.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        unfiltered.raise_if_invalid()
        np.testing.assert_array_equal(unfiltered.status.numpy(), DEFORMABLE_CONTACT_STATUS_VALID)
        np.testing.assert_array_equal(unfiltered.world_contact_count.numpy(), (len(records),))

    def test_keep_close_contact_before_normal_cone_filtering(self):
        """Keep a nearly coincident contact whose computed normal fails the cone test."""
        model, shapes, _ = _build_contact_model(device=self.device, shape_margin=0.0)
        state, _, contact = self._make_system(
            model,
            2,
            enable_rigid_normal_cone_filtering=True,
            normal_cone_filtering_min_distance=1.0e-4,
        )
        feature_position = state.particle_q.numpy()[0]
        normal = np.array((-1.0, 0.0, 0.0), dtype=np.float32)
        separations = (5.0e-5, 2.0e-4)
        contacts = _make_contacts(
            device=self.device,
            capacity=2,
            records=[
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shapes[0],
                    "body_position": feature_position - separation * normal,
                    "normal": normal,
                }
                for separation in separations
            ],
        )
        contacts._enable_rigid_soft_full_surface_contact = True

        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        np.testing.assert_array_equal(
            contact.status.numpy(),
            (DEFORMABLE_CONTACT_STATUS_VALID, DEFORMABLE_CONTACT_STATUS_UNUSED),
        )

    def test_filter_boundary_edge_with_outward_half_plane(self):
        """Treat a boundary edge as a tolerant outward 180-degree normal cone."""
        positions = wp.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=wp.vec3,
            device=self.device,
        )
        edge_indices = wp.array(
            ((2, -1, 0, 1), (0, -1, 1, 2), (1, -1, 2, 0)),
            dtype=wp.int32,
            device=self.device,
        )
        cone_axis = wp.zeros(3, dtype=wp.vec3, device=self.device)
        cone_cosine = wp.zeros(3, dtype=wp.float32, device=self.device)
        wp.launch(
            compute_soft_edge_normal_cones,
            dim=3,
            inputs=[positions, edge_indices],
            outputs=[cone_axis, cone_cosine],
            device=self.device,
        )

        angle_4 = np.deg2rad(4.0)
        angle_6 = np.deg2rad(6.0)
        directions = wp.array(
            (
                (0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
                (0.0, np.sin(angle_4), np.cos(angle_4)),
                (0.0, np.sin(angle_6), np.cos(angle_6)),
            ),
            dtype=wp.vec3,
            device=self.device,
        )
        accepted = wp.empty(len(directions), dtype=wp.int32, device=self.device)
        wp.launch(
            _evaluate_edge_normal_cone,
            dim=len(directions),
            inputs=[directions, cone_axis, cone_cosine],
            outputs=[accepted],
            device=self.device,
        )

        np.testing.assert_allclose(
            cone_axis.numpy(),
            ((0.0, -1.0, 0.0), (np.sqrt(0.5), np.sqrt(0.5), 0.0), (-1.0, 0.0, 0.0)),
            atol=1.0e-7,
        )
        np.testing.assert_array_equal(cone_cosine.numpy(), (-1.0, -1.0, -1.0))
        np.testing.assert_array_equal(accepted.numpy(), (1, 0, 1, 1, 1, 0))

    def test_keep_tetrahedral_particle_contact_without_surface_topology(self):
        """Keep a rigid contact on a tetrahedron that has no triangle or edge topology."""
        model, shape = _build_tetrahedral_contact_model(self.device)
        state, _, contact = self._make_system(model, 1)
        positions = state.particle_q.numpy()
        normal = np.array((0.0, 0.0, 1.0))
        contacts = _make_contacts(
            device=self.device,
            capacity=1,
            records=[
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shape,
                    "body_position": _surface_position_for_gap(
                        model,
                        positions,
                        (0, -1, -1),
                        (1.0, 0.0, 0.0),
                        shape,
                        0.0,
                        normal,
                    ),
                    "normal": normal,
                }
            ],
        )
        contacts._enable_rigid_soft_full_surface_contact = True

        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        np.testing.assert_array_equal(contact.status.numpy(), DEFORMABLE_CONTACT_STATUS_VALID)

    def test_compute_scalar_delassus_with_mass_split_and_pins(self):
        """Match scalar mass-split Delassus values and leave pinned nodes unchanged."""
        model, shapes, _ = _build_contact_model(device=self.device, fix_left=True)
        state, cloth, contact = self._make_system(model, 2)
        positions = state.particle_q.numpy()
        normal = np.array((0.0, 0.0, 1.0))
        records = []
        for indices, coefficients in (
            ((0, 1, -1), (0.25, 0.75, 0.0)),
            ((1, -1, -1), (1.0, 0.0, 0.0)),
        ):
            records.append(
                {
                    "indices": indices,
                    "barycentric": coefficients,
                    "shape": shapes[0],
                    "body_position": _surface_position_for_gap(
                        model,
                        positions,
                        indices,
                        coefficients,
                        shapes[0],
                        0.0,
                        normal,
                    ),
                }
            )
        contacts = _make_contacts(device=self.device, capacity=2, records=records)
        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        inverse_weight = cloth.inverse_weight.numpy()
        multiplicity = contact.particle_multiplicity.numpy()
        expected_split = multiplicity * inverse_weight
        expected_delassus = np.array(
            (
                0.25**2 * expected_split[0] + 0.75**2 * expected_split[1],
                expected_split[1],
            )
        )
        np.testing.assert_allclose(contact.split_inverse_weight.numpy(), expected_split, atol=1.0e-7)
        np.testing.assert_allclose(contact.delassus.numpy(), expected_delassus, rtol=2.0e-6, atol=2.0e-6)

        bias = np.zeros((2, 3), dtype=np.float32)
        bias[:, 2] = -1.0
        contact.bias.assign(bias)
        projected = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        contact.project(projected)
        projected_np = projected.numpy()
        self.assertEqual(float(cloth.inverse_weight.numpy()[0]), 0.0)
        np.testing.assert_array_equal(projected_np[0], np.zeros(3))
        self.assertGreater(float(projected_np[1, 2]), 0.0)

    def test_avbd_metric_uses_true_delassus_without_multiplicity(self):
        """Keep the AVBD contact metric unchanged by shared-particle incidence."""
        model, shapes, _ = _build_contact_model(device=self.device)
        state, cloth, contact = self._make_system(model, 2)
        positions = state.particle_q.numpy()
        normal = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        records = []
        for gap in (-0.01, 0.0):
            records.append(
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shapes[0],
                    "body_position": _surface_position_for_gap(
                        model,
                        positions,
                        (0, -1, -1),
                        (1.0, 0.0, 0.0),
                        shapes[0],
                        gap,
                        normal,
                    ),
                    "normal": normal,
                }
            )
        contacts = _make_contacts(device=self.device, capacity=2, records=records)
        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        projected = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        baseline = wp.zeros_like(projected)
        world_active = wp.ones(model.world_count, dtype=wp.bool, device=self.device)
        project_deformable_constraints_avbd(20, contact, world_active, baseline, projected)

        true_delassus = float(cloth.inverse_weight.numpy()[0])
        expected_delassus = np.tile(true_delassus * np.eye(3, dtype=np.float32), (2, 1, 1))
        expected_inverse = np.tile(np.eye(3, dtype=np.float32) / true_delassus, (2, 1, 1))
        np.testing.assert_allclose(contact.avbd_delassus.numpy(), expected_delassus, rtol=2.0e-6)
        np.testing.assert_allclose(contact.avbd_inverse_delassus.numpy(), expected_inverse, rtol=2.0e-6)
        self.assertEqual(int(contact.particle_multiplicity.numpy()[0]), 2)
        self.assertTrue(np.all(np.isfinite(contact.world_avbd_stationarity_max.numpy())))
        self.assertLess(float(contact.world_avbd_stationarity_max.numpy()[0]), 1.0e-4)

    def test_project_all_coulomb_branches_and_rotate(self):
        """Match separation, sticking, and sliding formulas under rotation."""
        model, shapes, _ = _build_contact_model(device=self.device)
        state, _, contact = self._make_system(model, 1)
        positions = state.particle_q.numpy()
        normal = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        record = {
            "indices": (0, -1, -1),
            "barycentric": (1.0, 0.0, 0.0),
            "shape": shapes[0],
            "body_position": _surface_position_for_gap(
                model,
                positions,
                (0, -1, -1),
                (1.0, 0.0, 0.0),
                shapes[0],
                0.0,
                normal,
            ),
        }
        contacts = _make_contacts(device=self.device, capacity=1, records=[record])
        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()
        scalar = float(contact.delassus.numpy()[0])

        cases = (
            ("separation", np.array((0.3, 0.0, 0.2)), np.zeros(3)),
            ("sticking", np.array((0.2, 0.0, -1.0)), np.array((-0.2, 0.0, 1.0)) / scalar),
            ("sliding", np.array((1.0, 0.0, -1.0)), np.array((-0.5, 0.0, 1.0)) / scalar),
        )
        for _name, free_velocity, expected_reaction in cases:
            projected = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
            contact.reaction.zero_()
            contact.normal.assign(normal.reshape((1, 3)))
            contact.bias.assign(free_velocity.astype(np.float32).reshape((1, 3)))
            contact.project(projected)
            np.testing.assert_allclose(
                contact.reaction.numpy()[0],
                expected_reaction,
                rtol=2.0e-6,
                atol=2.0e-6,
            )

        rotation = np.array(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
        free_velocity = np.array((1.0, 0.0, -1.0))
        reference_reaction = np.array((-0.5, 0.0, 1.0)) / scalar
        projected = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        contact.reaction.zero_()
        contact.normal.assign((rotation @ normal).astype(np.float32).reshape((1, 3)))
        contact.bias.assign((rotation @ free_velocity).astype(np.float32).reshape((1, 3)))
        contact.project(projected)
        np.testing.assert_allclose(
            contact.reaction.numpy()[0],
            rotation @ reference_reaction,
            rtol=2.0e-6,
            atol=2.0e-6,
        )

    def test_apgd_uses_isotropic_step_for_pure_deformable_contact(self):
        """Use the scalar Delassus metric for pure deformable APGD contacts."""
        model, shapes, _ = _build_contact_model(device=self.device)
        state, _, contact = self._make_system(model, 1)
        positions = state.particle_q.numpy()
        normal = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        contacts = _make_contacts(
            device=self.device,
            capacity=1,
            records=[
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shapes[0],
                    "body_position": _surface_position_for_gap(
                        model,
                        positions,
                        (0, -1, -1),
                        (1.0, 0.0, 0.0),
                        shapes[0],
                        0.0,
                        normal,
                    ),
                    "normal": normal,
                }
            ],
        )
        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        projected_np = np.zeros((model.particle_count, 3), dtype=np.float32)
        projected_np[0] = (1.0, 0.0, -1.0)
        projected = wp.array(projected_np, dtype=wp.vec3, device=self.device)
        world_active = wp.ones(model.world_count, dtype=wp.bool, device=self.device)
        projection_status = wp.full(model.world_count, PROJECTION_STATUS_VALID, dtype=wp.int32, device=self.device)
        restart_dot = wp.zeros(model.world_count, dtype=wp.float32, device=self.device)
        contact.initialize_apgd(world_active, rigid_coordinates=False)

        trial = contact.apgd_trial.numpy()[0]
        coefficients = contact.coefficients.numpy()[0]
        velocity = contact.bias.numpy()[0].copy()
        for slot, particle in enumerate(contact.particle_indices.numpy()[0]):
            if particle >= 0:
                velocity += coefficients[slot] * projected_np[particle]
        tangent = velocity - np.dot(normal, velocity) * normal
        corrected = velocity + float(contact.friction.numpy()[0]) * np.linalg.norm(tangent) * normal
        metric = float(contact.delassus.numpy()[0])
        value = trial - corrected / metric
        normal_value = float(np.dot(normal, value))
        tangent_value = value - normal_value * normal
        tangent_norm = float(np.linalg.norm(tangent_value))
        friction = float(contact.friction.numpy()[0])
        if friction * tangent_norm <= -normal_value:
            expected = np.zeros(3, dtype=np.float32)
        elif tangent_norm <= friction * normal_value:
            expected = value
        else:
            projected_normal = (friction * tangent_norm + normal_value) / (friction * friction + 1.0)
            expected = projected_normal * normal + friction * projected_normal * tangent_value / tangent_norm

        contact.project_apgd(
            world_active,
            projected,
            None,
            restart_dot,
            projection_status,
            rigid_coordinates=False,
        )

        np.testing.assert_allclose(contact.apgd_next.numpy()[0], expected, atol=2.0e-6)

    def test_accumulate_multiple_contacts_with_mass_split(self):
        """Accumulate simultaneous mass-split corrections from incident contacts."""
        model, shapes, _ = _build_contact_model(device=self.device)
        state, cloth, contact = self._make_system(model, 2)
        positions = state.particle_q.numpy()
        normal = np.array((0.0, 0.0, 1.0))
        body_position = _surface_position_for_gap(
            model,
            positions,
            (0, -1, -1),
            (1.0, 0.0, 0.0),
            shapes[0],
            0.0,
            normal,
        )
        records = [
            {
                "indices": (0, -1, -1),
                "barycentric": (1.0, 0.0, 0.0),
                "shape": shapes[0],
                "body_position": body_position,
            }
            for _contact in range(2)
        ]
        contacts = _make_contacts(device=self.device, capacity=2, records=records)
        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        inverse_weight = float(cloth.inverse_weight.numpy()[0])
        self.assertEqual(int(contact.particle_multiplicity.numpy()[0]), 2)
        np.testing.assert_allclose(contact.delassus.numpy(), 2.0 * inverse_weight, atol=2.0e-6)
        bias = np.zeros((2, 3), dtype=np.float32)
        bias[:, 2] = -1.0
        contact.bias.assign(bias)
        projected = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        contact.project(projected)

        expected_reaction = 1.0 / (2.0 * inverse_weight)
        np.testing.assert_allclose(contact.reaction.numpy()[:, 2], expected_reaction, atol=2.0e-6)
        np.testing.assert_allclose(projected.numpy()[0], (0.0, 0.0, 1.0), atol=2.0e-6)

    def test_warm_start_and_compute_residuals(self):
        """Restore within-step reactions and compute contact and consensus residuals."""
        model, shapes, _ = _build_contact_model(device=self.device)
        state, _, contact = self._make_system(model, 1)
        positions = state.particle_q.numpy()
        normal = np.array((0.0, 0.0, 1.0))
        record = {
            "indices": (0, -1, -1),
            "barycentric": (1.0, 0.0, 0.0),
            "shape": shapes[0],
            "body_position": _surface_position_for_gap(
                model,
                positions,
                (0, -1, -1),
                (1.0, 0.0, 0.0),
                shapes[0],
                0.0,
                normal,
            ),
        }
        contacts = _make_contacts(device=self.device, capacity=1, records=[record])
        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.bias.assign(np.array(((0.1, 0.0, -1.0),), dtype=np.float32))

        projected = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        contact.project(projected)
        projected_after_sweep = projected.numpy()
        reaction_after_sweep = contact.reaction.numpy()
        contact.compute_contact_residuals(projected)
        self.assertLess(float(contact.contact_residual.numpy()[0]), 2.0e-6)

        restored = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        contact.apply_reaction_warm_start(restored)
        np.testing.assert_allclose(restored.numpy(), projected_after_sweep, atol=2.0e-6)
        np.testing.assert_array_equal(contact.reaction.numpy(), reaction_after_sweep)

        global_velocity = wp.full(model.particle_count, wp.vec3(1.0, 0.0, 0.0), device=self.device)
        projected_velocity = wp.full(model.particle_count, wp.vec3(0.75, 0.0, 0.0), device=self.device)
        global_previous = wp.full(model.particle_count, wp.vec3(0.5, 0.0, 0.0), device=self.device)
        projected_previous = wp.full(model.particle_count, wp.vec3(0.7, 0.0, 0.0), device=self.device)
        contact.compute_consensus_residuals(
            global_velocity,
            projected_velocity,
            global_previous,
            projected_previous,
            _world_dt(model, 0.1, self.device),
        )
        np.testing.assert_allclose(contact.world_consensus_residual.numpy(), (0.25,), atol=1.0e-7)
        np.testing.assert_allclose(contact.world_iterate_residual.numpy(), (0.05,), atol=1.0e-7)
        np.testing.assert_allclose(contact.world_displacement_residual.numpy(), (0.075,), atol=1.0e-7)

    def test_include_kinematic_collider_velocity_in_bias(self):
        """Include kinematic linear, angular, and local surface velocity with the correct sign."""
        model, shapes, bodies = _build_contact_model(device=self.device, collider="kinematic")
        state, _, contact = self._make_system(model, 1, stabilization_fraction=0.0)
        state.body_qd.assign(np.array(((0.2, -0.1, 0.3, 0.0, 0.0, 2.0),), dtype=np.float32))
        positions = state.particle_q.numpy()
        normal = np.array((0.0, 0.0, 1.0))
        local_point = np.array((0.5, 0.0, 0.0))
        local_velocity = np.array((0.0, 0.4, 0.0))
        record = {
            "indices": (0, -1, -1),
            "barycentric": (1.0, 0.0, 0.0),
            "shape": shapes[0],
            "body_position": local_point,
            "body_velocity": local_velocity,
            "normal": normal,
        }
        contacts = _make_contacts(device=self.device, capacity=1, records=[record])
        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        linear = np.array((0.2, -0.1, 0.3))
        angular = np.array((0.0, 0.0, 2.0))
        expected_collider_velocity = linear + np.cross(angular, local_point) + local_velocity
        np.testing.assert_allclose(contact.collider_velocity.numpy()[0], expected_collider_velocity, atol=1.0e-6)
        gap = float(np.dot(normal, positions[0] - local_point))
        gap -= float(model.particle_radius.numpy()[0] + model.shape_margin.numpy()[shapes[0]])
        expected_target = -gap / 0.1 if gap > 0.0 else 0.0
        expected_bias = -expected_collider_velocity - expected_target * normal
        np.testing.assert_allclose(contact.bias.numpy()[0], expected_bias, atol=2.0e-6)
        self.assertEqual(int(model.body_flags.numpy()[bodies[0]]), int(newton.BodyFlags.KINEMATIC))

    def test_include_prescribed_particle_velocity_in_bias(self):
        """Include prescribed particle motion in the deformable contact bias."""
        model, shapes, _ = _build_contact_model(device=self.device)
        masses = model.particle_mass.numpy()
        masses[0] = 0.0
        model.particle_mass.assign(masses)
        state, cloth, contact = self._make_system(model, 1, stabilization_fraction=0.0)
        prescribed_velocity = np.array((0.4, -0.2, 0.0), dtype=np.float32)
        velocities = state.particle_qd.numpy()
        velocities[0] = prescribed_velocity
        state.particle_qd.assign(velocities)
        cloth.assemble(state, _world_dt(model, 0.1, self.device))
        normal = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        coefficients = (0.5, 0.5, 0.0)
        record = {
            "indices": (0, 1, -1),
            "barycentric": coefficients,
            "shape": shapes[0],
            "body_position": _surface_position_for_gap(
                model,
                state.particle_q.numpy(),
                (0, 1, -1),
                coefficients,
                shapes[0],
                0.0,
                normal,
            ),
            "normal": normal,
        }
        contacts = _make_contacts(device=self.device, capacity=1, records=[record])

        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        np.testing.assert_allclose(contact.bias.numpy()[0], 0.5 * prescribed_velocity, atol=1.0e-6)
        packed = cloth.topology.newton_to_packed.numpy()
        self.assertEqual(float(cloth.full_inverse_weight.numpy()[packed[0]]), 0.0)
        self.assertGreater(float(cloth.full_inverse_weight.numpy()[packed[1]]), 0.0)

    def test_adapt_dynamic_rigid_frame_jacobian_and_bias(self):
        """Build the rigid contact endpoint at an off-COM surface point."""
        model, shapes, bodies = _build_contact_model(device=self.device, collider="dynamic")
        state, _, contact = self._make_system(model, 1, stabilization_fraction=0.0)
        body_twist = np.array((0.2, -0.1, 0.3, 0.4, -0.5, 0.6), dtype=np.float32)
        state.body_qd.assign(body_twist.reshape((1, 6)))
        surface_point = np.array((0.3, -0.2, 0.4), dtype=np.float32)
        prescribed_velocity = np.array((0.1, 0.2, -0.05), dtype=np.float32)
        contacts = _make_contacts(
            device=self.device,
            capacity=1,
            records=[
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shapes[0],
                    "body_position": surface_point,
                    "body_velocity": prescribed_velocity,
                }
            ],
        )

        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.raise_if_invalid()

        expected_jacobian = np.zeros((3, 6), dtype=np.float32)
        expected_jacobian[:, :3] = -np.identity(3)
        expected_jacobian[:, 3:] = np.array(
            (
                (0.0, -surface_point[2], surface_point[1]),
                (surface_point[2], 0.0, -surface_point[0]),
                (-surface_point[1], surface_point[0], 0.0),
            ),
            dtype=np.float32,
        )
        full_surface_velocity = body_twist[:3] + np.cross(body_twist[3:], surface_point) + prescribed_velocity
        self.assertEqual(int(contact.body.numpy()[0]), bodies[0])
        np.testing.assert_allclose(contact.frame.numpy()[0], np.identity(3), atol=1.0e-6)
        np.testing.assert_allclose(contact.body_jacobian.numpy()[0], expected_jacobian, atol=1.0e-6)
        np.testing.assert_allclose(contact.collider_velocity.numpy()[0], full_surface_velocity, atol=1.0e-6)
        np.testing.assert_allclose(contact.bias.numpy()[0], -full_surface_velocity, atol=1.0e-6)
        np.testing.assert_allclose(contact.rigid_bias.numpy()[0], -prescribed_velocity, atol=1.0e-6)

    def test_reject_malformed_cross_world_and_adapt_dynamic_rigid_records(self):
        """Reject invalid records and adapt a dynamic rigid endpoint."""
        model, shapes, _ = _build_contact_model(device=self.device)
        state, _, contact = self._make_system(model, 1)
        malformed = _make_contacts(
            device=self.device,
            capacity=1,
            records=[
                {
                    "indices": (0, -1, 1),
                    "barycentric": (0.5, 0.0, 0.5),
                    "shape": shapes[0],
                    "body_position": (0.0, 0.0, 0.0),
                }
            ],
        )
        contact.prepare(malformed, state, _world_dt(model, 0.1, self.device))
        self.assertEqual(int(contact.status.numpy()[0]), DEFORMABLE_CONTACT_STATUS_MALFORMED)
        with self.assertRaisesRegex(ValueError, "malformed"):
            contact.raise_if_invalid()

        cross_model, cross_shapes, _ = _build_contact_model(device=self.device, world_count=2)
        cross_state, _, cross_contact = self._make_system(cross_model, 1)
        cross_world = _make_contacts(
            device=self.device,
            capacity=1,
            records=[
                {
                    "indices": (0, 4, -1),
                    "barycentric": (0.5, 0.5, 0.0),
                    "shape": cross_shapes[0],
                    "body_position": (0.0, 0.0, 0.0),
                }
            ],
        )
        cross_contact.prepare(cross_world, cross_state, _world_dt(cross_model, 0.1, self.device))
        self.assertEqual(int(cross_contact.status.numpy()[0]), DEFORMABLE_CONTACT_STATUS_CROSS_WORLD)
        with self.assertRaisesRegex(ValueError, "cross-world"):
            cross_contact.raise_if_invalid()

        dynamic_model, dynamic_shapes, dynamic_bodies = _build_contact_model(
            device=self.device,
            collider="dynamic",
        )
        dynamic_state, _, dynamic_contact = self._make_system(dynamic_model, 1)
        dynamic = _make_contacts(
            device=self.device,
            capacity=1,
            records=[
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": dynamic_shapes[0],
                    "body_position": (0.0, 0.0, 0.0),
                }
            ],
        )
        dynamic_contact.prepare(dynamic, dynamic_state, _world_dt(dynamic_model, 0.1, self.device))
        dynamic_contact.raise_if_invalid()
        self.assertEqual(int(dynamic_contact.status.numpy()[0]), DEFORMABLE_CONTACT_STATUS_VALID)
        self.assertEqual(int(dynamic_contact.body.numpy()[0]), dynamic_bodies[0])
        np.testing.assert_allclose(
            dynamic_contact.frame.numpy()[0],
            np.identity(3),
            atol=1.0e-6,
        )
        expected_jacobian = np.zeros((3, 6), dtype=np.float32)
        expected_jacobian[:, :3] = -np.identity(3)
        np.testing.assert_allclose(dynamic_contact.body_jacobian.numpy()[0], expected_jacobian, atol=1.0e-6)

    def test_reject_contact_with_only_pinned_nodes(self):
        """Reject a contact whose incident nodes all have zero inverse weight."""
        model, shapes, _ = _build_contact_model(device=self.device, fix_left=True)
        state, _, contact = self._make_system(model, 1)
        contacts = _make_contacts(
            device=self.device,
            capacity=1,
            records=[
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shapes[0],
                    "body_position": (0.0, 0.0, 0.0),
                }
            ],
        )
        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        self.assertEqual(
            int(contact.status.numpy()[0]),
            DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS,
        )
        with self.assertRaisesRegex(ValueError, "Delassus"):
            contact.raise_if_invalid()

    def test_capture_contact_prepare_projection_and_residuals(self):
        """Capture and replay soft-contact adaptation, projection, and residuals."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model, shapes, _ = _build_contact_model(device=self.device)
        state, _, contact = self._make_system(model, 1)
        positions = state.particle_q.numpy()
        normal = np.array((0.0, 0.0, 1.0))
        contacts = _make_contacts(
            device=self.device,
            capacity=1,
            records=[
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shapes[0],
                    "body_position": _surface_position_for_gap(
                        model,
                        positions,
                        (0, -1, -1),
                        (1.0, 0.0, 0.0),
                        shapes[0],
                        -0.01,
                        normal,
                    ),
                }
            ],
        )
        projected = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        baseline = wp.zeros_like(projected)
        world_active = wp.ones(model.world_count, dtype=wp.bool, device=self.device)
        contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
        contact.project(projected, iterations=2)
        project_deformable_constraints_avbd(2, contact, world_active, baseline, projected)
        contact.compute_contact_residuals(projected)

        with wp.ScopedCapture(device=self.device) as capture:
            projected.zero_()
            contact.prepare(contacts, state, _world_dt(model, 0.1, self.device))
            contact.project(projected, iterations=2)
            project_deformable_constraints_avbd(2, contact, world_active, baseline, projected)
            contact.compute_contact_residuals(projected)
        wp.capture_launch(capture.graph)

        np.testing.assert_array_equal(contact.status.numpy(), DEFORMABLE_CONTACT_STATUS_VALID)
        self.assertTrue(np.all(np.isfinite(projected.numpy())))
        self.assertTrue(np.all(np.isfinite(contact.reaction.numpy())))
        self.assertTrue(np.all(np.isfinite(contact.contact_residual.numpy())))


if __name__ == "__main__":
    unittest.main(verbosity=2)
