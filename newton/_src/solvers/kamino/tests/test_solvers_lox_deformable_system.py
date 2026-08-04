# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LOX frozen cloth smooth system."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.solvers.lox import (
    DeformableClothSystem,
    DeformableSplittingState,
    validate_deformable_cloth_model,
)
from newton._src.solvers.kamino._src.solvers.lox.deformable_assembly import (
    _TETRAHEDRON_NEGATIVE_CURVATURE_MARGIN,
)
from newton._src.solvers.kamino._src.solvers.lox.deformable_energy import mat99, tet_cofactor
from newton._src.solvers.kamino._src.solvers.lox.deformable_tetrahedron_energy import (
    tet_stable_neo_hookean_hessian,
    tet_stable_neo_hookean_spectral_metrics,
)
from newton._src.solvers.kamino._src.solvers.lox.deformable_tetrahedron_proximal import (
    _factor_gauss_newton_base,
    _solve_gauss_newton_step,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _world_dt(model: newton.Model, value: float, device: wp.DeviceLike) -> wp.array[wp.float32]:
    """Construct an explicit uniform per-world time-step array."""
    return wp.full(model.world_count, value, dtype=wp.float32, device=device)


# Match the production unroll cap for the standalone constitutive test kernel.
wp.set_module_options({"enable_backward": False, "max_unroll": 4})


@wp.kernel
def _evaluate_tetrahedron_hessian(
    deformation: wp.array[wp.mat33],
    rest_volume: float,
    k_mu: float,
    k_lambda: float,
    activation: float,
    hessian: wp.array[mat99],
    projected_hessian: wp.array[mat99],
    majorizing_metric: wp.array[mat99],
):
    hessian[0] = tet_stable_neo_hookean_hessian(
        deformation[0],
        rest_volume,
        k_mu,
        k_lambda,
        activation,
    )
    metrics = tet_stable_neo_hookean_spectral_metrics(
        deformation[0],
        rest_volume,
        k_mu,
        k_lambda,
        activation,
        0.0,
        _TETRAHEDRON_NEGATIVE_CURVATURE_MARGIN,
    )
    projected_hessian[0] = metrics.projected
    majorizing_metric[0] = metrics.majorizer


@wp.kernel
def _evaluate_tetrahedron_proximal_step(
    right_hand_side: wp.array[wp.mat33],
    deformation: wp.array[wp.mat33],
    frozen_metric: wp.array[mat99],
    rest_volume: float,
    step: wp.array[wp.mat33],
    succeeded: wp.array[wp.int32],
):
    frozen_factor, factor_succeeded = _factor_gauss_newton_base(
        frozen_metric[0],
        tet_cofactor(deformation[0]),
        rest_volume,
        120.0,
        75.0,
    )
    step_value, solve_succeeded = _solve_gauss_newton_step(
        right_hand_side[0],
        deformation[0],
        frozen_factor,
        rest_volume,
        120.0,
        75.0,
    )
    step[0] = step_value
    succeeded[0] = 1 if factor_succeeded and solve_succeeded else 0


def _build_grid_model(
    *,
    device: wp.DeviceLike,
    world_count: int = 1,
    dim_x: int = 1,
    dim_y: int = 1,
    fix_left: bool = False,
    add_springs: bool = False,
    color: bool = False,
    include_bending_in_coloring: bool = True,
    tri_ke: float = 100.0,
    tri_ka: float = 80.0,
    tri_kd: float = 0.0,
    edge_ke: float = 2.0,
    edge_kd: float = 0.0,
) -> newton.Model:
    """Build one square cloth grid in each world."""
    builder = newton.ModelBuilder()
    for world in range(world_count):
        builder.begin_world()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, float(world)),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
            fix_left=fix_left,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=edge_ke,
            edge_kd=edge_kd,
            add_springs=add_springs,
        )
        builder.end_world()
    if color:
        builder.color(include_bending=include_bending_in_coloring)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_triangle_model(
    *,
    device: wp.DeviceLike,
    tri_ke: float,
    tri_ka: float,
    tri_kd: float,
) -> newton.Model:
    """Build one triangle without a valid bending stencil."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        density=3.0,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    builder.end_world()
    model = builder.finalize(device=device)
    return model


def _build_tet_model(
    *,
    device: wp.DeviceLike,
    k_mu: float = 100.0,
    k_lambda: float = 80.0,
    k_damp: float = 0.0,
) -> newton.Model:
    """Build one regular tetrahedral deformable without surface triangles."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    for position in (
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
    ):
        builder.add_particle(position, wp.vec3(0.0), mass=1.0, radius=0.0)
    builder.add_tetrahedron(0, 1, 2, 3, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_soft_grid_model(*, device: wp.DeviceLike, fix_left: bool = False) -> newton.Model:
    """Build one tetrahedral cell with its generated collision surface."""
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=1.0,
        cell_y=1.0,
        cell_z=1.0,
        density=1.0,
        k_mu=100.0,
        k_lambda=80.0,
        k_damp=1.0,
        fix_left=fix_left,
    )
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _build_mixed_deformable_model(*, device: wp.DeviceLike) -> newton.Model:
    """Build a tetrahedron and triangle cloth in one Newton world."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    for position in (
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(1.0, 0.0, 1.0),
        wp.vec3(0.0, 1.0, 1.0),
        wp.vec3(0.0, 0.0, 2.0),
    ):
        builder.add_particle(position, wp.vec3(0.0), mass=1.0, radius=0.0)
    builder.add_tetrahedron(0, 1, 2, 3, k_mu=100.0, k_lambda=80.0, k_damp=1.0)
    builder.add_cloth_grid(
        pos=wp.vec3(2.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        cell_x=1.0,
        cell_y=1.0,
        mass=1.0,
        tri_ke=100.0,
        tri_ka=80.0,
        tri_kd=1.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=1.0,
        edge_kd=0.0,
        particle_radius=0.0,
    )
    builder.end_world()
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _bsr_to_dense(matrix) -> np.ndarray:
    """Convert the active BSR topology to a dense scalar matrix."""
    offsets = matrix.offsets.numpy()
    columns = matrix.columns.numpy()
    values = matrix.values.numpy()
    block_rows, block_columns = matrix.block_shape
    dense = np.zeros((matrix.nrow * block_rows, matrix.ncol * block_columns), dtype=np.float64)
    for row in range(matrix.nrow):
        for block in range(int(offsets[row]), int(offsets[row + 1])):
            column = int(columns[block])
            dense[
                row * block_rows : (row + 1) * block_rows,
                column * block_columns : (column + 1) * block_columns,
            ] = values[block]
    return dense


def _reference_triangle_system(
    model: newton.Model,
    state: newton.State,
    dt: float,
    linearization_velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble an independent dense membrane reference."""
    positions_start = state.particle_q.numpy().astype(np.float64)
    velocity_start = state.particle_qd.numpy().astype(np.float64)
    velocities = (
        velocity_start if linearization_velocity is None else np.asarray(linearization_velocity, dtype=np.float64)
    )
    positions = positions_start + dt * velocities
    external_forces = state.particle_f.numpy().astype(np.float64)
    masses = model.particle_mass.numpy().astype(np.float64)
    flags = model.particle_flags.numpy()
    dynamic = ((flags & int(newton.ParticleFlags.ACTIVE)) != 0) & (masses > 0.0)
    gravity = model.gravity.numpy()[0].astype(np.float64)

    count = model.particle_count
    matrix = np.zeros((3 * count, 3 * count), dtype=np.float64)
    force = np.zeros((count, 3), dtype=np.float64)
    for particle in range(count):
        block = slice(3 * particle, 3 * particle + 3)
        if dynamic[particle]:
            matrix[block, block] = masses[particle] * np.eye(3)
            force[particle] = external_forces[particle] + masses[particle] * gravity
        else:
            matrix[block, block] = np.eye(3)

    indices = model.tri_indices.numpy()
    poses = model.tri_poses.numpy().astype(np.float64)
    areas = model.tri_areas.numpy().astype(np.float64)
    materials = model.tri_materials.numpy().astype(np.float64)
    activations = model.tri_activations.numpy().astype(np.float64)

    for triangle, vertices in enumerate(indices):
        rest_pose = poses[triangle]
        x10 = positions[vertices[1]] - positions[vertices[0]]
        x20 = positions[vertices[2]] - positions[vertices[0]]
        deformation = np.column_stack(
            (
                x10 * rest_pose[0, 0] + x20 * rest_pose[1, 0],
                x10 * rest_pose[0, 1] + x20 * rest_pose[1, 1],
            )
        )
        metric = deformation.T @ deformation
        area_ratio = np.sqrt(max(np.linalg.det(metric), 1.0e-20))
        area_gradients = np.column_stack(
            (
                (metric[1, 1] * deformation[:, 0] - metric[0, 1] * deformation[:, 1]) / area_ratio,
                (metric[0, 0] * deformation[:, 1] - metric[0, 1] * deformation[:, 0]) / area_ratio,
            )
        )

        mu = materials[triangle, 0]
        lmbd = materials[triangle, 1] + mu
        alpha = 1.0 + mu / lmbd if lmbd > 1.0e-6 else 1.0
        stress = mu * deformation + lmbd * (area_ratio - alpha + activations[triangle]) * area_gradients
        coefficients = (
            np.array(
                (
                    -rest_pose[0, 0] - rest_pose[1, 0],
                    -rest_pose[0, 1] - rest_pose[1, 1],
                )
            ),
            np.array((rest_pose[0, 0], rest_pose[0, 1])),
            np.array((rest_pose[1, 0], rest_pose[1, 1])),
        )

        deformation_start = np.column_stack(
            (
                (positions_start[vertices[1]] - positions_start[vertices[0]]) * rest_pose[0, 0]
                + (positions_start[vertices[2]] - positions_start[vertices[0]]) * rest_pose[1, 0],
                (positions_start[vertices[1]] - positions_start[vertices[0]]) * rest_pose[0, 1]
                + (positions_start[vertices[2]] - positions_start[vertices[0]]) * rest_pose[1, 1],
            )
        )
        metric_rate = (deformation.T @ deformation - deformation_start.T @ deformation_start) / dt
        metric_derivatives = []

        for order, vertex in enumerate(vertices):
            coefficient = coefficients[order]
            dc00 = 2.0 * coefficient[0] * deformation[:, 0]
            dc01 = coefficient[0] * deformation[:, 1] + coefficient[1] * deformation[:, 0]
            dc11 = 2.0 * coefficient[1] * deformation[:, 1]
            metric_derivatives.append((dc00, dc01, dc11))
            local_force = -(stress @ coefficient)
            damping = materials[triangle, 2]
            local_force -= damping * (
                metric_rate[0, 0] * dc00 + 2.0 * metric_rate[0, 1] * dc01 + metric_rate[1, 1] * dc11
            )
            if dynamic[vertex]:
                force[vertex] += areas[triangle] * local_force

        for row, row_vertex in enumerate(vertices):
            for column, column_vertex in enumerate(vertices):
                if not dynamic[row_vertex] or not dynamic[column_vertex]:
                    continue
                row_coefficient = coefficients[row]
                column_coefficient = coefficients[column]
                darea_row = area_gradients @ row_coefficient
                darea_column = area_gradients @ column_coefficient
                tangent = mu * np.dot(row_coefficient, column_coefficient) * np.eye(3)
                tangent += lmbd * np.outer(darea_row, darea_column)
                damping = materials[triangle, 2]
                if damping > 0.0:
                    row_derivatives = metric_derivatives[row]
                    column_derivatives = metric_derivatives[column]
                    tangent += (
                        damping
                        / dt
                        * (
                            np.outer(row_derivatives[0], column_derivatives[0])
                            + 2.0 * np.outer(row_derivatives[1], column_derivatives[1])
                            + np.outer(row_derivatives[2], column_derivatives[2])
                        )
                    )
                row_block = slice(3 * row_vertex, 3 * row_vertex + 3)
                column_block = slice(3 * column_vertex, 3 * column_vertex + 3)
                matrix[row_block, column_block] += dt * dt * areas[triangle] * tangent

    inertial_correction = np.zeros_like(velocities)
    inertial_correction[dynamic] = masses[dynamic, None] * (velocity_start[dynamic] - velocities[dynamic])
    rhs = matrix @ velocities.reshape(-1) + inertial_correction.reshape(-1) + dt * force.reshape(-1)
    rhs.reshape((-1, 3))[~dynamic] = 0.0
    return matrix, rhs


def _tetrahedron_local_hessian(model: newton.Model, deformation: np.ndarray, tetrahedron: int = 0) -> np.ndarray:
    """Evaluate the full tetrahedron energy Hessian with respect to ``F``."""
    cofactor = np.column_stack(
        (
            np.cross(deformation[:, 1], deformation[:, 2]),
            np.cross(deformation[:, 2], deformation[:, 0]),
            np.cross(deformation[:, 0], deformation[:, 1]),
        )
    )
    rest_pose = model.tet_poses.numpy()[tetrahedron].astype(np.float64)
    rest_volume = 1.0 / (6.0 * np.linalg.det(rest_pose))
    material = model.tet_materials.numpy()[tetrahedron]
    mu = float(material[0])
    lmbd = float(material[1]) + mu
    alpha = 1.0 + mu / lmbd if lmbd > 1.0e-6 else 1.0
    constraint = np.linalg.det(deformation) - alpha + float(model.tet_activations.numpy()[tetrahedron])
    cofactor_vector = cofactor.reshape(-1, order="F")
    hessian = rest_volume * (mu * np.eye(9) + lmbd * np.outer(cofactor_vector, cofactor_vector))
    levi_civita = np.zeros((3, 3, 3), dtype=np.float64)
    levi_civita[0, 1, 2] = levi_civita[1, 2, 0] = levi_civita[2, 0, 1] = 1.0
    levi_civita[0, 2, 1] = levi_civita[2, 1, 0] = levi_civita[1, 0, 2] = -1.0
    determinant_hessian = np.empty((9, 9), dtype=np.float64)
    for first in range(9):
        first_row, first_column = first % 3, first // 3
        for second in range(9):
            second_row, second_column = second % 3, second // 3
            determinant_hessian[first, second] = np.einsum(
                "m,n,mn->",
                levi_civita[first_row, second_row],
                levi_civita[first_column, second_column],
                deformation,
            )
    return hessian + rest_volume * lmbd * constraint * determinant_hessian


def _reference_tetrahedron_system(
    model: newton.Model,
    state: newton.State,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble an independent dense tetrahedron reference."""
    positions_start = state.particle_q.numpy().astype(np.float64)
    velocities = state.particle_qd.numpy().astype(np.float64)
    positions = positions_start + dt * velocities
    masses = model.particle_mass.numpy().astype(np.float64)
    flags = model.particle_flags.numpy()
    dynamic = ((flags & int(newton.ParticleFlags.ACTIVE)) != 0) & (masses > 0.0)
    count = model.particle_count
    matrix = np.zeros((3 * count, 3 * count), dtype=np.float64)
    force = np.zeros((count, 3), dtype=np.float64)
    for particle in range(count):
        block = slice(3 * particle, 3 * particle + 3)
        matrix[block, block] = (masses[particle] if dynamic[particle] else 1.0) * np.eye(3)

    indices = model.tet_indices.numpy()
    poses = model.tet_poses.numpy().astype(np.float64)
    materials = model.tet_materials.numpy().astype(np.float64)
    activations = model.tet_activations.numpy().astype(np.float64)
    for tetrahedron, vertices in enumerate(indices):
        rest_pose = poses[tetrahedron]
        deformation = (
            np.column_stack(
                (
                    positions[vertices[1]] - positions[vertices[0]],
                    positions[vertices[2]] - positions[vertices[0]],
                    positions[vertices[3]] - positions[vertices[0]],
                )
            )
            @ rest_pose
        )
        deformation_start = (
            np.column_stack(
                (
                    positions_start[vertices[1]] - positions_start[vertices[0]],
                    positions_start[vertices[2]] - positions_start[vertices[0]],
                    positions_start[vertices[3]] - positions_start[vertices[0]],
                )
            )
            @ rest_pose
        )
        cofactor = np.column_stack(
            (
                np.cross(deformation[:, 1], deformation[:, 2]),
                np.cross(deformation[:, 2], deformation[:, 0]),
                np.cross(deformation[:, 0], deformation[:, 1]),
            )
        )
        rest_volume = 1.0 / (6.0 * np.linalg.det(rest_pose))
        k_mu, k_lambda, damping = materials[tetrahedron]
        lambda_hat = k_lambda + k_mu
        pressure = lambda_hat * (np.linalg.det(deformation) - 1.0 + activations[tetrahedron])
        if lambda_hat > 0.0:
            pressure -= lambda_hat * k_mu / max(lambda_hat, 1.0e-6)
        elif lambda_hat < 0.0:
            pressure += lambda_hat * k_mu / max(-lambda_hat, 1.0e-6)
        stress_density = k_mu * deformation + pressure * cofactor
        stress = rest_volume * stress_density
        full_hessian = _tetrahedron_local_hessian(model, deformation, tetrahedron)
        eigenvalues, eigenvectors = np.linalg.eigh(full_hessian)
        negative_curvature = max(0.0, -float(np.min(eigenvalues)))
        metric_eigenvalues = np.abs(eigenvalues) + _TETRAHEDRON_NEGATIVE_CURVATURE_MARGIN * negative_curvature
        majorizing_metric = (eigenvectors * metric_eigenvalues) @ eigenvectors.T
        coefficients = (
            -np.sum(rest_pose, axis=0),
            rest_pose[0],
            rest_pose[1],
            rest_pose[2],
        )
        metric_rate = (deformation.T @ deformation - deformation_start.T @ deformation_start) / dt

        def metric_derivatives(
            deformation_value: np.ndarray,
            coefficient: np.ndarray,
        ) -> tuple[np.ndarray, ...]:
            return (
                2.0 * coefficient[0] * deformation_value[:, 0],
                coefficient[0] * deformation_value[:, 1] + coefficient[1] * deformation_value[:, 0],
                coefficient[0] * deformation_value[:, 2] + coefficient[2] * deformation_value[:, 0],
                2.0 * coefficient[1] * deformation_value[:, 1],
                coefficient[1] * deformation_value[:, 2] + coefficient[2] * deformation_value[:, 1],
                2.0 * coefficient[2] * deformation_value[:, 2],
            )

        metric_rates = (
            metric_rate[0, 0],
            metric_rate[0, 1],
            metric_rate[0, 2],
            metric_rate[1, 1],
            metric_rate[1, 2],
            metric_rate[2, 2],
        )
        metric_weights = (1.0, 2.0, 2.0, 1.0, 2.0, 1.0)
        derivatives = tuple(metric_derivatives(deformation, coefficient) for coefficient in coefficients)

        for row, row_vertex in enumerate(vertices):
            local_force = -(stress @ coefficients[row])
            local_force -= (
                rest_volume
                * damping
                * sum(
                    weight * rate * derivative
                    for weight, rate, derivative in zip(metric_weights, metric_rates, derivatives[row], strict=True)
                )
            )
            if dynamic[row_vertex]:
                force[row_vertex] += local_force

            for column, column_vertex in enumerate(vertices):
                if not dynamic[row_vertex] or not dynamic[column_vertex]:
                    continue
                row_slice = slice(3 * row_vertex, 3 * row_vertex + 3)
                column_slice = slice(3 * column_vertex, 3 * column_vertex + 3)
                row_map = np.zeros((9, 3), dtype=np.float64)
                column_map = np.zeros((9, 3), dtype=np.float64)
                for deformation_column in range(3):
                    for axis in range(3):
                        row_map[3 * deformation_column + axis, axis] = coefficients[row][deformation_column]
                        column_map[3 * deformation_column + axis, axis] = coefficients[column][deformation_column]
                tangent = row_map.T @ majorizing_metric @ column_map
                tangent += (
                    rest_volume
                    * damping
                    / dt
                    * sum(
                        weight * np.outer(row_derivative, column_derivative)
                        for weight, row_derivative, column_derivative in zip(
                            metric_weights,
                            derivatives[row],
                            derivatives[column],
                            strict=True,
                        )
                    )
                )
                matrix[row_slice, column_slice] += dt * dt * tangent

    return matrix, force


def _membrane_energy(model: newton.Model, positions: np.ndarray) -> float:
    """Evaluate the stable membrane energy used by the cloth system."""
    energy = 0.0
    for triangle, vertices in enumerate(model.tri_indices.numpy()):
        rest_pose = model.tri_poses.numpy()[triangle].astype(np.float64)
        x10 = positions[vertices[1]] - positions[vertices[0]]
        x20 = positions[vertices[2]] - positions[vertices[0]]
        deformation = np.column_stack(
            (
                x10 * rest_pose[0, 0] + x20 * rest_pose[1, 0],
                x10 * rest_pose[0, 1] + x20 * rest_pose[1, 1],
            )
        )
        metric = deformation.T @ deformation
        area_ratio = np.sqrt(max(np.linalg.det(metric), 1.0e-20))
        material = model.tri_materials.numpy()[triangle]
        mu = float(material[0])
        lmbd = float(material[1]) + mu
        alpha = 1.0 + mu / lmbd if lmbd > 1.0e-6 else 1.0
        constraint = area_ratio - alpha + float(model.tri_activations.numpy()[triangle])
        density = 0.5 * mu * (np.sum(deformation * deformation) - 2.0) + 0.5 * lmbd * constraint**2
        energy += float(model.tri_areas.numpy()[triangle]) * density
    return energy


def _membrane_local_gradient(model: newton.Model, deformation: np.ndarray, triangle: int = 0) -> np.ndarray:
    """Evaluate the local membrane energy gradient with respect to ``F``."""
    metric = deformation.T @ deformation
    area_ratio = np.sqrt(max(np.linalg.det(metric), 1.0e-20))
    area_gradient = np.column_stack(
        (
            (metric[1, 1] * deformation[:, 0] - metric[0, 1] * deformation[:, 1]) / area_ratio,
            (metric[0, 0] * deformation[:, 1] - metric[0, 1] * deformation[:, 0]) / area_ratio,
        )
    )
    material = model.tri_materials.numpy()[triangle]
    mu = float(material[0])
    lmbd = float(material[1]) + mu
    alpha = 1.0 + mu / lmbd if lmbd > 1.0e-6 else 1.0
    constraint = area_ratio - alpha + float(model.tri_activations.numpy()[triangle])
    return float(model.tri_areas.numpy()[triangle]) * (mu * deformation + lmbd * constraint * area_gradient)


def _tetrahedron_local_gradient(model: newton.Model, deformation: np.ndarray, tetrahedron: int = 0) -> np.ndarray:
    """Evaluate the local tetrahedron energy gradient with respect to ``F``."""
    cofactor = np.column_stack(
        (
            np.cross(deformation[:, 1], deformation[:, 2]),
            np.cross(deformation[:, 2], deformation[:, 0]),
            np.cross(deformation[:, 0], deformation[:, 1]),
        )
    )
    rest_pose = model.tet_poses.numpy()[tetrahedron].astype(np.float64)
    rest_volume = 1.0 / (6.0 * np.linalg.det(rest_pose))
    material = model.tet_materials.numpy()[tetrahedron]
    mu = float(material[0])
    lmbd = float(material[1]) + mu
    alpha = 1.0 + mu / lmbd if lmbd > 1.0e-6 else 1.0
    constraint = np.linalg.det(deformation) - alpha + float(model.tet_activations.numpy()[tetrahedron])
    return rest_volume * (mu * deformation + lmbd * constraint * cofactor)


def _tetrahedron_force(model: newton.Model, positions: np.ndarray) -> np.ndarray:
    """Evaluate the nodal tetrahedron force from local energy gradients."""
    force = np.zeros_like(positions, dtype=np.float64)
    for tetrahedron, vertices in enumerate(model.tet_indices.numpy()):
        rest_pose = model.tet_poses.numpy()[tetrahedron].astype(np.float64)
        deformation = (
            np.column_stack(
                (
                    positions[vertices[1]] - positions[vertices[0]],
                    positions[vertices[2]] - positions[vertices[0]],
                    positions[vertices[3]] - positions[vertices[0]],
                )
            )
            @ rest_pose
        )
        gradient = _tetrahedron_local_gradient(model, deformation, tetrahedron)
        coefficients = (
            -np.sum(rest_pose, axis=0),
            rest_pose[0],
            rest_pose[1],
            rest_pose[2],
        )
        for order, vertex in enumerate(vertices):
            force[vertex] -= gradient @ coefficients[order]
    return force


def _membrane_force(model: newton.Model, positions: np.ndarray) -> np.ndarray:
    """Evaluate the nodal membrane force from local energy gradients."""
    force = np.zeros_like(positions, dtype=np.float64)
    for triangle, vertices in enumerate(model.tri_indices.numpy()):
        rest_pose = model.tri_poses.numpy()[triangle].astype(np.float64)
        deformation = np.column_stack(
            (
                (positions[vertices[1]] - positions[vertices[0]]) * rest_pose[0, 0]
                + (positions[vertices[2]] - positions[vertices[0]]) * rest_pose[1, 0],
                (positions[vertices[1]] - positions[vertices[0]]) * rest_pose[0, 1]
                + (positions[vertices[2]] - positions[vertices[0]]) * rest_pose[1, 1],
            )
        )
        gradient = _membrane_local_gradient(model, deformation, triangle)
        coefficients = (
            np.array(
                (
                    -rest_pose[0, 0] - rest_pose[1, 0],
                    -rest_pose[0, 1] - rest_pose[1, 1],
                )
            ),
            np.array((rest_pose[0, 0], rest_pose[0, 1])),
            np.array((rest_pose[1, 0], rest_pose[1, 1])),
        )
        for order, vertex in enumerate(vertices):
            force[vertex] -= gradient @ coefficients[order]
    return force


def _bending_energy(model: newton.Model, positions: np.ndarray) -> float:
    """Evaluate Newton's dihedral-angle bending energy."""
    energy = 0.0
    edge_indices = model.edge_indices.numpy()
    rest_angles = model.edge_rest_angle.numpy()
    rest_lengths = model.edge_rest_length.numpy()
    properties = model.edge_bending_properties.numpy()
    for edge, vertices in enumerate(edge_indices):
        if np.any(vertices < 0):
            continue
        x0, x1, x2, x3 = positions[vertices]
        normal_0 = np.cross(x2 - x0, x3 - x0)
        normal_1 = np.cross(x3 - x1, x2 - x1)
        edge_vector = x3 - x2
        normal_0 /= np.linalg.norm(normal_0)
        normal_1 /= np.linalg.norm(normal_1)
        edge_vector /= np.linalg.norm(edge_vector)
        angle = np.arctan2(
            np.dot(np.cross(normal_0, normal_1), edge_vector),
            np.dot(normal_0, normal_1),
        )
        stiffness = float(properties[edge, 0]) * float(rest_lengths[edge])
        energy += 0.5 * stiffness * (angle - float(rest_angles[edge])) ** 2
    return energy


def _finite_difference_force(energy, positions: np.ndarray, epsilon: float = 1.0e-4) -> np.ndarray:
    """Differentiate an energy with centered finite differences."""
    force = np.zeros_like(positions, dtype=np.float64)
    for particle in range(positions.shape[0]):
        for axis in range(3):
            positive = positions.copy()
            negative = positions.copy()
            positive[particle, axis] += epsilon
            negative[particle, axis] -= epsilon
            force[particle, axis] = -(energy(positive) - energy(negative)) / (2.0 * epsilon)
    return force


class TestLOXDeformableSystem(unittest.TestCase):
    """Test the LOX cloth topology and smooth-system assembly."""

    def setUp(self):
        """Select the configured Newton test device."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_validate_supported_and_unsupported_cloth(self):
        """Reject unsupported deformable components and malformed topology."""
        model = _build_grid_model(device=self.device)
        validate_deformable_cloth_model(model)

        spring_model = _build_grid_model(device=self.device, add_springs=True)
        with self.assertRaisesRegex(ValueError, "springs"):
            validate_deformable_cloth_model(spring_model)

        builder = newton.ModelBuilder()
        builder.begin_world()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
        )
        builder.add_particle(pos=wp.vec3(2.0, 2.0, 0.0), vel=wp.vec3(0.0), mass=1.0)
        builder.end_world()
        unattached_model = builder.finalize(device=self.device)
        with self.assertRaisesRegex(ValueError, "unattached particles"):
            validate_deformable_cloth_model(unattached_model)

        cross_world_model = _build_grid_model(device=self.device, world_count=2)
        cross_world_triangles = cross_world_model.tri_indices.numpy()
        cross_world_triangles[0, 2] = 4
        cross_world_model.tri_indices.assign(cross_world_triangles)
        with self.assertRaisesRegex(ValueError, "cannot span Newton worlds"):
            validate_deformable_cloth_model(cross_world_model)

    def test_accept_pure_tetrahedral_deformable(self):
        """Accept and assemble a pure tetrahedral deformable model."""
        model = _build_tet_model(device=self.device)

        validate_deformable_cloth_model(model)
        system = DeformableClothSystem(model)
        state_in = model.state()
        system.assemble(state_in, _world_dt(model, 0.01, self.device))

        self.assertEqual(system.triplet_count, model.particle_count + 16 * model.tet_count)
        self.assertTrue(np.all(np.isfinite(system.smooth_matrix.values.numpy())))

        state_out = model.state()
        solver = newton.solvers.SolverKamino(
            model,
            config=newton.solvers.SolverKamino.Config(dynamics_solver="lox"),
        )
        solver.step(state_in, state_out, None, None, 0.01)
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def test_initialize_deformable_inertial_warmstart_fraction(self):
        """Pre-apply a fraction of State force and gravity to the deformable initial guess."""
        model = _build_grid_model(device=self.device)
        model.set_gravity((0.4, -1.2, -9.0))
        state = model.state()
        initial_velocity = np.tile(np.asarray((0.2, -0.1, 0.3), dtype=np.float32), (model.particle_count, 1))
        initial_force = np.tile(np.asarray((0.7, -0.5, 1.1), dtype=np.float32), (model.particle_count, 1))
        state.particle_qd.assign(initial_velocity)
        state.particle_f.assign(initial_force)
        fraction = 0.25
        time_step = 0.02
        system = DeformableClothSystem(model)
        splitting = DeformableSplittingState(system)

        world_dt = _world_dt(model, time_step, self.device)
        system.assemble(state, world_dt)
        splitting.begin_time_step(world_dt, 0.0)

        packed = system.topology.packed_to_newton.numpy()
        np.testing.assert_array_equal(splitting.projected_velocity.numpy(), initial_velocity[packed])

        retained = np.full_like(initial_velocity, 17.0)
        next_velocity = np.tile(np.asarray((-0.3, 0.5, -0.4), dtype=np.float32), (model.particle_count, 1))
        next_force = np.tile(np.asarray((-0.8, 0.6, 1.4), dtype=np.float32), (model.particle_count, 1))
        splitting.projected_velocity.assign(retained[packed])
        state.particle_qd.assign(next_velocity)
        state.particle_f.assign(next_force)
        world_dt = _world_dt(model, time_step, self.device)
        system.assemble(state, world_dt)
        splitting.begin_time_step(world_dt, fraction)

        particle_world = model.particle_world.numpy()
        acceleration = next_force / model.particle_mass.numpy()[:, None] + model.gravity.numpy()[particle_world]
        np.testing.assert_allclose(
            splitting.projected_velocity.numpy(),
            (next_velocity + fraction * time_step * acceleration)[packed],
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_deformable_dual_impulse_round_trips_state_order(self):
        """Round-trip the nodal consensus impulse through Newton particle order."""
        model = _build_grid_model(device=self.device)
        system = DeformableClothSystem(model)
        splitting = DeformableSplittingState(system)
        newton_impulse = np.arange(3 * model.particle_count, dtype=np.float32).reshape((-1, 3))
        state_impulse = wp.array(newton_impulse, dtype=wp.vec3, device=self.device)
        packed = system.topology.packed_to_newton.numpy()

        splitting.load_state_dual_impulse(state_impulse)
        np.testing.assert_array_equal(splitting.dual_impulse.numpy(), newton_impulse[packed])

        packed_impulse = -0.5 * splitting.dual_impulse.numpy()
        splitting.dual_impulse.assign(packed_impulse)
        state_impulse.zero_()
        splitting.write_state_dual_impulse(state_impulse)
        expected = np.empty_like(newton_impulse)
        expected[packed] = packed_impulse
        np.testing.assert_array_equal(state_impulse.numpy(), expected)

    def test_assemble_tetrahedron_against_dense_reference(self):
        """Match tet forces and all pair blocks against a dense reference."""
        model = _build_tet_model(device=self.device, k_mu=120.0, k_lambda=75.0, k_damp=3.0)
        model.tet_activations.assign(np.array([0.07], dtype=np.float32))
        state = model.state()
        state.particle_qd.assign(
            np.array(
                (
                    (0.1, -0.2, 0.3),
                    (0.4, 0.1, -0.2),
                    (-0.3, 0.5, 0.15),
                    (0.2, -0.4, 0.35),
                ),
                dtype=np.float32,
            )
        )
        dt = 0.02
        for proximal_iterations in (0, 1):
            with self.subTest(proximal_iterations=proximal_iterations):
                system = DeformableClothSystem(model, proximal_iterations=proximal_iterations)
                system.assemble(state, _world_dt(model, dt, self.device))
                reference_matrix, reference_force = _reference_tetrahedron_system(model, state, dt)

                packed = system.topology.packed_to_newton.numpy()
                scalar_order = (3 * packed[:, None] + np.arange(3)).reshape(-1)
                np.testing.assert_allclose(
                    _bsr_to_dense(system.smooth_matrix),
                    reference_matrix[np.ix_(scalar_order, scalar_order)],
                    rtol=2.0e-5,
                    atol=2.0e-5,
                )
                np.testing.assert_allclose(
                    system.smooth_force.numpy(),
                    reference_force[packed],
                    rtol=2.0e-5,
                    atol=2.0e-5,
                )

    def test_tetrahedron_damping_ignores_finite_rigid_rotation(self):
        """Keep objective metric-rate damping invariant under rigid rotation."""
        model = _build_tet_model(device=self.device, k_mu=120.0, k_lambda=75.0, k_damp=1.0e6)
        state = model.state()
        positions_start = model.particle_q.numpy()
        rotation_end = np.array(
            (
                (0.0, -1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        positions_end = positions_start @ rotation_end.T
        dt = 0.25
        state.particle_q.assign(positions_start)
        state.particle_qd.assign((positions_end - positions_start) / dt)

        system = DeformableClothSystem(model, proximal_iterations=0)
        system.assemble(state, _world_dt(model, dt, self.device))
        force_with_damping = system.smooth_force.numpy().copy()

        materials = model.tet_materials.numpy()
        materials[:, 2] = 0.0
        model.tet_materials.assign(materials)
        system.assemble(state, _world_dt(model, dt, self.device))
        force_without_damping = system.smooth_force.numpy().copy()

        np.testing.assert_allclose(system.position_linearized.numpy(), positions_end, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(force_with_damping, force_without_damping, rtol=0.0, atol=1.0e-6)

    def test_tetrahedron_objective_damping_is_first_order_at_rest(self):
        """Retain nonzero first-order strain damping at the rest state."""
        model = _build_tet_model(device=self.device, k_mu=120.0, k_lambda=75.0, k_damp=4.0)
        state = model.state()
        system = DeformableClothSystem(model, proximal_iterations=0)
        system.assemble(state, _world_dt(model, 0.02, self.device))
        matrix_with_damping = _bsr_to_dense(system.smooth_matrix)

        materials = model.tet_materials.numpy()
        materials[:, 2] = 0.0
        model.tet_materials.assign(materials)
        system.assemble(state, _world_dt(model, 0.02, self.device))
        damping_tangent = matrix_with_damping - _bsr_to_dense(system.smooth_matrix)

        eigenvalues = np.linalg.eigvalsh(damping_tangent)
        self.assertGreaterEqual(float(eigenvalues[0]), -1.0e-6)
        self.assertEqual(int(np.count_nonzero(eigenvalues > 1.0e-5)), 6)

    def test_tetrahedron_objective_damping_dissipates(self):
        """Make objective metric-rate damping dissipative with a PSD tangent."""
        model = _build_tet_model(device=self.device, k_mu=120.0, k_lambda=75.0, k_damp=4.0)
        state = model.state()
        state.particle_q.assign(
            np.array(
                (
                    (0.05, -0.02, 0.03),
                    (0.82, 0.08, -0.04),
                    (0.12, 0.91, 0.09),
                    (-0.08, 0.11, 0.76),
                ),
                dtype=np.float32,
            )
        )
        velocity = np.array(
            (
                (0.1, -0.2, 0.3),
                (0.4, 0.1, -0.2),
                (-0.3, 0.5, 0.15),
                (0.2, -0.4, 0.35),
            ),
            dtype=np.float32,
        )
        state.particle_qd.assign(velocity)
        system = DeformableClothSystem(model, proximal_iterations=0)
        system.assemble(state, _world_dt(model, 0.02, self.device))
        force_with_damping = system.smooth_force.numpy().copy()
        matrix_with_damping = _bsr_to_dense(system.smooth_matrix)

        materials = model.tet_materials.numpy()
        materials[:, 2] = 0.0
        model.tet_materials.assign(materials)
        system.assemble(state, _world_dt(model, 0.02, self.device))
        damping_force = force_with_damping - system.smooth_force.numpy()
        damping_tangent = matrix_with_damping - _bsr_to_dense(system.smooth_matrix)

        self.assertLess(float(np.sum(damping_force * velocity)), 0.0)
        np.testing.assert_allclose(damping_tangent, damping_tangent.T, rtol=0.0, atol=1.0e-5)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(damping_tangent))), -1.0e-4)

    def test_tetrahedron_damping_stays_absolute_under_stiffness_scaling(self):
        """Keep fixed absolute damping unchanged under uniform stiffness scaling."""
        positions = np.array(
            (
                (0.05, -0.02, 0.03),
                (0.82, 0.08, -0.04),
                (0.12, 0.91, 0.09),
                (-0.08, 0.11, 0.76),
            ),
            dtype=np.float32,
        )
        velocity = np.array(
            (
                (0.1, -0.2, 0.3),
                (0.4, 0.1, -0.2),
                (-0.3, 0.5, 0.15),
                (0.2, -0.4, 0.35),
            ),
            dtype=np.float32,
        )

        def damping_contribution(stiffness_scale: float) -> tuple[np.ndarray, np.ndarray]:
            model = _build_tet_model(
                device=self.device,
                k_mu=stiffness_scale * 120.0,
                k_lambda=stiffness_scale * 75.0,
                k_damp=4.0,
            )
            state = model.state()
            state.particle_q.assign(positions)
            state.particle_qd.assign(velocity)
            system = DeformableClothSystem(model, proximal_iterations=0)
            system.assemble(state, _world_dt(model, 0.02, self.device))
            force_with_damping = system.smooth_force.numpy().copy()
            matrix_with_damping = _bsr_to_dense(system.smooth_matrix)

            materials = model.tet_materials.numpy()
            materials[:, 2] = 0.0
            model.tet_materials.assign(materials)
            system.assemble(state, _world_dt(model, 0.02, self.device))
            return (
                force_with_damping - system.smooth_force.numpy(),
                matrix_with_damping - _bsr_to_dense(system.smooth_matrix),
            )

        reference_force, reference_tangent = damping_contribution(1.0)
        scaled_force, scaled_tangent = damping_contribution(16.0)
        np.testing.assert_allclose(scaled_force, reference_force, rtol=2.0e-3, atol=5.0e-6)
        np.testing.assert_allclose(scaled_tangent, reference_tangent, rtol=2.0e-3, atol=5.0e-6)

    def test_tetrahedron_full_hessian_matches_gradient_differential(self):
        """Match the full tet Hessian to the exact gradient differential."""
        model = _build_tet_model(device=self.device, k_mu=120.0, k_lambda=75.0, k_damp=0.0)
        model.tet_activations.assign(np.array([0.07], dtype=np.float32))
        deformation = np.array(
            (
                (0.45, 0.2, -0.1),
                (0.1, 0.8, 0.15),
                (-0.05, 0.2, 0.9),
            ),
            dtype=np.float32,
        )
        rest_pose = model.tet_poses.numpy()[0].astype(np.float64)
        rest_volume = 1.0 / (6.0 * np.linalg.det(rest_pose))
        material = model.tet_materials.numpy()[0]
        hessian = wp.empty(1, dtype=mat99, device=self.device)
        projected_hessian = wp.empty(1, dtype=mat99, device=self.device)
        majorizing_metric = wp.empty(1, dtype=mat99, device=self.device)
        wp.launch(
            _evaluate_tetrahedron_hessian,
            dim=1,
            inputs=[
                wp.array([deformation], dtype=wp.mat33, device=self.device),
                rest_volume,
                float(material[0]),
                float(material[1]),
                float(model.tet_activations.numpy()[0]),
            ],
            outputs=[hessian, projected_hessian, majorizing_metric],
            device=self.device,
        )

        epsilon = 1.0e-4
        reference = np.empty((9, 9), dtype=np.float64)
        for column in range(9):
            row_index = column % 3
            column_index = column // 3
            plus = deformation.astype(np.float64)
            minus = deformation.astype(np.float64)
            plus[row_index, column_index] += epsilon
            minus[row_index, column_index] -= epsilon
            reference[:, column] = (
                (_tetrahedron_local_gradient(model, plus) - _tetrahedron_local_gradient(model, minus)) / (2.0 * epsilon)
            ).reshape(-1, order="F")

        actual = hessian.numpy()[0].astype(np.float64)
        np.testing.assert_allclose(actual, reference, rtol=2.0e-4, atol=2.0e-4)
        np.testing.assert_allclose(actual, actual.T, rtol=0.0, atol=1.0e-6)
        self.assertLess(float(np.min(np.linalg.eigvalsh(actual))), -1.0)
        eigenvalues, eigenvectors = np.linalg.eigh(reference)
        reference_projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        actual_projected = projected_hessian.numpy()[0].astype(np.float64)
        np.testing.assert_allclose(actual_projected, reference_projected, rtol=2.0e-4, atol=2.0e-4)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(actual_projected))), -1.0e-4)

        negative_curvature = max(0.0, -float(np.min(eigenvalues)))
        reference_metric_eigenvalues = np.abs(eigenvalues) + _TETRAHEDRON_NEGATIVE_CURVATURE_MARGIN * negative_curvature
        reference_majorizer = (eigenvectors * reference_metric_eigenvalues) @ eigenvectors.T
        actual_majorizer = majorizing_metric.numpy()[0].astype(np.float64)
        np.testing.assert_allclose(actual_majorizer, reference_majorizer, rtol=2.0e-4, atol=2.0e-4)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(actual + actual_majorizer))), 1.0)

    def test_tetrahedron_majorizer_preserves_rest_rotation_null_modes(self):
        """Preserve the exact rest Hessian and its rotational null modes."""
        model = _build_tet_model(device=self.device, k_mu=120.0, k_lambda=75.0, k_damp=0.0)
        rest_pose = model.tet_poses.numpy()[0].astype(np.float64)
        rest_volume = 1.0 / (6.0 * np.linalg.det(rest_pose))
        material = model.tet_materials.numpy()[0]
        hessian = wp.empty(1, dtype=mat99, device=self.device)
        projected_hessian = wp.empty(1, dtype=mat99, device=self.device)
        majorizing_metric = wp.empty(1, dtype=mat99, device=self.device)
        wp.launch(
            _evaluate_tetrahedron_hessian,
            dim=1,
            inputs=[
                wp.array([np.eye(3, dtype=np.float32)], dtype=wp.mat33, device=self.device),
                rest_volume,
                float(material[0]),
                float(material[1]),
                0.0,
            ],
            outputs=[hessian, projected_hessian, majorizing_metric],
            device=self.device,
        )

        actual_hessian = hessian.numpy()[0].astype(np.float64)
        actual_metric = majorizing_metric.numpy()[0].astype(np.float64)
        np.testing.assert_allclose(actual_metric, actual_hessian, rtol=2.0e-4, atol=2.0e-4)
        expected_eigenvalues = rest_volume * np.array(
            [0.0, 0.0, 0.0, 240.0, 240.0, 240.0, 240.0, 240.0, 465.0],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            np.linalg.eigvalsh(actual_metric),
            expected_eigenvalues,
            rtol=2.0e-4,
            atol=2.0e-4,
        )

    def test_tetrahedron_spectral_metrics_preserve_small_energy_scales(self):
        """Preserve small tet scales and match the dense Gauss-Newton solve."""
        deformation = np.array(
            (
                (0.45, 0.2, -0.1),
                (0.1, 0.8, 0.15),
                (-0.05, 0.2, 0.9),
            ),
            dtype=np.float32,
        )
        hessian = wp.empty(1, dtype=mat99, device=self.device)
        projected_hessian = wp.empty(1, dtype=mat99, device=self.device)
        majorizing_metric = wp.empty(1, dtype=mat99, device=self.device)

        def evaluate(rest_volume: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            wp.launch(
                _evaluate_tetrahedron_hessian,
                dim=1,
                inputs=[
                    wp.array([deformation], dtype=wp.mat33, device=self.device),
                    rest_volume,
                    120.0,
                    75.0,
                    0.07,
                ],
                outputs=[hessian, projected_hessian, majorizing_metric],
                device=self.device,
            )
            return (
                hessian.numpy()[0].copy(),
                projected_hessian.numpy()[0].copy(),
                majorizing_metric.numpy()[0].copy(),
            )

        reference = evaluate(1.0)
        energy_scale = 1.0e-15
        scaled = evaluate(energy_scale)
        for actual, expected in zip(scaled, reference, strict=True):
            np.testing.assert_allclose(actual, energy_scale * expected, rtol=1.0e-3, atol=1.0e-18)

        right_hand_side = np.array(
            (
                (0.2, -0.1, 0.3),
                (-0.4, 0.5, -0.2),
                (0.1, 0.2, -0.3),
            ),
            dtype=np.float32,
        )
        step = wp.empty(1, dtype=wp.mat33, device=self.device)
        succeeded = wp.empty(1, dtype=wp.int32, device=self.device)

        def solve(scale: float, metric: np.ndarray) -> np.ndarray:
            wp.launch(
                _evaluate_tetrahedron_proximal_step,
                dim=1,
                inputs=[
                    wp.array([scale * right_hand_side], dtype=wp.mat33, device=self.device),
                    wp.array([deformation], dtype=wp.mat33, device=self.device),
                    wp.array(metric[None], dtype=mat99, device=self.device),
                    scale,
                ],
                outputs=[step, succeeded],
                device=self.device,
            )
            self.assertEqual(int(succeeded.numpy()[0]), 1)
            return step.numpy()[0].copy()

        reference_step = solve(1.0, reference[2])
        cofactor = np.column_stack(
            (
                np.cross(deformation[:, 1], deformation[:, 2]),
                np.cross(deformation[:, 2], deformation[:, 0]),
                np.cross(deformation[:, 0], deformation[:, 1]),
            )
        ).reshape(-1, order="F")
        base = reference[2].astype(np.float64) + 120.0 * np.eye(9)
        volume_scale = 75.0 + 120.0
        dense_system = base + volume_scale * np.outer(cofactor, cofactor)
        dense_system += 1.0e-6 * np.max(np.abs(dense_system)) * np.eye(9)
        expected_step = np.linalg.solve(
            dense_system,
            right_hand_side.reshape(-1, order="F"),
        ).reshape((3, 3), order="F")
        np.testing.assert_allclose(reference_step, expected_step, rtol=2.0e-5, atol=2.0e-7)

        scaled_step = solve(energy_scale, scaled[2])
        np.testing.assert_allclose(scaled_step, reference_step, rtol=2.0e-3, atol=2.0e-6)

    def test_step_soft_grid_in_rigid_free_fall(self):
        """Advance a standard soft grid in rigid free fall."""
        model = _build_soft_grid_model(device=self.device)
        validate_deformable_cloth_model(model)
        solver = newton.solvers.SolverKamino(
            model,
            config=newton.solvers.SolverKamino.Config(dynamics_solver="lox"),
        )
        state_in = model.state()
        state_out = model.state()
        dt = 0.01

        solver.step(state_in, state_out, None, None, dt)

        np.testing.assert_allclose(
            state_out.particle_qd.numpy(),
            np.tile((0.0, 0.0, -9.81 * dt), (model.particle_count, 1)),
            rtol=3.0e-4,
            atol=3.0e-5,
        )
        np.testing.assert_allclose(
            state_out.particle_q.numpy(),
            state_in.particle_q.numpy() + dt * state_out.particle_qd.numpy(),
            rtol=3.0e-5,
            atol=3.0e-6,
        )

    def test_step_kinematic_soft_grid_particles(self):
        """Advance active zero-mass particles using their prescribed velocities."""
        model = _build_soft_grid_model(device=self.device, fix_left=True)
        masses = model.particle_mass.numpy()
        flags = model.particle_flags.numpy()
        pinned = masses == 0.0
        self.assertTrue(np.any(pinned))
        self.assertTrue(np.all((flags[pinned] & int(newton.ParticleFlags.ACTIVE)) != 0))
        validate_deformable_cloth_model(model)

        solver = newton.solvers.SolverKamino(
            model,
            config=newton.solvers.SolverKamino.Config(dynamics_solver="lox"),
        )
        state_in = model.state()
        state_out = model.state()
        prescribed_velocity = np.array((0.5, -0.25, 0.75), dtype=np.float32)
        velocities = state_in.particle_qd.numpy()
        velocities[pinned] = prescribed_velocity
        state_in.particle_qd.assign(velocities)
        dt = 0.01
        solver.step(state_in, state_out, None, None, dt)

        np.testing.assert_allclose(
            state_out.particle_q.numpy()[pinned],
            state_in.particle_q.numpy()[pinned] + dt * prescribed_velocity,
        )
        np.testing.assert_allclose(
            state_out.particle_qd.numpy()[pinned],
            np.broadcast_to(prescribed_velocity, (np.count_nonzero(pinned), 3)),
        )

    def test_step_mixed_cloth_and_tetrahedron(self):
        """Advance cloth and tetrahedral elasticity in one packed solve."""
        model = _build_mixed_deformable_model(device=self.device)
        validate_deformable_cloth_model(model)
        system = DeformableClothSystem(model)
        self.assertGreater(system.triangle_count, 0)
        self.assertGreater(system.tetrahedron_count, 0)
        solver = newton.solvers.SolverKamino(
            model,
            config=newton.solvers.SolverKamino.Config(dynamics_solver="lox"),
        )
        state_in = model.state()
        state_out = model.state()

        solver.step(state_in, state_out, None, None, 0.01)

        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
        self.assertTrue(np.all(state_out.particle_qd.numpy()[:, 2] < 0.0))

    def test_validate_tetrahedron_data_and_coloring(self):
        """Validate tet data and replace a coloring that misses tet couplings."""
        model = _build_tet_model(device=self.device)
        model.particle_colors = wp.zeros(model.particle_count, dtype=wp.int32, device=self.device)
        system = DeformableClothSystem(model)
        self.assertEqual(system.coloring_source, "topology")
        packed_color = system.topology.packed_color.numpy()
        for vertices in system.topology.tetrahedron_indices.numpy():
            self.assertEqual(np.unique(packed_color[vertices]).size, 4)

        repeated_model = _build_tet_model(device=self.device)
        repeated_indices = repeated_model.tet_indices.numpy()
        repeated_indices[0, 3] = repeated_indices[0, 2]
        repeated_model.tet_indices.assign(repeated_indices)
        with self.assertRaisesRegex(ValueError, "four distinct particles"):
            validate_deformable_cloth_model(repeated_model)

        singular_model = _build_tet_model(device=self.device)
        singular_poses = singular_model.tet_poses.numpy()
        singular_poses[0, 2] = 0.0
        singular_model.tet_poses.assign(singular_poses)
        with self.assertRaisesRegex(ValueError, "finite positive determinants"):
            validate_deformable_cloth_model(singular_model)

        negative_material_model = _build_tet_model(device=self.device)
        negative_materials = negative_material_model.tet_materials.numpy()
        negative_materials[0, 1] = -1.0
        negative_material_model.tet_materials.assign(negative_materials)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            validate_deformable_cloth_model(negative_material_model)

        large_rest_tet_model = _build_tet_model(device=self.device)
        large_rest_tet_poses = large_rest_tet_model.tet_poses.numpy()
        large_rest_tet_poses *= 1.0e-5
        large_rest_tet_model.tet_poses.assign(large_rest_tet_poses)
        validate_deformable_cloth_model(large_rest_tet_model)

    def test_replace_invalid_model_coloring_with_topology_coloring(self):
        """Replace a coloring that misses cloth couplings with a valid fallback."""
        model = _build_grid_model(
            device=self.device,
            dim_x=8,
            dim_y=4,
            color=True,
        )
        model.particle_colors.zero_()
        system = DeformableClothSystem(model)

        self.assertEqual(system.coloring_source, "topology")
        self.assertGreater(system.color_count, 1)
        self.assertLessEqual(system.preconditioner.level_count, system.color_count)
        packed_color = system.topology.packed_color.numpy()
        for elements in (system.topology.triangle_indices, system.topology.bending_indices):
            for vertices in elements.numpy():
                self.assertEqual(np.unique(packed_color[vertices]).size, np.unique(vertices).size)

    def test_assemble_triangle_against_dense_reference(self):
        """Match membrane matrix and right-hand side against a dense reference."""
        model = _build_triangle_model(device=self.device, tri_ke=120.0, tri_ka=70.0, tri_kd=1.5)
        model.set_gravity((0.0, -9.81, 0.0))
        state = model.state()
        state.particle_qd.assign(
            np.array(
                (
                    (0.1, -0.2, 0.05),
                    (-0.05, 0.15, 0.1),
                    (0.2, 0.05, -0.1),
                ),
                dtype=np.float32,
            )
        )
        state.particle_f.assign(
            np.array(
                (
                    (1.0, 0.0, 0.0),
                    (0.0, 2.0, 0.0),
                    (0.0, 0.0, 3.0),
                ),
                dtype=np.float32,
            )
        )
        model.tri_activations.fill_(0.08)

        dt = 0.02
        system = DeformableClothSystem(model)
        system.assemble(state, _world_dt(model, dt, self.device))
        reference_matrix, reference_rhs = _reference_triangle_system(model, state, dt)

        np.testing.assert_allclose(_bsr_to_dense(system.smooth_matrix), reference_matrix, rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(system.smooth_rhs.numpy().reshape(-1), reference_rhs, rtol=3.0e-5, atol=3.0e-5)
        np.testing.assert_allclose(reference_matrix, reference_matrix.T, rtol=1.0e-10, atol=1.0e-10)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(reference_matrix).min()), -1.0e-8)

    def test_reassemble_triangle_preserves_step_start_inertia(self):
        """Reassemble around a candidate velocity without shifting inertial data."""
        model = _build_triangle_model(device=self.device, tri_ke=120.0, tri_ka=70.0, tri_kd=1.5)
        model.set_gravity((0.0, -9.81, 0.0))
        state = model.state()
        state.particle_qd.assign(
            np.array(
                (
                    (0.1, -0.2, 0.05),
                    (-0.05, 0.15, 0.1),
                    (0.2, 0.05, -0.1),
                ),
                dtype=np.float32,
            )
        )
        candidate_velocity = np.array(
            (
                (0.4, -0.1, 0.2),
                (-0.2, 0.35, -0.15),
                (0.3, -0.25, 0.05),
            ),
            dtype=np.float32,
        )
        dt = 0.02
        system = DeformableClothSystem(model)
        system.assemble(state, _world_dt(model, dt, self.device))
        packed_velocity = candidate_velocity[system.topology.packed_to_newton.numpy()]
        system.reassemble(
            wp.array(packed_velocity, dtype=wp.vec3, device=self.device),
            _world_dt(model, dt, self.device),
        )

        reference_matrix, reference_rhs = _reference_triangle_system(
            model,
            state,
            dt,
            linearization_velocity=candidate_velocity,
        )
        np.testing.assert_allclose(_bsr_to_dense(system.smooth_matrix), reference_matrix, rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(system.smooth_rhs.numpy().reshape(-1), reference_rhs, rtol=3.0e-5, atol=3.0e-5)
        self.assertEqual(int(system.assembly_count.numpy()[0]), 2)
        self.assertEqual(int(system.preconditioner.factorization_count.numpy()[0]), 2)

    def test_membrane_force_matches_energy_gradient(self):
        """Match the membrane force to a finite-difference energy gradient."""
        model = _build_triangle_model(device=self.device, tri_ke=90.0, tri_ka=60.0, tri_kd=0.0)
        model.set_gravity((0.0, 0.0, 0.0))
        model.tri_activations.fill_(0.05)
        state = model.state()
        positions = state.particle_q.numpy()
        positions[1] += np.array((0.15, 0.03, 0.1), dtype=np.float32)
        positions[2] += np.array((-0.04, -0.08, -0.06), dtype=np.float32)
        state.particle_q.assign(positions)

        system = DeformableClothSystem(model)
        system.assemble(state, _world_dt(model, 0.01, self.device))
        expected_force = _finite_difference_force(lambda q: _membrane_energy(model, q), positions)

        np.testing.assert_allclose(system.smooth_force.numpy(), expected_force, rtol=2.0e-3, atol=2.0e-2)

    def test_update_membrane_proximal_rhs_without_changing_matrix(self):
        """Update the nonlinear membrane RHS while retaining the frozen matrix."""
        model = _build_triangle_model(device=self.device, tri_ke=90.0, tri_ka=60.0, tri_kd=0.0)
        model.set_gravity((0.0, 0.0, 0.0))
        state = model.state()
        positions = state.particle_q.numpy()
        positions[1] += np.array((0.25, 0.05, 0.15), dtype=np.float32)
        positions[2] += np.array((-0.08, -0.12, -0.1), dtype=np.float32)
        state.particle_q.assign(positions)
        velocities = state.particle_qd.numpy()
        velocities[1] = np.array((1.5, -0.4, 0.8), dtype=np.float32)
        velocities[2] = np.array((-0.6, 0.9, -0.5), dtype=np.float32)
        state.particle_qd.assign(velocities)

        dt = 0.03
        system = DeformableClothSystem(model, proximal_iterations=6)
        system.assemble(state, _world_dt(model, dt, self.device))
        matrix_before = system.system_matrix.values.numpy().copy()
        np.testing.assert_array_equal(system.nonlinear_rhs.numpy(), 0.0)

        candidate_velocity = system.smooth_velocity.numpy()
        candidate_velocity[1] += np.array((2.0, -1.0, 0.5), dtype=np.float32)
        candidate_velocity[2] += np.array((-1.0, 1.5, -0.75), dtype=np.float32)
        system.smooth_velocity.assign(candidate_velocity)
        system.update_proximal(_world_dt(model, dt, self.device))

        np.testing.assert_array_equal(system.system_matrix.values.numpy(), matrix_before)
        self.assertGreater(np.linalg.norm(system.nonlinear_rhs.numpy()), 0.0)
        self.assertEqual(int(system.proximal_failed.numpy()[0]), 0)
        self.assertGreater(float(system.proximal_position_residual.numpy()[0]), 0.0)

        proximal = system.membrane_proximal
        coordinate = proximal.proximal_coordinate.numpy()[0].T
        multiplier = proximal.multiplier.numpy()[0].T
        gradient = _membrane_local_gradient(model, coordinate)
        np.testing.assert_allclose(multiplier, gradient, rtol=2.0e-3, atol=5.0e-3)

    def test_update_tetrahedron_proximal_rhs_without_changing_matrix(self):
        """Update the nonlinear tetrahedron RHS while retaining the frozen matrix."""
        model = _build_tet_model(device=self.device, k_mu=120.0, k_lambda=75.0, k_damp=0.0)
        state = model.state()
        positions = state.particle_q.numpy()
        positions[1] += np.array((0.2, 0.05, -0.08), dtype=np.float32)
        positions[2] += np.array((-0.1, 0.12, 0.06), dtype=np.float32)
        positions[3] += np.array((0.07, -0.09, 0.15), dtype=np.float32)
        state.particle_q.assign(positions)
        velocities = state.particle_qd.numpy()
        velocities[1] = np.array((1.2, -0.4, 0.6), dtype=np.float32)
        velocities[2] = np.array((-0.5, 0.8, -0.3), dtype=np.float32)
        velocities[3] = np.array((0.4, -0.7, 0.9), dtype=np.float32)
        state.particle_qd.assign(velocities)

        dt = 0.03
        system = DeformableClothSystem(model, proximal_iterations=6)
        system.assemble(state, _world_dt(model, dt, self.device))
        matrix_before = system.system_matrix.values.numpy().copy()
        np.testing.assert_array_equal(system.nonlinear_rhs.numpy(), 0.0)

        candidate_velocity = system.smooth_velocity.numpy()
        candidate_velocity[1] += np.array((1.5, -0.8, 0.4), dtype=np.float32)
        candidate_velocity[2] += np.array((-0.9, 1.2, -0.5), dtype=np.float32)
        candidate_velocity[3] += np.array((0.7, -0.6, 1.1), dtype=np.float32)
        system.smooth_velocity.assign(candidate_velocity)
        system.update_proximal(_world_dt(model, dt, self.device))

        np.testing.assert_array_equal(system.system_matrix.values.numpy(), matrix_before)
        self.assertGreater(np.linalg.norm(system.nonlinear_rhs.numpy()), 0.0)
        self.assertEqual(int(system.proximal_failed.numpy()[0]), 0)
        self.assertGreater(float(system.proximal_position_residual.numpy()[0]), 0.0)

        proximal = system.tetrahedron_proximal
        self.assertIsNotNone(proximal)
        self.assertIs(proximal.frozen_metric, system.tetrahedron_metric)
        coordinate = proximal.proximal_coordinate.numpy()[0].astype(np.float64)
        multiplier = proximal.multiplier.numpy()[0].astype(np.float64)
        gradient = _tetrahedron_local_gradient(model, coordinate)
        np.testing.assert_allclose(multiplier, gradient, rtol=3.0e-3, atol=5.0e-3)

    def test_regularize_volume_only_tetrahedron_proximal_locally(self):
        """Regularize a volume-only tet prox without changing its frozen global metric."""
        model = _build_tet_model(device=self.device, k_mu=0.0, k_lambda=75.0, k_damp=0.0)
        state = model.state()
        positions = state.particle_q.numpy()
        positions[1] += np.array((0.2, 0.05, -0.08), dtype=np.float32)
        positions[2] += np.array((-0.1, 0.12, 0.06), dtype=np.float32)
        positions[3] += np.array((0.07, -0.09, 0.15), dtype=np.float32)
        state.particle_q.assign(positions)

        dt = 0.03
        system = DeformableClothSystem(model, proximal_iterations=8)
        system.assemble(state, _world_dt(model, dt, self.device))
        matrix_before = system.system_matrix.values.numpy().copy()
        candidate_velocity = system.smooth_velocity.numpy()
        candidate_velocity[1] += np.array((1.5, -0.8, 0.4), dtype=np.float32)
        candidate_velocity[2] += np.array((-0.9, 1.2, -0.5), dtype=np.float32)
        candidate_velocity[3] += np.array((0.7, -0.6, 1.1), dtype=np.float32)
        system.smooth_velocity.assign(candidate_velocity)

        system.update_proximal(_world_dt(model, dt, self.device))

        np.testing.assert_array_equal(system.system_matrix.values.numpy(), matrix_before)
        self.assertEqual(int(system.proximal_failed.numpy()[0]), 0)
        self.assertTrue(np.all(np.isfinite(system.nonlinear_rhs.numpy())))
        self.assertGreater(np.linalg.norm(system.nonlinear_rhs.numpy()), 0.0)

    def test_accumulate_membrane_and_tetrahedron_proximal_rhs(self):
        """Accumulate membrane and tetrahedron corrections in one packed RHS."""
        model = _build_mixed_deformable_model(device=self.device)
        state = model.state()
        dt = 0.03
        system = DeformableClothSystem(model, proximal_iterations=6)
        system.assemble(state, _world_dt(model, dt, self.device))
        candidate_velocity = system.smooth_velocity.numpy()
        candidate_velocity += np.random.default_rng(4321).normal(0.0, 1.0, candidate_velocity.shape).astype(np.float32)
        system.smooth_velocity.assign(candidate_velocity)

        system.update_proximal(_world_dt(model, dt, self.device))

        self.assertIsNotNone(system.membrane_proximal)
        self.assertIsNotNone(system.tetrahedron_proximal)
        nonlinear_rhs = system.nonlinear_rhs.numpy()
        self.assertGreater(np.linalg.norm(nonlinear_rhs[:4]), 0.0)
        self.assertGreater(np.linalg.norm(nonlinear_rhs[4:]), 0.0)
        self.assertEqual(int(system.proximal_failed.numpy()[0]), 0)

    def test_regularize_area_only_membrane_proximal_locally(self):
        """Regularize an area-only prox without changing its frozen global metric."""
        model = _build_triangle_model(device=self.device, tri_ke=0.0, tri_ka=60.0, tri_kd=0.0)
        model.set_gravity((0.0, 0.0, 0.0))
        state = model.state()
        positions = state.particle_q.numpy()
        positions[1] += np.array((0.25, 0.05, 0.15), dtype=np.float32)
        positions[2] += np.array((-0.08, -0.12, -0.1), dtype=np.float32)
        state.particle_q.assign(positions)

        dt = 0.03
        system = DeformableClothSystem(model, proximal_iterations=8)
        system.assemble(state, _world_dt(model, dt, self.device))
        matrix_before = system.system_matrix.values.numpy().copy()
        candidate_velocity = system.smooth_velocity.numpy()
        candidate_velocity[1] += np.array((2.0, -1.0, 0.5), dtype=np.float32)
        candidate_velocity[2] += np.array((-1.0, 1.5, -0.75), dtype=np.float32)
        system.smooth_velocity.assign(candidate_velocity)

        system.update_proximal(_world_dt(model, dt, self.device))

        np.testing.assert_array_equal(system.system_matrix.values.numpy(), matrix_before)
        self.assertEqual(int(system.proximal_failed.numpy()[0]), 0)
        self.assertTrue(np.all(np.isfinite(system.nonlinear_rhs.numpy())))
        self.assertGreater(np.linalg.norm(system.nonlinear_rhs.numpy()), 0.0)

    def test_relax_membrane_proximal_state_and_rhs(self):
        """Blend the complete membrane proximal update with its previous state."""
        model = _build_triangle_model(device=self.device, tri_ke=90.0, tri_ka=60.0, tri_kd=0.0)
        model.set_gravity((0.0, 0.0, 0.0))
        state = model.state()
        positions = state.particle_q.numpy()
        positions[1] += np.array((0.25, 0.05, 0.15), dtype=np.float32)
        positions[2] += np.array((-0.08, -0.12, -0.1), dtype=np.float32)
        state.particle_q.assign(positions)
        dt = 0.03

        def update(relaxation: float) -> DeformableClothSystem:
            system = DeformableClothSystem(
                model,
                proximal_iterations=6,
                proximal_relaxation=relaxation,
            )
            system.assemble(state, _world_dt(model, dt, self.device))
            candidate_velocity = system.smooth_velocity.numpy()
            candidate_velocity[1] += np.array((2.0, -1.0, 0.5), dtype=np.float32)
            candidate_velocity[2] += np.array((-1.0, 1.5, -0.75), dtype=np.float32)
            system.smooth_velocity.assign(candidate_velocity)
            system.update_proximal(_world_dt(model, dt, self.device))
            return system

        full = update(1.0)
        half = update(0.5)
        full_proximal = full.membrane_proximal
        half_proximal = half.membrane_proximal
        frozen_coordinate = full_proximal.frozen_coordinate.numpy()
        frozen_multiplier = full_proximal.frozen_gradient.numpy()

        np.testing.assert_allclose(
            half_proximal.proximal_coordinate.numpy(),
            frozen_coordinate + 0.5 * (full_proximal.proximal_coordinate.numpy() - frozen_coordinate),
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            half_proximal.multiplier.numpy(),
            frozen_multiplier + 0.5 * (full_proximal.multiplier.numpy() - frozen_multiplier),
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            half.nonlinear_rhs.numpy(),
            0.5 * full.nonlinear_rhs.numpy(),
            rtol=3.0e-6,
            atol=3.0e-6,
        )

        disabled = update(0.0)
        self.assertIsNone(disabled.membrane_proximal)
        np.testing.assert_array_equal(disabled.nonlinear_rhs.numpy(), 0.0)

    def test_apply_triangle_drag_and_lift_explicitly(self):
        """Apply Newton triangle drag and lift without adding an aerodynamic tangent."""
        model = _build_triangle_model(device=self.device, tri_ke=0.0, tri_ka=0.0, tri_kd=0.0)
        model.set_gravity((0.0, 0.0, 0.0))
        materials = model.tri_materials.numpy()
        materials[0, 3] = 1.0
        materials[0, 4] = 2.0
        model.tri_materials.assign(materials)
        state = model.state()
        state.particle_qd.fill_(wp.vec3(0.0, 0.0, 2.0))

        system = DeformableClothSystem(model)
        system.assemble(state, _world_dt(model, 0.01, self.device))

        drag = 2.0
        lift = 2.0 * 0.5 * (0.5 * np.pi) * 4.0
        expected_force = np.tile((0.0, 0.0, -drag - lift), (3, 1))
        np.testing.assert_allclose(system.smooth_force.numpy(), expected_force, rtol=2.0e-6, atol=2.0e-6)

        mass_matrix = np.diag(np.repeat(model.particle_mass.numpy(), 3))
        np.testing.assert_allclose(_bsr_to_dense(system.smooth_matrix), mass_matrix, atol=1.0e-7)

    def test_bending_force_matches_energy_gradient(self):
        """Match the dihedral force and verify its Gauss-Newton tangent is PSD."""
        model = _build_grid_model(
            device=self.device,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=5.0,
            edge_kd=0.0,
        )
        state = model.state()
        positions = state.particle_q.numpy()
        positions[3, 2] = 0.25
        state.particle_q.assign(positions)

        dt = 0.01
        system = DeformableClothSystem(model)
        system.assemble(state, _world_dt(model, dt, self.device))
        expected_force = _finite_difference_force(lambda q: _bending_energy(model, q), positions)

        np.testing.assert_allclose(system.smooth_force.numpy(), expected_force, rtol=3.0e-3, atol=3.0e-3)
        dense = _bsr_to_dense(system.smooth_matrix)
        mass = np.repeat(model.particle_mass.numpy(), 3)
        tangent = dense - np.diag(mass)
        np.testing.assert_allclose(tangent, tangent.T, rtol=1.0e-6, atol=1.0e-6)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(tangent).min()), -2.0e-5)

    def test_pin_rows_and_columns(self):
        """Replace pinned particle rows by identity without dynamic coupling."""
        model = _build_grid_model(device=self.device, fix_left=True)
        state = model.state()
        system = DeformableClothSystem(model)
        system.assemble(state, _world_dt(model, 0.01, self.device))

        dense = _bsr_to_dense(system.smooth_matrix)
        rhs = system.smooth_rhs.numpy()
        flags = model.particle_flags.numpy()
        masses = model.particle_mass.numpy()
        pinned = np.flatnonzero(((flags & int(newton.ParticleFlags.ACTIVE)) == 0) | (masses <= 0.0))
        self.assertGreater(pinned.size, 0)
        for particle in pinned:
            block = slice(3 * particle, 3 * particle + 3)
            expected_row = np.zeros((3, dense.shape[1]))
            expected_row[:, block] = np.eye(3)
            np.testing.assert_allclose(dense[block, :], expected_row, atol=1.0e-7)
            np.testing.assert_allclose(dense[:, block], expected_row.T, atol=1.0e-7)
            np.testing.assert_allclose(rhs[particle], 0.0, atol=1.0e-7)

    def test_replace_masked_triplet_values_between_steps(self):
        """Replace duplicate triplet values without changing or accumulating topology."""
        model = _build_triangle_model(device=self.device, tri_ke=100.0, tri_ka=70.0, tri_kd=1.0)
        state = model.state()
        system = DeformableClothSystem(model)
        system.assemble(state, _world_dt(model, 0.01, self.device))
        first_matrix = _bsr_to_dense(system.smooth_matrix)

        model.tri_materials.zero_()
        system.assemble(state, _world_dt(model, 0.01, self.device))
        second_matrix = _bsr_to_dense(system.smooth_matrix)
        mass_matrix = np.diag(np.repeat(model.particle_mass.numpy(), 3))

        self.assertGreater(float(np.linalg.norm(first_matrix - mass_matrix)), 0.0)
        np.testing.assert_allclose(second_matrix, mass_matrix, atol=1.0e-7)

    def test_capture_frozen_assembly(self):
        """Capture and replay fixed-topology assembly on CUDA."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model = _build_grid_model(device=self.device)
        state = model.state()
        system = DeformableClothSystem(model)
        system.assemble(state, _world_dt(model, 0.01, self.device))

        with wp.ScopedCapture(device=self.device) as capture:
            system.assemble(state, _world_dt(model, 0.01, self.device))
        wp.capture_launch(capture.graph)
        self.assertTrue(np.all(np.isfinite(system.smooth_rhs.numpy())))

    def test_capture_tetrahedron_proximal_update(self):
        """Capture and replay a tetrahedron proximal RHS update on CUDA."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model = _build_tet_model(device=self.device)
        state = model.state()
        system = DeformableClothSystem(model, proximal_iterations=4)
        system.assemble(state, _world_dt(model, 0.01, self.device))
        system.update_proximal(_world_dt(model, 0.01, self.device))

        with wp.ScopedCapture(device=self.device) as capture:
            system.update_proximal(_world_dt(model, 0.01, self.device))
        wp.capture_launch(capture.graph)

        self.assertEqual(int(system.proximal_failed.numpy()[0]), 0)
        self.assertTrue(np.all(np.isfinite(system.nonlinear_rhs.numpy())))


if __name__ == "__main__":
    unittest.main(verbosity=2)
