# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shallow two-level preconditioning for LOX deformables."""

from __future__ import annotations

from collections import deque

import numpy as np
import warp as wp
import warp.sparse as wps
from warp.optim.linear import LinearOperator

from ...linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from .deformable_jacobi import DeformableJacobi
from .deformable_preconditioner import DEFORMABLE_PRECONDITIONER_STATUS_FAILED
from .linear import HybridLLTBlockedSolver

__all__ = ["DeformableTwoLevel"]

_AGGREGATE_PARTICLE_COUNT = 64
"""Target particle count for deterministic graph aggregates."""

_AGGREGATE_COUNT_LIMIT = 128
"""Largest coarse vertex count allocated for one iterative component."""

_AGGREGATE_BLOCK_DIM = 64
"""Thread count used to restrict one graph aggregate."""


def _aggregate_component(
    component_rows: np.ndarray,
    adjacency: list[list[int]],
    target_size: int,
) -> list[list[int]]:
    """Partition a deterministic breadth-first traversal into aggregates."""
    unvisited = {int(row) for row in component_rows}
    traversal: list[int] = []
    while unvisited:
        seed = min(unvisited)
        unvisited.remove(seed)
        queue = deque([seed])
        while queue:
            row = queue.popleft()
            traversal.append(row)
            for neighbor in adjacency[row]:
                if neighbor not in unvisited:
                    continue
                unvisited.remove(neighbor)
                queue.append(neighbor)
    return [traversal[start : start + target_size] for start in range(0, len(traversal), target_size)]


@wp.kernel
def _assemble_coarse_matrix(
    fine_values: wp.array[wp.mat33],
    coarse_slot_base: wp.array[wp.int32],
    coarse_slot_dimension: wp.array[wp.int32],
    coarse_matrix: wp.array[wp.float32],
):
    slot = wp.tid()
    base = coarse_slot_base[slot]
    if base < 0:
        return
    dimension = coarse_slot_dimension[slot]
    value = fine_values[slot]
    for row in range(3):
        for column in range(3):
            wp.atomic_add(coarse_matrix, base + row * dimension + column, value[row, column])


@wp.kernel
def _restrict_residual_by_aggregate(
    right_hand_side: wp.array[wp.vec3],
    aggregate_offsets: wp.array[wp.int32],
    aggregate_particles: wp.array[wp.int32],
    aggregate_coarse_offset: wp.array[wp.int32],
    aggregate_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    coarse_right_hand_side: wp.array[wp.float32],
):
    thread = wp.tid()
    aggregate = thread // _AGGREGATE_BLOCK_DIM
    lane = thread - aggregate * _AGGREGATE_BLOCK_DIM
    value = wp.vec3(0.0)
    world = aggregate_world[aggregate]
    if world_active[world] != 0 and world_status[world] != DEFORMABLE_PRECONDITIONER_STATUS_FAILED:
        slot = wp.int32(aggregate_offsets[aggregate] + lane)
        slot_end = aggregate_offsets[aggregate + 1]
        while slot < slot_end:
            value += right_hand_side[aggregate_particles[slot]]
            slot += _AGGREGATE_BLOCK_DIM
    value = wp.tile_reduce(wp.add, wp.tile(value, preserve_type=True))[0]
    if lane == 0:
        coarse_offset = aggregate_coarse_offset[aggregate]
        for axis in range(3):
            coarse_right_hand_side[coarse_offset + axis] = value[axis]


@wp.kernel
def _apply_two_level(
    inverse_diagonal: wp.array[wp.mat33],
    coarse_solution: wp.array[wp.float32],
    particle_coarse_offset: wp.array[wp.int32],
    right_hand_side: wp.array[wp.vec3],
    addend: wp.array[wp.vec3],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    alpha: float,
    beta: float,
    result: wp.array[wp.vec3],
):
    particle = wp.tid()
    coarse_offset = particle_coarse_offset[particle]
    world = packed_world[particle]
    value = right_hand_side[particle]
    if (
        coarse_offset >= 0
        and world_active[world] != 0
        and world_status[world] != DEFORMABLE_PRECONDITIONER_STATUS_FAILED
    ):
        value = inverse_diagonal[particle] * value
        value += wp.vec3(
            coarse_solution[coarse_offset],
            coarse_solution[coarse_offset + 1],
            coarse_solution[coarse_offset + 2],
        )
    value *= alpha
    if beta != 0.0:
        value += beta * addend[particle]
    result[particle] = value


class DeformableTwoLevel:
    """Apply ``D^-1 + P (P^T A P)^-1 P^T`` using graph aggregates."""

    def __init__(
        self,
        system_matrix: wps.BsrMatrix,
        diagonal_slots: wp.array[wp.int32],
        packed_component: wp.array[wp.int32],
        packed_world: wp.array[wp.int32],
        world_active: wp.array[wp.int32],
        batch_offsets: wp.array[wp.int32],
        regularization: float = 1.0e-6,
        row_active: wp.array[wp.int32] | None = None,
    ):
        """Cache deterministic graph aggregates and dense coarse storage.

        Args:
            system_matrix: Symmetric 3-by-3 block system matrix.
            diagonal_slots: BSR value slot for every diagonal block.
            packed_component: Structural component for every packed particle.
            packed_world: World index for every packed particle.
            world_active: Mutable active flag for every world.
            batch_offsets: Scalar degree-of-freedom offsets for Warp batching.
            regularization: Relative positive fine-diagonal floor.
            row_active: Optional nonzero flag for rows assigned to CR.
        """
        if system_matrix.block_shape != (3, 3) or system_matrix.nrow != system_matrix.ncol:
            raise ValueError("LOX deformable two-level preconditioning requires a square 3-by-3 BSR matrix.")
        row_count = int(system_matrix.nrow)
        if packed_component.shape != (row_count,) or packed_component.dtype != wp.int32:
            raise ValueError("LOX deformable two-level preconditioning requires one int32 component per row.")
        if row_active is None:
            row_active = wp.ones(row_count, dtype=wp.int32, device=system_matrix.device)
        elif row_active.shape != (row_count,) or row_active.dtype != wp.int32:
            raise ValueError("LOX deformable two-level preconditioning requires one int32 active flag per row.")

        self.system_matrix = system_matrix
        self.device = system_matrix.device
        self.row_count = row_count
        self.packed_world = packed_world
        self.world_active = world_active
        self.row_active = row_active
        self.fine_preconditioner = DeformableJacobi(
            system_matrix,
            diagonal_slots,
            packed_world,
            world_active,
            batch_offsets,
            regularization=regularization,
            row_active=row_active,
            block_diagonal=True,
        )
        self.world_status = self.fine_preconditioner.world_status
        self.factorization_count = self.fine_preconditioner.factorization_count

        offsets = system_matrix.offsets.numpy().astype(np.int32, copy=False)
        columns = system_matrix.columns.numpy().astype(np.int32, copy=False)
        slot_rows = np.repeat(np.arange(row_count, dtype=np.int32), np.diff(offsets))
        component = packed_component.numpy().astype(np.int32, copy=False)
        active = row_active.numpy().astype(bool, copy=False)
        if np.any(component[slot_rows] != component[columns]):
            raise ValueError("LOX deformable two-level matrix entries cannot span structural components.")

        adjacency: list[list[int]] = [[] for _ in range(row_count)]
        for row in range(row_count):
            adjacency[row] = sorted(
                int(column)
                for column in columns[int(offsets[row]) : int(offsets[row + 1])]
                if column != row and active[column]
            )

        particle_aggregate = np.full(row_count, -1, dtype=np.int32)
        iterative_components: list[int] = []
        aggregate_counts: list[int] = []
        aggregate_offsets = [0]
        aggregate_particles: list[int] = []
        next_aggregate = 0
        for component_index in range(int(np.max(component)) + 1):
            rows = np.flatnonzero((component == component_index) & active).astype(np.int32)
            if rows.size == 0:
                continue
            target_size = max(
                _AGGREGATE_PARTICLE_COUNT,
                (int(rows.size) + _AGGREGATE_COUNT_LIMIT - 1) // _AGGREGATE_COUNT_LIMIT,
            )
            aggregates = _aggregate_component(rows, adjacency, target_size)
            for local_aggregate, aggregate_rows in enumerate(aggregates):
                particle_aggregate[aggregate_rows] = next_aggregate + local_aggregate
                aggregate_particles.extend(aggregate_rows)
                aggregate_offsets.append(len(aggregate_particles))
            iterative_components.append(component_index)
            aggregate_counts.append(len(aggregates))
            next_aggregate += len(aggregates)

        if not aggregate_counts:
            raise ValueError("LOX deformable two-level preconditioning requires at least one iterative component.")

        self.aggregate_count = next_aggregate
        self.component_aggregate_counts = tuple(aggregate_counts)
        self.info = DenseSquareMultiLinearInfo()
        self.info.finalize(
            dimensions=[3 * count for count in aggregate_counts],
            dtype=wp.float32,
            itype=wp.int32,
            device=self.device,
        )
        matrix_offsets = np.asarray(self.info.mio.numpy(), dtype=np.int64)
        vector_offsets = np.asarray(self.info.vio.numpy(), dtype=np.int64)

        component_to_coarse_block = {
            component_index: coarse_block for coarse_block, component_index in enumerate(iterative_components)
        }
        aggregate_block = np.empty(self.aggregate_count, dtype=np.int32)
        aggregate_local = np.empty(self.aggregate_count, dtype=np.int32)
        first_aggregate = 0
        for coarse_block, count in enumerate(aggregate_counts):
            aggregate_block[first_aggregate : first_aggregate + count] = coarse_block
            aggregate_local[first_aggregate : first_aggregate + count] = np.arange(count, dtype=np.int32)
            first_aggregate += count

        particle_coarse_offset = np.full(row_count, -1, dtype=np.int32)
        for particle in np.flatnonzero(active):
            aggregate = int(particle_aggregate[particle])
            block = component_to_coarse_block[int(component[particle])]
            particle_coarse_offset[particle] = int(vector_offsets[block]) + 3 * int(aggregate_local[aggregate])

        aggregate_offsets_np = np.asarray(aggregate_offsets, dtype=np.int32)
        aggregate_particles_np = np.asarray(aggregate_particles, dtype=np.int32)
        aggregate_first_particles = aggregate_particles_np[aggregate_offsets_np[:-1]]
        aggregate_coarse_offset = particle_coarse_offset[aggregate_first_particles]
        aggregate_world = packed_world.numpy().astype(np.int32, copy=False)[aggregate_first_particles]

        coarse_slot_base = np.full(columns.shape[0], -1, dtype=np.int32)
        coarse_slot_dimension = np.zeros(columns.shape[0], dtype=np.int32)
        for slot, (row, column) in enumerate(zip(slot_rows, columns, strict=True)):
            row_aggregate = int(particle_aggregate[row])
            column_aggregate = int(particle_aggregate[column])
            if row_aggregate < 0 or column_aggregate < 0:
                continue
            block = int(aggregate_block[row_aggregate])
            if aggregate_block[column_aggregate] != block:
                raise ValueError("LOX deformable two-level aggregates cannot span coarse blocks.")
            dimension = 3 * aggregate_counts[block]
            coarse_slot_base[slot] = (
                int(matrix_offsets[block])
                + 3 * int(aggregate_local[row_aggregate]) * dimension
                + 3 * int(aggregate_local[column_aggregate])
            )
            coarse_slot_dimension[slot] = dimension

        self.particle_aggregate = wp.array(particle_aggregate, dtype=wp.int32, device=self.device)
        self.particle_coarse_offset = wp.array(particle_coarse_offset, dtype=wp.int32, device=self.device)
        self.aggregate_offsets = wp.array(aggregate_offsets_np, dtype=wp.int32, device=self.device)
        self.aggregate_particles = wp.array(aggregate_particles_np, dtype=wp.int32, device=self.device)
        self.aggregate_coarse_offset = wp.array(aggregate_coarse_offset, dtype=wp.int32, device=self.device)
        self.aggregate_world = wp.array(aggregate_world, dtype=wp.int32, device=self.device)
        self.coarse_slot_base = wp.array(coarse_slot_base, dtype=wp.int32, device=self.device)
        self.coarse_slot_dimension = wp.array(coarse_slot_dimension, dtype=wp.int32, device=self.device)
        self.coarse_matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)
        self.coarse_right_hand_side = wp.zeros(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self.coarse_solution = wp.zeros(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self.coarse_operator = DenseLinearOperatorData(info=self.info, mat=self.coarse_matrix)
        self.coarse_solver = HybridLLTBlockedSolver(
            operator=self.coarse_operator,
            factorize_block_size=64,
            solve_block_size=32,
            solve_block_dim=256,
            dtype=wp.float32,
            device=self.device,
        )
        self.linear_operator = LinearOperator(
            shape=system_matrix.shape,
            dtype=system_matrix.dtype,
            device=self.device,
            matvec=self._matvec,
            batch_offsets=batch_offsets,
        )

    def factorize(self) -> None:
        """Factor the fine diagonal and exact Galerkin coarse matrix."""
        self.fine_preconditioner.factorize()
        self.coarse_matrix.zero_()
        wp.launch(
            _assemble_coarse_matrix,
            dim=self.system_matrix.values.shape[0],
            inputs=[
                self.system_matrix.values,
                self.coarse_slot_base,
                self.coarse_slot_dimension,
            ],
            outputs=[self.coarse_matrix],
            device=self.device,
        )
        self.coarse_solver.compute(self.coarse_matrix)

    def _matvec(
        self,
        x: wp.array[wp.vec3],
        y: wp.array[wp.vec3],
        z: wp.array[wp.vec3],
        alpha: float,
        beta: float,
    ) -> None:
        wp.launch(
            _restrict_residual_by_aggregate,
            dim=self.aggregate_count * _AGGREGATE_BLOCK_DIM,
            inputs=[
                x,
                self.aggregate_offsets,
                self.aggregate_particles,
                self.aggregate_coarse_offset,
                self.aggregate_world,
                self.world_active,
                self.world_status,
            ],
            outputs=[self.coarse_right_hand_side],
            block_dim=_AGGREGATE_BLOCK_DIM,
            device=self.device,
        )
        self.coarse_solver.solve(self.coarse_right_hand_side, self.coarse_solution)
        wp.launch(
            _apply_two_level,
            dim=self.row_count,
            inputs=[
                self.fine_preconditioner.inverse_diagonal,
                self.coarse_solution,
                self.particle_coarse_offset,
                x,
                y,
                self.packed_world,
                self.world_active,
                self.world_status,
                alpha,
                beta,
            ],
            outputs=[z],
            device=self.device,
        )
