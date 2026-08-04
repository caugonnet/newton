# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preallocated Jacobi preconditioning for the LOX cloth system."""

import numpy as np
import warp as wp
import warp.sparse as wps
from warp.optim.linear import LinearOperator

from .deformable_preconditioner import (
    DEFORMABLE_PRECONDITIONER_STATUS_FAILED,
    DEFORMABLE_PRECONDITIONER_STATUS_REGULARIZED,
    DEFORMABLE_PRECONDITIONER_STATUS_VALID,
)

__all__ = ["DeformableJacobi"]


@wp.kernel
def _begin_factorization(
    world_status: wp.array[wp.int32],
    factorization_count: wp.array[wp.int32],
):
    world = wp.tid()
    world_status[world] = DEFORMABLE_PRECONDITIONER_STATUS_VALID
    if world == 0:
        wp.atomic_add(factorization_count, 0, 1)


@wp.kernel
def _factorize_jacobi(
    diagonal_slots: wp.array[wp.int32],
    system_values: wp.array[wp.mat33],
    packed_world: wp.array[wp.int32],
    row_active: wp.array[wp.int32],
    regularization: float,
    inverse_diagonal: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    row = wp.tid()
    if row_active[row] == 0:
        inverse_diagonal[row] = 1.0
        return
    block = system_values[diagonal_slots[row]]
    diagonal = (block[0, 0] + block[1, 1] + block[2, 2]) / 3.0
    scale = wp.max(
        wp.max(wp.abs(block[0, 0]), wp.abs(block[1, 1])),
        wp.abs(block[2, 2]),
    )
    floor = wp.max(regularization * scale, 1.0e-12)
    status = wp.int32(DEFORMABLE_PRECONDITIONER_STATUS_VALID)
    if not wp.isfinite(diagonal):
        inverse_diagonal[row] = 1.0
        status = DEFORMABLE_PRECONDITIONER_STATUS_FAILED
    else:
        if diagonal < floor:
            diagonal = floor
            status = DEFORMABLE_PRECONDITIONER_STATUS_REGULARIZED
        inverse_diagonal[row] = 1.0 / diagonal
    if status != DEFORMABLE_PRECONDITIONER_STATUS_VALID:
        wp.atomic_max(world_status, packed_world[row], status)


@wp.kernel
def _apply_jacobi(
    inverse_diagonal: wp.array[wp.float32],
    right_hand_side: wp.array[wp.vec3],
    addend: wp.array[wp.vec3],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    row_active: wp.array[wp.int32],
    alpha: float,
    beta: float,
    result: wp.array[wp.vec3],
):
    row = wp.tid()
    world = packed_world[row]
    value = right_hand_side[row]
    if (
        row_active[row] != 0
        and world_active[world] != 0
        and world_status[world] != DEFORMABLE_PRECONDITIONER_STATUS_FAILED
    ):
        value *= inverse_diagonal[row]
    value *= alpha
    if beta != 0.0:
        value += beta * addend[row]
    result[row] = value


class DeformableJacobi:
    """Apply a preallocated isotropic diagonal preconditioner."""

    def __init__(
        self,
        system_matrix: wps.BsrMatrix,
        diagonal_slots: wp.array[wp.int32],
        packed_world: wp.array[wp.int32],
        world_active: wp.array[wp.int32],
        batch_offsets: wp.array[wp.int32],
        regularization: float = 1.0e-6,
        row_active: wp.array[wp.int32] | None = None,
    ):
        """Allocate fixed-topology Jacobi storage.

        Args:
            system_matrix: Symmetric 3-by-3 block system matrix.
            diagonal_slots: BSR value slot for every diagonal block.
            packed_world: World index for every packed block row.
            world_active: Mutable active flag for every world.
            batch_offsets: Scalar degree-of-freedom offsets for Warp batching.
            regularization: Relative positive diagonal floor.
            row_active: Optional nonzero flag for rows assigned to CR.
        """
        if system_matrix.block_shape != (3, 3) or system_matrix.nrow != system_matrix.ncol:
            raise ValueError("LOX cloth Jacobi requires a square 3-by-3 BSR matrix.")
        if diagonal_slots.shape != (system_matrix.nrow,) or diagonal_slots.dtype != wp.int32:
            raise ValueError("LOX cloth Jacobi requires one int32 diagonal slot per block row.")
        if not np.isfinite(regularization) or regularization <= 0.0:
            raise ValueError(f"LOX cloth Jacobi regularization must be finite and positive, got {regularization}.")

        self.system_matrix = system_matrix
        self.device = system_matrix.device
        self.row_count = int(system_matrix.nrow)
        self.diagonal_slots = diagonal_slots
        self.packed_world = packed_world
        self.world_active = world_active
        self.regularization = float(regularization)
        if row_active is None:
            row_active = wp.ones(self.row_count, dtype=wp.int32, device=self.device)
        elif row_active.shape != (self.row_count,) or row_active.dtype != wp.int32:
            raise ValueError("LOX cloth Jacobi requires one int32 active flag per row.")
        self.row_active = row_active
        self.world_status = wp.zeros(world_active.shape[0], dtype=wp.int32, device=self.device)
        self.factorization_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.inverse_diagonal = wp.empty(self.row_count, dtype=wp.float32, device=self.device)
        self.linear_operator = LinearOperator(
            shape=system_matrix.shape,
            dtype=system_matrix.dtype,
            device=self.device,
            matvec=self._matvec,
            batch_offsets=batch_offsets,
        )

    def factorize(self) -> None:
        """Extract the inverse diagonal of the current system matrix."""
        wp.launch(
            _begin_factorization,
            dim=self.world_status.shape[0],
            inputs=[self.world_status, self.factorization_count],
            device=self.device,
        )
        wp.launch(
            _factorize_jacobi,
            dim=self.row_count,
            inputs=[
                self.diagonal_slots,
                self.system_matrix.values,
                self.packed_world,
                self.row_active,
                self.regularization,
            ],
            outputs=[self.inverse_diagonal, self.world_status],
            device=self.device,
        )

    def _matvec(
        self,
        x: wp.array[wp.vec3],
        y: wp.array[wp.vec3],
        z: wp.array[wp.vec3],
        alpha: float,
        beta: float,
    ) -> None:
        wp.launch(
            _apply_jacobi,
            dim=self.row_count,
            inputs=[
                self.inverse_diagonal,
                x,
                y,
                self.packed_world,
                self.world_active,
                self.world_status,
                self.row_active,
                alpha,
                beta,
            ],
            outputs=[z],
            device=self.device,
        )
