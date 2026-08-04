# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LOX-specific hybrid dense Cholesky solver."""

from typing import Any

import warp as wp

from ......core.types import override
from ...linalg import DenseLinearOperatorData, LLTBlockedSolver, factorize


@wp.kernel(enable_backward=False)
def _gather_block_dimensions(
    block_indices: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    gathered_dimensions: wp.array[wp.int32],
):
    block = wp.tid()
    gathered_dimensions[block] = dimensions[block_indices[block]]


class HybridLLTBlockedSolver(LLTBlockedSolver):
    """Use sequential kernels for small blocks and tiled kernels otherwise."""

    def __init__(self, *args: Any, sequential_block_size: int = 6, **kwargs: Any):
        self._sequential_block_size = sequential_block_size
        self._sequential_num_blocks = 0
        self._sequential_block_indices: wp.array[wp.int32] | None = None
        self._sequential_dim: wp.array[wp.int32] | None = None
        self._sequential_mio: wp.array[wp.int32] | None = None
        self._sequential_vio: wp.array[wp.int32] | None = None
        self._tiled_num_blocks = 0
        self._tiled_block_indices: wp.array[wp.int32] | None = None
        self._tiled_dim: wp.array[wp.int32] | None = None
        self._tiled_mio: wp.array[wp.int32] | None = None
        self._tiled_vio: wp.array[wp.int32] | None = None
        super().__init__(*args, **kwargs)

    @property
    def block_size(self) -> int:
        """Return the tile size used by the solve kernels."""
        return self._solve_block_size

    @override
    def _allocate_impl(self, A: DenseLinearOperatorData, **kwargs: Any) -> None:
        super()._allocate_impl(A, **kwargs)

        dimensions = self._operator.info.dimensions
        if dimensions is None:
            raise ValueError("LOX LLT factorization requires host-side block dimensions.")

        matrix_offsets = self._operator.info.mio.numpy().astype(int).tolist()
        vector_offsets = self._operator.info.vio.numpy().astype(int).tolist()
        sequential_indices = [
            block for block, dimension in enumerate(dimensions) if dimension <= self._sequential_block_size
        ]
        tiled_indices = [block for block, dimension in enumerate(dimensions) if dimension > self._sequential_block_size]

        with wp.ScopedDevice(self._device):
            self._sequential_num_blocks = len(sequential_indices)
            self._sequential_block_indices = wp.array(sequential_indices, dtype=wp.int32)
            self._sequential_dim = wp.empty(self._sequential_num_blocks, dtype=wp.int32)
            self._sequential_mio = wp.array([matrix_offsets[block] for block in sequential_indices], dtype=wp.int32)
            self._sequential_vio = wp.array([vector_offsets[block] for block in sequential_indices], dtype=wp.int32)

            self._tiled_num_blocks = len(tiled_indices)
            self._tiled_block_indices = wp.array(tiled_indices, dtype=wp.int32)
            self._tiled_dim = wp.empty(self._tiled_num_blocks, dtype=wp.int32)
            self._tiled_mio = wp.array([matrix_offsets[block] for block in tiled_indices], dtype=wp.int32)
            self._tiled_vio = wp.array([vector_offsets[block] for block in tiled_indices], dtype=wp.int32)

    @override
    def _factorize_impl(self, A: wp.array) -> None:
        self._gather_partition_dimensions()
        if self._sequential_num_blocks > 0:
            factorize.llt_sequential_factorize(
                num_blocks=self._sequential_num_blocks,
                dim=self._sequential_dim,
                mio=self._sequential_mio,
                A=A,
                L=self._L,
            )
        if self._tiled_num_blocks > 0:
            factorize.llt_blocked_factorize(
                kernel=self._factorize_kernel,
                num_blocks=self._tiled_num_blocks,
                block_dim=self._factorize_block_dim,
                dim=self._tiled_dim,
                mio=self._tiled_mio,
                A=A,
                L=self._L,
            )

    @override
    def _solve_impl(self, b: wp.array, x: wp.array) -> None:
        if self._sequential_num_blocks > 0:
            factorize.llt_sequential_solve(
                num_blocks=self._sequential_num_blocks,
                dim=self._sequential_dim,
                mio=self._sequential_mio,
                vio=self._sequential_vio,
                L=self._L,
                b=b,
                y=self._y,
                x=x,
            )
        if self._tiled_num_blocks > 0:
            factorize.llt_blocked_solve(
                kernel=self._solve_kernel,
                num_blocks=self._tiled_num_blocks,
                block_dim=self._solve_block_dim,
                dim=self._tiled_dim,
                mio=self._tiled_mio,
                vio=self._tiled_vio,
                L=self._L,
                b=b,
                y=self._y,
                x=x,
            )

    @override
    def _solve_inplace_impl(self, x: wp.array) -> None:
        if self._sequential_num_blocks > 0:
            factorize.llt_sequential_solve_inplace(
                num_blocks=self._sequential_num_blocks,
                dim=self._sequential_dim,
                mio=self._sequential_mio,
                vio=self._sequential_vio,
                L=self._L,
                x=x,
            )
        if self._tiled_num_blocks > 0:
            factorize.llt_blocked_solve_inplace(
                kernel=self._solve_inplace_kernel,
                num_blocks=self._tiled_num_blocks,
                block_dim=self._solve_block_dim,
                dim=self._tiled_dim,
                mio=self._tiled_mio,
                vio=self._tiled_vio,
                L=self._L,
                y=self._y,
                x=x,
            )

    def _gather_partition_dimensions(self) -> None:
        if self._sequential_num_blocks > 0:
            wp.launch(
                _gather_block_dimensions,
                dim=self._sequential_num_blocks,
                inputs=[self._sequential_block_indices, self._operator.info.dim],
                outputs=[self._sequential_dim],
                device=self._device,
            )
        if self._tiled_num_blocks > 0:
            wp.launch(
                _gather_block_dimensions,
                dim=self._tiled_num_blocks,
                inputs=[self._tiled_block_indices, self._operator.info.dim],
                outputs=[self._tiled_dim],
                device=self._device,
            )
