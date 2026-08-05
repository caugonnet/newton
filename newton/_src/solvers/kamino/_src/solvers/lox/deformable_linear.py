# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Allocation-free fixed-count iterative linear solves for LOX cloth."""

from collections.abc import Callable

import warp as wp
from warp._src.optim import linear as _wpl_internal
from warp.optim import linear as wpl

__all__ = ["DeformableCRState"]

_TILED_DOT_DEFAULT_TILE_SIZE = 512
_TILED_DOT_MIN_TREE_TILE_SIZE = 128
_TILED_DOT_MAX_ITEMS_PER_LANE = 128


def _select_tiled_dot_tile_size(scalar_length: int, batch_count: int) -> int:
    """Select a bounded-tree tile without dropping below 128 threads."""
    if batch_count != 1:
        return _TILED_DOT_DEFAULT_TILE_SIZE
    tile_size = _TILED_DOT_DEFAULT_TILE_SIZE
    while tile_size >= _TILED_DOT_MIN_TREE_TILE_SIZE:
        if scalar_length > _TILED_DOT_MAX_ITEMS_PER_LANE * tile_size:
            return tile_size
        tile_size //= 2
    return _TILED_DOT_DEFAULT_TILE_SIZE


def _run_capturable_loop(
    do_cycle: Callable[[], None],
    r_norm_sq: wp.array[wp.float32],
    maxiter: int,
    atol_sq: wp.array[wp.float32],
    current_iteration: wp.array[wp.int32],
    *,
    cycle_size: int = 1,
):
    """Run a fixed number of preallocated solver cycles without device allocation."""
    for _iteration in range(0, maxiter, cycle_size):
        do_cycle()
    current_iteration.fill_(maxiter)
    return current_iteration, r_norm_sq, atol_sq


class DeformableCRState(wpl.CR):
    """Stateful fixed-count CR with preallocated loop-control storage.

    This local specialization works around Warp allocating its two-element
    iteration/condition array inside every stateful solver invocation.
    """

    def _allocate(self) -> None:
        super()._allocate()
        scalar_length = self._b.shape[0] * self._dofs_per_entry
        tile_size = _TILED_DOT_DEFAULT_TILE_SIZE
        if self._device.is_cuda and self._A.batch_offsets is not None:
            tile_size = _select_tiled_dot_tile_size(scalar_length, self._batch_count)
        if tile_size != _TILED_DOT_DEFAULT_TILE_SIZE:
            self._tiled_dot = _wpl_internal.TiledDot(
                max_length=scalar_length,
                scalar_type=self._scalar_type,
                tile_size=tile_size,
                device=self._device,
                max_column_count=2,
                batch_offsets=self._A.batch_offsets,
            )
        self._current_iteration = wp.empty(1, dtype=wp.int32, device=self._device)

    def _run(self, A, b, x, M):
        device = self._device
        batch_offsets = A.batch_offsets
        dofs_per_entry = self._dofs_per_entry
        tiled_dot = self._tiled_dot
        r_and_z_buf = self._r_and_z_buf
        y_and_Ap_buf = self._y_and_Ap_buf
        r_and_Az = self._r_and_Az
        p = self._p

        if M is None:
            r_and_z = self._r_and_z_repeated
            y_and_Ap = self._y_and_Ap_repeated
        else:
            r_and_z = r_and_z_buf
            y_and_Ap = y_and_Ap_buf

        r, z = r_and_z[0], r_and_z[1]
        r_copy, Az = r_and_Az[0], r_and_Az[1]
        y, Ap = y_and_Ap[0], y_and_Ap[1]

        r_norm_sq = tiled_dot.col(0)
        zAz_new = tiled_dot.col(1)
        zAz_old, atol_sq = self._residuals[0], self._residuals[1]

        _wpl_internal._initialize_absolute_tolerance(b, self._tol, self._atol, tiled_dot, atol_sq)
        A.matvec(x, b, r, alpha=-1.0, beta=1.0)

        y_and_Ap_buf.zero_()
        if M is not None:
            z.zero_()
            M.matvec(r, z, z, alpha=1.0, beta=0.0)

        def update_rr_zAz() -> None:
            A.matvec(z, Az, Az, alpha=1.0, beta=0.0)
            r_copy.assign(r)
            tiled_dot.compute(r_and_z, r_and_Az)

        update_rr_zAz()
        p.assign(z)
        Ap.assign(Az)

        def do_iteration() -> None:
            zAz_old.assign(zAz_new)

            if M is not None:
                M.matvec(Ap, y, y, alpha=1.0, beta=0.0)
            tiled_dot.compute(Ap, y, col_offset=1)
            y_Ap = tiled_dot.col(1)

            if M is None:
                wp.launch(
                    kernel=_wpl_internal._cg_kernel_1,
                    dim=x.shape[0],
                    device=device,
                    inputs=[atol_sq, r_norm_sq, zAz_old, y_Ap, x, r, p, Ap, batch_offsets, dofs_per_entry],
                )
            else:
                wp.launch(
                    kernel=_wpl_internal._cr_kernel_1,
                    dim=x.shape[0],
                    device=device,
                    inputs=[atol_sq, r_norm_sq, zAz_old, y_Ap, x, r, z, p, Ap, y, batch_offsets, dofs_per_entry],
                )

            update_rr_zAz()
            wp.launch(
                kernel=_wpl_internal._cr_kernel_2,
                dim=z.shape[0],
                device=device,
                inputs=[atol_sq, r_norm_sq, zAz_old, zAz_new, z, p, Az, Ap, batch_offsets, dofs_per_entry],
            )

        return _run_capturable_loop(
            do_iteration,
            r_norm_sq,
            self._maxiter,
            atol_sq,
            self._current_iteration,
        )
