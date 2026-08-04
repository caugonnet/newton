# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-space state for LOX splitting iterations."""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from ...core.types import mat66f, vec6f
from .projection import PROJECTION_STATUS_VALID

__all__ = ["SplittingState"]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _prepare_projection(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    body_split_enabled: wp.array[wp.int32],
    global_solution: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
    global_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if not world_active[world]:
        return
    global_twist_previous[body] = global_twist[body]
    global_twist[body] = global_solution[body]
    projected_twist_previous[body] = projected_twist[body]
    if body_split_enabled[body] != 0:
        projected_twist[body] = global_solution[body] - splitting_dual[body]
    else:
        projected_twist[body] = global_solution[body]
        splitting_dual[body] = vec6f(0.0)


@wp.kernel
def _replace_global_twist(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    body_split_enabled: wp.array[wp.int32],
    global_solution: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
):
    body = wp.tid()
    if world_active[body_world[body]]:
        global_twist[body] = global_solution[body]
        if body_split_enabled[body] == 0:
            projected_twist[body] = global_solution[body]
            splitting_dual[body] = vec6f(0.0)


@wp.kernel
def _store_dual_impulse(
    body_has_unilateral: wp.array[wp.int32],
    weight: wp.array[mat66f],
    splitting_dual: wp.array[vec6f],
    splitting_dual_impulse: wp.array[vec6f],
):
    body = wp.tid()
    if body_has_unilateral[body] != 0:
        splitting_dual_impulse[body] = weight[body] @ splitting_dual[body]
    else:
        splitting_dual[body] = vec6f(0.0)
        splitting_dual_impulse[body] = vec6f(0.0)


@wp.kernel
def _restore_dual_from_impulse(
    body_has_unilateral: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    splitting_dual_impulse: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
):
    body = wp.tid()
    if body_has_unilateral[body] != 0:
        splitting_dual[body] = inverse_weight[body] @ splitting_dual_impulse[body]
    else:
        splitting_dual_impulse[body] = vec6f(0.0)
        splitting_dual[body] = vec6f(0.0)


@wp.kernel
def _initialize_finish_iteration(
    projection_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    iteration_failed: wp.array[wp.int32],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_dual: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    iteration_count[world] += 1
    iteration_failed[world] = 0
    residual_change[world] = 0.0
    residual_split[world] = 0.0
    residual_dual[world] = 0.0
    if projection_status[world] != PROJECTION_STATUS_VALID:
        world_active[world] = False
        world_failed[world] = True


@wp.kernel
def _finish_iteration(
    time_step: wp.float32,
    position_tolerance: wp.float32,
    rotation_tolerance: wp.float32,
    velocity_tolerance: wp.float32,
    body_world: wp.array[wp.int32],
    global_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    splitting_dual: wp.array[vec6f],
    iteration_failed: wp.array[wp.int32],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_dual: wp.array[wp.float32],
):
    body = wp.tid()
    world = body_world[body]
    if not world_active[world]:
        return

    previous = global_twist_previous[body]
    current = global_twist[body]
    projected_previous = projected_twist_previous[body]
    projected = projected_twist[body]
    dual = splitting_dual[body]
    finite = wp.bool(True)
    for axis in range(6):
        finite = (
            finite
            and wp.isfinite(previous[axis])
            and wp.isfinite(current[axis])
            and wp.isfinite(projected_previous[axis])
            and wp.isfinite(projected[axis])
            and wp.isfinite(dual[axis])
        )
    if not finite:
        wp.atomic_max(iteration_failed, world, 1)
        return

    linear_change = wp.float32(0.0)
    angular_change = wp.float32(0.0)
    linear_split = wp.float32(0.0)
    angular_split = wp.float32(0.0)
    dual_residual = wp.float32(0.0)
    for axis in range(3):
        linear_change = wp.max(linear_change, wp.abs(current[axis] - previous[axis]))
        angular_change = wp.max(angular_change, wp.abs(current[axis + 3] - previous[axis + 3]))
        linear_split = wp.max(linear_split, wp.abs(current[axis] - projected[axis]))
        angular_split = wp.max(angular_split, wp.abs(current[axis + 3] - projected[axis + 3]))
    change = wp.max(
        time_step * linear_change / position_tolerance,
        time_step * angular_change / rotation_tolerance,
    )
    split = wp.max(
        time_step * linear_split / position_tolerance,
        time_step * angular_split / rotation_tolerance,
    )
    for axis in range(6):
        dual_residual = wp.max(dual_residual, wp.abs(current[axis] - projected_previous[axis]) / velocity_tolerance)
    wp.atomic_max(residual_change, world, change)
    wp.atomic_max(residual_split, world, split)
    wp.atomic_max(residual_dual, world, dual_residual)
    splitting_dual[body] += projected - current


@wp.kernel
def _finalize_finish_iteration(
    structural_residual: wp.array[wp.float32],
    projected_structural_residual: wp.array[wp.float32],
    iteration_failed: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_structural: wp.array[wp.float32],
    residual_structural_projected: wp.array[wp.float32],
    residual_dual: wp.array[wp.float32],
    residual_total: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    if iteration_failed[world] != 0:
        world_active[world] = False
        world_failed[world] = True
        return

    change = residual_change[world]
    split = residual_split[world]
    dual_residual = residual_dual[world]
    structural = structural_residual[world]
    total = change + split + structural + dual_residual
    residual_structural[world] = structural
    residual_structural_projected[world] = projected_structural_residual[world]
    residual_total[world] = total
    if total <= 1.0:
        world_active[world] = False
        world_converged[world] = True


@wp.kernel
def _mark_iteration_limit(
    world_active: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
):
    world = wp.tid()
    if world_active[world]:
        world_active[world] = False
        world_iteration_limit[world] = True


class SplittingState:
    """Persistent body/world state for a batched LOX solve."""

    def __init__(self, body_counts: Sequence[int], device: wp.DeviceLike = None):
        if len(body_counts) == 0:
            raise ValueError("At least one world is required.")
        if any(not isinstance(count, int) or count <= 0 for count in body_counts):
            raise ValueError("Each prototype world must contain at least one active body.")

        self.device = wp.get_device(device)
        self.body_counts = tuple(body_counts)
        self.num_worlds = len(body_counts)
        self.num_bodies = sum(body_counts)
        offsets = [0]
        body_world = []
        for world, count in enumerate(body_counts):
            offsets.append(offsets[-1] + count)
            body_world.extend([world] * count)

        self.body_offset = wp.array(offsets, dtype=wp.int32, device=self.device)
        self.body_world = wp.array(body_world, dtype=wp.int32, device=self.device)
        self.projected_twist = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.projected_twist_previous = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.global_twist = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.global_twist_previous = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.splitting_dual = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.splitting_dual_impulse = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.world_active = wp.ones(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_converged = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_failed = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_iteration_limit = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.iteration_count = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.residual_change = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_split = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_structural = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_structural_projected = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_unilateral_dual = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_total = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self._iteration_failed = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self._all_bodies_split_enabled = wp.ones(self.num_bodies, dtype=wp.int32, device=self.device)

    def reset(self) -> None:
        """Reset all body-space warm starts and world diagnostics."""
        self.projected_twist.zero_()
        self.projected_twist_previous.zero_()
        self.global_twist.zero_()
        self.global_twist_previous.zero_()
        self.splitting_dual.zero_()
        self.splitting_dual_impulse.zero_()
        self.world_active.fill_(True)
        self.world_converged.zero_()
        self.world_failed.zero_()
        self.world_iteration_limit.zero_()
        self.iteration_count.zero_()
        self.residual_change.zero_()
        self.residual_split.zero_()
        self.residual_structural.zero_()
        self.residual_structural_projected.zero_()
        self.residual_unilateral_dual.zero_()
        self.residual_total.zero_()
        self._iteration_failed.zero_()

    def begin(self, initial_twist: wp.array[vec6f], reset_dual: bool = False) -> None:
        """Initialize one nonlinear solve while optionally retaining its impulse warm start."""
        if initial_twist.shape[0] != self.num_bodies:
            raise ValueError("initial_twist must contain one entry per packed active body.")
        wp.copy(self.projected_twist, initial_twist)
        wp.copy(self.projected_twist_previous, initial_twist)
        wp.copy(self.global_twist, initial_twist)
        wp.copy(self.global_twist_previous, initial_twist)
        if reset_dual:
            self.splitting_dual.zero_()
            self.splitting_dual_impulse.zero_()
        self.world_active.fill_(True)
        self.world_converged.zero_()
        self.world_failed.zero_()
        self.world_iteration_limit.zero_()
        self.iteration_count.zero_()
        self.residual_change.zero_()
        self.residual_split.zero_()
        self.residual_structural.zero_()
        self.residual_structural_projected.zero_()
        self.residual_unilateral_dual.zero_()
        self.residual_total.zero_()
        self._iteration_failed.zero_()

    def store_dual_impulse(
        self,
        weight: wp.array[mat66f],
        body_has_unilateral: wp.array[wp.int32],
    ) -> None:
        """Store the physical body impulse ``W u`` for later warm starting."""
        if weight.shape[0] != self.num_bodies or body_has_unilateral.shape[0] != self.num_bodies:
            raise ValueError("Weight and unilateral mask arrays must contain one entry per body.")
        wp.launch(
            _store_dual_impulse,
            dim=self.num_bodies,
            inputs=[body_has_unilateral, weight],
            outputs=[self.splitting_dual, self.splitting_dual_impulse],
            device=self.device,
        )

    def restore_dual_from_impulse(
        self,
        inverse_weight: wp.array[mat66f],
        body_has_unilateral: wp.array[wp.int32],
    ) -> None:
        """Recover the scaled dual ``u = W^-1 (W u)`` for the current weight."""
        if inverse_weight.shape[0] != self.num_bodies or body_has_unilateral.shape[0] != self.num_bodies:
            raise ValueError("Inverse weight and unilateral mask arrays must contain one entry per body.")
        wp.launch(
            _restore_dual_from_impulse,
            dim=self.num_bodies,
            inputs=[body_has_unilateral, inverse_weight],
            outputs=[self.splitting_dual_impulse, self.splitting_dual],
            device=self.device,
        )

    def prepare_projection(
        self,
        global_solution: wp.array[vec6f],
        body_split_enabled: wp.array[wp.int32] | None = None,
    ) -> None:
        """Store the global solution and initialize ``p = v - u``."""
        if global_solution.shape[0] != self.num_bodies:
            raise ValueError("global_solution must contain one entry per packed active body.")
        if body_split_enabled is None:
            body_split_enabled = self._all_bodies_split_enabled
        elif body_split_enabled.shape[0] != self.num_bodies:
            raise ValueError("body_split_enabled must contain one entry per packed active body.")
        wp.launch(
            _prepare_projection,
            dim=self.num_bodies,
            inputs=[self.body_world, self.world_active, body_split_enabled, global_solution],
            outputs=[
                self.splitting_dual,
                self.global_twist_previous,
                self.global_twist,
                self.projected_twist_previous,
                self.projected_twist,
            ],
            device=self.device,
        )

    def replace_global_twist(
        self,
        global_solution: wp.array[vec6f],
        body_split_enabled: wp.array[wp.int32] | None = None,
    ) -> None:
        """Replace the active global twist while preserving its iteration history."""
        if global_solution.shape[0] != self.num_bodies:
            raise ValueError("global_solution must contain one entry per packed active body.")
        if body_split_enabled is None:
            body_split_enabled = self._all_bodies_split_enabled
        elif body_split_enabled.shape[0] != self.num_bodies:
            raise ValueError("body_split_enabled must contain one entry per packed active body.")
        wp.launch(
            _replace_global_twist,
            dim=self.num_bodies,
            inputs=[self.body_world, self.world_active, body_split_enabled, global_solution],
            outputs=[self.global_twist, self.projected_twist, self.splitting_dual],
            device=self.device,
        )

    def finish_iteration(
        self,
        projection_status: wp.array[wp.int32],
        time_step: float,
        position_tolerance: float,
        rotation_tolerance: float,
        structural_residual: wp.array[wp.float32] | None = None,
        projected_structural_residual: wp.array[wp.float32] | None = None,
        velocity_tolerance: float = 1.0e-5,
    ) -> None:
        """Update the dual state, residuals, and per-world convergence mask."""
        if projection_status.shape[0] != self.num_worlds:
            raise ValueError("projection_status must contain one entry per world.")
        if structural_residual is None:
            self.residual_structural.zero_()
            structural_residual = self.residual_structural
        elif structural_residual.shape[0] != self.num_worlds:
            raise ValueError("structural_residual must contain one entry per world.")
        if projected_structural_residual is None:
            self.residual_structural_projected.zero_()
            projected_structural_residual = self.residual_structural_projected
        elif projected_structural_residual.shape[0] != self.num_worlds:
            raise ValueError("projected_structural_residual must contain one entry per world.")
        if time_step <= 0.0 or position_tolerance <= 0.0 or rotation_tolerance <= 0.0 or velocity_tolerance <= 0.0:
            raise ValueError("Time step and convergence tolerances must be positive.")
        wp.launch(
            _initialize_finish_iteration,
            dim=self.num_worlds,
            inputs=[
                projection_status,
            ],
            outputs=[
                self.world_active,
                self.world_failed,
                self.iteration_count,
                self._iteration_failed,
                self.residual_change,
                self.residual_split,
                self.residual_unilateral_dual,
            ],
            device=self.device,
        )
        wp.launch(
            _finish_iteration,
            dim=self.num_bodies,
            inputs=[
                time_step,
                position_tolerance,
                rotation_tolerance,
                velocity_tolerance,
                self.body_world,
                self.global_twist_previous,
                self.global_twist,
                self.projected_twist_previous,
                self.projected_twist,
                self.world_active,
            ],
            outputs=[
                self.splitting_dual,
                self._iteration_failed,
                self.residual_change,
                self.residual_split,
                self.residual_unilateral_dual,
            ],
            device=self.device,
        )
        wp.launch(
            _finalize_finish_iteration,
            dim=self.num_worlds,
            inputs=[
                structural_residual,
                projected_structural_residual,
                self._iteration_failed,
            ],
            outputs=[
                self.world_active,
                self.world_converged,
                self.world_failed,
                self.residual_change,
                self.residual_split,
                self.residual_structural,
                self.residual_structural_projected,
                self.residual_unilateral_dual,
                self.residual_total,
            ],
            device=self.device,
        )

    def mark_iteration_limit(self) -> None:
        """Deactivate worlds that remain unconverged at the iteration limit."""
        wp.launch(
            _mark_iteration_limit,
            dim=self.num_worlds,
            inputs=[],
            outputs=[self.world_active, self.world_iteration_limit],
            device=self.device,
        )
