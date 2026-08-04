# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-contact orchestration for the LOX rigid-body solve."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ...core.types import vec6f
from .adapter import LOXKaminoAdapter
from .joint import BatchedStructuralJointSolver
from .sweep import (
    compute_projection_residuals,
    prepare_contact_projection_data,
    prepare_jacobi_projection_data,
    project_constraints_jacobi,
    project_constraints_sequential,
)
from .weight import BODY_WEIGHT_BETA_DEFAULT, BODY_WEIGHT_SIGMA_DEFAULT

__all__ = [
    "LOX_STATUS_ACTIVE",
    "LOX_STATUS_CONVERGED",
    "LOX_STATUS_FAILED",
    "LOX_STATUS_ITERATION_LIMIT",
    "LOXSolver",
]

LOX_STATUS_ACTIVE = 0
"""The world is active in the current splitting solve."""

LOX_STATUS_CONVERGED = 1
"""The world met the configured splitting tolerances."""

LOX_STATUS_FAILED = 2
"""A unilateral projection failed for the world."""

LOX_STATUS_ITERATION_LIMIT = 3
"""The world reached the configured splitting iteration limit."""

wp.set_module_options({"enable_backward": False})

_BODY_WEIGHT_MASS_PROPORTIONAL = 0
_BODY_WEIGHT_ANISOTROPIC = 1
_JOINT_METRIC_SIMPLE = 0
_JOINT_METRIC_FULL_BLOCK = 1
_JOINT_METRIC_MASS_SPLIT = 2
_JOINT_SOLVE_ALM = 0
_JOINT_SOLVE_SCHUR_DIRECT = 1
_JOINT_PENALTY_SEED_PERCENTILE = 0.02


def _low_mode_eigenvalue(eigenvalues: np.ndarray) -> float:
    """Select the discrete low-mode percentile used for ALM scale seeding."""
    index = int(_JOINT_PENALTY_SEED_PERCENTILE * eigenvalues.size)
    return float(eigenvalues[index])


@wp.kernel
def _finalize_world_status(
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
    world_accepted: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    converged = world_converged[world] and not world_failed[world]
    world_accepted[world] = not world_failed[world]
    if world_failed[world]:
        world_status[world] = LOX_STATUS_FAILED
    elif converged:
        world_status[world] = LOX_STATUS_CONVERGED
    elif world_iteration_limit[world]:
        world_status[world] = LOX_STATUS_ITERATION_LIMIT
    else:
        world_status[world] = LOX_STATUS_ACTIVE


@wp.kernel
def _update_iteration_condition(
    max_iterations: wp.int32,
    world_active: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    condition: wp.array[wp.int32],
):
    world = wp.tid()
    if world_active[world] and iteration_count[world] < max_iterations:
        wp.atomic_max(condition, 0, 1)


def _validate_fixed_iteration_count(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to one.")


class LOXSolver:
    """Run a fixed LOX splitting solve on one frozen linearization.

    The caller owns collision detection, Jacobian construction, pose
    integration, and nonlinear relinearization. This class owns the smooth
    system update, weight construction, blocked dense LLT solve, unilateral
    projections, convergence freezing, and output conversion for one such
    linearization.
    """

    def __init__(
        self,
        adapter: LOXKaminoAdapter,
        max_iterations: int = 25,
        projection_iterations: int = 3,
        projection_method: str = "jacobi",
        position_tolerance: float = 1.0e-5,
        rotation_tolerance: float = 1.0e-5,
        velocity_tolerance: float = 1.0e-5,
        weight_sigma: float = BODY_WEIGHT_SIGMA_DEFAULT,
        weight_beta: float = BODY_WEIGHT_BETA_DEFAULT,
        joint_penalty_scale: float = 10.0,
        joint_multiplier_projected_fraction: float = 0.0,
        joint_warmstart_factor: float = 0.5,
        _body_weight_mode: int = _BODY_WEIGHT_MASS_PROPORTIONAL,
        _joint_metric_mode: int = _JOINT_METRIC_SIMPLE,
        _joint_solve_mode: int = _JOINT_SOLVE_ALM,
    ):
        _validate_fixed_iteration_count("max_iterations", max_iterations)
        _validate_fixed_iteration_count("projection_iterations", projection_iterations)
        if projection_method not in ("jacobi", "gauss_seidel"):
            raise ValueError("projection_method must be 'jacobi' or 'gauss_seidel'.")
        if not math.isfinite(position_tolerance) or position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be finite and positive.")
        if not math.isfinite(rotation_tolerance) or rotation_tolerance <= 0.0:
            raise ValueError("rotation_tolerance must be finite and positive.")
        if not math.isfinite(velocity_tolerance) or velocity_tolerance <= 0.0:
            raise ValueError("velocity_tolerance must be finite and positive.")
        if not math.isfinite(weight_sigma) or not 0.0 < weight_sigma <= 1.0:
            raise ValueError("weight_sigma must be finite and in (0, 1].")
        if not math.isfinite(weight_beta) or weight_beta < 1.0:
            raise ValueError("weight_beta must be finite and at least one.")
        if not math.isfinite(joint_penalty_scale) or joint_penalty_scale <= 0.0:
            raise ValueError("joint_penalty_scale must be finite and positive.")
        if (
            not math.isfinite(joint_multiplier_projected_fraction)
            or not 0.0 <= joint_multiplier_projected_fraction <= 1.0
        ):
            raise ValueError("joint_multiplier_projected_fraction must be finite and in [0, 1].")
        if not math.isfinite(joint_warmstart_factor) or not 0.0 <= joint_warmstart_factor <= 1.0:
            raise ValueError("joint_warmstart_factor must be finite and in [0, 1].")
        if _body_weight_mode not in (_BODY_WEIGHT_MASS_PROPORTIONAL, _BODY_WEIGHT_ANISOTROPIC):
            raise ValueError("_body_weight_mode must select a supported internal body weight.")
        if _joint_metric_mode not in (_JOINT_METRIC_SIMPLE, _JOINT_METRIC_FULL_BLOCK, _JOINT_METRIC_MASS_SPLIT):
            raise ValueError("_joint_metric_mode must select a supported internal joint metric.")
        if _joint_solve_mode not in (_JOINT_SOLVE_ALM, _JOINT_SOLVE_SCHUR_DIRECT):
            raise ValueError("_joint_solve_mode must select a supported internal joint solve.")

        self.adapter = adapter
        self.device = adapter.device
        self.max_iterations = max_iterations
        self.projection_iterations = projection_iterations
        self.projection_method = projection_method
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.weight_sigma = weight_sigma
        self.weight_beta = weight_beta
        self.joint_penalty_scale = wp.full(
            adapter.num_worlds, joint_penalty_scale, dtype=wp.float32, device=self.device
        )
        self.joint_multiplier_projected_fraction = joint_multiplier_projected_fraction
        self.joint_warmstart_factor = joint_warmstart_factor
        self._body_weight_mode = _body_weight_mode
        self._joint_metric_mode = _joint_metric_mode
        self._joint_solve_mode = _joint_solve_mode

        self.system = adapter.system
        self.splitting = adapter.splitting
        self.projected_twist = self.splitting.projected_twist
        self.world_active = self.splitting.world_active
        self.world_converged = self.splitting.world_converged
        self.world_failed = self.splitting.world_failed
        self.world_iteration_limit = self.splitting.world_iteration_limit
        self.iteration_count = self.splitting.iteration_count
        self.world_accepted = wp.zeros(adapter.num_worlds, dtype=wp.bool, device=self.device)
        self.world_status = wp.zeros(adapter.num_worlds, dtype=wp.int32, device=self.device)
        self._iteration_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.contact_residual_max = adapter.world_contact_residual_max
        self.limit_residual_max = adapter.world_limit_residual_max
        self.friction_residual_max = adapter.world_friction_residual_max
        self._initial_twist = wp.zeros(adapter.model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        self.structural_joint_solver: BatchedStructuralJointSolver | None = None
        if _joint_solve_mode == _JOINT_SOLVE_SCHUR_DIRECT and adapter.structural_row_count > 0:
            self.structural_joint_solver = self._make_structural_joint_solver()

    def _make_structural_joint_solver(self) -> BatchedStructuralJointSolver:
        adapter = self.adapter
        return BatchedStructuralJointSolver(
            body_system=self.system,
            row_world=adapter.structural_row_world,
            body_first_global=adapter.structural_body_first_global,
            body_second_global=adapter.structural_body_second_global,
            jacobian_first=adapter.structural_jacobian_first,
            jacobian_second=adapter.structural_jacobian_second,
            block_row_offset=adapter.structural_block_row_offset,
            block_row_count=adapter.structural_block_row_count,
            row_block=adapter.structural_row_block,
        )

    def joint_penalty_scale_seed(self, time_step: float) -> list[float]:
        """Estimate and apply a timestep-aware structural ALM scale.

        The estimate is the reciprocal of the discrete second-percentile
        positive eigenvalue of the effective-mass-normalized structural
        Delassus. Both the percentile and the resulting scale are evaluated
        independently for each world. Systems with fewer than 50 positive
        modes therefore retain the minimum-eigenvalue estimate. The smooth
        operator excludes the body contact-consensus metric, which is a
        separate splitting concern.

        The adapter must already contain Jacobians for the state to analyze.
        This operation performs a one-time dense structural assembly and a
        host eigensolve, so it is intended for initialization rather than the
        captured simulation loop.

        Args:
            time_step: Simulation time step [s].

        Returns:
            The estimated dimensionless structural ALM penalty scale for each
            world.
        """
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        if self._joint_solve_mode != _JOINT_SOLVE_ALM:
            raise ValueError("joint penalty scale seeding is only available for the structural ALM solve.")

        adapter = self.adapter
        if adapter.structural_row_count == 0:
            return self.joint_penalty_scale.numpy().tolist()

        self.reset()
        adapter.begin_time_step(time_step)
        try:
            adapter.body_linearization_twist.zero_()
            adapter.update(
                time_step,
                joint_penalty_scale=1.0,
                linearization_twist=adapter.body_linearization_twist,
                block_joint_metrics=False,
                mass_split_joint_metrics=False,
                assemble_structural_penalty=False,
            )
            wp.copy(self.system.weighted_matrix, self.system.smooth_matrix)
            self.system.factorize()

            structural_solver = self._make_structural_joint_solver()
            structural_solver.assemble_delassus()
            matrix_values = structural_solver.unscaled_schur_matrix.numpy()
            matrix_offsets = structural_solver.info.mio.numpy()
            vector_offsets = structural_solver.info.vio.numpy()
            vector_rows = structural_solver.vector_row.numpy()
            effective_mass = adapter.structural_effective_mass.numpy()
            matrix_epsilon = np.finfo(matrix_values.dtype).eps

            seeds = self.joint_penalty_scale.numpy().astype(np.float64)
            positive_by_world: list[list[np.ndarray]] = [[] for _ in range(adapter.num_worlds)]
            for component, row_count in enumerate(structural_solver.component_row_counts):
                if row_count == 0:
                    continue
                matrix_offset = int(matrix_offsets[component])
                vector_offset = int(vector_offsets[component])
                rows = vector_rows[vector_offset : vector_offset + row_count]
                row_metric = effective_mass[rows].astype(np.float64)
                metric_sqrt = np.sqrt(row_metric)
                matrix = (
                    matrix_values[matrix_offset : matrix_offset + row_count * row_count]
                    .reshape(row_count, row_count)
                    .astype(np.float64)
                )
                normalized = metric_sqrt[:, None] * matrix * metric_sqrt[None, :]
                eigenvalues = np.linalg.eigvalsh(normalized)
                positive_threshold = matrix_epsilon * row_count * float(eigenvalues[-1])
                positive = eigenvalues[eigenvalues > positive_threshold]
                if positive.size == 0:
                    world = structural_solver.component_world_host[component]
                    raise RuntimeError(
                        f"World {world} component {component} structural Delassus has no resolvable positive eigenvalue."
                    )
                positive_by_world[structural_solver.component_world_host[component]].append(positive)

            for world, component_eigenvalues in enumerate(positive_by_world):
                if not component_eigenvalues:
                    continue
                positive = np.sort(np.concatenate(component_eigenvalues))
                seed = 1.0 / _low_mode_eigenvalue(positive)
                if not math.isfinite(seed) or seed > np.finfo(np.float32).max:
                    raise RuntimeError(f"World {world} structural ALM penalty seed is not representable in float32.")
                seeds[world] = seed

            applied_seeds = seeds.astype(np.float32)
            self.joint_penalty_scale.assign(applied_seeds)
            return applied_seeds.tolist()
        finally:
            self.reset()

    def reset(self) -> None:
        """Clear structural/splitting warm starts and per-world diagnostics."""
        self.splitting.reset()
        self.adapter.reset_structural_multipliers()
        self.adapter.projection_status.zero_()
        self.world_accepted.zero_()
        self.world_status.fill_(LOX_STATUS_ACTIVE)
        self.adapter.contact_residual.zero_()
        self.adapter.limit_residual.zero_()
        self.adapter.friction_residual.zero_()
        self.contact_residual_max.zero_()
        self.limit_residual_max.zero_()
        self.friction_residual_max.zero_()
        self.adapter.friction_reaction.zero_()
        self._initial_twist.zero_()

    def begin_time_step(
        self,
        time_step: float,
        limit_stabilization_fraction: float = 0.01,
        contact_stabilization_fraction: float = 0.01,
        contact_dead_zone: float = 1.0e-6,
        impact_velocity_threshold: float = 1.0e-3,
        reset_dual: bool = False,
    ) -> None:
        """Freeze inertial velocities and import damped constraint warm starts.

        The body consensus warm start is retained as the generalized impulse
        ``W u`` and converted back to the scaled dual after the next solve
        builds its current body weight.
        """
        self.adapter.scale_structural_multipliers(self.joint_warmstart_factor)
        self.adapter.begin_time_step(
            time_step,
            limit_stabilization_fraction=limit_stabilization_fraction,
            contact_stabilization_fraction=contact_stabilization_fraction,
            contact_dead_zone=contact_dead_zone,
            impact_velocity_threshold=impact_velocity_threshold,
        )
        self.splitting.begin(self.adapter.body_velocity_begin, reset_dual=reset_dual)
        self.world_accepted.zero_()
        self.world_status.fill_(LOX_STATUS_ACTIVE)

    def _prepare_body_space_projection(self) -> None:
        adapter = self.adapter
        if self.projection_method == "jacobi":
            prepare_jacobi_projection_data(
                adapter.friction_world,
                adapter.friction_local,
                adapter.world_friction_count,
                adapter.friction_body_first,
                adapter.friction_body_second,
                adapter.friction_jacobian_first,
                adapter.friction_jacobian_second,
                adapter.contact_world,
                adapter.contact_local,
                adapter.world_contact_count,
                adapter.contact_body_first,
                adapter.contact_body_second,
                adapter.contact_jacobian_first,
                adapter.contact_jacobian_second,
                adapter.contact_bias,
                adapter.contact_friction,
                adapter.limit_world,
                adapter.limit_local,
                adapter.world_limit_count,
                adapter.limit_body_first,
                adapter.limit_body_second,
                adapter.limit_jacobian_first,
                adapter.limit_jacobian_second,
                adapter.body_constraint_count,
                adapter.static_body_constraint_count,
                self.system.inverse_weight,
                adapter.friction_projection_delassus,
                adapter.contact_projection_delassus,
                adapter.contact_projection_delassus_normal_first,
                adapter.limit_projection_delassus,
                adapter.world_jacobi_projection_status,
            )
        else:
            prepare_contact_projection_data(
                adapter.contact_world,
                adapter.contact_local,
                adapter.world_contact_count,
                adapter.contact_body_first,
                adapter.contact_body_second,
                adapter.contact_jacobian_first,
                adapter.contact_jacobian_second,
                adapter.contact_bias,
                adapter.contact_friction,
                self.system.inverse_weight,
                adapter.contact_projection_delassus,
                adapter.contact_projection_delassus_normal_first,
                adapter.world_contact_projection_status,
            )

    def _project_body_space_constraints(self) -> None:
        adapter = self.adapter
        splitting = self.splitting
        if self.projection_method == "jacobi":
            project_constraints_jacobi(
                self.projection_iterations,
                splitting.world_active,
                splitting.body_world,
                adapter.friction_world,
                adapter.friction_local,
                adapter.world_friction_count,
                adapter.friction_body_first,
                adapter.friction_body_second,
                adapter.friction_jacobian_first,
                adapter.friction_jacobian_second,
                adapter.friction_impulse_bound,
                adapter.friction_projection_delassus,
                adapter.contact_world,
                adapter.contact_local,
                adapter.world_contact_count,
                adapter.contact_body_first,
                adapter.contact_body_second,
                adapter.contact_jacobian_first,
                adapter.contact_jacobian_second,
                adapter.contact_bias,
                adapter.contact_friction,
                adapter.contact_projection_delassus,
                adapter.contact_projection_delassus_normal_first,
                adapter.limit_world,
                adapter.limit_local,
                adapter.world_limit_count,
                adapter.limit_body_first,
                adapter.limit_body_second,
                adapter.limit_jacobian_first,
                adapter.limit_jacobian_second,
                adapter.limit_bias,
                adapter.limit_projection_delassus,
                self.system.inverse_weight,
                splitting.projected_twist,
                adapter.projection_twist_delta,
                adapter.contact_reaction,
                adapter.limit_reaction,
                adapter.friction_reaction,
                adapter.world_jacobi_projection_status,
                adapter.projection_status,
            )
        else:
            project_constraints_sequential(
                self.projection_iterations,
                splitting.world_active,
                adapter.world_friction_offset,
                adapter.world_friction_count,
                adapter.friction_body_first,
                adapter.friction_body_second,
                adapter.friction_jacobian_first,
                adapter.friction_jacobian_second,
                adapter.friction_impulse_bound,
                adapter.world_contact_offset,
                adapter.world_contact_count,
                adapter.contact_body_first,
                adapter.contact_body_second,
                adapter.contact_jacobian_first,
                adapter.contact_jacobian_second,
                adapter.contact_bias,
                adapter.contact_friction,
                adapter.world_limit_offset,
                adapter.world_limit_count,
                adapter.limit_body_first,
                adapter.limit_body_second,
                adapter.limit_jacobian_first,
                adapter.limit_jacobian_second,
                adapter.limit_bias,
                self.system.inverse_weight,
                splitting.projected_twist,
                adapter.friction_reaction,
                adapter.friction_velocity,
                adapter.contact_reaction,
                adapter.contact_velocity,
                adapter.limit_reaction,
                adapter.limit_velocity,
                adapter.projection_status,
                adapter.contact_projection_delassus,
                adapter.contact_projection_delassus_normal_first,
                adapter.world_contact_projection_status,
            )

    def _update_conditional_iteration(self) -> None:
        self._iteration_condition.zero_()
        wp.launch(
            _update_iteration_condition,
            dim=self.adapter.num_worlds,
            inputs=[
                self.max_iterations,
                self.splitting.world_active,
                self.splitting.iteration_count,
            ],
            outputs=[self._iteration_condition],
            device=self.device,
        )

    def _body_space_iteration(self, time_step: float, linearization_twist: wp.array[vec6f], conditional: bool) -> None:
        adapter = self.adapter
        system = self.system
        splitting = self.splitting
        if self.structural_joint_solver is None:
            system.solve_candidate(splitting.projected_twist, splitting.splitting_dual)
            adapter.unpack_system_solution()
        else:
            system.build_candidate_right_hand_side(splitting.projected_twist, splitting.splitting_dual)
            self.structural_joint_solver.solve_free_body()
            self.structural_joint_solver.project(
                time_step,
                linearization_twist,
                splitting.world_active,
                adapter.structural_residual,
                adapter.structural_multiplier_index,
                adapter.structural_multiplier,
                adapter.data.joints.lambda_j,
            )
            adapter.unpack_system_solution()
        splitting.prepare_projection(adapter.system_solution_twist)
        self._project_body_space_constraints()
        if self.structural_joint_solver is None:
            adapter.update_structural_multipliers_from_twist(
                time_step,
                min(self.position_tolerance, self.rotation_tolerance),
                linearization_twist,
                splitting.global_twist,
                splitting.projected_twist,
                splitting.world_active,
                projected_fraction=self.joint_multiplier_projected_fraction,
            )
        else:
            # Larger direct corrections can reflect the projected structural
            # residual and introduce temporal oscillations.
            projected_fraction = min(self.joint_multiplier_projected_fraction, 0.5)
            if projected_fraction > 0.0:
                self.structural_joint_solver.refine_from_twists(
                    time_step,
                    linearization_twist,
                    splitting.global_twist,
                    splitting.projected_twist,
                    projected_fraction,
                    adapter.world_has_unilateral,
                    splitting.world_active,
                    adapter.structural_residual,
                    adapter.structural_multiplier_index,
                    adapter.structural_multiplier,
                    adapter.data.joints.lambda_j,
                )
                adapter.unpack_system_solution()
                splitting.replace_global_twist(adapter.system_solution_twist)
            adapter.evaluate_structural_residuals_from_twists(
                time_step,
                min(self.position_tolerance, self.rotation_tolerance),
                linearization_twist,
                splitting.global_twist,
                splitting.projected_twist,
                splitting.world_active,
            )
        splitting.finish_iteration(
            adapter.projection_status,
            time_step,
            self.position_tolerance,
            self.rotation_tolerance,
            structural_residual=adapter.world_structural_residual,
            projected_structural_residual=adapter.world_projected_structural_residual,
            velocity_tolerance=self.velocity_tolerance,
        )
        if conditional:
            self._update_conditional_iteration()

    def solve(
        self,
        time_step: float,
        initial_twist: wp.array[vec6f] | None = None,
        linearization_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Assemble and solve one frozen-contact smooth linearization.

        Args:
            time_step: Simulation time step [s].
            initial_twist: Optional body-space splitting initial guess. If
                omitted, the last projected twist is retained as a warm start.
            linearization_twist: Body twist defining the structural Newton
                linearization pose. Pass zero for the first linearly implicit
                solve about the begin-step pose.
        """
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be finite and positive.")

        if initial_twist is None:
            wp.copy(self._initial_twist, self.projected_twist)
            initial_twist = self._initial_twist
        self.splitting.begin(initial_twist, reset_dual=False)
        self.world_accepted.zero_()
        self.world_status.fill_(LOX_STATUS_ACTIVE)
        self.adapter.contact_residual.zero_()
        self.adapter.limit_residual.zero_()
        self.adapter.friction_residual.zero_()
        self.contact_residual_max.zero_()
        self.limit_residual_max.zero_()
        self.friction_residual_max.zero_()

        adapter = self.adapter
        system = self.system
        splitting = self.splitting
        use_structural_schur = self._joint_solve_mode == _JOINT_SOLVE_SCHUR_DIRECT
        adapter.update(
            time_step,
            joint_penalty_scale=self.joint_penalty_scale,
            linearization_twist=linearization_twist,
            block_joint_metrics=not use_structural_schur and self._joint_metric_mode != _JOINT_METRIC_SIMPLE,
            mass_split_joint_metrics=not use_structural_schur and self._joint_metric_mode == _JOINT_METRIC_MASS_SPLIT,
            assemble_structural_penalty=not use_structural_schur,
        )
        if linearization_twist is None:
            linearization_twist = adapter.body_linearization_twist
        weight_metric = None
        if use_structural_schur:
            weight_metric = system.build_simple_joint_aware_weight_metric(
                adapter.structural_row_world,
                adapter.structural_body_first_global,
                adapter.structural_body_second_global,
                adapter.structural_jacobian_first,
                adapter.structural_jacobian_second,
                adapter.structural_effective_mass,
                joint_metric_scale=self.joint_penalty_scale,
            )
        if self._body_weight_mode == _BODY_WEIGHT_MASS_PROPORTIONAL:
            system.build_weighted_matrix(
                metric_matrix=weight_metric,
                body_has_unilateral=adapter.body_has_unilateral,
                sigma=self.weight_sigma,
                beta=self.weight_beta,
            )
        else:
            system.build_anisotropic_weighted_matrix(
                metric_matrix=weight_metric,
                body_has_unilateral=adapter.body_has_unilateral,
                sigma=self.weight_sigma,
                beta=self.weight_beta,
            )
        splitting.restore_dual_from_impulse(system.inverse_weight, adapter.body_has_unilateral)
        system.factorize()
        self._prepare_body_space_projection()
        if self.structural_joint_solver is not None:
            self.structural_joint_solver.factorize()
            self.structural_joint_solver.warmstart(time_step, adapter.structural_multiplier)

        if self.device.is_cuda and self.device.is_capturing and wp.is_conditional_graph_supported():
            self._iteration_condition.fill_(1)
            wp.capture_while(
                self._iteration_condition,
                self._body_space_iteration,
                time_step=time_step,
                linearization_twist=linearization_twist,
                conditional=True,
            )
        else:
            for _iteration in range(self.max_iterations):
                self._body_space_iteration(time_step, linearization_twist, conditional=False)

        splitting.store_dual_impulse(system.weight, adapter.body_has_unilateral)
        splitting.mark_iteration_limit()
        wp.launch(
            _finalize_world_status,
            dim=adapter.num_worlds,
            inputs=[splitting.world_converged, splitting.world_failed, splitting.world_iteration_limit],
            outputs=[self.world_accepted, self.world_status],
            device=self.device,
        )
        compute_projection_residuals(
            self.world_accepted,
            adapter.projection_status,
            adapter.friction_world,
            adapter.friction_local,
            adapter.world_friction_count,
            adapter.friction_body_first,
            adapter.friction_body_second,
            adapter.friction_jacobian_first,
            adapter.friction_jacobian_second,
            adapter.friction_impulse_bound,
            adapter.friction_reaction,
            adapter.contact_world,
            adapter.contact_local,
            adapter.world_contact_count,
            adapter.contact_body_first,
            adapter.contact_body_second,
            adapter.contact_jacobian_first,
            adapter.contact_jacobian_second,
            adapter.contact_bias,
            adapter.contact_friction,
            adapter.contact_reaction,
            adapter.limit_world,
            adapter.limit_local,
            adapter.world_limit_count,
            adapter.limit_body_first,
            adapter.limit_body_second,
            adapter.limit_jacobian_first,
            adapter.limit_jacobian_second,
            adapter.limit_bias,
            adapter.limit_reaction,
            system.inverse_weight,
            splitting.projected_twist,
            adapter.friction_velocity,
            adapter.contact_velocity,
            adapter.limit_velocity,
            adapter.contact_residual,
            adapter.limit_residual,
            adapter.friction_residual,
            adapter.world_contact_residual_max,
            adapter.world_limit_residual_max,
            adapter.world_friction_residual_max,
        )
        adapter.write_outputs(time_step, body_velocity=splitting.projected_twist)
