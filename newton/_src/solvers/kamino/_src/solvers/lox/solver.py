# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-contact orchestration for the LOX rigid-body solve."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...core.types import vec6f
from .adapter import LOXKaminoAdapter
from .apgd import project_constraints_apgd, project_deformable_constraints_apgd
from .avbd import prepare_constraints_avbd, project_constraints_avbd, project_deformable_constraints_avbd
from .cable import validate_cable_model
from .colored_gauss_seidel import ColoredGaussSeidelProjection
from .contact import CoulombSolveStatistics
from .deformable_contact import DeformableContactSystem
from .deformable_penetration import DeformablePenetrationFreeLimiter
from .deformable_self_contact import DeformableSelfContactDetector
from .deformable_splitting import DeformableSplittingState
from .deformable_system import DeformableFEMSystem
from .joint import BatchedStructuralJointSolver
from .sweep import (
    compute_projection_residuals,
    prepare_contact_projection_data,
    prepare_jacobi_projection_data,
    project_constraints_jacobi,
    project_constraints_sequential,
    sweep_constraints_sequential,
    warm_start_constraints_sequential,
)
from .time import validate_world_time_steps
from .weight import BODY_WEIGHT_BETA_DEFAULT, BODY_WEIGHT_SIGMA_DEFAULT, DEFORMABLE_WEIGHT_BETA_DEFAULT

if TYPE_CHECKING:
    from ......sim import Contacts, Model, State
    from ....config import ConstraintStabilizationConfig, LOXSolverConfig
    from ...core.data import DataKamino
    from ...core.model import ModelKamino
    from ...geometry.contacts import ContactsKamino
    from ...kinematics.limits import LimitsKamino

__all__ = [
    "LOX_STATUS_ACTIVE",
    "LOX_STATUS_CONVERGED",
    "LOX_STATUS_FAILED",
    "LOX_STATUS_ITERATION_LIMIT",
    "LOXProblem",
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


@wp.kernel
def _reset_newton_particle_state(
    particle_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    particle_q_default: wp.array[wp.vec3],
    particle_qd_default: wp.array[wp.vec3],
    reset_q: bool,
    reset_qd: bool,
    reset_f: bool,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_dual_impulse: wp.array[wp.vec3],
):
    particle = wp.tid()
    world = wp.max(particle_world[particle], 0)
    if world_mask[world]:
        if reset_q:
            particle_q[particle] = particle_q_default[particle]
        if reset_qd:
            particle_qd[particle] = particle_qd_default[particle]
            particle_dual_impulse[particle] = wp.vec3(0.0)
        if reset_f:
            particle_f[particle] = wp.vec3(0.0)


@wp.kernel
def _reset_newton_body_dual_impulse(
    body_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    body_dual_impulse: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    world = wp.max(body_world[body], 0)
    if world_mask[world]:
        body_dual_impulse[body] = wp.spatial_vectorf(0.0)


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
def _reset_solver_worlds_masked(
    world_mask: wp.array[wp.bool],
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    cloth_only_residual_total: wp.array[wp.float32],
    world_accepted: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    contact_residual_max: wp.array[wp.float32],
    limit_residual_max: wp.array[wp.float32],
    friction_residual_max: wp.array[wp.float32],
):
    world = wp.tid()
    if world_mask[world]:
        world_active[world] = True
        world_converged[world] = False
        world_failed[world] = False
        world_iteration_limit[world] = False
        iteration_count[world] = 0
        cloth_only_residual_total[world] = 0.0
        world_accepted[world] = False
        world_status[world] = LOX_STATUS_ACTIVE
        contact_residual_max[world] = 0.0
        limit_residual_max[world] = 0.0
        friction_residual_max[world] = 0.0


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


@wp.kernel
def _mark_iteration_limit(
    world_active: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
):
    world = wp.tid()
    if world_active[world]:
        world_active[world] = False
        world_iteration_limit[world] = True


@wp.kernel
def _initialize_body_velocity_guess(
    body_block: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    velocity_start: wp.array[vec6f],
    external_wrench: wp.array[wp.spatial_vectorf],
    gravity: wp.array[wp.vec3f],
    time_step: wp.array[wp.float32],
    fraction: float,
    velocity_guess: wp.array[vec6f],
):
    body = wp.tid()
    dt = time_step[body_world[body]]
    guess = velocity_start[body]
    if body_block[body] >= 0 and fraction > 0.0:
        wrench = external_wrench[body]
        inv_mass = inverse_mass[body]
        linear_acceleration = inv_mass * wp.vec3f(wrench[0], wrench[1], wrench[2])
        if inv_mass > 0.0:
            linear_acceleration += gravity[body_world[body]]
        angular_acceleration = inverse_inertia_world[body] @ wp.vec3f(wrench[3], wrench[4], wrench[5])
        for axis in range(3):
            guess[axis] += fraction * dt * linear_acceleration[axis]
            guess[axis + 3] += fraction * dt * angular_acceleration[axis]
    velocity_guess[body] = guess


def _validate_fixed_iteration_count(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to one.")


class LOXProblem:
    """Store the inputs for one LOX forward-dynamics solve."""

    def __init__(
        self,
        *,
        limit_stabilization_fraction: float = 0.01,
        contact_stabilization_fraction: float = 0.01,
        contact_dead_zone: float = 1.0e-6,
        impact_velocity_threshold: float = 1.0e-3,
        contact_recoverable_response: bool = False,
    ):
        self.initial_twist: wp.array[vec6f] | None = None
        self.linearization_twist: wp.array[vec6f] | None = None
        self.limit_stabilization_fraction = limit_stabilization_fraction
        self.contact_stabilization_fraction = contact_stabilization_fraction
        self.contact_dead_zone = contact_dead_zone
        self.impact_velocity_threshold = impact_velocity_threshold
        self.contact_recoverable_response = contact_recoverable_response
        self._newton_model: Model | None = None
        self._constraints_config: ConstraintStabilizationConfig | None = None
        self._lox_config: LOXSolverConfig | None = None
        self._use_fk_solver = False
        self._newton_solver: LOXSolver | None = None
        self._newton_state_in: State | None = None
        self._newton_state_out: State | None = None
        self._newton_contacts: Contacts | None = None
        self._newton_body_pose: wp.array[wp.transform] | None = None

    @property
    def contact_conversion_policy(self) -> tuple[bool, bool]:
        """Return speculative-contact culling and prescribed-contact filtering."""
        return False, True

    def attach_newton_solver(self, solver: LOXSolver) -> None:
        """Attach the LOX solver that owns Newton-facing warm-start state."""
        self._newton_solver = solver

    def attach_newton_model(
        self,
        model: Model,
        constraints_config: ConstraintStabilizationConfig,
        lox_config: LOXSolverConfig,
        *,
        use_fk_solver: bool,
    ) -> None:
        """Attach Newton-facing model and configuration data."""
        self._newton_model = model
        self._newton_body_pose = (
            wp.empty_like(model.body_q) if model.particle_count > 0 and model.body_count > 0 else None
        )
        self._constraints_config = constraints_config
        self._lox_config = lox_config
        self._use_fk_solver = use_fk_solver
        self._built_friction_active = model.joint_friction.numpy() > 0.0
        adapter = self._require_newton_solver().adapter
        if adapter is None:
            self._built_effort_driven = np.zeros(model.joint_dof_count, dtype=bool)
            self._built_effort_bounded = np.zeros(model.joint_dof_count, dtype=bool)
        else:
            self._built_effort_driven = np.asarray(adapter.effort_driven_dof_mask, dtype=bool)
            self._built_effort_bounded = np.asarray(adapter.effort_bounded_dof_mask, dtype=bool)

    def validate_newton_model_changed(self, *, check_dof: bool) -> None:
        """Validate LOX topology frozen while attaching the Newton model."""
        model = self._require_newton_model()
        validate_cable_model(model, use_fk_solver=self._use_fk_solver)
        if not check_dof:
            return

        friction = model.joint_friction.numpy()
        if not np.isfinite(friction).all() or np.any(friction < 0.0):
            raise ValueError("Joint friction values must be finite and nonnegative.")
        changed = np.flatnonzero((friction > 0.0) != self._built_friction_active)
        if changed.size > 0:
            dof = int(changed[0])
            raise RuntimeError(
                f"Changing joint-friction constraint topology for DOF {dof} is not supported; "
                "recreate SolverKamino to apply a zero-to-positive or positive-to-zero friction change."
            )

        effort_limit = model.joint_effort_limit.numpy()
        if np.isnan(effort_limit).any() or np.any(effort_limit < 0.0):
            raise ValueError("Joint effort limits must be nonnegative or positive infinity.")
        current_bounded = self._built_effort_driven & np.isfinite(effort_limit)
        changed = np.flatnonzero(current_bounded != self._built_effort_bounded)
        if changed.size > 0:
            dof = int(changed[0])
            raise RuntimeError(
                f"Changing joint-effort constraint topology for DOF {dof} is not supported; "
                "recreate SolverKamino to apply a finite-to-infinite or infinite-to-finite effort-limit change."
            )

    def _ensure_dual_impulse_state(
        self,
        state: State,
    ) -> tuple[wp.array[wp.spatial_vector] | None, wp.array[wp.vec3] | None]:
        model = self._require_newton_model()

        def ensure(name: str, count: int, dtype: type) -> wp.array | None:
            if count == 0:
                return None
            value = getattr(state, name, None)
            if value is None:
                value = wp.zeros(count, dtype=dtype, device=model.device)
                setattr(state, name, value)
            if not isinstance(value, wp.array) or value.shape != (count,) or value.dtype != dtype:
                raise ValueError(f"State.{name} must have shape ({count},) and dtype {dtype}.")
            if value.device != model.device:
                raise ValueError(f"State.{name} must be allocated on {model.device}, found {value.device}.")
            return value

        return (
            ensure("body_lox_dual_impulse", model.body_count, wp.spatial_vector),
            ensure("particle_lox_dual_impulse", model.particle_count, wp.vec3),
        )

    def prepare_newton_step(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
    ) -> None:
        """Import Newton warm starts and retain deformable step inputs."""
        solver = self._require_newton_solver()
        body_in, particle_in = self._ensure_dual_impulse_state(state_in)
        self._ensure_dual_impulse_state(state_out)
        self._newton_state_in = state_in
        self._newton_state_out = state_out
        self._newton_contacts = contacts
        if self._newton_body_pose is not None:
            if state_in.body_q is None:
                raise ValueError("The LOX deformable path requires Newton body-origin poses.")
            wp.copy(self._newton_body_pose, state_in.body_q)
        solver.load_state_dual_impulses(body_in, particle_in)

    def begin_newton_deformable_time_step(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
    ) -> None:
        """Prepare deformable coupling from retained Newton inputs."""
        solver = self._require_newton_solver()
        if solver.deformable_system is None:
            return
        state_in = self._newton_state_in
        if state_in is None:
            raise ValueError("The LOX deformable path requires the original Newton input state.")
        constraints_config = self._constraints_config
        lox_config = self._lox_config
        solver.begin_deformable_time_step(
            state_in,
            self._newton_contacts,
            time_step,
            inverse_time_step,
            contact_stabilization_fraction=constraints_config.gamma,
            contact_dead_zone=constraints_config.delta,
            impact_velocity_threshold=lox_config.impact_velocity_threshold,
            contact_recoverable_response=lox_config.contact_recoverable_response,
            body_pose=self._newton_body_pose,
        )

    def finish_newton_step(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
    ) -> None:
        """Write deformable state and persistent impulses to Newton output."""
        solver = self._require_newton_solver()
        state_out = self._newton_state_out
        if solver.deformable_system is not None:
            if state_out is None:
                raise ValueError("The LOX deformable path requires the original Newton output state.")
            solver.write_deformable_output(state_out, time_step, inverse_time_step)
        if state_out is not None:
            body_out, particle_out = self._ensure_dual_impulse_state(state_out)
            solver.write_state_dual_impulses(body_out, particle_out)

    def reset_newton_state(
        self,
        state: State,
        state_flags: int,
        world_mask: wp.array[wp.bool] | None,
    ) -> None:
        """Reset selected Newton arrays and LOX impulse history."""
        from ......sim import StateFlags  # noqa: PLC0415

        model = self._require_newton_model()
        body_dual_impulse, particle_dual_impulse = self._ensure_dual_impulse_state(state)
        if model.particle_count > 0:
            arrays = {
                "particle_q": state.particle_q,
                "particle_qd": state.particle_qd,
                "particle_f": state.particle_f,
            }
            expected_shape = (model.particle_count,)
            for name, value in arrays.items():
                if value is None or value.shape != expected_shape:
                    raise ValueError(f"LOX deformable reset requires State.{name} with shape {expected_shape}.")
                if value.device != model.device:
                    raise ValueError(
                        f"LOX deformable reset expected State.{name} on {model.device}, found {value.device}."
                    )

            reset_q = bool(state_flags & int(StateFlags.PARTICLE_Q))
            reset_qd = bool(state_flags & int(StateFlags.PARTICLE_QD))
            reset_f = bool(state_flags & int(StateFlags.PARTICLE_F))
            if world_mask is None:
                if reset_q:
                    wp.copy(state.particle_q, model.particle_q)
                if reset_qd:
                    wp.copy(state.particle_qd, model.particle_qd)
                    particle_dual_impulse.zero_()
                if reset_f:
                    state.particle_f.zero_()
            else:
                self._validate_world_mask(world_mask, "LOX deformable reset")
                wp.launch(
                    _reset_newton_particle_state,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_world,
                        world_mask,
                        model.particle_q,
                        model.particle_qd,
                        reset_q,
                        reset_qd,
                        reset_f,
                    ],
                    outputs=[state.particle_q, state.particle_qd, state.particle_f, particle_dual_impulse],
                    device=model.device,
                )

        if body_dual_impulse is None or not (state_flags & int(StateFlags.BODY_QD)):
            return
        if world_mask is None:
            body_dual_impulse.zero_()
        else:
            self._validate_world_mask(world_mask, "LOX reset")
            wp.launch(
                _reset_newton_body_dual_impulse,
                dim=model.body_count,
                inputs=[model.body_world, world_mask],
                outputs=[body_dual_impulse],
                device=model.device,
            )

    def _require_newton_model(self) -> Model:
        if self._newton_model is None:
            raise RuntimeError("LOXProblem is not attached to a Newton model.")
        return self._newton_model

    def _require_newton_solver(self) -> LOXSolver:
        if self._newton_solver is None:
            raise RuntimeError("LOXProblem is not attached to a LOX solver.")
        return self._newton_solver

    def _validate_world_mask(self, world_mask: wp.array[wp.bool], operation: str) -> None:
        model = self._require_newton_model()
        if world_mask.shape != (model.world_count,) or world_mask.dtype != wp.bool:
            raise ValueError(f"{operation} world_mask must have shape ({model.world_count},) and dtype bool.")
        if world_mask.device != model.device:
            raise ValueError(f"{operation} expected world_mask on {model.device}, found {world_mask.device}.")


class LOXSolver:
    """Run a fixed LOX splitting solve on one frozen linearization.

    The caller owns collision detection, Jacobian construction, pose
    integration, and nonlinear relinearization. This class owns the smooth
    system update, weight construction, blocked dense LLT solve, unilateral
    projections, convergence freezing, and output conversion for one such
    linearization. Optional cloth uses a separately assembled Warp BSR system
    and shares the same per-world iteration and acceptance state. A nonfailed
    world that reaches the iteration limit is accepted using its last projected
    iterate, matching the rigid LOX behavior.
    """

    @classmethod
    def from_config(
        cls,
        adapter: LOXKaminoAdapter | None,
        config: LOXSolverConfig,
        deformable_model: Model,
    ) -> LOXSolver:
        """Construct a solver from the Kamino LOX configuration."""
        return cls(
            adapter=adapter,
            max_iterations=config.max_iterations,
            use_graph_conditionals=config.use_graph_conditionals,
            projection_iterations=config.projection_iterations,
            projection_method=config.projection_method,
            gauss_seidel_max_colors=config.gauss_seidel_max_colors,
            inertial_warmstart_fraction=config.inertial_warmstart_fraction,
            position_tolerance=config.position_tolerance,
            rotation_tolerance=config.rotation_tolerance,
            velocity_tolerance=config.velocity_tolerance,
            weight_sigma=config.weight_sigma,
            weight_beta=config.weight_beta,
            deformable_weight_beta=config.deformable_weight_beta,
            selective_weights=config.selective_weights,
            joint_penalty_scale=config.joint_penalty_scale,
            joint_multiplier_projected_fraction=config.joint_multiplier_projected_fraction,
            joint_warmstart_factor=config.joint_warmstart_factor,
            deformable_model=deformable_model,
            deformable_cr_iterations=config.deformable_cr_iterations,
            deformable_direct_max_particles=config.deformable_direct_max_particles,
            deformable_proximal_iterations=config.deformable_proximal_iterations,
            deformable_proximal_relaxation=config.deformable_proximal_relaxation,
            deformable_preconditioner=config.deformable_preconditioner,
            deformable_preconditioner_fill_level=config.deformable_preconditioner_fill_level,
            deformable_hessian_regularization=config.deformable_hessian_regularization,
            deformable_enable_self_contact=config.deformable_enable_self_contact,
            deformable_enable_normal_cone_filtering=config.deformable_enable_normal_cone_filtering,
            deformable_enable_rigid_contact_normal_cone_filtering=(
                config.deformable_enable_rigid_contact_normal_cone_filtering
            ),
            deformable_normal_cone_filtering_min_distance=config.deformable_normal_cone_filtering_min_distance,
            deformable_self_contact_margin=config.deformable_self_contact_margin,
            deformable_self_contact_gap=config.deformable_self_contact_gap,
            deformable_self_contact_vertex_buffer_size=config.deformable_self_contact_vertex_buffer_size,
            deformable_self_contact_edge_buffer_size=config.deformable_self_contact_edge_buffer_size,
            deformable_self_contact_topological_filter_threshold=(
                config.deformable_self_contact_topological_filter_threshold
            ),
            deformable_self_contact_rest_exclusion_radius=config.deformable_self_contact_rest_exclusion_radius,
            deformable_self_contact_edge_parallel_epsilon=config.deformable_self_contact_edge_parallel_epsilon,
            deformable_enable_penetration_free_contact=config.deformable_enable_penetration_free_contact,
            deformable_penetration_free_contact_relaxation=(config.deformable_penetration_free_contact_relaxation),
            _joint_solve_mode=int(config.joint_solve_direct),
        )

    def __init__(
        self,
        adapter: LOXKaminoAdapter | None,
        max_iterations: int = 25,
        use_graph_conditionals: bool = True,
        projection_iterations: int = 3,
        projection_method: str = "jacobi",
        gauss_seidel_max_colors: int = 0,
        inertial_warmstart_fraction: float = 0.0,
        position_tolerance: float = 1.0e-5,
        rotation_tolerance: float = 1.0e-5,
        velocity_tolerance: float = 1.0e-5,
        weight_sigma: float = BODY_WEIGHT_SIGMA_DEFAULT,
        weight_beta: float = BODY_WEIGHT_BETA_DEFAULT,
        deformable_weight_beta: float = DEFORMABLE_WEIGHT_BETA_DEFAULT,
        selective_weights: bool = False,
        joint_penalty_scale: float = 10.0,
        joint_multiplier_projected_fraction: float = 1.0,
        joint_warmstart_factor: float = 0.5,
        deformable_model=None,
        deformable_cr_iterations: int = 4,
        deformable_direct_max_particles: int = 128,
        deformable_proximal_iterations: int = 1,
        deformable_proximal_relaxation: float = 1.0,
        deformable_preconditioner: str = "incomplete_ldlt",
        deformable_preconditioner_fill_level: int = 0,
        deformable_hessian_regularization: float = 1.0e-6,
        deformable_enable_self_contact: bool = False,
        deformable_enable_normal_cone_filtering: bool = True,
        deformable_enable_rigid_contact_normal_cone_filtering: bool = False,
        deformable_normal_cone_filtering_min_distance: float = 1.0e-4,
        deformable_self_contact_margin: float = 0.2,
        deformable_self_contact_gap: float = 0.0,
        deformable_self_contact_vertex_buffer_size: int = 32,
        deformable_self_contact_edge_buffer_size: int = 64,
        deformable_self_contact_topological_filter_threshold: int = 2,
        deformable_self_contact_rest_exclusion_radius: float = 0.0,
        deformable_self_contact_edge_parallel_epsilon: float = 1.0e-5,
        deformable_enable_penetration_free_contact: bool = False,
        deformable_penetration_free_contact_relaxation: float = 0.85,
        _body_weight_mode: int = _BODY_WEIGHT_MASS_PROPORTIONAL,
        _joint_metric_mode: int = _JOINT_METRIC_SIMPLE,
        _joint_solve_mode: int = _JOINT_SOLVE_ALM,
    ):
        _validate_fixed_iteration_count("max_iterations", max_iterations)
        _validate_fixed_iteration_count("projection_iterations", projection_iterations)
        if not isinstance(use_graph_conditionals, bool):
            raise ValueError("use_graph_conditionals must be a bool.")
        if projection_method not in ("jacobi", "gauss_seidel", "apgd", "avbd"):
            raise ValueError("projection_method must be 'jacobi', 'gauss_seidel', 'apgd', or 'avbd'.")
        if (
            not isinstance(gauss_seidel_max_colors, int)
            or isinstance(gauss_seidel_max_colors, bool)
            or gauss_seidel_max_colors < 0
        ):
            raise ValueError("gauss_seidel_max_colors must be a non-negative integer.")
        if not math.isfinite(inertial_warmstart_fraction) or not 0.0 <= inertial_warmstart_fraction <= 1.0:
            raise ValueError("inertial_warmstart_fraction must be finite and in [0, 1].")
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
        if not math.isfinite(deformable_weight_beta) or deformable_weight_beta < 1.0:
            raise ValueError("deformable_weight_beta must be finite and at least one.")
        if not isinstance(selective_weights, bool):
            raise ValueError("selective_weights must be a bool.")
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
        if adapter is None and deformable_model is None:
            raise ValueError("LOX requires a rigid adapter, a deformable model, or both.")
        if (
            not isinstance(deformable_direct_max_particles, int)
            or isinstance(deformable_direct_max_particles, bool)
            or deformable_direct_max_particles < 0
        ):
            raise ValueError("deformable_direct_max_particles must be a non-negative integer.")
        if (
            not isinstance(deformable_proximal_iterations, int)
            or isinstance(deformable_proximal_iterations, bool)
            or deformable_proximal_iterations < 0
        ):
            raise ValueError("deformable_proximal_iterations must be a non-negative integer.")
        if not math.isfinite(deformable_proximal_relaxation) or not 0.0 <= deformable_proximal_relaxation <= 1.0:
            raise ValueError("deformable_proximal_relaxation must be finite and in [0, 1].")
        if deformable_preconditioner not in ("incomplete_ldlt", "two_level", "block_jacobi", "jacobi"):
            raise ValueError(
                "deformable_preconditioner must be 'incomplete_ldlt', 'two_level', 'block_jacobi', or 'jacobi'."
            )
        if (
            not isinstance(deformable_preconditioner_fill_level, int)
            or isinstance(deformable_preconditioner_fill_level, bool)
            or deformable_preconditioner_fill_level < 0
        ):
            raise ValueError("LOX deformable incomplete-factor fill level must be a non-negative integer.")
        if not isinstance(deformable_enable_penetration_free_contact, bool):
            raise ValueError("deformable_enable_penetration_free_contact must be a bool.")
        if not isinstance(deformable_enable_normal_cone_filtering, bool):
            raise ValueError("deformable_enable_normal_cone_filtering must be a bool.")
        if not isinstance(deformable_enable_rigid_contact_normal_cone_filtering, bool):
            raise ValueError("deformable_enable_rigid_contact_normal_cone_filtering must be a bool.")
        if (
            not math.isfinite(deformable_normal_cone_filtering_min_distance)
            or deformable_normal_cone_filtering_min_distance < 0.0
        ):
            raise ValueError("deformable_normal_cone_filtering_min_distance must be finite and non-negative.")
        if (
            not math.isfinite(deformable_penetration_free_contact_relaxation)
            or not 0.0 < deformable_penetration_free_contact_relaxation <= 1.0
        ):
            raise ValueError("deformable_penetration_free_contact_relaxation must be finite and in (0, 1].")
        self.adapter = adapter
        self.device = adapter.device if adapter is not None else deformable_model.device
        self.coulomb_solve_statistics: CoulombSolveStatistics | None = None
        self.num_worlds = adapter.num_worlds if adapter is not None else int(deformable_model.world_count)
        if deformable_model is not None and int(deformable_model.world_count) != self.num_worlds:
            raise ValueError("Rigid and deformable LOX systems must contain the same number of worlds.")
        self.has_rigid = adapter is not None
        self.max_iterations = max_iterations
        self.use_graph_conditionals = use_graph_conditionals
        self.projection_iterations = projection_iterations
        self.projection_method = projection_method
        self.gauss_seidel_max_colors = gauss_seidel_max_colors
        self._colored_gauss_seidel = None
        self.inertial_warmstart_fraction = inertial_warmstart_fraction
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.weight_sigma = weight_sigma
        self.weight_beta = weight_beta
        self.selective_weights = selective_weights
        self.joint_penalty_scale = wp.full(self.num_worlds, joint_penalty_scale, dtype=wp.float32, device=self.device)
        self.joint_multiplier_projected_fraction = joint_multiplier_projected_fraction
        self.joint_warmstart_factor = joint_warmstart_factor
        self._body_weight_mode = _body_weight_mode
        self._joint_metric_mode = _joint_metric_mode
        if adapter is not None and adapter.has_massless_dynamic_body and _joint_solve_mode == _JOINT_SOLVE_SCHUR_DIRECT:
            _joint_solve_mode = _JOINT_SOLVE_ALM
        self._joint_solve_mode = _joint_solve_mode

        self.system = adapter.system if adapter is not None else None
        if self.system is not None:
            self.system.selective_body_weights = selective_weights
        self.splitting = adapter.splitting if adapter is not None else None
        self.projected_twist = (
            self.splitting.projected_twist
            if self.splitting is not None
            else wp.empty(0, dtype=vec6f, device=self.device)
        )
        self.world_active = (
            self.splitting.world_active
            if self.splitting is not None
            else wp.ones(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.world_converged = (
            self.splitting.world_converged
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.world_failed = (
            self.splitting.world_failed
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.world_iteration_limit = (
            self.splitting.world_iteration_limit
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.iteration_count = (
            self.splitting.iteration_count
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        )
        self._cloth_only_residual_total = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.world_accepted = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_status = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self._iteration_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.contact_residual_max = (
            adapter.world_contact_residual_max
            if adapter is not None
            else wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        )
        self.limit_residual_max = (
            adapter.world_limit_residual_max
            if adapter is not None
            else wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        )
        self.friction_residual_max = (
            adapter.world_friction_residual_max
            if adapter is not None
            else wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        )
        rigid_body_count = adapter.model.size.sum_of_num_bodies if adapter is not None else 0
        self._initial_twist = wp.zeros(rigid_body_count, dtype=vec6f, device=self.device)
        self._apgd_body_baseline = wp.zeros(rigid_body_count, dtype=vec6f, device=self.device)
        self._avbd_body_baseline = wp.zeros(rigid_body_count, dtype=vec6f, device=self.device)
        self._apgd_theta = wp.ones(self.num_worlds, dtype=wp.float32, device=self.device)
        self._apgd_beta = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self._apgd_restart_dot = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.deformable_system = None
        self.deformable_splitting = None
        self.deformable_self_contact_detector = None
        self.deformable_penetration_free_limiter = None
        self._deformable_self_contacts_enabled = deformable_enable_self_contact
        self._deformable_normal_cone_filtering_enabled = deformable_enable_normal_cone_filtering
        self._deformable_rigid_contact_normal_cone_filtering_enabled = (
            deformable_enable_rigid_contact_normal_cone_filtering
        )
        self._deformable_normal_cone_filtering_min_distance = float(deformable_normal_cone_filtering_min_distance)
        if deformable_model is not None and deformable_model.particle_count > 0:
            self.deformable_system = DeformableFEMSystem(
                deformable_model,
                cr_iterations=deformable_cr_iterations,
                direct_max_particles=deformable_direct_max_particles,
                weight_sigma=weight_sigma,
                weight_beta=deformable_weight_beta,
                preconditioner=deformable_preconditioner,
                preconditioner_regularization=deformable_hessian_regularization,
                preconditioner_fill_level=deformable_preconditioner_fill_level,
                proximal_iterations=deformable_proximal_iterations,
                proximal_relaxation=deformable_proximal_relaxation,
            )
            self.deformable_system.selective_consensus = selective_weights
            self.deformable_splitting = DeformableSplittingState(self.deformable_system)
            self._apgd_particle_baseline = wp.zeros(
                self.deformable_system.particle_count,
                dtype=wp.vec3,
                device=self.device,
            )
            self._avbd_particle_baseline = wp.zeros(
                self.deformable_system.particle_count,
                dtype=wp.vec3,
                device=self.device,
            )
            if deformable_enable_self_contact and deformable_model.tri_count < 1:
                raise ValueError("LOX deformable self-contact requires a triangulated surface.")
            if deformable_model.tri_count > 0 and (
                deformable_enable_self_contact or deformable_enable_penetration_free_contact
            ):
                self.deformable_self_contact_detector = DeformableSelfContactDetector(
                    deformable_model,
                    margin=deformable_self_contact_margin,
                    gap=deformable_self_contact_gap,
                    vertex_contact_buffer_size=deformable_self_contact_vertex_buffer_size,
                    edge_contact_buffer_size=deformable_self_contact_edge_buffer_size,
                    topological_contact_filter_threshold=deformable_self_contact_topological_filter_threshold,
                    rest_contact_exclusion_radius=deformable_self_contact_rest_exclusion_radius,
                    edge_parallel_epsilon=deformable_self_contact_edge_parallel_epsilon,
                    enable_normal_cone_filtering=deformable_enable_normal_cone_filtering,
                    normal_cone_filtering_min_distance=deformable_normal_cone_filtering_min_distance,
                )
            if deformable_enable_penetration_free_contact and self.deformable_self_contact_detector is not None:
                # This limiter can truncate every dynamic node, not only nodes in current contact stencils.
                self.deformable_system.selective_consensus = False
                self.deformable_penetration_free_limiter = DeformablePenetrationFreeLimiter(
                    self.deformable_self_contact_detector,
                    self.deformable_system.topology,
                    self.deformable_system.inverse_weight,
                    deformable_penetration_free_contact_relaxation,
                )
        use_parallel_candidates = self.device.is_cuda and adapter is not None and self.deformable_system is not None
        self._deformable_stream = wp.Stream(self.device) if use_parallel_candidates else None
        self._deformable_start_event = wp.Event(self.device) if use_parallel_candidates else None
        self._deformable_complete_event = wp.Event(self.device) if use_parallel_candidates else None
        self.deformable_contacts = None
        self._deformable_contacts_active = False
        self._deformable_prepared = False
        self.structural_joint_solver: BatchedStructuralJointSolver | None = None
        # The Schur path factors unconstrained body inertia before applying joints,
        # so a massless dynamic frame requires the maximal-coordinate ALM path.
        if (
            adapter is not None
            and self._joint_solve_mode == _JOINT_SOLVE_SCHUR_DIRECT
            and adapter.structural_row_count > 0
            and not adapter.has_massless_dynamic_body
        ):
            self.structural_joint_solver = self._make_structural_joint_solver()
        self.has_bounded_effort = adapter is not None and adapter.has_bounded_effort
        self._time_step: wp.array[wp.float32] | None = None
        self._inverse_time_step: wp.array[wp.float32] | None = None
        self._time_step_prepared = False
        self._coldstart_requested = False
        if self.has_bounded_effort:
            self._prepare_body_space_candidate = self._prepare_body_space_candidate_with_effort
            self._finish_body_space_iteration = self._finish_body_space_iteration_with_effort

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
            prescribed_twist=adapter.body_velocity_begin,
        )

    def joint_penalty_scale_seed(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
    ) -> list[float]:
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
        validate_world_time_steps(time_step, inverse_time_step, self.num_worlds, self.device)
        if self._joint_solve_mode != _JOINT_SOLVE_ALM:
            raise ValueError("joint penalty scale seeding is only available for the structural ALM solve.")

        adapter = self.adapter
        if adapter.structural_row_count == 0:
            return self.joint_penalty_scale.numpy().tolist()

        self.reset()
        adapter.begin_time_step(time_step, inverse_time_step)
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

    def reset(
        self,
        problem: LOXProblem | None = None,
        world_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Clear structural/splitting warm starts and per-world diagnostics."""
        del problem
        if world_mask is not None:
            if world_mask.shape != (self.num_worlds,) or world_mask.dtype != wp.bool:
                raise ValueError(f"world_mask must have shape ({self.num_worlds},) and dtype bool.")
            if world_mask.device != self.device:
                raise ValueError(f"world_mask must be allocated on {self.device}, found {world_mask.device}.")
        if self.coulomb_solve_statistics is not None:
            self.coulomb_solve_statistics.reset()
        if self.splitting is not None:
            self.splitting.reset(world_mask=world_mask)
            self.adapter.reset_structural_multipliers(world_mask=world_mask)
            if self.has_bounded_effort:
                self.adapter.reset_effort_counters(world_mask=world_mask)
            self.adapter.cables.reset()
            self.adapter.projection_status.zero_()
            self.adapter.contact_residual.zero_()
            self.adapter.limit_residual.zero_()
            self.adapter.friction_residual.zero_()
            self.adapter.reset_friction_reactions(world_mask=world_mask)
        elif world_mask is None:
            self.world_active.fill_(True)
            self.world_converged.zero_()
            self.world_failed.zero_()
            self.world_iteration_limit.zero_()
            self.iteration_count.zero_()
            self._cloth_only_residual_total.zero_()
        if self.deformable_splitting is not None:
            self.deformable_splitting.reset(world_mask=world_mask)
        if self.deformable_contacts is not None:
            self.deformable_contacts.reset()
        if world_mask is None:
            self.world_accepted.zero_()
            self.world_status.fill_(LOX_STATUS_ACTIVE)
            self.contact_residual_max.zero_()
            self.limit_residual_max.zero_()
            self.friction_residual_max.zero_()
        else:
            wp.launch(
                _reset_solver_worlds_masked,
                dim=self.num_worlds,
                inputs=[world_mask],
                outputs=[
                    self.world_active,
                    self.world_converged,
                    self.world_failed,
                    self.world_iteration_limit,
                    self.iteration_count,
                    self._cloth_only_residual_total,
                    self.world_accepted,
                    self.world_status,
                    self.contact_residual_max,
                    self.limit_residual_max,
                    self.friction_residual_max,
                ],
                device=self.device,
            )
        self._initial_twist.zero_()
        self._deformable_prepared = False
        self._time_step_prepared = False
        self._coldstart_requested = False

    def coldstart(self) -> None:
        """Clear persistent state before the next solve."""
        self.reset()
        self._coldstart_requested = True

    def warmstart(
        self,
        problem: LOXProblem,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ) -> None:
        """Prepare one LOX time step from Kamino's warm-started containers."""
        del data, limits, contacts
        self._begin_problem(problem, model.time.dt, model.time.inv_dt, reset_dual=False)

    def _begin_problem(
        self,
        problem: LOXProblem,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        reset_dual: bool,
    ) -> None:
        """Initialize a LOX problem using its time-step configuration."""
        if not isinstance(problem, LOXProblem):
            raise TypeError(f"Expected a LOXProblem, got {type(problem)}.")
        self.begin_time_step(
            time_step,
            inverse_time_step,
            limit_stabilization_fraction=problem.limit_stabilization_fraction,
            contact_stabilization_fraction=problem.contact_stabilization_fraction,
            contact_dead_zone=problem.contact_dead_zone,
            impact_velocity_threshold=problem.impact_velocity_threshold,
            contact_recoverable_response=problem.contact_recoverable_response,
            reset_dual=reset_dual,
        )
        self._time_step_prepared = True
        self._coldstart_requested = False

    def enable_coulomb_solve_statistics(self) -> CoulombSolveStatistics:
        """Enable and clear local Coulomb-solve instrumentation."""
        if self.coulomb_solve_statistics is None:
            self.coulomb_solve_statistics = CoulombSolveStatistics(self.device)
        else:
            self.coulomb_solve_statistics.reset()
        return self.coulomb_solve_statistics

    def load_state_dual_impulses(
        self,
        body_dual_impulse: wp.array[wp.spatial_vector] | None,
        particle_dual_impulse: wp.array[wp.vec3] | None,
    ) -> None:
        """Load consensus impulse warm starts from the input Newton state."""
        if self.splitting is not None:
            if body_dual_impulse is None:
                raise ValueError("LOX rigid bodies require State.body_lox_dual_impulse.")
            if body_dual_impulse.shape != (self.adapter.model.size.sum_of_num_bodies,):
                raise ValueError("State.body_lox_dual_impulse must contain one entry per body.")
            if body_dual_impulse.dtype != wp.spatial_vectorf or body_dual_impulse.device != self.device:
                raise ValueError("State.body_lox_dual_impulse has an incompatible dtype or device.")
            wp.copy(self.splitting.splitting_dual_impulse, body_dual_impulse.view(dtype=vec6f))
        if self.deformable_splitting is not None:
            if particle_dual_impulse is None:
                raise ValueError("LOX deformables require State.particle_lox_dual_impulse.")
            self.deformable_splitting.load_state_dual_impulse(particle_dual_impulse)

    def write_state_dual_impulses(
        self,
        body_dual_impulse: wp.array[wp.spatial_vector] | None,
        particle_dual_impulse: wp.array[wp.vec3] | None,
    ) -> None:
        """Write consensus impulse warm starts to the output Newton state."""
        if self.splitting is not None:
            if body_dual_impulse is None:
                raise ValueError("LOX rigid bodies require State.body_lox_dual_impulse.")
            if body_dual_impulse.shape != (self.adapter.model.size.sum_of_num_bodies,):
                raise ValueError("State.body_lox_dual_impulse must contain one entry per body.")
            if body_dual_impulse.dtype != wp.spatial_vectorf or body_dual_impulse.device != self.device:
                raise ValueError("State.body_lox_dual_impulse has an incompatible dtype or device.")
            wp.copy(body_dual_impulse.view(dtype=vec6f), self.splitting.splitting_dual_impulse)
        if self.deformable_splitting is not None:
            if particle_dual_impulse is None:
                raise ValueError("LOX deformables require State.particle_lox_dual_impulse.")
            self.deformable_splitting.write_state_dual_impulse(particle_dual_impulse)

    def begin_time_step(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        limit_stabilization_fraction: float = 0.01,
        contact_stabilization_fraction: float = 0.01,
        contact_dead_zone: float = 1.0e-6,
        impact_velocity_threshold: float = 1.0e-3,
        contact_recoverable_response: bool = False,
        reset_dual: bool = False,
    ) -> None:
        """Freeze inertial velocities and import damped constraint warm starts.

        The body consensus warm start is loaded from the Newton input state as
        the generalized impulse ``W u`` and converted back to the scaled dual
        after the solve builds its current body weight.
        """
        validate_world_time_steps(time_step, inverse_time_step, self.num_worlds, self.device)
        self._time_step = time_step
        self._inverse_time_step = inverse_time_step
        if self.adapter is not None:
            self.adapter.scale_structural_multipliers(self.joint_warmstart_factor)
            self.adapter.begin_time_step(
                time_step,
                inverse_time_step,
                limit_stabilization_fraction=limit_stabilization_fraction,
                contact_stabilization_fraction=contact_stabilization_fraction,
                contact_dead_zone=contact_dead_zone,
                impact_velocity_threshold=impact_velocity_threshold,
                contact_recoverable_response=contact_recoverable_response,
            )
            if self._deformable_contacts_active:
                self.deformable_contacts.accumulate_rigid_incidence(
                    self.adapter.body_constraint_count,
                    self.adapter.body_has_unilateral,
                    self.adapter.world_has_unilateral,
                )
            wp.launch(
                _initialize_body_velocity_guess,
                dim=self.adapter.model.size.sum_of_num_bodies,
                inputs=[
                    self.system.body_block,
                    self.adapter.model.bodies.wid,
                    self.adapter.model.bodies.inv_m_i,
                    self.adapter.data.bodies.inv_I_i,
                    self.adapter.body_velocity_begin,
                    self.adapter.data.bodies.w_e_i,
                    self.adapter.model.gravity.vector,
                    time_step,
                    self.inertial_warmstart_fraction,
                ],
                outputs=[self._initial_twist],
                device=self.device,
            )
            self.splitting.begin(self._initial_twist, reset_dual=reset_dual)
        else:
            self.world_active.fill_(True)
            self.world_converged.zero_()
            self.world_failed.zero_()
            self.world_iteration_limit.zero_()
            self.iteration_count.zero_()
            self._cloth_only_residual_total.zero_()
        self.world_accepted.zero_()
        self.world_status.fill_(LOX_STATUS_ACTIVE)
        self._time_step_prepared = True
        self._coldstart_requested = False

    def begin_deformable_time_step(
        self,
        state,
        contacts,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        contact_stabilization_fraction: float = 0.01,
        contact_dead_zone: float = 1.0e-6,
        impact_velocity_threshold: float = 1.0e-3,
        contact_recoverable_response: bool = False,
        body_pose: wp.array[wp.transform] | None = None,
    ) -> None:
        """Assemble and factor the first deformable linearization for the step."""
        if self.deformable_system is None or self.deformable_splitting is None:
            return
        validate_world_time_steps(time_step, inverse_time_step, self.num_worlds, self.device)
        # Contact preparation needs the broad weights to discover and size the
        # support. Factor only after that support has been frozen for this step.
        self.deformable_system.assemble(state, time_step, finalize_consensus=False)
        rigid_contact_capacity = contacts.soft_contact_max if contacts is not None else 0
        self_contact_capacity = (
            self.deformable_self_contact_detector.capacity
            if self._deformable_self_contacts_enabled and self.deformable_self_contact_detector is not None
            else 0
        )
        self._deformable_contacts_active = rigid_contact_capacity + self_contact_capacity > 0
        if self.deformable_self_contact_detector is not None:
            self.deformable_self_contact_detector.detect(state.particle_q)
        if self.deformable_penetration_free_limiter is not None:
            self.deformable_penetration_free_limiter.begin_time_step(state.particle_q)
        if self._deformable_contacts_active:
            capacities_changed = self.deformable_contacts is not None and (
                self.deformable_contacts.rigid_contact_capacity != rigid_contact_capacity
                or self.deformable_contacts.self_contact_capacity != self_contact_capacity
            )
            if self.deformable_contacts is None or capacities_changed:
                if self.device.is_cuda and self.device.is_capturing and not self.device.is_mempool_enabled:
                    raise RuntimeError(
                        "LOX deformable contact storage cannot be initialized during CUDA capture without "
                        "a stream-ordered memory pool. Enable the Warp memory pool or run one uncaptured step "
                        "before capture."
                    )
                self.deformable_contacts = DeformableContactSystem(
                    self.deformable_system.model,
                    self.deformable_system,
                    contact_capacity=rigid_contact_capacity,
                    self_contact_capacity=self_contact_capacity,
                    stabilization_fraction=contact_stabilization_fraction,
                    dead_zone=contact_dead_zone,
                    impact_velocity_threshold=impact_velocity_threshold,
                    recoverable_response=contact_recoverable_response,
                    enable_rigid_normal_cone_filtering=(self._deformable_rigid_contact_normal_cone_filtering_enabled),
                    normal_cone_filtering_min_distance=self._deformable_normal_cone_filtering_min_distance,
                )
            self.deformable_contacts.prepare(
                contacts,
                state,
                time_step,
                self_contact_detector=(
                    self.deformable_self_contact_detector if self._deformable_self_contacts_enabled else None
                ),
                body_pose=body_pose,
            )
            self.deformable_system.set_unilateral_incidence(self.deformable_contacts.particle_multiplicity)
            self.deformable_contacts.update_weight_metric()
            if self.projection_method == "gauss_seidel" and self.adapter is None:
                if self.gauss_seidel_max_colors == 0:
                    self.deformable_contacts.prepare_gauss_seidel_projection()
                elif self.gauss_seidel_max_colors > 1:
                    self._prepare_colored_gauss_seidel_projection()
            elif self.projection_method == "apgd" and self.adapter is None:
                self.deformable_contacts.prepare_apgd_projection(rigid_coordinates=False)
        else:
            self.deformable_system.set_unilateral_incidence(None)
        self.deformable_splitting.begin_time_step(time_step, self.inertial_warmstart_fraction)
        wp.copy(self.deformable_system.smooth_velocity, self.deformable_splitting.projected_velocity)
        self._deformable_prepared = True

    def prepare_deformable_nonlinear_iteration(
        self,
        time_step: wp.array[wp.float32],
        body_pose: wp.array[wp.transform] | None = None,
        body_velocity_begin: wp.array[vec6f] | None = None,
    ) -> None:
        """Reassemble cloth around the last accepted outer iterate."""
        if self.deformable_system is None or self.deformable_splitting is None:
            return
        wp.copy(
            self.deformable_splitting.projected_velocity,
            self.deformable_splitting.accepted_velocity,
        )
        self.deformable_system.reassemble(
            self.deformable_splitting.accepted_velocity,
            time_step,
        )
        self.deformable_splitting.begin(time_step)
        if self._deformable_contacts_active and self.deformable_contacts is not None:
            self.deformable_contacts.refresh_geometry(
                time_step,
                body_pose,
                body_velocity_begin,
                body_pose_is_center_of_mass=self.adapter is not None,
            )
            self.deformable_contacts.update_weight_metric()
            if self.projection_method == "gauss_seidel" and self.adapter is None:
                if self.gauss_seidel_max_colors == 0:
                    self.deformable_contacts.prepare_gauss_seidel_projection()
                elif self.gauss_seidel_max_colors > 1:
                    self._prepare_colored_gauss_seidel_projection()
            elif self.projection_method == "apgd" and self.adapter is None:
                self.deformable_contacts.prepare_apgd_projection(rigid_coordinates=False)
        self._deformable_prepared = True

    def _prepare_colored_gauss_seidel_projection(self) -> None:
        deformable_contacts = self.deformable_contacts if self._deformable_contacts_active else None
        if self._colored_gauss_seidel is None or not self._colored_gauss_seidel.matches(deformable_contacts):
            self._colored_gauss_seidel = ColoredGaussSeidelProjection(
                self.adapter,
                deformable_contacts,
                self.gauss_seidel_max_colors,
            )
        if self.adapter is not None:
            prepared_status = self.adapter.world_jacobi_projection_status
            inverse_weight = self.system.inverse_weight
        else:
            prepared_status = self.deformable_contacts.sequential_projection_status
            inverse_weight = None
        self._colored_gauss_seidel.prepare(inverse_weight, prepared_status)

    def _prepare_body_space_projection(self) -> None:
        adapter = self.adapter
        if self.projection_method in ("jacobi", "apgd") or (
            self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors == 1
        ):
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
                adapter.limit_projection_delassus,
                adapter.world_jacobi_projection_status,
            )
            if self._deformable_contacts_active:
                self.deformable_contacts.prepare_rigid_projection(
                    adapter.body_constraint_count,
                    adapter.static_body_constraint_count,
                    self.system.inverse_weight,
                    adapter.world_jacobi_projection_status,
                )
            if self.projection_method == "apgd":
                if self._deformable_contacts_active:
                    self.deformable_contacts.prepare_apgd_projection(
                        rigid_coordinates=True,
                        prepared_status=adapter.world_jacobi_projection_status,
                    )
        elif self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors > 1:
            self._prepare_colored_gauss_seidel_projection()
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
                adapter.world_contact_projection_status,
            )
            if self._deformable_contacts_active:
                if self.projection_method == "gauss_seidel":
                    self.deformable_contacts.prepare_gauss_seidel_projection(
                        self.system.inverse_weight,
                        adapter.world_contact_projection_status,
                    )
            if self.projection_method == "avbd":
                prepare_constraints_avbd(adapter, self.system.inverse_weight)

    def _project_body_space_constraints(self) -> None:
        adapter = self.adapter
        splitting = self.splitting
        if self.projection_method == "jacobi" or (
            self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors == 1
        ):
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
                deformable_contacts=self.deformable_contacts if self._deformable_contacts_active else None,
                deformable_projected_velocity=(
                    self.deformable_splitting.projected_velocity if self._deformable_contacts_active else None
                ),
                coulomb_statistics=self.coulomb_solve_statistics,
            )
        elif self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors > 1:
            self._colored_gauss_seidel.project(
                self.projection_iterations,
                splitting.world_active,
                splitting.body_world,
                self.system.inverse_weight,
                splitting.projected_twist,
                adapter.projection_twist_delta,
                (self.deformable_splitting.projected_velocity if self._deformable_contacts_active else None),
                adapter.world_jacobi_projection_status,
                adapter.projection_status,
            )
        elif self.projection_method == "apgd":
            project_constraints_apgd(
                self.projection_iterations,
                adapter,
                splitting.world_active,
                splitting.body_world,
                self.system.inverse_weight,
                self._apgd_body_baseline,
                splitting.projected_twist,
                self._apgd_theta,
                self._apgd_beta,
                self._apgd_restart_dot,
                deformable_contacts=self.deformable_contacts if self._deformable_contacts_active else None,
                particle_baseline=(self._apgd_particle_baseline if self._deformable_contacts_active else None),
                projected_velocity=(
                    self.deformable_splitting.projected_velocity if self._deformable_contacts_active else None
                ),
            )
        elif self.projection_method == "avbd":
            project_constraints_avbd(
                self.projection_iterations,
                adapter,
                splitting.world_active,
                self.system.weight,
                self.system.inverse_weight,
                self._avbd_body_baseline,
                splitting.projected_twist,
                deformable_contacts=self.deformable_contacts if self._deformable_contacts_active else None,
                particle_baseline=(self._avbd_particle_baseline if self._deformable_contacts_active else None),
                projected_velocity=(
                    self.deformable_splitting.projected_velocity if self._deformable_contacts_active else None
                ),
            )
        elif not self._deformable_contacts_active:
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
                adapter.world_contact_projection_status,
            )
        elif self.projection_method == "gauss_seidel":
            warm_start_constraints_sequential(
                splitting.world_active,
                adapter.world_friction_offset,
                adapter.world_friction_count,
                adapter.friction_body_first,
                adapter.friction_body_second,
                adapter.friction_jacobian_first,
                adapter.friction_jacobian_second,
                adapter.world_contact_offset,
                adapter.world_contact_count,
                adapter.contact_body_first,
                adapter.contact_body_second,
                adapter.contact_jacobian_first,
                adapter.contact_jacobian_second,
                adapter.world_limit_offset,
                adapter.world_limit_count,
                adapter.limit_body_first,
                adapter.limit_body_second,
                adapter.limit_jacobian_first,
                adapter.limit_jacobian_second,
                self.system.inverse_weight,
                splitting.projected_twist,
                adapter.friction_reaction,
                adapter.contact_reaction,
                adapter.limit_reaction,
                adapter.world_contact_projection_status,
                adapter.projection_status,
            )
            self.deformable_contacts.warm_start_gauss_seidel(
                splitting.world_active,
                self.deformable_splitting.projected_velocity,
                self.system.inverse_weight,
                splitting.projected_twist,
                adapter.projection_status,
            )
            for _sweep in range(self.projection_iterations):
                sweep_constraints_sequential(
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
                    adapter.contact_projection_delassus,
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
                )
                self.deformable_contacts.sweep_gauss_seidel(
                    splitting.world_active,
                    self.deformable_splitting.projected_velocity,
                    self.system.inverse_weight,
                    splitting.projected_twist,
                    adapter.projection_status,
                )

    def _update_conditional_iteration(self) -> None:
        self._iteration_condition.zero_()
        wp.launch(
            _update_iteration_condition,
            dim=self.num_worlds,
            inputs=[
                self.max_iterations,
                self.world_active,
                self.iteration_count,
            ],
            outputs=[self._iteration_condition],
            device=self.device,
        )

    def _deformable_candidate_projection(self, time_step: wp.array[wp.float32]) -> None:
        deformable_system = self.deformable_system
        deformable_splitting = self.deformable_splitting
        if deformable_system is None or deformable_splitting is None:
            return
        deformable_splitting.copy_world_active(self.world_active)
        center = deformable_splitting.build_consensus_center()
        deformable_system.solve_candidate(center)
        deformable_system.update_proximal(time_step)
        deformable_splitting.prepare_projection(deformable_system.smooth_velocity)
        contact_system = self.deformable_contacts if self._deformable_contacts_active else None
        if contact_system is not None and self.adapter is None:
            if self.projection_method == "jacobi" or (
                self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors == 1
            ):
                contact_system.apply_reaction_warm_start(deformable_splitting.projected_velocity)
                contact_system.project(
                    deformable_splitting.projected_velocity,
                    iterations=self.projection_iterations,
                )
            elif self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors == 0:
                contact_system.warm_start_gauss_seidel(
                    self.world_active,
                    deformable_splitting.projected_velocity,
                )
                for _sweep in range(self.projection_iterations):
                    contact_system.sweep_gauss_seidel(
                        self.world_active,
                        deformable_splitting.projected_velocity,
                    )
            elif self.projection_method == "gauss_seidel":
                self._colored_gauss_seidel.project(
                    self.projection_iterations,
                    self.world_active,
                    None,
                    None,
                    None,
                    None,
                    deformable_splitting.projected_velocity,
                    contact_system.sequential_projection_status,
                    contact_system.sequential_projection_status,
                )
            elif self.projection_method == "apgd":
                project_deformable_constraints_apgd(
                    self.projection_iterations,
                    contact_system,
                    self.world_active,
                    self._apgd_particle_baseline,
                    deformable_splitting.projected_velocity,
                    self._apgd_theta,
                    self._apgd_beta,
                    self._apgd_restart_dot,
                )
            elif self.projection_method == "avbd":
                project_deformable_constraints_avbd(
                    self.projection_iterations,
                    contact_system,
                    self.world_active,
                    self._avbd_particle_baseline,
                    deformable_splitting.projected_velocity,
                )

    def _finish_deformable_iteration(self, time_step: wp.array[wp.float32]) -> None:
        deformable_splitting = self.deformable_splitting
        if deformable_splitting is None:
            return
        contact_system = self.deformable_contacts if self._deformable_contacts_active else None
        residual_total = (
            self.splitting.residual_total if self.splitting is not None else self._cloth_only_residual_total
        )
        if self.splitting is None:
            residual_total.zero_()
            self.world_converged.fill_(True)
        deformable_splitting.finish_iteration(
            self.world_active,
            self.world_converged,
            self.world_failed,
            self.iteration_count,
            residual_total,
            time_step,
            self.position_tolerance,
            self.velocity_tolerance,
            contact_system=contact_system,
            rigid_projected_twist=self.splitting.projected_twist if self.splitting is not None else None,
            increment_iteration_count=self.splitting is None,
        )

    def _truncate_deformable_projection(self, time_step: wp.array[wp.float32]) -> None:
        """Apply deformable-only directional limiting before ADMM updates."""
        if self.deformable_penetration_free_limiter is None or self.deformable_splitting is None:
            return
        self.deformable_penetration_free_limiter.truncate(
            self.deformable_splitting.projected_velocity,
            self.deformable_splitting.world_active,
            time_step,
        )

    def _combined_iteration(
        self,
        time_step: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f] | None,
        conditional: bool,
    ) -> None:
        if self._deformable_stream is None:
            self._deformable_candidate_projection(time_step)
            if self.adapter is not None:
                self._prepare_body_space_iteration(time_step, linearization_twist)
        else:
            main_stream = wp.get_stream(self.device)
            main_stream.record_event(self._deformable_start_event)
            self._deformable_stream.wait_event(self._deformable_start_event)
            with wp.ScopedStream(self._deformable_stream):
                self._deformable_candidate_projection(time_step)
            self._deformable_stream.record_event(self._deformable_complete_event)
            self._prepare_body_space_candidate(time_step, linearization_twist)
            main_stream.wait_event(self._deformable_complete_event)
            self._project_body_space_constraints()

        self._truncate_deformable_projection(time_step)
        if self.adapter is not None:
            self._finish_body_space_iteration(time_step, linearization_twist)
        self._finish_deformable_iteration(time_step)
        if conditional:
            self._update_conditional_iteration()

    def _prepare_body_space_candidate(
        self, time_step: wp.array[wp.float32], linearization_twist: wp.array[vec6f]
    ) -> None:
        """Solve and prepare the rigid candidate without unilateral projection."""
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
                self._inverse_time_step,
                linearization_twist,
                splitting.world_active,
                adapter.structural_residual,
                adapter.structural_multiplier_index,
                adapter.structural_multiplier,
                adapter.data.joints.lambda_j,
            )
            adapter.unpack_system_solution()
        splitting.prepare_projection(adapter.system_solution_twist)

    def _prepare_body_space_candidate_with_effort(
        self,
        time_step: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
    ) -> None:
        """Solve a rigid candidate using the currently promoted actuator correction."""
        adapter = self.adapter
        system = self.system
        splitting = self.splitting
        adapter.promote_effort_counters(splitting.world_active)
        if self.structural_joint_solver is None:
            system.solve_candidate_with_effort(
                splitting.projected_twist,
                splitting.splitting_dual,
                adapter.body_effort_offset,
                adapter.body_effort_index,
                adapter.body_effort_side,
                adapter.effort_dynamic_row_index,
                adapter.dynamic_jacobian_first,
                adapter.dynamic_jacobian_second,
                adapter.effort_counter_applied,
            )
            adapter.unpack_system_solution()
        else:
            system.build_candidate_right_hand_side_with_effort(
                splitting.projected_twist,
                splitting.splitting_dual,
                adapter.body_effort_offset,
                adapter.body_effort_index,
                adapter.body_effort_side,
                adapter.effort_dynamic_row_index,
                adapter.dynamic_jacobian_first,
                adapter.dynamic_jacobian_second,
                adapter.effort_counter_applied,
            )
            self.structural_joint_solver.solve_free_body()
            self.structural_joint_solver.project(
                time_step,
                self._inverse_time_step,
                linearization_twist,
                splitting.world_active,
                adapter.structural_residual,
                adapter.structural_multiplier_index,
                adapter.structural_multiplier,
                adapter.data.joints.lambda_j,
            )
            adapter.unpack_system_solution()
        splitting.prepare_projection(adapter.system_solution_twist)

    def _prepare_body_space_iteration(
        self, time_step: wp.array[wp.float32], linearization_twist: wp.array[vec6f]
    ) -> None:
        """Prepare the rigid candidate and run the shared unilateral sweep."""
        self._prepare_body_space_candidate(time_step, linearization_twist)
        self._project_body_space_constraints()

    def _finish_body_space_iteration(
        self, time_step: wp.array[wp.float32], linearization_twist: wp.array[vec6f]
    ) -> None:
        """Update structural state and finish the rigid splitting iteration."""
        adapter = self.adapter
        splitting = self.splitting
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
                    self._inverse_time_step,
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
        adapter.evaluate_lagged_velocity_consistency(
            self.velocity_tolerance,
            splitting.global_twist,
            splitting.projected_twist_previous,
            splitting.world_active,
        )
        splitting.finish_iteration(
            adapter.projection_status,
            time_step,
            self.position_tolerance,
            self.rotation_tolerance,
            self.velocity_tolerance,
            structural_residual=adapter.world_structural_residual,
            projected_structural_residual=adapter.world_projected_structural_residual,
            lagged_velocity_residual=adapter.world_lagged_velocity_residual,
            lagged_velocity_required=adapter.world_lagged_velocity_required,
        )

    def _finish_body_space_iteration_with_effort(
        self,
        time_step: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
    ) -> None:
        """Update structural and finite-drive states before testing convergence."""
        adapter = self.adapter
        splitting = self.splitting
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
            projected_fraction = min(self.joint_multiplier_projected_fraction, 0.5)
            if projected_fraction > 0.0:
                self.structural_joint_solver.refine_from_twists(
                    time_step,
                    self._inverse_time_step,
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
        adapter.update_effort_counters(
            time_step,
            self.velocity_tolerance,
            splitting.projected_twist,
            splitting.world_active,
        )
        adapter.evaluate_lagged_velocity_consistency(
            self.velocity_tolerance,
            splitting.global_twist,
            splitting.projected_twist_previous,
            splitting.world_active,
        )
        splitting.finish_iteration_with_effort(
            adapter.projection_status,
            time_step,
            self.position_tolerance,
            self.rotation_tolerance,
            self.velocity_tolerance,
            adapter.world_effort_residual_max,
            structural_residual=adapter.world_structural_residual,
            projected_structural_residual=adapter.world_projected_structural_residual,
            lagged_velocity_residual=adapter.world_lagged_velocity_residual,
            lagged_velocity_required=adapter.world_lagged_velocity_required,
        )

    def _body_space_iteration(
        self,
        time_step: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
        conditional: bool,
    ) -> None:
        self._prepare_body_space_iteration(time_step, linearization_twist)
        self._finish_body_space_iteration(time_step, linearization_twist)
        if conditional:
            self._update_conditional_iteration()

    def solve(self, problem: LOXProblem) -> None:
        """Assemble and solve one frozen-contact smooth linearization.

        Args:
            problem: LOX forward-dynamics inputs for the current time step.
        """
        if not isinstance(problem, LOXProblem):
            raise TypeError(f"Expected a LOXProblem, got {type(problem)}.")
        if not self._time_step_prepared:
            raise RuntimeError("warmstart() or begin_time_step() must be called before solving LOX.")
        time_step = self._time_step
        inverse_time_step = self._inverse_time_step
        if time_step is None or inverse_time_step is None:
            raise RuntimeError("LOX per-world timestep arrays are not prepared.")
        initial_twist = problem.initial_twist
        linearization_twist = problem.linearization_twist
        if self.splitting is not None:
            if initial_twist is None:
                wp.copy(self._initial_twist, self.projected_twist)
                initial_twist = self._initial_twist
            self.splitting.begin(initial_twist, reset_dual=False)
        else:
            self.world_active.fill_(True)
            self.world_converged.zero_()
            self.world_failed.zero_()
            self.world_iteration_limit.zero_()
            self.iteration_count.zero_()
            self._cloth_only_residual_total.zero_()
        self.world_accepted.zero_()
        self.world_status.fill_(LOX_STATUS_ACTIVE)
        if self.adapter is not None:
            self.adapter.contact_residual.zero_()
            self.adapter.limit_residual.zero_()
            self.adapter.friction_residual.zero_()
        self.contact_residual_max.zero_()
        self.limit_residual_max.zero_()
        self.friction_residual_max.zero_()
        if self.deformable_system is not None and not self._deformable_prepared:
            raise RuntimeError("begin_deformable_time_step() must be called before solving LOX deformables.")

        adapter = self.adapter
        system = self.system
        splitting = self.splitting
        if adapter is not None:
            use_structural_schur = self._joint_solve_mode == _JOINT_SOLVE_SCHUR_DIRECT
            adapter.update(
                time_step,
                joint_penalty_scale=self.joint_penalty_scale,
                linearization_twist=linearization_twist,
                block_joint_metrics=not use_structural_schur and self._joint_metric_mode != _JOINT_METRIC_SIMPLE,
                mass_split_joint_metrics=not use_structural_schur
                and self._joint_metric_mode == _JOINT_METRIC_MASS_SPLIT,
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

        use_conditional_loop = self.use_graph_conditionals and (
            not self.device.is_cuda or not self.device.is_capturing or wp.is_conditional_graph_supported()
        )
        if use_conditional_loop:
            self._iteration_condition.fill_(1)
            if self.deformable_system is None:
                wp.capture_while(
                    self._iteration_condition,
                    self._body_space_iteration,
                    time_step=time_step,
                    linearization_twist=linearization_twist,
                    conditional=True,
                )
            else:
                wp.capture_while(
                    self._iteration_condition,
                    self._combined_iteration,
                    time_step=time_step,
                    linearization_twist=linearization_twist,
                    conditional=True,
                )
        else:
            for _iteration in range(self.max_iterations):
                if self.deformable_system is None:
                    self._body_space_iteration(time_step, linearization_twist, conditional=False)
                else:
                    self._combined_iteration(time_step, linearization_twist, conditional=False)

        if splitting is not None:
            splitting.store_dual_impulse(system.weight, adapter.body_has_unilateral)
            splitting.mark_iteration_limit()
        else:
            wp.launch(
                _mark_iteration_limit,
                dim=self.num_worlds,
                inputs=[],
                outputs=[self.world_active, self.world_iteration_limit],
                device=self.device,
            )
        if self.deformable_splitting is not None:
            self.deformable_splitting.store_dual_impulse()
        wp.launch(
            _finalize_world_status,
            dim=self.num_worlds,
            inputs=[self.world_converged, self.world_failed, self.world_iteration_limit],
            outputs=[self.world_accepted, self.world_status],
            device=self.device,
        )
        if self.deformable_splitting is not None:
            self.deformable_splitting.accept_projected(self.world_accepted)
        if adapter is not None:
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
            adapter.write_outputs(time_step, inverse_time_step, body_velocity=splitting.projected_twist)
        self._deformable_prepared = False

    def write_deformable_output(
        self,
        state_out,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
    ) -> None:
        """Write accepted projected cloth state into Newton output arrays."""
        validate_world_time_steps(time_step, inverse_time_step, self.num_worlds, self.device)
        if self.deformable_splitting is not None:
            self.deformable_splitting.write_output(state_out, time_step)
