"""Steel intra-step fork bench (plan N3 follow-up, 2026-11-07).

Standalone SolverSteel scene (two Franka arms sweeping six boxes, from
newton.examples.steel.example_steel_determinism_arms.build_scene, replicated
BENCH_WORLDS times, Steel defaults) so that Steel itself is the critical path;
in the Franka+cable ADMM scene the rigid entry hides behind the cable, so the
intra-step fork would be invisible there.

Three arms, one process each or all in one (BENCH_ARM=native|flat|fork|all):
  native  the frame captured by wp.ScopedCapture, replayed (Newton as shipped)
  flat    the same frame recorded as one STF task per substep (STF overhead control)
  fork    SolverSteel._advance re-orchestrated into four tasks per substep:
            pre   = preamble + FK/inverse-dynamics bias (stage 1a)
            mass  = joint tau, CRBA, Cholesky, trisolve, v_hat (stages 1b-3)
            rows  = _stage4_build_rows (friction patches, slots, contact rows, J)
            join  = responses, PGS sweep, integrate, publish, patch carry
          mass and rows are siblings after pre (trace: topics/python-stf/
          steel-step-dependencies-2026-11-07.md, Q2). Valid for Steel defaults
          only: with enable_joint_limits / joint friction / velocity limits /
          sparse-diagonal responses the row build reads _v_hat and the fork is
          wrong; the asserts below refuse those configurations.

Same rules as newton_admm_stf_bench.py: warm eagerly before recording, record
once, replay; fingerprint = sum |body_q| after the same total frame count;
BENCH_DOT dumps the recorded graph and prints its topology.
"""

import os
import time

import numpy as np
import warp as wp

import newton
import cuda.stf._experimental as stf
from newton.examples.steel.example_steel_determinism_arms import build_scene
from newton.solvers import SolverSteel

from newton.examples.multiphysics.admm_stf_bench import _task_scope, topology_summary

FRAMES = int(os.environ.get("BENCH_FRAMES", 200))
WORLDS = int(os.environ.get("BENCH_WORLDS", 8))
SUBSTEPS = int(os.environ.get("BENCH_SUBSTEPS", 4))  # even: state_0 holds the frame result
ARM = os.environ.get("BENCH_ARM", "all")
DOT = os.environ.get("BENCH_DOT")
DETERMINISTIC = os.environ.get("BENCH_DETERMINISTIC", "0") == "1"  # Steel RUN_TO_RUN + deterministic collision
FPS = 50


class Scene:
    def __init__(self):
        scene = build_scene()
        builder = newton.ModelBuilder()
        builder.rigid_gap = scene.rigid_gap
        builder.replicate(scene, WORLDS)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.solver = SolverSteel(
            self.model,
            articulation_constraint_max=384,
            free_body_constraint_max=128,
            deterministic=wp.DeterministicMode.RUN_TO_RUN if DETERMINISTIC else wp.DeterministicMode.NOT_GUARANTEED,
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model, deterministic=DETERMINISTIC)
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.dt = 1.0 / FPS / SUBSTEPS

    def substep_native(self):
        self.state_0.clear_forces()
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def frame_native(self):
        for _ in range(SUBSTEPS):
            self.substep_native()

    def fingerprint(self):
        wp.synchronize()
        return float(np.abs(self.state_0.body_q.numpy()).sum())


def _assert_default_steel(sv, state_out):
    """The fork below mirrors SolverSteel._advance for its default configuration only."""
    for flag in (
        "_contact_compliance", "_contact_torsion_enabled", "_double_buffered", "_enable_joint_friction",
        "_contact_warmstart", "_row_watermark", "_enable_joint_limits", "_enable_joint_velocity_limits",
        "_joint_friction_load_active", "_reuse_kinematics_enabled", "_preelim_active",
    ):
        assert not getattr(sv, flag), flag
    assert sv._sleeping is None and sv._device_torsion is None
    assert sv._velocity_iterations == 0
    assert sv._sparse_mass_matrix_size is None and sv._sparse_diagonal_response_size is None
    assert not sv._sparse_diagonal_contact_solve
    assert sv._rigid_velocity_limit_slot is None
    assert state_out.body_parent_f is None
    assert not sv._coupling_patch_history_restore_pending and not sv._coupling_patch_history_snapshot_pending
    assert sv.model.articulation_count


def steel_advance_forked(sv, state_in, state_out, control, contacts, dt, task_pre, task_mass, task_rows, task_join, pre_hook=None):
    """SolverSteel.step() + _advance() (Newton main 1a497c00, defaults) with the
    mass-matrix chain and the row build recorded as sibling tasks. The four
    task_* arguments are factories returning the context manager of one task
    each; the tasks are opened one after the other (never nested). pre_hook
    runs first inside the pre task (clear forces, collide)."""
    _assert_default_steel(sv, state_out)
    model = sv.model
    with task_pre():
        if pre_hook is not None:
            pre_hook()
        stage3_qd = _steel_pre(sv, state_in, state_out, control, contacts, dt)
    state_aug = sv  # _prepare_augmented_state returns the solver
    if control is None:
        control = model.control(clone_variables=False)
    _steel_fork_tail(sv, state_aug, stage3_qd, state_in, state_out, control, contacts, dt, task_mass, task_rows, task_join)


def _steel_pre(sv, state_in, state_out, control, contacts, dt):
    model = sv.model
    # step() checks
    sv.validate_observables(None, contacts)
    if contacts is not None and contacts.rigid_contact_max > sv._max_contacts_alloc:
        raise ValueError("Steel contact capacity mismatch")
    # _advance() preamble
    if sv._last_step_dt is not None and abs(sv._last_step_dt - dt) > 1.0e-8:
        sv._force_mass_update = True
    sv._last_step_dt = dt
    if control is None:
        control = model.control(clone_variables=False)
    if sv._unarticulated_bodies.size:
        from newton._src.solvers.steel.solver_steel import copy_unarticulated_body_state

        wp.launch(
            copy_unarticulated_body_state,
            dim=sv._unarticulated_bodies.size,
            inputs=[sv._unarticulated_bodies, state_in.body_q, state_in.body_qd],
            outputs=[state_out.body_q, state_out.body_qd],
            device=model.device,
        )
    from newton._src.solvers.steel.solver_steel import _flag_disabled_tree_joints, finish_patch_impulses

    wp.launch(
        _flag_disabled_tree_joints,
        dim=model.joint_count,
        inputs=[model.joint_enabled, model.joint_articulation, model.joint_world, int(model.world_count)],
        outputs=[sv._disabled_tree_joint],
        device=model.device,
    )
    state_aug = sv._prepare_augmented_state(state_in)
    sv._begin_dense_rows()
    # Stage 1a: FK + inverse-dynamics bias (both chains need it)
    assert state_aug is sv
    return sv._stage1_fk_id(state_in, state_aug, state_out)


def _steel_fork_tail(sv, state_aug, stage3_qd, state_in, state_out, control, contacts, dt, task_mass, task_rows, task_join):
    from newton._src.solvers.steel.solver_steel import finish_patch_impulses

    model = sv.model
    with task_mass():
        # Stage 1b, 1c
        sv._stage1_joint_tau(state_in, state_aug, state_out, control, dt)
        sv._stage1_crba(state_aug)
        # Stage 2: factor (verbatim from _advance, sparse branches excluded by the asserts)
        for size in sv._for_sizes():
            if sv._crba_cholesky_kernels_by_size[size] or sv._crba_cholesky_warp_kernels_by_size[size]:
                pass
            elif sv._execution_plan.use_diagonal_mass(size):
                sv._stage2_factor_diagonal(size)
            elif sv._execution_plan.use_tiled_cholesky(size):
                sv._stage2_cholesky_tiled(size)
            else:
                sv._stage2_cholesky_loop(size)
            if size in sv._Linv_by_size:
                sv._stage2_invert_lower_factor(size)
            if sv._inverse_cholesky_kernels_by_size[size] is not None:
                wp.launch(
                    sv._inverse_cholesky_kernels_by_size[size],
                    dim=sv._n_arts_by_size[size] * 32,
                    inputs=[sv._L_by_size[size], sv._group_to_art[size], sv._mass_update_mask],
                    outputs=[sv._Hinv_by_size[size]],
                    block_dim=256,
                    device=model.device,
                )
        # Stage 3: unconstrained acceleration and velocity prediction
        state_aug._joint_qdd.zero_()
        for size in sv._for_sizes():
            if sv._execution_plan.use_diagonal_mass(size):
                sv._stage3_solve_diagonal(size, state_aug)
                continue
            use_tiled = sv._trisolve_kernel == "tiled" or (
                sv._trisolve_kernel == "auto" and size > sv._small_dof_threshold
            )
            if use_tiled:
                sv._stage3_trisolve_tiled(size, state_aug)
            else:
                sv._stage3_trisolve_loop(size, state_aug)
        sv._stage3_compute_v_hat(state_in, state_aug, dt, stage3_qd)

    with task_rows():
        # Stage 4a: rows (reads FK outputs and contacts; nothing from stages 1b-3 in the default config)
        sv._stage4_build_rows(state_in, state_aug, control, contacts, dt)

    with task_join():
        # Stages 4b-6: responses Y = H^-1 J^T, diagonals, rhs, PGS sweep
        sv._solve_matrix_free(state_in, state_aug, contacts, dt)
        # Stage 7
        sv._integrate(state_in, state_aug, state_out, dt, sv._v_out)
        sv._stage7_update_kinematics(state_out)
        has_contacts = contacts is not None and contacts.rigid_contact_max > 0
        if sv._friction_anchors_enabled:
            if has_contacts:
                for route, parents, mu, impulses in sv._patch_row_arrays():
                    wp.launch(
                        finish_patch_impulses,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            contacts.rigid_contact_count, sv._friction_patches.current, state_in.body_q,
                            state_out.body_q, state_out.body_qd, model.body_com, sv._body_to_articulation,
                            contacts.rigid_contact_surface_velocity, sv._contact_world, sv._contact_slot,
                            sv._contact_path, sv._contact_slots_needed, route, parents, mu, impulses, dt,
                        ],
                        device=model.device,
                    )
                sv._friction_patches.store(state_in)
            else:
                sv._friction_patches.previous.valid.zero_()
        sv._step += 1


def run_native(sc):
    sc.frame_native()  # warm (allocations, module loads)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        sc.frame_native()
    graph = cap.graph
    wp.capture_launch(graph)  # warm replay
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(FRAMES):
        wp.capture_launch(graph)
    wp.synchronize()
    return (time.perf_counter() - t0) / FRAMES * 1e3, None


def run_stf(sc, fork):
    sc.frame_native()  # warm eagerly
    wp.synchronize()
    ctx = stf.stackable_context()
    t_s, t_m, t_r = ctx.token(), ctx.token(), ctx.token()

    def task(*deps, symbol):
        return _task_scope(ctx.task(*deps, symbol=symbol))

    topo = None
    with ctx.launchable_graph_scope() as scope:
        for i in range(SUBSTEPS):
            s_in, s_out = (sc.state_0, sc.state_1) if i % 2 == 0 else (sc.state_1, sc.state_0)
            if not fork:
                with task(t_s.rw(), t_m.rw(), t_r.rw(), symbol=f"step{i}"):
                    s_in.clear_forces()
                    sc.collision_pipeline.collide(s_in, sc.contacts)
                    sc.solver.step(s_in, s_out, sc.control, sc.contacts, sc.dt)
                continue
            def pre_hook(s_in=s_in):
                s_in.clear_forces()
                sc.collision_pipeline.collide(s_in, sc.contacts)

            steel_advance_forked(
                    sc.solver, s_in, s_out, sc.control, sc.contacts, sc.dt,
                    task_pre=lambda i=i: task(t_s.rw(), t_m.rw(), t_r.rw(), symbol=f"pre{i}"),
                    task_mass=lambda i=i: task(t_s.read(), t_m.rw(), symbol=f"mass{i}"),
                    task_rows=lambda i=i: task(t_s.read(), t_r.rw(), symbol=f"rows{i}"),
                    task_join=lambda i=i: task(t_s.rw(), t_m.read(), t_r.read(), symbol=f"join{i}"),
                    pre_hook=pre_hook,
            )
        if DOT:
            from cuda.bindings import runtime as cudart

            (err,) = cudart.cudaGraphDebugDotPrint(
                scope.graph, DOT.encode(), cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose
            )
            assert err == cudart.cudaError_t.cudaSuccess, err
            topo = topology_summary(DOT)
        scope.launch()
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(FRAMES):
            scope.launch()
        wp.synchronize()
        ms = (time.perf_counter() - t0) / FRAMES * 1e3
    ctx.finalize()
    return ms, topo


def main():
    wp.init()
    arms = ["native", "flat", "fork"] if ARM == "all" else [ARM]
    results = {}
    for arm in arms:
        sc = Scene()
        if arm == "native":
            ms, topo = run_native(sc)
        else:
            ms, topo = run_stf(sc, fork=arm == "fork")
        fp = sc.fingerprint()
        results[arm] = (ms, fp)
        if topo is not None:
            print(f"TOPOLOGY arm={arm} {topo}")
        print(f"RESULT arm={arm} det={int(DETERMINISTIC)} worlds={WORLDS} substeps={SUBSTEPS} ms_per_frame={ms:.3f} fingerprint={fp!r}")
        del sc
    if "native" in results:
        for arm in ("flat", "fork"):
            if arm in results:
                print(f"CHECK {arm}: bit_identical={results[arm][1] == results['native'][1]} "
                      f"speedup={results['native'][0] / results[arm][0]:.2f}x")


if __name__ == "__main__":
    main()
