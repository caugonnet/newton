# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Newton-shipped SolverCoupledADMM scenes, native graph replay vs. the same
frame re-orchestrated as cuda.stf tasks (plan N1 of
docs/plan-fresh-branches-newton-warp-isaac-2026-11-07.md; this is the port of
the August bench to Newton main 1a497c00, 2026-11-07).

Recipe (topics/python-stf/README.md): stock Warp main, `cuda.stf._experimental`
imported directly; every task body runs under `wp.ScopedStream` +
`wp.capture_begin(external=True)` on the task's stream. Per ADMM iteration the
per-entry prepare and solve loops become width-N task groups (one token per
entry), the consensus steps (`_accumulate_admm_forces`, `_update_admm_dual`)
are barrier tasks on every token.

Postmortem rule 1 (topology before timing): with BENCH_DOT=<path> the recorded
STF graph is written with cudaGraphDebugDotPrint and summarised: kernel nodes
per entry class, longest path, and the number of direct edges between kernels
of two different entry classes (spurious capture_end leaf joins would show up
there). Timing is reported regardless; the topology line is what the
measurement README must quote first.

Environment: BENCH_EXAMPLE (admm_contact | mujoco_vbd_admm | kamino_mujoco_admm
| franka_vbd_cable_admm | franka_pulley), BENCH_FRAMES (default 200), BENCH_DOT (dot path),
BENCH_ARGS (extra example CLI args, e.g. "--substeps 4"). franka_pulley is the
NVIDIA/newton-samples scene (examples/multiphysics/example_franka_pulley_mechanical_advantage.py);
BENCH_SAMPLES_ROOT must point at a checkout of that repository.

Rules carried from August: warm up before any capture; even substep counts so
the state double-buffer returns to state_0; never cache a capture-time
allocation on a persistent attribute (the STF arm disables the example's own
capture so Newton allocates for real during warm-up).
"""

import contextlib
import importlib
import os
import re
import shlex
import sys
import time

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer
import cuda.stf._experimental as stf
from newton._src.solvers.coupled.solver_coupled import _copy_state

FRAMES = int(os.environ.get("BENCH_FRAMES", 200))
EXAMPLE = os.environ.get("BENCH_EXAMPLE", "admm_contact")
DOT = os.environ.get("BENCH_DOT")
EXTRA_ARGS = shlex.split(os.environ.get("BENCH_ARGS", ""))
ARM = os.environ.get("BENCH_ARM", "both")
STEEL_FORK = os.environ.get("BENCH_STEEL_FORK", "0") == "1"  # also fork SolverSteel entries (level 2 inside level 1)  # native | stf | both (one arm per process for nsys)
# 1: the per-frame prologue (IK solve + control write) runs on its own token, as a sibling of the
# first substep's head/prepare/accumulate phases; the solve tasks read that token (they read control).
PRE_SIBLING = os.environ.get("BENCH_PRE_SIBLING", "0") == "1"
assert not (PRE_SIBLING and STEEL_FORK), "the forked Steel solve tasks do not take the prologue dependency yet"


def _eval_ik(ex, s_in, s_out):
    newton.eval_ik(ex.model, s_out, s_out.joint_q, s_out.joint_qd)


def _collide(ex, s_in):
    ex.collision_pipeline.collide(s_in, ex.contacts)


def _franka_frame_pre(ex):
    # mirrors Example.simulate() of example_mujoco_franka_vbd_cable_admm_solver:
    # IK solve, gripper targets, control copy, all Warp work on the current stream
    ex.ik_solver.step(ex.ik_joint_q, ex.ik_joint_q, iterations=ex.ik_iters)
    mod = importlib.import_module("newton.examples.multiphysics.example_mujoco_franka_vbd_cable_admm_solver")
    wp.launch(
        mod.set_gripper_q,
        dim=ex.world_count,
        inputs=[ex.ik_joint_q, ex.finger_pos_buf, ex.finger_idx0, ex.finger_idx1],
        device=ex.device,
    )
    wp.copy(dest=ex.control_joint_target_q[:, : ex.n_coords], src=ex.ik_joint_q)


def _pulley_frame_pre(ex):
    # mirrors Example.simulate() of newton-samples' example_franka_pulley_mechanical_advantage
    ex.ik_solver.step(ex.ik_joint_q, ex.ik_joint_q, iterations=ex.ik_iters)
    mod = importlib.import_module("examples.multiphysics.example_franka_pulley_mechanical_advantage")
    wp.launch(
        mod.set_gripper_target,
        dim=1,
        inputs=[ex.ik_joint_q, ex.finger_position, ex.finger_index_0, ex.finger_index_1],
        device=ex.device,
    )
    wp.copy(dest=ex.control_joint_target_q[:, : ex.n_coords], src=ex.ik_joint_q)


# name -> (module, per-frame prologue, per-substep pre-step, per-substep post-step)
# Each mirrors what the example's own simulate() does around solver.step().
EXAMPLES = {
    "admm_contact": ("newton.examples.multiphysics.example_admm_contact_solver", None, None, None),
    "mujoco_vbd_admm": ("newton.examples.multiphysics.example_mujoco_vbd_admm_solver", None, None, _eval_ik),
    "kamino_mujoco_admm": ("newton.examples.multiphysics.example_kamino_mujoco_admm_solver", None, None, None),
    "franka_vbd_cable_admm": (
        "newton.examples.multiphysics.example_mujoco_franka_vbd_cable_admm_solver", _franka_frame_pre, _collide, _eval_ik
    ),
    # NVIDIA/newton-samples (external checkout, BENCH_SAMPLES_ROOT): MuJoCo Franka + VBD cable/pulleys, ADMM
    "franka_pulley": (
        "examples.multiphysics.example_franka_pulley_mechanical_advantage", _pulley_frame_pre, _collide, _eval_ik
    ),
}
_mod_name, FRAME_PRE, SUBSTEP_PRE, SUBSTEP_POST = EXAMPLES[EXAMPLE]
if _mod_name.startswith("examples."):
    sys.path.insert(0, os.environ["BENCH_SAMPLES_ROOT"])
Example = importlib.import_module(_mod_name).Example

# Kernel -> entry class for the topology summary, built from the Warp kernels
# registered by each solver package (the CUDA symbol is <key>_<hash>_cuda_kernel_forward).
# Coupled-solver (barrier phase) kernels are allowed to touch every entry.
KERNEL_CLASS_PREFIXES = [
    ("coupled", ("newton._src.solvers.coupled",)),
    ("mujoco", ("mujoco_warp", "newton._src.solvers.mujoco")),
    ("kamino", ("newton._src.solvers.kamino", "kamino")),
    ("vbd", ("newton._src.solvers.vbd",)),
    ("steel", ("newton._src.solvers.steel",)),
    ("xpbd", ("newton._src.solvers.xpbd",)),
    ("newton-core", ("newton._src.sim", "newton._src.geometry", "newton._src.core", "newton._src.utils", "newton._src.solvers.ik", "newton._src.solvers.solver", "newton.examples")),
    ("warp", ("warp",)),
]


def kernel_class_map():
    """Every Warp kernel registered so far (including closure-built ones), keyed
    by kernel key, mapped to the entry class of its defining Python module."""
    from warp._src.context import user_modules

    out = {}
    for modname, module in list(user_modules.items()):
        cls = next((c for c, prefixes in KERNEL_CLASS_PREFIXES if any(modname == p or modname.startswith(p + ".") for p in prefixes)), None)
        if cls is None:
            continue
        for key in getattr(module, "kernels", {}):
            out.setdefault(key, cls)
    return out


class _task_scope:
    """Run a task body on the task's stream with Warp adopting the external
    capture, so allocator/sort paths are capture-safe."""

    def __init__(self, t):
        self._t = t

    def __enter__(self):
        self._t.__enter__()
        self.stream = wp.Stream(wp.get_cuda_device(), cuda_stream=self._t.stream_ptr())
        self.scope = wp.ScopedStream(self.stream, sync_enter=False, sync_exit=False)
        self.scope.__enter__()
        wp.capture_begin(stream=self.stream, external=True)
        return self

    # Warp's Graph.__del__ destroys the captured cudaGraph_t (release_or_defer(Graph)) even for an
    # external capture, i.e. the STF-owned graph; keep every returned handle alive for the process.
    retained = []

    def __exit__(self, *exc):
        _task_scope.retained.append(wp.capture_end(stream=self.stream))
        self.scope.__exit__(*exc)
        return self._t.__exit__(*exc)


def _no_capture(self):
    # STF arm: skip the example's native capture so every Newton-internal
    # buffer is allocated for real during the eager warm-up frame
    self.graph = None


_ExampleNoCapture = type("_ExampleNoCapture", (Example,), {"capture": _no_capture})


class _NoScopedCapture:
    """Stand-in for wp.ScopedCapture while an example that captures in __init__
    (newton-samples) is constructed for the STF arm: the frame runs eagerly."""

    graph = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_example(cls=Example):
    parser = cls.create_parser()
    headless = ["--headless"] if "--headless" in parser.format_help() else []
    args = parser.parse_args(["--viewer", "null", *headless, *EXTRA_ARGS])
    viewer = newton.viewer.ViewerNull()
    if cls is _ExampleNoCapture and not hasattr(Example, "capture"):
        with _patched(wp, "ScopedCapture", _NoScopedCapture):
            return cls(viewer, args)
    return cls(viewer, args)


@contextlib.contextmanager
def _patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def fingerprint(ex):
    wp.synchronize()
    out = [float(np.abs(ex.state_0.body_q.numpy()).sum())]
    if ex.state_0.particle_q is not None and ex.state_0.particle_q.shape[0] > 0:
        out.append(float(np.abs(ex.state_0.particle_q.numpy()).sum()))
    return tuple(out)


def run_native(ex):
    assert ex.graph is not None, "native arm needs the example's own graph capture"
    # two warm replays: matches the stf arm's eager frame + warm replay,
    # so both arms fingerprint after the same total frame count
    wp.capture_launch(ex.graph)
    wp.capture_launch(ex.graph)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(FRAMES):
        wp.capture_launch(ex.graph)
    wp.synchronize()
    return (time.perf_counter() - t0) / FRAMES * 1e3


def record_stf_solver_step(task, ex, toks, t_shared, state_in, state_out, solve_extra=()):
    """SolverCoupled.step() + SolverCoupledADMM._step_coupled() (Newton main
    1a497c00) re-orchestrated as STF tasks, with the per-entry loops split into
    width-N task groups. `solve_extra`: extra dependencies of the solve tasks
    (the control written by a sibling prologue task)."""
    s = ex.solver
    coupling = s._coupling
    dt = ex.sim_dt
    control, contacts = ex.control, ex.contacts
    all_rw = [t_shared.rw()] + [toks[n].rw() for n in s._entries]

    with task(*all_rw, symbol="head"):
        state_in.clear_forces()
        newton.examples.apply_coupled_viewer_forces(ex, state_in)
        if SUBSTEP_PRE is not None:
            SUBSTEP_PRE(ex, state_in)
        # SolverCoupled.step() preamble
        s.validate_observables(None, contacts)
        s._require_unreduced_contacts(contacts)
        s._distribute_state(state_in, dt=dt)
        # SolverCoupledADMM._step_coupled() preamble
        s._refresh_collision_contact_groups(state_in)
        if float(coupling.gamma) > 0.0:
            s._refresh_admm_proximal_masks()
            s._refresh_admm_proximal_view_overrides(refresh_supported_solvers=True)
        for name, entry in s._entries.items():
            buf = s._admm_buffers[name]
            if buf.body_q_n is not None:
                wp.copy(buf.body_q_n, entry.state_0.body_q)
                wp.copy(buf.body_qd_n, entry.state_0.body_qd)
                wp.copy(buf.body_qd_k, entry.state_0.body_qd)
            if buf.particle_q_n is not None:
                wp.copy(buf.particle_q_n, entry.state_0.particle_q)
                wp.copy(buf.particle_qd_n, entry.state_0.particle_qd)
                wp.copy(buf.particle_qd_k, entry.state_0.particle_qd)
            if buf.joint_q_n is not None:
                wp.copy(buf.joint_q_n, entry.state_0.joint_q)
                wp.copy(buf.joint_qd_n, entry.state_0.joint_qd)
                wp.copy(buf.joint_qd_k, entry.state_0.joint_qd)
        s._admm_begin_step(dt)

    for k in range(int(coupling.iterations)):
        for name, entry in s._entries.items():
            with task(toks[name].rw(), t_shared.read(), symbol=f"prep-{name}"):
                s._prepare_admm_iteration_state(
                    entry, s._admm_buffers[name], state_in, dt, iteration_restart=k > 0
                )
        # consensus barrier: touches contact groups spanning entry pairs
        with task(*all_rw, symbol="accum"):
            s._accumulate_admm_forces(k, dt, refresh_jv=k == 0, initialize_contact_u=k == 0)
        for name, entry in s._entries.items():
            if STEEL_FORK and type(entry.solver).__name__ == "SolverSteel":
                _record_steel_entry_forked(task, s, name, entry, toks, t_shared, control, contacts, dt)
                continue
            with task(toks[name].rw(), t_shared.read(), *solve_extra, symbol=f"solve-{name}"):
                s._apply_admm_force_inputs(entry, s._admm_buffers[name], dt)
                s._step_entry(entry, control, contacts, dt)
                _copy_qd_k(s, name, entry)
        with task(*all_rw, symbol="dual"):
            s._update_admm_dual(k, dt)

    with task(*all_rw, symbol="tail"):
        _copy_state(state_in, state_out)
        s._reconcile_state(state_out)
        if SUBSTEP_POST is not None:
            SUBSTEP_POST(ex, state_in, state_out)
    s._entry_output_state_valid = True


def _copy_qd_k(s, name, entry):
    buf = s._admm_buffers[name]
    if buf.body_qd_k is not None:
        wp.copy(buf.body_qd_k, entry.state_1.body_qd)
    if buf.particle_qd_k is not None:
        wp.copy(buf.particle_qd_k, entry.state_1.particle_qd)
    if buf.joint_qd_k is not None:
        wp.copy(buf.joint_qd_k, entry.state_1.joint_qd)


def _record_steel_entry_forked(task, s, name, entry, toks, t_shared, control, contacts, dt):
    """Level 2 inside level 1: the Steel entry's solve task becomes the four
    tasks of newton_steel_fork_bench.steel_advance_forked (pre, mass || rows,
    join). SolverCoupled._step_entry's wrapping (contact filtering, control
    copy) goes into the pre task; single-substep, non-in-place entries only."""
    from newton._src.solvers.coupled.solver_coupled import _copy_control_to_entry
    from newton.examples.steel.steel_fork_bench import steel_advance_forked

    assert entry.substeps == 1 and not entry.in_place, "forked Steel entry: single substep only"
    tm, tr = toks[f"{name}.mass"], toks[f"{name}.rows"]

    def pre_hook():
        s._apply_admm_force_inputs(entry, s._admm_buffers[name], dt)
        # SolverCoupled._step_entry wrapping; both return persistent per-entry objects
        return _copy_control_to_entry(control, entry), s._contacts_for_entry(entry, contacts)

    steel_advance_forked(
        entry.solver, entry.state_0, entry.state_1, None, None, dt,
        task_pre=lambda: task(toks[name].rw(), t_shared.read(), symbol=f"solve-{name}-pre"),
        task_mass=lambda: task(toks[name].read(), t_shared.read(), tm.rw(), symbol=f"solve-{name}-mass"),
        task_rows=lambda: task(toks[name].read(), t_shared.read(), tr.rw(), symbol=f"solve-{name}-rows"),
        task_join=lambda: task(toks[name].rw(), t_shared.read(), tm.read(), tr.read(), symbol=f"solve-{name}-join"),
        pre_hook=pre_hook,
        post_hook=lambda: _copy_qd_k(s, name, entry),
    )


_KERNEL_RX = re.compile(r"\|\s*([A-Za-z_]\w*?)_[0-9a-f]{8}_cuda_kernel_(?:forward|backward)")


def _classify(label, kmap):
    m = _KERNEL_RX.search(label)
    if not m:
        # Warp builtins (memtile/fill, radix sort, ...) carry mangled C++ names
        return "warp-builtin"
    key = m.group(1)
    if key in kmap:
        return kmap[key]
    # Warp kernel keys may carry a module prefix; fall back to a suffix match
    for k, c in kmap.items():
        if key.endswith(k):
            return c
    return "other:" + key[:40]


def topology_summary(dot_path):
    """Parse the cudaGraphDebugDotPrint output: kernel nodes per entry class,
    longest path, direct edges between kernels of two different entry classes
    (the coupled/barrier class excluded)."""
    text = open(dot_path).read()
    nodes, edges = {}, []
    for m in re.finditer(r'^\s*"?(\w+)"?\s*\[(.*?)\];?\s*$', text, re.M | re.S):
        nid, attrs = m.group(1), m.group(2)
        lab = re.search(r'label\s*=\s*"(.*?)"', attrs, re.S)
        if lab:
            nodes[nid] = lab.group(1).replace("\\n", " ").replace("\\l", " ")
    for m in re.finditer(r'"?(\w+)"?\s*->\s*"?(\w+)"?', text):
        if m.group(1) in nodes and m.group(2) in nodes:
            edges.append((m.group(1), m.group(2)))
    kmap = kernel_class_map()
    kind = {n: ("kernel" if l.lstrip("{").startswith("KERNEL") else "other") for n, l in nodes.items()}
    cls = {n: _classify(l, kmap) for n, l in nodes.items()}
    per_class = {}
    for n in nodes:
        if kind[n] == "kernel":
            per_class[cls[n]] = per_class.get(cls[n], 0) + 1
    cross = [(a, b) for a, b in edges if kind[a] == kind[b] == "kernel"
             and cls[a] != cls[b]
             and not any(c in ("coupled", "newton-core", "warp", "warp-builtin") or c.startswith("other") for c in (cls[a], cls[b]))]
    # longest path (DAG)
    succ = {}
    indeg = {n: 0 for n in nodes}
    for a, b in edges:
        succ.setdefault(a, []).append(b)
        indeg[b] += 1
    order, stack = [], [n for n in nodes if indeg[n] == 0]
    dist = {n: 1 for n in nodes}
    while stack:
        n = stack.pop()
        order.append(n)
        for m in succ.get(n, []):
            dist[m] = max(dist[m], dist[n] + 1)
            indeg[m] -= 1
            if indeg[m] == 0:
                stack.append(m)
    return {
        "nodes": len(nodes),
        "kernel_nodes": sum(1 for n in nodes if kind[n] == "kernel"),
        "edges": len(edges),
        "longest_path": max(dist.values()) if dist else 0,
        "per_class": per_class,
        "cross_entry_direct_edges": len(cross),
        "cross_examples": [(_KERNEL_RX.search(nodes[a]).group(1), _KERNEL_RX.search(nodes[b]).group(1)) for a, b in cross[:6]],
    }


def run_stf(ex):
    # Odd substep counts: mirror the example's own graph-mode path
    # (need_state_copy): after the last substep the newest state is in state_1,
    # so copy it back into state_0 instead of swapping. Both arms pay that copy.
    odd = ex.sim_substeps % 2 == 1
    # warm eagerly (allocations, module loads must precede recording)
    ex.simulate()
    wp.synchronize()
    ctx = stf.stackable_context()
    toks = {name: ctx.token() for name in ex.solver._entries}
    if STEEL_FORK:
        for name, entry in ex.solver._entries.items():
            if type(entry.solver).__name__ == "SolverSteel":
                toks[f"{name}.mass"], toks[f"{name}.rows"] = ctx.token(), ctx.token()
    t_shared = ctx.token()
    all_rw = [t_shared.rw()] + [toks[n].rw() for n in ex.solver._entries]

    def task(*deps, symbol):
        return _task_scope(ctx.task(*deps, symbol=symbol))

    topo = None
    with ctx.launchable_graph_scope() as scope:
        solve_extra = ()
        if FRAME_PRE is not None:
            if PRE_SIBLING:
                t_pre = ctx.token()
                with task(t_pre.rw(), symbol="frame-pre"):
                    FRAME_PRE(ex)
                solve_extra = (t_pre.read(),)
            else:
                with task(*all_rw, symbol="frame-pre"):
                    FRAME_PRE(ex)
        pair = (ex.state_0, ex.state_1)
        for i in range(ex.sim_substeps):
            s_in, s_out = pair if i % 2 == 0 else (pair[1], pair[0])
            record_stf_solver_step(task, ex, toks, t_shared, s_in, s_out, solve_extra if i == 0 else ())
        if odd:
            with task(*all_rw, symbol="state-return"):
                ex.state_0.assign(ex.state_1)

        if DOT:
            from cuda.bindings import runtime as cudart

            (err,) = cudart.cudaGraphDebugDotPrint(
                scope.graph, DOT.encode(), cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose
            )
            assert err == cudart.cudaError_t.cudaSuccess, err
            topo = topology_summary(DOT)

        scope.launch()  # warm replay
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
    ex_n = make_example()
    print(f"SETUP example={EXAMPLE} substeps={ex_n.sim_substeps} admm_iterations={int(ex_n.solver._coupling.iterations)} "
          f"entries={list(ex_n.solver._entries)} frames={FRAMES} newton={newton.__file__} warp={wp.config.version}")
    if ARM == "stf":
        del ex_n
    else:
        ms_n = run_native(ex_n)
        fp_n = fingerprint(ex_n)
        print(f"RESULT arm=native ms_per_frame={ms_n:.3f} fingerprint={fp_n}")
        del ex_n
        if ARM == "native":
            return

    ex_s = make_example(_ExampleNoCapture)
    ms_s, topo = run_stf(ex_s)
    fp_s = fingerprint(ex_s)
    if topo is not None:
        print(f"TOPOLOGY {topo}")
    print(f"RESULT arm=stf ms_per_frame={ms_s:.3f} fingerprint={fp_s}")
    if ARM == "stf":
        return

    rel = np.abs(np.array(fp_s) - np.array(fp_n)) / np.maximum(np.abs(np.array(fp_n)), 1e-9)
    print(f"CHECK rel_dev={rel.max():.2e} bit_identical={fp_s == fp_n}")
    print(f"SPEEDUP example={EXAMPLE} {ms_n / ms_s:.2f}x")


if __name__ == "__main__":
    main()
