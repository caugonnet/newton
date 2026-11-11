# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Newton-shipped SolverCoupledADMM scenes: native graph replay vs. the same
frame recorded as a cuda.stf task graph.

Recipe: the frame is one launchable scope of cuda.stf's Warp adapter around the
example's own `simulate()`. Under a recording every launch belongs to a task:
SolverCoupledADMM records its own (its per-entry prepare and solve phases may
overlap), and gives a scope to an entry that records its own (SolverSteel:
mass-matrix chain and row build) and one task to any other
(newton._src.solvers.tasks). The example's frame is not written with tasks, so
the bench records what it does around the coupled solver's step as tasks of
the frame's sequence (tasks_around_step).

Postmortem rule 1 (topology before timing): with BENCH_DOT=<path> the recorded
STF graph is written with cudaGraphDebugDotPrint and summarised.

Environment: BENCH_EXAMPLE (admm_contact | mujoco_vbd_admm | kamino_mujoco_admm
| franka_vbd_cable_admm | franka_pulley), BENCH_FRAMES (default 200), BENCH_DOT (dot path),
BENCH_ARGS (extra example CLI args, e.g. "--substeps 4"), BENCH_ARM (native | stf | both),
BENCH_STEEL_FORK=0 (SolverSteel entries without their fork), BENCH_MJW_FORK=1 (fork
SolverMuJoCo entries inside mujoco_warp's step through mjwarp_stf_fork.py; mujoco_warp
unchanged). franka_pulley is the NVIDIA/newton-samples scene; BENCH_SAMPLES_ROOT must
point at a checkout of that repository.

Rules carried from August: warm up before any capture; never cache a
capture-time allocation on a persistent attribute (the STF arm disables the
example's own capture so Newton allocates for real during warm-up).
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
from cuda.stf._experimental.interop.warp import release_graph_handles, scope, scope_token, warp_task

from newton.examples.multiphysics.mjwarp_stf_fork import MjwForkShim, installed as _mjw_installed

FRAMES = int(os.environ.get("BENCH_FRAMES", 200))
EXAMPLE = os.environ.get("BENCH_EXAMPLE", "admm_contact")
DOT = os.environ.get("BENCH_DOT")
EXTRA_ARGS = shlex.split(os.environ.get("BENCH_ARGS", ""))
ARM = os.environ.get("BENCH_ARM", "both")  # native | stf | both (one arm per process for nsys)
STEEL_FORK = os.environ.get("BENCH_STEEL_FORK", "1") == "1"  # SolverSteel entries fork inside their step
MJW_FORK = os.environ.get("BENCH_MJW_FORK", "0") == "1"  # SolverMuJoCo entries fork inside mujoco_warp's step

EXAMPLES = {
    "admm_contact": "newton.examples.multiphysics.example_admm_contact_solver",
    "mujoco_vbd_admm": "newton.examples.multiphysics.example_mujoco_vbd_admm_solver",
    "kamino_mujoco_admm": "newton.examples.multiphysics.example_kamino_mujoco_admm_solver",
    "franka_vbd_cable_admm": "newton.examples.multiphysics.example_mujoco_franka_vbd_cable_admm_solver",
    # NVIDIA/newton-samples (external checkout, BENCH_SAMPLES_ROOT): MuJoCo Franka + VBD cable/pulleys, ADMM
    "franka_pulley": "examples.multiphysics.example_franka_pulley_mechanical_advantage",
}
_mod_name = EXAMPLES[EXAMPLE]
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


@contextlib.contextmanager
def tasks_around_step(solver, symbol="frame"):
    """Record a frame that is not written with tasks: everything it launches
    outside ``solver.step`` goes into tasks that take the scope token read-write
    (the frame's sequence), and ``solver.step``, which records its own tasks, is
    called from the scope."""
    seq = scope_token()
    block = [warp_task(seq.rw(), symbol=symbol)]
    block[0].__enter__()
    step = solver.step

    def step_from_scope(*args, **kwargs):
        block[0].__exit__(None, None, None)
        try:
            return step(*args, **kwargs)
        finally:
            block[0] = warp_task(seq.rw(), symbol=symbol)
            block[0].__enter__()

    solver.step = step_from_scope
    try:
        yield
    finally:
        del solver.step
        block[0].__exit__(None, None, None)


def run_stf(ex):
    # warm eagerly (allocations, module loads must precede recording)
    ex.simulate()
    wp.synchronize()
    entries = ex.solver._entries
    for entry in entries.values():
        if type(entry.solver).__name__ == "SolverSteel":
            entry.solver.fork_tasks = STEEL_FORK
    ctx = stf.stackable_context()
    topo = None
    with contextlib.ExitStack() as shims:
        if MJW_FORK:
            for entry in entries.values():
                if type(entry.solver).__name__ == "SolverMuJoCo":
                    shims.enter_context(_mjw_installed(entry.solver, MjwForkShim(type(entry.solver)._mujoco_warp)))
        with scope(ctx, launchable=True, symbol=f"frame: {EXAMPLE}") as frame:
            with tasks_around_step(ex.solver):
                ex.simulate()  # the example's frame, as shipped

            if DOT:
                from cuda.bindings import runtime as cudart

                (err,) = cudart.cudaGraphDebugDotPrint(
                    frame.graph, DOT.encode(), cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose
                )
                assert err == cudart.cudaError_t.cudaSuccess, err
                topo = topology_summary(DOT)

            frame.launch()  # warm replay
            wp.synchronize()
            t0 = time.perf_counter()
            for _ in range(FRAMES):
                frame.launch()
            wp.synchronize()
            ms = (time.perf_counter() - t0) / FRAMES * 1e3
    ctx.finalize()
    release_graph_handles()
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
