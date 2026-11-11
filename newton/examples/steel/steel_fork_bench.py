"""Steel intra-step fork bench.

Standalone SolverSteel scene (two Franka arms sweeping six boxes, from
newton.examples.steel.example_steel_determinism_arms.build_scene, replicated
BENCH_WORLDS times, Steel defaults) so that Steel itself is the critical path.

Three arms, one process each or all in one (BENCH_ARM=native|flat|fork|all):
  native  the frame captured by wp.ScopedCapture, replayed (Newton as shipped)
  flat    the same frame recorded as cuda.stf tasks, every task in the frame's
          sequence (STF overhead control)
  fork    the same frame: SolverSteel records its mass-matrix chain and its row
          build as two tasks that may overlap (newton._src.solvers.tasks)

Under a recording every launch belongs to a task: the bench puts its own
launches (forces, collision) in a task of the frame's sequence and calls the
solver, which records its own tasks, from the scope.

Rules: warm eagerly before recording, record once, replay; fingerprint =
sum |body_q| after the same total frame count; BENCH_DOT dumps the recorded
graph and prints its topology.
"""

import os
import time

import numpy as np
import warp as wp

import newton
import cuda.stf._experimental as stf
from cuda.stf._experimental.interop.warp import release_graph_handles, scope, scope_token, warp_task
from newton.examples.steel.example_steel_determinism_arms import build_scene
from newton.solvers import SolverSteel

from newton.examples.multiphysics.admm_stf_bench import topology_summary

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
        with warp_task(scope_token().rw(), symbol="collide"):  # no-ops when nothing is recorded
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
    sc.solver.fork_tasks = fork
    ctx = stf.stackable_context()
    topo = None
    with scope(ctx, launchable=True, symbol="frame: steel") as frame:
        sc.frame_native()  # SolverSteel records its own tasks
        if DOT:
            from cuda.bindings import runtime as cudart

            (err,) = cudart.cudaGraphDebugDotPrint(
                frame.graph, DOT.encode(), cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose
            )
            assert err == cudart.cudaError_t.cudaSuccess, err
            topo = topology_summary(DOT)
        frame.launch()
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
