# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Probe two independent MuJoCo solvers recorded as sibling STF tasks.

This is intentionally a standalone experiment script, not a committed test.
It checks whether Newton can call ordinary SolverMuJoCo instances from
independent CUDASTF tasks while MuJoCo Warp records its normal conditional
solver loop inside each task.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import warp as wp

import mujoco_warp
import newton
from newton.solvers import SolverMuJoCo


def _make_solver_case(device: wp.Device, *, x_offset: float = 0.0):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.8
    builder.add_ground_plane()

    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(x_offset, 0.0, 0.35), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-2),
    )
    builder.add_shape_sphere(body, radius=0.12)

    model = builder.finalize(device=device)
    solver = SolverMuJoCo(
        model,
        use_mujoco_contacts=True,
        iterations=24,
        ls_iterations=8,
        njmax=32,
        nconmax=16,
        solver="newton",
        integrator="implicitfast",
    )
    if hasattr(solver.mjw_model.opt, "graph_conditional"):
        solver.mjw_model.opt.graph_conditional = True

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    return model, solver, state_0, state_1, control, contacts


def _step_case(solver: SolverMuJoCo, state_0, state_1, control, contacts, dt: float):
    state_0.clear_forces()
    solver.step(state_0, state_1, control, contacts, dt)


def run_probe(dot_path: str, launches: int) -> None:
    wp.init()
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("This probe requires a CUDA device")

    from warp import stf_experimental as wp_stf  # noqa: PLC0415

    if not wp_stf.is_available():
        raise RuntimeError("warp.stf_experimental is not available")

    with wp.ScopedDevice(device):
        case_a = _make_solver_case(device, x_offset=-0.3)
        case_b = _make_solver_case(device, x_offset=0.3)

        graph = wp_stf.task_graph()
        ctx = graph.context
        token_a = ctx.token()
        token_b = ctx.token()
        joined = ctx.token()

        wp_stf.warmup(device=device)

        with graph:
            with ctx.task(token_a.write(), symbol="mujoco_solver_a") as (_stream,):
                _step_case(case_a[1], case_a[2], case_a[3], case_a[4], case_a[5], 1.0 / 60.0)

            with ctx.task(token_b.write(), symbol="mujoco_solver_b") as (_stream,):
                _step_case(case_b[1], case_b[2], case_b[3], case_b[4], case_b[5], 1.0 / 60.0)

            with ctx.task(token_a.read(), token_b.read(), joined.write(), symbol="mujoco_join"):
                pass

        wp_stf.dump_dot(graph.raw, dot_path, flags=0)

        for _ in range(launches):
            graph.launch()
        wp.synchronize_device(device)

        body_q_a = case_a[3].body_q.numpy()
        body_q_b = case_b[3].body_q.numpy()
        graph.finalize()

    print(f"mujoco_warp: {mujoco_warp.__file__}")
    print(f"dot: {os.path.abspath(dot_path)}")
    print(f"body_q_a: {body_q_a}")
    print(f"body_q_b: {body_q_b}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dot", default="mujoco_two_solver_stf.dot")
    parser.add_argument("--launches", type=int, default=2)
    args = parser.parse_args()
    run_probe(args.dot, args.launches)


if __name__ == "__main__":
    main()
