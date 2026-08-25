# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the captured CUDASTF scheduling experiment for Kamino LOX."""

import argparse
import json
import statistics
import time

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino.config import LOXSolverConfig


def build_model(device, world_count, cloth_dim, links):
    builder = newton.ModelBuilder()
    for world in range(world_count):
        builder.begin_world()
        offset = 3.0 * world
        builder.add_cloth_grid(
            pos=wp.vec3(offset, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=cloth_dim,
            dim_y=cloth_dim,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.1,
            tri_ke=1000.0,
            tri_ka=1000.0,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=10.0,
            edge_kd=0.0,
        )
        joints = []
        parent = -1
        for link in range(links):
            body = builder.add_link(
                xform=wp.transform((offset, 0.0, 2.0 + 0.1 * link), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            )
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    body,
                    axis=(0.0, 1.0, 0.0),
                    parent_xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()),
                    child_xform=wp.transform_identity(),
                )
            )
            parent = body
        builder.add_articulation(joints)
        builder.end_world()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    positions = model.particle_q.numpy()
    positions[:, 2] += 0.02 * np.sin(7.0 * positions[:, 0]) * np.sin(9.0 * positions[:, 1])
    model.particle_q.assign(positions)
    return model


def make_graph(device, args, use_stf):
    model = build_model(device, args.world_count, args.cloth_dim, args.links)
    state_in, state_out = model.state(), model.state()
    solver = newton.solvers.SolverKamino(
        model,
        config=newton.solvers.SolverKamino.Config(
            dynamics_solver="lox",
            lox=LOXSolverConfig(
                use_stf=use_stf,
                use_graph_conditionals=True,
                max_iterations=args.lox_iterations,
                deformable_cr_iterations=args.cr_iterations,
                deformable_direct_max_particles=0,
                position_tolerance=1.0e-12,
                rotation_tolerance=1.0e-12,
                velocity_tolerance=1.0e-12,
            ),
        ),
    )

    def step():
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)

    step()
    wp.synchronize_device(device)
    solver.reset(state_in)
    with wp.ScopedCapture(device=device) as capture:
        step()
    return capture.graph


def time_batch(graph, device, batch):
    start = time.perf_counter()
    for _ in range(batch):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    return 1000.0 * (time.perf_counter() - start) / batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-count", type=int, default=16)
    parser.add_argument("--cloth-dim", type=int, default=12)
    parser.add_argument("--links", type=int, default=16)
    parser.add_argument("--lox-iterations", type=int, default=8)
    parser.add_argument("--cr-iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--profile-replays", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    wp.init()
    device = wp.get_device(args.device)
    graphs = {
        "baseline": make_graph(device, args, use_stf=False),
        "stf": make_graph(device, args, use_stf=True),
    }

    for graph in graphs.values():
        for _ in range(args.warmup):
            wp.capture_launch(graph)
    wp.synchronize_device(device)

    samples = {name: [] for name in graphs}
    for repeat in range(args.repeats):
        order = ("baseline", "stf") if repeat % 2 == 0 else ("stf", "baseline")
        for name in order:
            samples[name].append(time_batch(graphs[name], device, args.batch))

    medians = {name: statistics.median(values) for name, values in samples.items()}
    result = {
        "world_count": args.world_count,
        "cloth_dim": args.cloth_dim,
        "particles": args.world_count * (args.cloth_dim + 1) ** 2,
        "links": args.world_count * args.links,
        "lox_iterations": args.lox_iterations,
        "cr_iterations": args.cr_iterations,
        "baseline_ms": medians["baseline"],
        "stf_ms": medians["stf"],
        "speedup": medians["baseline"] / medians["stf"],
        "change_percent": 100.0 * (medians["stf"] / medians["baseline"] - 1.0),
        "baseline_samples": samples["baseline"],
        "stf_samples": samples["stf"],
    }
    print(json.dumps(result, indent=2))

    if args.profile_replays:
        with wp.ScopedCudaProfiler(device):
            for name, graph in graphs.items():
                with wp.ScopedTimer(name, print=False, use_nvtx=True):
                    for _ in range(args.profile_replays):
                        wp.capture_launch(graph)
                    wp.synchronize_device(device)


if __name__ == "__main__":
    main()
