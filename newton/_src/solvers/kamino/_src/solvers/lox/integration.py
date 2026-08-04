# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pose integration for projected LOX body twists."""

from __future__ import annotations

import warp as wp

from ...core.math import compute_body_pose_update_with_logmap
from ...core.types import vec6f

__all__ = ["accept_projected_body_state", "integrate_projected_body_poses"]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _integrate_projected_body_poses(
    body_world: wp.array[wp.int32],
    world_time_step: wp.array[wp.float32],
    pose_previous: wp.array[wp.transformf],
    projected_twist: wp.array[vec6f],
    pose_candidate: wp.array[wp.transformf],
):
    body = wp.tid()
    twist = projected_twist[body]
    pose_candidate[body] = compute_body_pose_update_with_logmap(
        world_time_step[body_world[body]],
        pose_previous[body],
        wp.vec3f(twist[0], twist[1], twist[2]),
        wp.vec3f(twist[3], twist[4], twist[5]),
    )


@wp.kernel
def _accept_projected_body_state(
    body_world: wp.array[wp.int32],
    world_accepted: wp.array[wp.bool],
    pose_candidate: wp.array[wp.transformf],
    twist_candidate: wp.array[vec6f],
    pose_accepted: wp.array[wp.transformf],
    twist_accepted: wp.array[vec6f],
):
    body = wp.tid()
    if world_accepted[body_world[body]]:
        pose_accepted[body] = pose_candidate[body]
        twist_accepted[body] = twist_candidate[body]


def integrate_projected_body_poses(
    body_world: wp.array[wp.int32],
    world_time_step: wp.array[wp.float32],
    pose_previous: wp.array[wp.transformf],
    projected_twist: wp.array[vec6f],
    pose_candidate: wp.array[wp.transformf],
) -> None:
    """Integrate body poses from a fixed begin-of-step state.

    Args:
        body_world: World index of each packed body.
        world_time_step: Time step of each world [s].
        pose_previous: Begin-of-step body poses.
        projected_twist: End-of-step linear-first body twists [m/s, rad/s].
        pose_candidate: Output end-of-step body poses.
    """
    body_count = body_world.shape[0]
    if (
        pose_previous.shape[0] != body_count
        or projected_twist.shape[0] != body_count
        or pose_candidate.shape[0] != body_count
    ):
        raise ValueError("Pose, twist, and body-world arrays must have identical lengths.")
    wp.launch(
        _integrate_projected_body_poses,
        dim=body_count,
        inputs=[body_world, world_time_step, pose_previous, projected_twist],
        outputs=[pose_candidate],
        device=pose_candidate.device,
    )


def accept_projected_body_state(
    body_world: wp.array[wp.int32],
    world_accepted: wp.array[wp.bool],
    pose_candidate: wp.array[wp.transformf],
    twist_candidate: wp.array[vec6f],
    pose_accepted: wp.array[wp.transformf],
    twist_accepted: wp.array[vec6f],
) -> None:
    """Update the last finite body state in accepted worlds only."""
    body_count = body_world.shape[0]
    if world_accepted.shape[0] == 0:
        raise ValueError("world_accepted must contain at least one world.")
    if any(array.shape[0] != body_count for array in (pose_candidate, twist_candidate, pose_accepted, twist_accepted)):
        raise ValueError("Pose, twist, and body-world arrays must have identical lengths.")
    wp.launch(
        _accept_projected_body_state,
        dim=body_count,
        inputs=[body_world, world_accepted, pose_candidate, twist_candidate],
        outputs=[pose_accepted, twist_accepted],
        device=pose_accepted.device,
    )
