from __future__ import annotations

import argparse
import math
import random

import sapien

from envs.base import TaskRuntime


def _add_static_box(scene, half_size, pose):
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    builder.add_box_collision(half_size=half_size)
    builder.set_initial_pose(pose)
    builder.build()


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ball-count", type=int, default=4, help="Number of balls")
    parser.add_argument("--ball-radius", type=float, default=0.02)
    parser.add_argument("--container-half-extent", type=float, default=0.6)
    parser.add_argument("--container-wall-height", type=float, default=0.45)
    parser.add_argument("--container-wall-thickness", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)


def build_pouring_balls(args) -> TaskRuntime:
    ball_count = args.ball_count
    radius = args.ball_radius
    container_half_extent = args.container_half_extent
    wall_height = args.container_wall_height
    wall_thickness = args.container_wall_thickness

    scene = sapien.Scene([sapien.physx.PhysxGpuSystem(device=args.device)])
    scene.set_timestep(args.dt)
    scene.add_ground(altitude=0.0, render=False)

    # Floor
    _add_static_box(
        scene=scene,
        half_size=[container_half_extent, container_half_extent, wall_thickness],
        pose=sapien.Pose(p=[0.0, 0.0, wall_thickness]),
    )

    # Four walls
    wall_z = wall_height + wall_thickness
    wall_half_z = wall_height
    x_offset = container_half_extent + wall_thickness
    y_offset = container_half_extent + wall_thickness
    _add_static_box(
        scene=scene,
        half_size=[wall_thickness, container_half_extent + wall_thickness, wall_half_z],
        pose=sapien.Pose(p=[x_offset, 0.0, wall_z]),
    )
    _add_static_box(
        scene=scene,
        half_size=[wall_thickness, container_half_extent + wall_thickness, wall_half_z],
        pose=sapien.Pose(p=[-x_offset, 0.0, wall_z]),
    )
    _add_static_box(
        scene=scene,
        half_size=[container_half_extent + wall_thickness, wall_thickness, wall_half_z],
        pose=sapien.Pose(p=[0.0, y_offset, wall_z]),
    )
    _add_static_box(
        scene=scene,
        half_size=[container_half_extent + wall_thickness, wall_thickness, wall_half_z],
        pose=sapien.Pose(p=[0.0, -y_offset, wall_z]),
    )

    rng = random.Random(args.seed)
    grid = max(1, math.ceil(ball_count ** (1.0 / 3.0)))
    spacing = radius * 2.2
    base_x = -0.5 * (grid - 1) * spacing
    base_y = -0.5 * (grid - 1) * spacing
    start_z = wall_height * 2.5 + radius

    for idx in range(ball_count):
        ix = idx % grid
        iy = (idx // grid) % grid
        iz = idx // (grid * grid)
        x = base_x + ix * spacing + rng.uniform(-0.05 * radius, 0.05 * radius)
        y = base_y + iy * spacing + rng.uniform(-0.05 * radius, 0.05 * radius)
        z = start_z + iz * spacing

        builder = scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        builder.add_sphere_collision(radius=radius)
        builder.set_initial_pose(sapien.Pose(p=[x, y, z]))
        builder.build(name=f"ball_{idx}")

    return TaskRuntime(
        name="pouring_balls",
        scene=scene,
        physx_system=scene.physx_system,
        metadata={
            "config": str(ball_count),
            "ball_count": ball_count,
            "ball_radius": radius,
            "container_half_extent": container_half_extent,
            "container_wall_height": wall_height,
            "container_wall_thickness": wall_thickness,
            "seed": args.seed,
        },
    )
