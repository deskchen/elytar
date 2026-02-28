from __future__ import annotations

import math
import random

import sapien

from .base import TaskRuntime


_DIFFICULTY_TO_COUNT = {
    "easy": 256,
    "medium": 1024,
    "hard": 4096,
}


def _add_static_box(scene, half_size, pose):
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    builder.add_box_collision(half_size=half_size)
    builder.set_initial_pose(pose)
    builder.build()


def build_pouring_balls(args) -> TaskRuntime:
    ball_count = (
        args.ball_count
        if args.ball_count is not None
        else _DIFFICULTY_TO_COUNT[args.difficulty]
    )
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
            "difficulty": args.difficulty,
            "ball_count": ball_count,
            "ball_radius": radius,
            "container_half_extent": container_half_extent,
            "container_wall_height": wall_height,
            "container_wall_thickness": wall_thickness,
            "seed": args.seed,
        },
    )

