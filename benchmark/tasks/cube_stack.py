from __future__ import annotations

import math

import sapien

from .base import TaskRuntime


_DIFFICULTY_TO_COUNT = {
    "easy": 27,
    "medium": 125,
    "hard": 343,
}


def build_cube_stack(args) -> TaskRuntime:
    cube_count = (
        args.cube_count
        if args.cube_count is not None
        else _DIFFICULTY_TO_COUNT[args.difficulty]
    )
    half_size = args.cube_half_size
    spacing = args.cube_spacing if args.cube_spacing > 0 else half_size * 2.2

    scene = sapien.Scene([sapien.physx.PhysxGpuSystem(device=args.device)])
    scene.set_timestep(args.dt)
    scene.add_ground(altitude=0.0, render=False)

    levels = max(1, int(round(cube_count ** (1.0 / 3.0))))
    grid_side = max(1, math.ceil(math.sqrt(cube_count / levels)))
    created = 0
    for level in range(levels):
        z = half_size * (2.0 * level + 1.0) + 0.01
        for ix in range(grid_side):
            for iy in range(grid_side):
                if created >= cube_count:
                    break
                x = (ix - 0.5 * (grid_side - 1)) * spacing
                y = (iy - 0.5 * (grid_side - 1)) * spacing

                builder = scene.create_actor_builder()
                builder.set_physx_body_type("dynamic")
                builder.add_box_collision(half_size=[half_size, half_size, half_size])
                builder.set_initial_pose(sapien.Pose(p=[x, y, z]))
                builder.build(name=f"cube_{created}")
                created += 1
            if created >= cube_count:
                break

    return TaskRuntime(
        name="cube_stack",
        scene=scene,
        physx_system=scene.physx_system,
        metadata={
            "difficulty": args.difficulty,
            "cube_count": cube_count,
            "cube_half_size": half_size,
            "cube_spacing": spacing,
        },
    )

