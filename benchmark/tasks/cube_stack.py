from __future__ import annotations

import math

import sapien

from .base import TaskRuntime

# Distinct colors for cubes (RGB)
_CUBE_COLORS = [
    [1.0, 0.2, 0.2],  # red
    [0.2, 0.8, 0.2],  # green
    [0.2, 0.2, 1.0],  # blue
    [1.0, 0.8, 0.2],  # yellow
    [0.2, 0.9, 0.9],  # cyan
    [1.0, 0.4, 0.8],  # pink
    [1.0, 0.5, 0.0],  # orange
    [0.6, 0.2, 0.8],  # purple
]


def _build_into_scene(scene: sapien.Scene, args, cube_count: int, half_size: float, spacing: float, render: bool) -> None:
    scene.set_timestep(args.dt)
    ground_material = [0.55, 0.45, 0.35] if render else None  # tan/brown
    scene.add_ground(altitude=0.0, render=render, render_material=ground_material)
    if render:
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [1, 1, 1])

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
                if render:
                    color = _CUBE_COLORS[created % len(_CUBE_COLORS)]
                    builder.add_box_visual(half_size=[half_size, half_size, half_size], material=color)
                builder.set_initial_pose(sapien.Pose(p=[x, y, z]))
                builder.build(name=f"cube_{created}")
                created += 1
            if created >= cube_count:
                break


def build_cube_stack(args) -> TaskRuntime:
    cube_count = args.cube_count
    half_size = args.cube_half_size
    spacing = args.cube_spacing if args.cube_spacing > 0 else half_size * 2.2
    render = getattr(args, "render", False)
    num_envs = getattr(args, "num_envs", 1)

    if num_envs > 1:
        px = sapien.physx.PhysxGpuSystem(device=args.device)
        scenes = []
        env_spacing = 4.0
        report_every = max(1, num_envs // 16)
        for i in range(num_envs):
            systems = [px]
            if render:
                systems.append(sapien.render.RenderSystem())
            scene = sapien.Scene(systems)
            px.set_scene_offset(scene, [i * env_spacing, 0.0, 0.0])
            _build_into_scene(scene, args, cube_count, half_size, spacing, render)
            scenes.append(scene)
        return TaskRuntime(
            name="cube_stack",
            scene=scenes[0],
            physx_system=px,
            metadata={
                "config": str(cube_count),
                "cube_count": cube_count,
                "cube_half_size": half_size,
                "cube_spacing": spacing,
                "num_envs": num_envs,
            },
            scenes=scenes,
        )

    systems = [sapien.physx.PhysxGpuSystem(device=args.device)]
    if render:
        systems.append(sapien.render.RenderSystem())
    scene = sapien.Scene(systems)
    _build_into_scene(scene, args, cube_count, half_size, spacing, render)

    return TaskRuntime(
        name="cube_stack",
        scene=scene,
        physx_system=scene.physx_system,
        metadata={
            "config": str(cube_count),
            "cube_count": cube_count,
            "cube_half_size": half_size,
            "cube_spacing": spacing,
        },
    )
