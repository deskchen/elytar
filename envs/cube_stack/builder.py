"""
ManiSkill StackCube-v1 scene port: table + ground + 2 cubes (red, green).
Matches mani_skill/envs/tasks/tabletop/stack_cube.py for setup verification.
No robot - physics-only benchmark.
"""
from __future__ import annotations

import argparse
import math

import sapien

from envs.base import TaskRuntime


def _build_into_scene(
    scene: sapien.Scene,
    args,
    render: bool,
) -> None:
    scene.set_timestep(args.dt)

    # Table: horizontal box. Use same pose for collision and visual so they align.
    # Actor at z=-table_half_z; box center at (0,0,table_half_z) in actor frame -> world z=0; top at z=table_half_z.
    table_half_xy = (2.418 / 2, 1.209 / 2)
    table_half_z = 0.9196429 / 2
    table_half = (*table_half_xy, table_half_z)
    box_pose = sapien.Pose(p=[0, 0, table_half_z])
    builder = scene.create_actor_builder()
    builder.add_box_collision(pose=box_pose, half_size=table_half)
    if render:
        mat = sapien.render.RenderMaterial(base_color=[0.72, 0.55, 0.42, 1.0], roughness=0.9)
        builder.add_box_visual(pose=box_pose, half_size=table_half, material=mat)
    builder.set_initial_pose(sapien.Pose(p=[-0.12, 0, -table_half_z]))
    builder.build_static(name="table")

    # Ground (below table)
    ground_material = [0.55, 0.45, 0.35] if render else None
    scene.add_ground(altitude=-table_half_z - 0.1, render=render, render_material=ground_material)

    if render:
        scene.set_ambient_light([0.25, 0.25, 0.25])
        scene.add_directional_light([0, 1, -1], [0.8, 0.8, 0.8])
        scene.add_directional_light([1, 0, -0.5], [0.4, 0.4, 0.4])

    # Cubes: cubeB (blue) on table, cubeA (red) stacked on top. half_size=0.05 when render for visibility.
    # Table top z = actor.z + box_offset.z + half_z = -table_half_z + table_half_z + table_half_z = table_half_z
    half_size = 0.05 if render else 0.02
    table_top_z = table_half_z
    cubeB_z = table_top_z + half_size
    cubeA_z = cubeB_z + 2 * half_size

    cubeB_builder = scene.create_actor_builder()
    cubeB_builder.set_physx_body_type("dynamic")
    cubeB_builder.add_box_collision(half_size=[half_size] * 3)
    if render:
        mat_blue = sapien.render.RenderMaterial(base_color=[0.0, 0.0, 1.0, 1.0], roughness=0.5)
        cubeB_builder.add_box_visual(half_size=[half_size] * 3, material=mat_blue)
    cubeB_builder.set_initial_pose(sapien.Pose(p=[0, 0, cubeB_z]))
    cubeB_builder.build(name="cubeB")

    cubeA_builder = scene.create_actor_builder()
    cubeA_builder.set_physx_body_type("dynamic")
    cubeA_builder.add_box_collision(half_size=[half_size] * 3)
    if render:
        mat_red = sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 1.0], roughness=0.5)
        cubeA_builder.add_box_visual(half_size=[half_size] * 3, material=mat_red)
    cubeA_builder.set_initial_pose(sapien.Pose(p=[0, 0, cubeA_z]))
    cubeA_builder.build(name="cubeA")


def add_args(parser: argparse.ArgumentParser) -> None:
    """Cube stack uses only common args (num_envs, render, dt, device). No extra args."""
    pass


def build_cube_stack(args) -> TaskRuntime:
    """ManiSkill StackCube scene: table + 2 cubes."""
    render = getattr(args, "render", False)
    num_envs = getattr(args, "num_envs", 1)

    if num_envs > 1:
        px = sapien.physx.PhysxGpuSystem(device=args.device)
        scenes = []
        env_spacing = 50.0
        scene_grid_length = int(math.ceil(math.sqrt(num_envs)))
        for scene_idx in range(num_envs):
            scene_x = scene_idx % scene_grid_length - scene_grid_length // 2
            scene_y = scene_idx // scene_grid_length - scene_grid_length // 2
            systems = [px]
            if render:
                systems.append(sapien.render.RenderSystem())
            scene = sapien.Scene(systems)
            px.set_scene_offset(scene, [scene_x * env_spacing, scene_y * env_spacing, 0.0])
            _build_into_scene(scene, args, render)
            scenes.append(scene)
        return TaskRuntime(
            name="cube_stack",
            scene=scenes[0],
            physx_system=px,
            metadata={"num_envs": num_envs},
            scenes=scenes,
        )

    systems = [sapien.physx.PhysxGpuSystem(device=args.device)]
    if render:
        systems.append(sapien.render.RenderSystem())
    scene = sapien.Scene(systems)
    _build_into_scene(scene, args, render)

    return TaskRuntime(
        name="cube_stack",
        scene=scene,
        physx_system=scene.physx_system,
        metadata={"num_envs": num_envs},
    )
