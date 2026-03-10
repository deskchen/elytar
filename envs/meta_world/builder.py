"""Shared builders for Meta-World tasks. No Isaac Gym or MTEnvs dependency."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import sapien

from envs.base import TaskRuntime

try:
    import torch
except ImportError:
    torch = None  # type: ignore
from envs.meta_world.objects import (
    CYLINDER_HALF_LENGTH,
    CYLINDER_RADIUS,
)

# Meta-World layout (from franka_base._create_envs)
TABLE_POS = [0.0, 0.0, 1.0]
TABLE_THICKNESS = 0.054
TABLE_HALF_XY = (0.6, 0.6)  # 1.2x1.2
TABLE_STAND_HEIGHT = 0.01
TABLE_STAND_HALF_XY = (0.1, 0.1)  # 0.2x0.2
TABLE_STAND_POS = [-0.6, 0.0, 1.0 + TABLE_THICKNESS / 2 + TABLE_STAND_HEIGHT / 2]
FRANKA_POS = [-0.55, 0.0, 1.0 + TABLE_THICKNESS / 2 + TABLE_STAND_HEIGHT]
TABLE_SURFACE_Z = TABLE_POS[2] + TABLE_THICKNESS  # Top of table (box center + half thickness)

FRANKA_DEFAULT_DOF = [0.0, 0.2, 0, -2.5, -0.17, 2.6, -0.7, 0.04, 0.04]


def _make_before_step_gpu(
    physx_system: sapien.physx.PhysxGpuSystem,
    frankas: list[sapien.Articulation],
    default_dof: list[float],
):
    """Create before_step that sets drive targets via GPU API (required when Direct GPU API is enabled)."""
    if torch is None:
        raise RuntimeError(
            "Meta-World Franka tasks with GPU PhysX require PyTorch for drive targets. "
            "Install: pip install torch"
        )
    default_dof_tup = tuple(default_dof)
    cached = [None, None]  # [default_tensor, zero_vel]

    def before_step(step_index: int, time_s: float) -> None:
        target_qpos = physx_system.cuda_articulation_target_qpos.torch()
        target_qvel = physx_system.cuda_articulation_target_qvel.torch()
        if cached[0] is None:
            device = target_qpos.device
            cached[0] = torch.tensor(default_dof_tup, dtype=torch.float32, device=device)
            cached[1] = torch.zeros(len(default_dof_tup), dtype=torch.float32, device=device)
        default_tensor, zero_vel = cached[0], cached[1]
        dof = min(len(default_dof_tup), target_qpos.shape[1])
        for franka in frankas:
            gpu_idx = franka.get_gpu_index()
            target_qpos[gpu_idx, :dof].copy_(default_tensor[:dof])
            target_qvel[gpu_idx, :dof].copy_(zero_vel[:dof])
        physx_system.gpu_apply_articulation_target_position()
        physx_system.gpu_apply_articulation_target_velocity()

    return before_step


def _assets_dir() -> Path:
    return Path(__file__).parent / "assets"


def _build_table(scene: sapien.Scene, render: bool) -> None:
    """Build table and table stand (static)."""
    table_builder = scene.create_actor_builder()
    table_builder.set_physx_body_type("static")
    table_half = (*TABLE_HALF_XY, TABLE_THICKNESS / 2)
    box_pose = sapien.Pose(p=[0, 0, TABLE_THICKNESS / 2])
    table_builder.add_box_collision(pose=box_pose, half_size=table_half)
    if render:
        mat = sapien.render.RenderMaterial(base_color=[0.6, 0.6, 0.65, 1.0], roughness=0.9)
        table_builder.add_box_visual(pose=box_pose, half_size=table_half, material=mat)
    table_builder.set_initial_pose(sapien.Pose(p=TABLE_POS))
    table_builder.build_static(name="table")

    stand_builder = scene.create_actor_builder()
    stand_builder.set_physx_body_type("static")
    stand_half = (*TABLE_STAND_HALF_XY, TABLE_STAND_HEIGHT / 2)
    stand_pose = sapien.Pose(p=[0, 0, TABLE_STAND_HEIGHT / 2])
    stand_builder.add_box_collision(pose=stand_pose, half_size=stand_half)
    if render:
        mat = sapien.render.RenderMaterial(base_color=[0.4, 0.4, 0.45, 1.0], roughness=0.9)
        stand_builder.add_box_visual(pose=stand_pose, half_size=stand_half, material=mat)
    stand_builder.set_initial_pose(sapien.Pose(p=TABLE_STAND_POS))
    stand_builder.build_static(name="table_stand")


def _build_cylinder(
    scene: sapien.Scene,
    name: str,
    pos: tuple[float, float, float],
    render: bool,
) -> None:
    """Build a dynamic cylinder (Meta-World push/pick_place object)."""
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("dynamic")
    builder.add_cylinder_collision(radius=CYLINDER_RADIUS, half_length=CYLINDER_HALF_LENGTH)
    if render:
        mat = sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 1.0], roughness=0.5)
        builder.add_cylinder_visual(radius=CYLINDER_RADIUS, half_length=CYLINDER_HALF_LENGTH, material=mat)
    builder.set_initial_pose(sapien.Pose(p=list(pos)))
    builder.build(name=name)


def _load_franka(
    scene: sapien.Scene,
    render: bool,
    default_dof: list[float],
) -> sapien.Articulation:
    """Load Franka from bundled URDF and set drive properties."""
    assets = _assets_dir()
    urdf_path = assets / "franka_description" / "robots" / "franka_panda_gripper.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Franka URDF not found: {urdf_path}")

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    package_dir = str(assets)
    articulation = loader.load(str(urdf_path), package_dir=package_dir)

    articulation.root_pose = sapien.Pose(p=FRANKA_POS)

    # Set drive properties: arm (position) + gripper (effort)
    # Do NOT access articulation.qpos here: GPU PhysX is not initialized until after builder returns.
    idx = 0
    for joint in articulation.active_joints:
        dof = int(joint.dof)
        if dof <= 0:
            continue
        if idx < 7:  # arm
            joint.set_drive_properties(
                stiffness=400.0,
                damping=40.0,
                force_limit=87.0,
                mode="force",
            )
        else:  # gripper
            joint.set_drive_properties(
                stiffness=0.0,
                damping=0.0,
                force_limit=80.0,
                mode="force",
            )
        idx += dof
    # Drive targets are set via before_step + GPU API (joint.set_drive_target is illegal with Direct GPU API).

    return articulation


CYLINDER_POS = (0.3, 0.0)


def _build_into_scene_cylinder(
    scene: sapien.Scene,
    args: argparse.Namespace,
    render: bool,
) -> None:
    """Build table + cylinder. No robot."""
    scene.set_timestep(args.dt)
    scene.add_ground(altitude=0.0, render=render)
    _build_table(scene, render)
    z = TABLE_SURFACE_Z + CYLINDER_HALF_LENGTH
    _build_cylinder(scene, "cylinder", (*CYLINDER_POS, z), render)
    if render:
        scene.set_ambient_light([0.25, 0.25, 0.25])
        scene.add_directional_light([0, 1, -1], [0.8, 0.8, 0.8])


def _build_into_scene_franka_cylinder(
    scene: sapien.Scene,
    args: argparse.Namespace,
    render: bool,
) -> tuple[sapien.Articulation, object]:
    """Build Franka + table + cylinder."""
    scene.set_timestep(args.dt)
    scene.add_ground(altitude=0.0, render=render)
    _build_table(scene, render)
    franka = _load_franka(scene, render, FRANKA_DEFAULT_DOF)
    z = TABLE_SURFACE_Z + CYLINDER_HALF_LENGTH
    _build_cylinder(scene, "cylinder", (*CYLINDER_POS, z), render)
    if render:
        scene.set_ambient_light([0.25, 0.25, 0.25])
        scene.add_directional_light([0, 1, -1], [0.8, 0.8, 0.8])
    return franka, None


def add_args(parser: argparse.ArgumentParser) -> None:
    pass


def build_cylinder(args: argparse.Namespace) -> TaskRuntime:
    """Table + cylinder. Physics-only, no robot."""
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
            _build_into_scene_cylinder(scene, args, render)
            scenes.append(scene)
        return TaskRuntime(
            name="cylinder",
            scene=scenes[0],
            physx_system=px,
            metadata={"num_envs": num_envs},
            scenes=scenes,
        )

    systems = [sapien.physx.PhysxGpuSystem(device=args.device)]
    if render:
        systems.append(sapien.render.RenderSystem())
    scene = sapien.Scene(systems)
    _build_into_scene_cylinder(scene, args, render)
    return TaskRuntime(
        name="cylinder",
        scene=scene,
        physx_system=scene.physx_system,
        metadata={"num_envs": num_envs},
    )


def build_franka_cylinder(args: argparse.Namespace) -> TaskRuntime:
    """Franka + table + cylinder."""
    render = getattr(args, "render", False)
    num_envs = getattr(args, "num_envs", 1)

    if num_envs > 1:
        px = sapien.physx.PhysxGpuSystem(device=args.device)
        scenes = []
        frankas = []
        env_spacing = 50.0
        scene_grid_length = int(math.ceil(math.sqrt(num_envs)))
        for scene_idx in range(num_envs):
            scene_x = scene_idx % scene_grid_length - scene_grid_length // 2
            scene_y = scene_idx // scene_grid_length - scene_grid_length // 2
            systems = [px, sapien.render.RenderSystem()]
            scene = sapien.Scene(systems)
            px.set_scene_offset(scene, [scene_x * env_spacing, scene_y * env_spacing, 0.0])
            franka, _ = _build_into_scene_franka_cylinder(scene, args, render)
            scenes.append(scene)
            frankas.append(franka)
        return TaskRuntime(
            name="franka_cylinder",
            scene=scenes[0],
            physx_system=px,
            before_step=_make_before_step_gpu(px, frankas, FRANKA_DEFAULT_DOF),
            metadata={"num_envs": num_envs},
            scenes=scenes,
        )

    systems = [sapien.physx.PhysxGpuSystem(device=args.device), sapien.render.RenderSystem()]
    scene = sapien.Scene(systems)
    franka, _ = _build_into_scene_franka_cylinder(scene, args, render)
    return TaskRuntime(
        name="franka_cylinder",
        scene=scene,
        physx_system=scene.physx_system,
        before_step=_make_before_step_gpu(scene.physx_system, [franka], FRANKA_DEFAULT_DOF),
        metadata={"num_envs": num_envs},
    )
