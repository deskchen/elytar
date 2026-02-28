from __future__ import annotations

import math
from pathlib import Path

import sapien

from .base import TaskRuntime


def build_humanoid_from_urdf(args) -> TaskRuntime:
    if not args.humanoid_urdf:
        raise ValueError("--humanoid-urdf is required when task includes humanoid_from_urdf")

    urdf_path = Path(args.humanoid_urdf).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Humanoid URDF not found: {urdf_path}")

    scene = sapien.Scene([sapien.physx.PhysxGpuSystem(device=args.device)])
    scene.set_timestep(args.dt)
    scene.add_ground(altitude=0.0, render=False)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    articulation = loader.load(str(urdf_path))

    # Keep the root above the ground plane at startup.
    root_pose = articulation.root_pose
    root_height = max(float(root_pose.p[2]), args.humanoid_root_height)
    articulation.root_pose = sapien.Pose(
        p=[float(root_pose.p[0]), float(root_pose.p[1]), root_height],
        q=root_pose.q,
    )

    qpos0 = articulation.qpos
    joint_specs: list[tuple[object, float, float]] = []
    qpos_offset = 0

    for joint_idx, joint in enumerate(articulation.active_joints):
        dof = int(joint.dof)
        if dof <= 0:
            continue

        if dof == 1:
            base = float(qpos0[qpos_offset])
            phase = joint_idx * (math.pi / 4.0)
            joint.set_drive_properties(
                stiffness=args.humanoid_joint_stiffness,
                damping=args.humanoid_joint_damping,
                force_limit=args.humanoid_joint_force_limit,
                mode="force",
            )
            joint.set_drive_target(base)
            joint.set_drive_velocity_target(0.0)
            joint_specs.append((joint, base, phase))
        qpos_offset += dof

    if args.humanoid_motion == "walk":
        frequency_hz = 1.5
        amplitude = args.humanoid_target_scale
    else:
        frequency_hz = 3.0
        amplitude = args.humanoid_target_scale * 1.8

    def before_step(step_index: int, time_s: float) -> None:
        phase = 2.0 * math.pi * frequency_hz * time_s
        for joint, base, joint_phase in joint_specs:
            target = base + amplitude * math.sin(phase + joint_phase)
            joint.set_drive_target(float(target))

    return TaskRuntime(
        name="humanoid_from_urdf",
        scene=scene,
        physx_system=scene.physx_system,
        before_step=before_step,
        metadata={
            "difficulty": args.difficulty,
            "humanoid_urdf": str(urdf_path),
            "humanoid_motion": args.humanoid_motion,
            "humanoid_target_scale": args.humanoid_target_scale,
            "humanoid_joint_stiffness": args.humanoid_joint_stiffness,
            "humanoid_joint_damping": args.humanoid_joint_damping,
            "humanoid_joint_force_limit": args.humanoid_joint_force_limit,
        },
    )

