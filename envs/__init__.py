"""Benchmark environments. Auto-discovered from subpackages."""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

from envs.base import TaskRuntime

TaskBuilder = Callable[[Any], TaskRuntime]

TASK_ALIASES = {
    "cube": "cube_stack",
    "balls": "pouring_balls",
    "humanoid": "humanoid_from_urdf",
    # Meta-World legacy names -> franka_cylinder
    "pick_place": "franka_cylinder",
    "push": "franka_cylinder",
    "reach": "franka_cylinder",
    "sweep": "franka_cylinder",
    "push_primitive": "cylinder",
    "sweep_primitive": "cylinder",
}


def discover_envs() -> dict[str, tuple[TaskBuilder, Callable[[Any], None] | None]]:
    """Discover envs: name -> (build_fn, add_args_fn or None)."""
    envs: dict[str, tuple[TaskBuilder, Callable[[Any], None] | None]] = {}
    for _importer, modname, _ispkg in pkgutil.iter_modules(__path__, prefix="envs."):
        if modname == "envs.base":
            continue
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        add_args_fn = getattr(mod, "add_args", None)
        for attr in dir(mod):
            if attr.startswith("build_") and callable(getattr(mod, attr)):
                build_fn = getattr(mod, attr)
                name = attr[6:]  # build_cube_stack -> cube_stack
                envs[name] = (build_fn, add_args_fn)
    return envs


_BUILDERS: dict[str, TaskBuilder] | None = None


def _get_builders() -> dict[str, TaskBuilder]:
    global _BUILDERS
    if _BUILDERS is None:
        discovered = discover_envs()
        _BUILDERS = {name: build for name, (build, _) in discovered.items()}
    return _BUILDERS


def resolve_task_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return TASK_ALIASES.get(normalized, normalized)


def get_task_builder(name: str) -> TaskBuilder:
    resolved = resolve_task_name(name)
    builders = _get_builders()
    if resolved not in builders:
        available = ", ".join(sorted(builders))
        raise ValueError(f"Unknown task '{name}'. Available tasks: {available}")
    return builders[resolved]


def list_tasks() -> list[str]:
    return sorted(_get_builders().keys())


def add_all_env_args(parser) -> None:
    """Call add_args from each discovered env."""
    discovered = discover_envs()
    for _name, (_build, add_args_fn) in discovered.items():
        if add_args_fn is not None:
            add_args_fn(parser)


__all__ = [
    "TaskRuntime",
    "discover_envs",
    "get_task_builder",
    "list_tasks",
    "resolve_task_name",
    "add_all_env_args",
]
