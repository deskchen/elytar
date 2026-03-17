"""Benchmark environments. Auto-discovered from subpackages."""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

from envs.base import TaskRuntime

TaskBuilder = Callable[[Any], TaskRuntime]
TaskSceneBuilder = Callable[[Any, int, int, int, Any], tuple[list[Any], Any, dict[str, Any]]]

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


def get_task_scene_builder(name: str) -> TaskSceneBuilder | None:
    """Return optional scene-builder hook for mixed task mode."""
    resolved = resolve_task_name(name)
    build_fn = get_task_builder(resolved)
    module_name = build_fn.__module__
    module = importlib.import_module(module_name)
    scene_builder_name = f"build_scenes_into_{resolved}"
    scene_builder = getattr(module, scene_builder_name, None)
    if scene_builder is None and hasattr(module, "__path__"):
        # Task builders are often re-exported from package __init__.py, while scene builders live in .builder.
        try:
            builder_module = importlib.import_module(f"{module_name}.builder")
            scene_builder = getattr(builder_module, scene_builder_name, None)
        except Exception:
            scene_builder = None
    if callable(scene_builder):
        return scene_builder
    return None


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
    "get_task_scene_builder",
    "list_tasks",
    "resolve_task_name",
    "add_all_env_args",
]
