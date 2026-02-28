from __future__ import annotations

from collections.abc import Callable

from .base import TaskRuntime
from .cube_stack import build_cube_stack
from .humanoid_urdf import build_humanoid_from_urdf
from .pouring_balls import build_pouring_balls


TaskBuilder = Callable[[object], TaskRuntime]


TASK_BUILDERS: dict[str, TaskBuilder] = {
    "cube_stack": build_cube_stack,
    "pouring_balls": build_pouring_balls,
    "humanoid_from_urdf": build_humanoid_from_urdf,
}

# Short aliases for convenience from CLI.
TASK_ALIASES = {
    "cube": "cube_stack",
    "balls": "pouring_balls",
    "humanoid": "humanoid_from_urdf",
}


def resolve_task_name(name: str) -> str:
    normalized = name.strip().lower()
    return TASK_ALIASES.get(normalized, normalized)


def get_task_builder(name: str) -> TaskBuilder:
    resolved = resolve_task_name(name)
    if resolved not in TASK_BUILDERS:
        available = ", ".join(sorted(TASK_BUILDERS))
        raise ValueError(f"Unknown task '{name}'. Available tasks: {available}")
    return TASK_BUILDERS[resolved]


def list_tasks() -> list[str]:
    return sorted(TASK_BUILDERS.keys())

