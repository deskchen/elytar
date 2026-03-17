"""Base types for benchmark environments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


BeforeStepHook = Callable[[int, float], None]


@dataclass
class SceneBuildResult:
    """Result of building one task scene."""

    before_step: BeforeStepHook | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRuntime:
    name: str
    scene: Any
    physx_system: Any
    before_step: BeforeStepHook | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # When set, multiple scenes share physx_system (vectorized). run_task uses scenes for viewer/clear.
    scenes: list[Any] | None = None
