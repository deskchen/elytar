from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


BeforeStepHook = Callable[[int, float], None]


@dataclass
class TaskRuntime:
    name: str
    scene: Any
    physx_system: Any
    before_step: BeforeStepHook | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

