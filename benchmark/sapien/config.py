"""Benchmark runner configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPUMemoryConfig:
    """PhysX GPU memory config. ManiSkill3 defaults."""

    temp_buffer_capacity: int = 2**24
    max_rigid_contact_count: int = 2**19
    max_rigid_patch_count: int = 2**18
    heap_capacity: int = 2**26
    found_lost_pairs_capacity: int = 2**25
    found_lost_aggregate_pairs_capacity: int = 2**10
    total_aggregate_pairs_capacity: int = 2**10
    collision_stack_size: int = 64 * 64 * 1024

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}

