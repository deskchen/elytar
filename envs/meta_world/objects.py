"""Object geometry for Meta-World tasks. Hardcoded from original MJCF specs."""

from __future__ import annotations

# Cylinder: r=0.02, h=0.04 (from cylinder.xml size="0.02 0.02" = radius, half-height)
CYLINDER_RADIUS = 0.02
CYLINDER_HALF_LENGTH = 0.02  # half of height

# Block: cube, half_size per side (from block.xml size="0.02 0.02 0.02")
BLOCK_HALF_SIZE = 0.02

# Puck: cylinder, similar to cylinder
PUCK_RADIUS = 0.02
PUCK_HALF_LENGTH = 0.01

# Soccer ball: sphere
SOCCER_BALL_RADIUS = 0.02
