"""Shared PhysX math types for Capybara DSL kernels.

Provides @cp.struct types mirroring PhysX foundation math (PxVec3)
with @cp.inline methods, reducing scalar decomposition in kernel code.
Source: ~/capybara-triton (see DSL_GRAMMAR_PHYSX_PORT.md section 7).
"""

import capybara as cp


@cp.struct
class PxVec3:
    x: cp.float32
    y: cp.float32
    z: cp.float32

    @cp.inline
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    @cp.inline
    def cross(self, other):
        return PxVec3(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
        )

    @cp.inline
    def add(self, other):
        return PxVec3(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    @cp.inline
    def sub(self, other):
        return PxVec3(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    @cp.inline
    def scale(self, s):
        return PxVec3(x=self.x * s, y=self.y * s, z=self.z * s)

    @cp.inline
    def magnitude_sq(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    @cp.inline
    def normalize_safe(self, thread):
        """Returns (normalized_vec, magnitude). Branchless fade-out for tiny vectors."""
        mag2 = self.x * self.x + self.y * self.y + self.z * self.z
        eps = cp.float32(1.0e-20)
        tiny = cp.float32(1.0e-30)
        mask = mag2 / (mag2 + eps)
        inv = thread.rsqrt(mag2 + tiny)
        mag = thread.sqrt(mag2 + tiny) * mask
        return PxVec3(x=self.x * mask * inv, y=self.y * mask * inv, z=self.z * mask * inv), mag
