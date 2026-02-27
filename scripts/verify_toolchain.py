#!/usr/bin/env python3
"""Verify that the installed SAPIEN wheel was built locally with local PhysX."""
import os
import sys
import warnings

# Vulkan/EGL driver warnings are irrelevant for this headless physics-only check.
warnings.filterwarnings("ignore", module=r"sapien\._vulkan_tricks")


def main():
    expected_marker = os.environ.get("SAPIEN_LOCAL_BUILD_MARKER", "elytar")
    expected_physx = os.environ.get("SAPIEN_LOCAL_PHYSX_VERSION", "")

    import sapien

    version = getattr(sapien, "__version__", "unknown")
    marker = getattr(sapien, "__local_build_marker__", "")
    physx_ver = getattr(sapien, "__local_physx_version__", "")

    print(f"sapien.__version__             = {version}")
    print(f"sapien.__local_build_marker__  = {marker!r}")
    print(f"sapien.__local_physx_version__ = {physx_ver!r}")

    ok = True

    if expected_marker and marker != expected_marker:
        print(
            f"FAIL [sapien]: expected marker {expected_marker!r}, got {marker!r}",
            file=sys.stderr,
        )
        ok = False

    if not physx_ver:
        print("FAIL [physx]: no local PhysX version embedded — wheel was not built with local PhysX", file=sys.stderr)
        ok = False
    elif expected_physx and physx_ver != expected_physx:
        print(
            f"FAIL [physx]: expected {expected_physx!r}, got {physx_ver!r}",
            file=sys.stderr,
        )
        ok = False

    if not ok:
        return 1

    scene = sapien.Scene()
    scene.set_timestep(1 / 240.0)
    scene.add_ground(altitude=0.0)
    for _ in range(5):
        scene.step()
        scene.update_render()

    print(f"OK — SAPIEN {version} with local PhysX {physx_ver} (5 physics steps completed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
