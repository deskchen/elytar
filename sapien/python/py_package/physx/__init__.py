#
# Copyright 2025 Hillbot Inc.
# Copyright 2020-2024 UCSD SU Lab
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import io
import os
import platform
from pathlib import Path
from zipfile import ZipFile

import requests

from ..pysapien.physx import *
from ..pysapien.physx import _enable_gpu

try:
    from ..version import __local_physx_version__
except ImportError:
    __local_physx_version__ = ""


def _resolve_gpu_lib_path():
    """Return (path, local_required). When local_required, path must exist; no prebuilt download."""
    physx_version = version()
    if platform.system() == "Windows":
        lib_name = "PhysXGpu_64.dll"
    else:
        lib_name = "libPhysXGpu_64.so"

    configs = [
        os.environ.get("PHYSX_CONFIG", "checked"),
        "checked",
        "profile",
        "release",
    ]

    if __local_physx_version__:
        # This wheel was built with local PhysX: require local GPU lib only; do not fall back to prebuilt.
        physx_root = os.environ.get("SAPIEN_PHYSX5_DIR")
        if not physx_root:
            return (None, True)
        root = Path(physx_root)
        if platform.system() == "Linux":
            if platform.machine() in ("x86_64", "AMD64"):
                subdirs = ["linux.x86_64", "linux.clang"]
            else:
                subdirs = ["linux.aarch64"]
            for cfg in configs:
                for subdir in subdirs:
                    candidate = root / "bin" / subdir / cfg / lib_name
                    if candidate.exists():
                        return (candidate, True)
            return (root / "bin" / subdirs[0] / configs[0] / lib_name, True)
        if platform.system() == "Windows":
            subdir = "win.x86_64.vc142.mt"
            for cfg in configs:
                candidate = root / "bin" / subdir / cfg / lib_name
                if candidate.exists():
                    return (candidate, True)
            return (root / "bin" / subdir / configs[0] / lib_name, True)
        return (None, True)

    # Non-local wheel: allow prebuilt path (download if missing).
    parent = Path.home() / ".sapien" / "physx" / physx_version
    return (parent / lib_name, False)


def enable_gpu():
    if is_gpu_enabled():
        return

    physx_version = version()
    dll_path, local_required = _resolve_gpu_lib_path()

    if local_required:
        if dll_path is None:
            raise RuntimeError(
                "This SAPIEN build requires a local PhysX GPU library. Set SAPIEN_PHYSX5_DIR to your "
                "PhysX source directory (e.g. /workspace/physx). Run from repo root after "
                "scripts/update_toolchain.sh so it is set automatically, or export SAPIEN_PHYSX5_DIR."
            )
        if not dll_path.exists():
            raise RuntimeError(
                f"Local PhysX GPU library not found at {dll_path}. "
                "Build PhysX with GPU support (scripts/update_toolchain.sh) and ensure "
                "SAPIEN_PHYSX5_DIR points to that PhysX tree. Tried configs: checked, profile, release."
            )
        if os.environ.get("SAPIEN_DEBUG_GPU_LIB"):
            print(f"[SAPIEN] Loading local PhysX GPU lib: {dll_path}", flush=True)
        dll = dll_path
        parent = dll.parent
    else:
        dll = dll_path
        parent = dll.parent
        if not dll.exists():
            parent.mkdir(exist_ok=True, parents=True)
            if platform.system() == "Windows":
                url = f"https://github.com/sapien-sim/physx-precompiled/releases/download/{physx_version}/windows-dll.zip"
            elif platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64"):
                url = f"https://github.com/sapien-sim/physx-precompiled/releases/download/{physx_version}/linux-so.zip"
            elif platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64"):
                url = f"https://github.com/sapien-sim/physx-precompiled/releases/download/{physx_version}/linux-aarch64-so.zip"
            else:
                raise RuntimeError("Unsupported platform")
            print(
                f"Downloading PhysX GPU library to {parent} from Github. This can take several minutes."
                f" If it fails to download, please manually download {url} and unzip at {parent}."
            )
            res = requests.get(url)
            z = ZipFile(io.BytesIO(res.content))
            z.extractall(parent)
            print("Download complete.")

    import ctypes

    if platform.system() == "Windows":
        ctypes.CDLL("cuda.dll", ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(dll), ctypes.RTLD_LOCAL)
    elif platform.system() == "Linux":
        ctypes.CDLL("libcuda.so", ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(dll), ctypes.RTLD_LOCAL)

    _enable_gpu()
