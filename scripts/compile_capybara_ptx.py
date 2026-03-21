#!/usr/bin/env python3
"""Compile Capybara DSL kernels to PTX from sibling JSON specs.

Default behavior:
1) scan under --root for folders named `capybara`
2) for each `<stem>.json` in those folders, require sibling `<stem>.py`
3) compile kernels listed in JSON
4) write merged PTX to sibling `../PTX/<stem>.capybara.ptx`

Supports single-target mode via --module-path + --spec-path.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from capybara.runtime import _compile, _resolve_kernel_args

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELYTAR_ROOT = _SCRIPT_DIR.parent
_DEFAULT_SCAN_ROOT = _ELYTAR_ROOT / "physx-5.6.1-capybara" / "source"

_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _load_module_from_path(module_path: Path):
    module_path = module_path.resolve()
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module_name = f"cp_mod_{module_path.stem}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_ptx_header(ptx: str) -> str:
    idx = ptx.find(".visible .entry")
    if idx <= 0:
        return ""
    return ptx[:idx].rstrip() + "\n"


def _extract_entry_body(ptx: str) -> str:
    idx = ptx.find(".visible .entry")
    if idx < 0:
        return ""
    return ptx[idx:].rstrip()


def _compile_one(jit_fn, args, constexprs):
    resolved = _resolve_kernel_args(jit_fn, *args, **constexprs)
    result = _compile(
        jit_fn,
        kernel_args=resolved["kernel_args"],
        block_size=resolved["block_size"],
        elem_type=resolved["elem_type"],
        ptr_elem_types=resolved["ptr_elem_types"],
        ptr_unsigned=resolved.get("ptr_unsigned"),
        constexpr_values=resolved["constexpr_values"],
        shape_arg_values=resolved["shape_arg_values"],
        soa_struct_params=resolved["soa_struct_params"],
        max_regs=getattr(jit_fn, "max_regs", None),
        verbose_ptxas=False,
        debug=False,
        estimate=True,
    )
    return result.ptx_text


def _materialize_spec(spec: dict[str, Any], module):
    kind = spec.get("kind")
    if kind == "tensor":
        dtype_name = spec["dtype"]
        if dtype_name not in _TORCH_DTYPE_MAP:
            raise ValueError(f"Unsupported tensor dtype in JSON spec: {dtype_name}")
        shape = tuple(spec["shape"])
        device = spec.get("device", "cuda")
        return torch.empty(*shape, dtype=_TORCH_DTYPE_MAP[dtype_name], device=device)

    if kind == "int":
        return int(spec["value"])

    if kind == "float":
        return float(spec["value"])

    if kind == "struct":
        class_name = spec["class"]
        struct_cls = getattr(module, class_name, None)
        if struct_cls is None:
            raise ValueError(f"Struct class '{class_name}' not found in module")
        fields = spec["fields"]
        materialized_fields = {
            field_name: _materialize_spec(field_spec, module)
            for field_name, field_spec in fields.items()
        }
        return struct_cls.from_arrays(**materialized_fields)

    raise ValueError(f"Unsupported arg kind in JSON spec: {kind}")


def _compile_from_json(module, cfg: dict[str, Any], verbose: bool):
    kernels = cfg.get("kernels")
    if not kernels:
        raise ValueError("JSON spec must contain non-empty 'kernels' list")

    compiled = []
    for kernel_cfg in kernels:
        kernel_name = kernel_cfg["name"]
        jit_fn = getattr(module, kernel_name, None)
        if jit_fn is None:
            raise ValueError(f"Kernel '{kernel_name}' not found in module")

        arg_specs = kernel_cfg.get("args", [])
        args = [_materialize_spec(arg_spec, module) for arg_spec in arg_specs]
        constexprs = dict(kernel_cfg.get("constexprs", {}))

        if verbose:
            print(f"Compiling {kernel_name}...")
        ptx = _compile_one(jit_fn, args, constexprs)
        compiled.append((kernel_name, ptx))
    return compiled


def _default_out_path(module_path: Path) -> Path:
    # .../<module>/src/capybara/<stem>.py -> .../<module>/src/PTX/<stem>.capybara.ptx
    capybara_dir = module_path.parent
    if capybara_dir.name != "capybara":
        raise ValueError(f"Expected module in a 'capybara' directory: {module_path}")
    src_dir = capybara_dir.parent
    return src_dir / "PTX" / f"{module_path.stem}.capybara.ptx"


def _discover_jobs(root: Path):
    jobs: list[tuple[Path, Path, Path]] = []
    for capybara_dir in root.rglob("capybara"):
        if not capybara_dir.is_dir():
            continue
        for spec_path in capybara_dir.glob("*.json"):
            module_path = capybara_dir / f"{spec_path.stem}.py"
            if not module_path.exists():
                continue
            out_ptx = _default_out_path(module_path)
            jobs.append((module_path, spec_path, out_ptx))
    return sorted(jobs, key=lambda x: str(x[0]))


def _compile_job(module_path: Path, spec_path: Path, out_ptx: Path, verbose: bool):
    module = _load_module_from_path(module_path)
    with open(spec_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    compiled = _compile_from_json(module, cfg, verbose)
    if not compiled:
        raise RuntimeError(f"No kernels compiled from spec: {spec_path}")

    header = _extract_ptx_header(compiled[0][1])
    entry_bodies = [_extract_entry_body(ptx) for _, ptx in compiled]
    entry_bodies = [body for body in entry_bodies if body]
    if len(entry_bodies) != len(compiled):
        print(f"Warning: missing .entry body in {spec_path}", file=sys.stderr)

    merged = header + "\n\n".join(entry_bodies) + "\n"
    out_ptx.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ptx, "w", encoding="utf-8") as f:
        f.write(merged)
    return len(entry_bodies)


def main():
    parser = argparse.ArgumentParser(
        description="Compile Capybara kernels from sibling JSON specs into PTX files",
    )
    parser.add_argument("--root", default=str(_DEFAULT_SCAN_ROOT))
    parser.add_argument("--module-path")
    parser.add_argument("--spec-path")
    parser.add_argument(
        "--out-ptx",
        help="Output PTX path override (default: ../PTX/<stem>.capybara.ptx)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    jobs: list[tuple[Path, Path, Path]] = []
    if args.module_path or args.spec_path or args.out_ptx:
        if not (args.module_path and args.spec_path):
            raise ValueError(
                "Single-target mode requires both --module-path and --spec-path"
            )
        module_path = Path(args.module_path).resolve()
        spec_path = Path(args.spec_path).resolve()
        out_ptx = (
            Path(args.out_ptx).resolve()
            if args.out_ptx
            else _default_out_path(module_path)
        )
        jobs = [(module_path, spec_path, out_ptx)]
    else:
        jobs = _discover_jobs(Path(args.root).resolve())

    if not jobs:
        print("No capybara <stem>.py + <stem>.json pairs found.")
        return

    total_kernels = 0
    for module_path, spec_path, out_ptx in jobs:
        if args.verbose:
            print(f"\n==> {module_path}")
            print(f"    spec: {spec_path}")
            print(f"    out : {out_ptx}")
        n = _compile_job(module_path, spec_path, out_ptx, args.verbose)
        total_kernels += n
        print(str(out_ptx))

    if args.verbose:
        print(f"\nCompiled {len(jobs)} module(s), {total_kernels} kernel entry block(s).")


if __name__ == "__main__":
    main()
