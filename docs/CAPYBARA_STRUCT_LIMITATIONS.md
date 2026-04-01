# Capybara @cp.struct Limitations

Limitations discovered while porting PhysX CUDA kernels to Capybara DSL using `@cp.struct` types with `@cp.inline` methods. Each section includes the failing pattern, the root cause in the compiler, and the workaround used.

---

## 1. No method chaining on struct values

### Failing pattern

```python
n = p1.sub(p0).cross(p2.sub(p0))
```

### Error

```
NotImplementedError: Unsupported call: Call(func=Attribute(value=Call(...), attr='cross', ...))
```

### Root cause

`exprs.py:_gen_call()` dispatches struct method calls via `Name.attr(...)` — it expects the receiver to be a simple variable name (`ast.Name`). When the receiver is itself a call result (`ast.Call`), the codegen has no dispatch path and raises `NotImplementedError`.

**File**: `python/capybara/compiler/codegen/exprs.py`, around line 505.

### Workaround

Break every chain into intermediate variables:

```python
e10 = p1.sub(p0)
e20 = p2.sub(p0)
n = e10.cross(e20)
```

### Impact

Adds ~1 extra line per chained call. Expressions like `a.scale(u).add(b.scale(v)).add(c.scale(w))` (common in barycentric interpolation) need 3 lines instead of 1.

---

## 2. Positional arguments rejected in struct constructors

### Failing pattern

```python
v = PxVec3(1.0, 2.0, 3.0)
```

### Error

```
RuntimeError: 'PxVec3' is a @cp.struct — use keyword arguments: PxVec3(x=..., y=..., z=...)
```

### Root cause

The struct constructor codegen explicitly requires keyword arguments to match field names. Positional dispatch is not implemented.

**File**: `python/capybara/compiler/codegen/exprs.py`, struct construction path.

### Workaround

Always use keyword arguments:

```python
v = PxVec3(x=1.0, y=2.0, z=3.0)
```

### Impact

Adds verbosity. `PxVec3(x=a, y=b, z=c)` is 12 characters longer than `PxVec3(a, b, c)` per call site. With ~20 PxVec3 constructions in utility.py, this adds ~240 characters.

---

## 3. Nested struct method calls lose `self`

### Failing pattern

```python
@cp.struct
class PxVec3:
    x: cp.float32
    y: cp.float32
    z: cp.float32

    @cp.inline
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    @cp.inline
    def magnitude_sq(self):
        return self.dot(self)  # calls another method on self

    @cp.inline
    def normalize_safe(self, thread):
        mag2 = self.magnitude_sq()  # calls magnitude_sq which calls dot
        ...
```

### Error

```
NameError: Undefined variable in kernel: self
```

### Root cause

When `normalize_safe` is inlined, the compiler binds `self` to the caller's struct value. But when `magnitude_sq` is then inlined *within* the already-inlined `normalize_safe` body, the compiler tries to resolve `self` in the inline expansion context where it's no longer a live binding. The nested `_do_inline` call in `inline.py` doesn't propagate the outer `self` binding into the inner inline scope.

**File**: `python/capybara/compiler/codegen/inline.py`, `_do_inline()` around line 435 — argument binding doesn't carry through nested method inlining.

### Workaround

Never call struct methods from within other struct methods. Inline the math directly:

```python
@cp.inline
def magnitude_sq(self):
    return self.x * self.x + self.y * self.y + self.z * self.z  # no self.dot()

@cp.inline
def normalize_safe(self, thread):
    mag2 = self.x * self.x + self.y * self.y + self.z * self.z  # no self.magnitude_sq()
    ...
```

### Impact

Forces code duplication inside struct methods. The `dot` formula appears in `dot`, `magnitude_sq`, and `normalize_safe` independently.

---

## 4. Tuple-returning @cp.inline fails inside while loops

### Failing pattern

```python
@cp.inline
def _apply_lock_linear(lock_flags, x, y, z):
    if lock_flags & cp.int32(1):
        x = cp.float32(0.0)
    if lock_flags & cp.int32(2):
        y = cp.float32(0.0)
    if lock_flags & cp.int32(4):
        z = cp.float32(0.0)
    return x, y, z

@cp.kernel
def my_kernel(...):
    with cp.Kernel(...) as (bx, block):
        for tid, thread in block.threads():
            while idx < count:
                # tuple unpack from inline, inside while loop
                x, y, z = _apply_lock_linear(flags, x, y, z)
                ...
                idx += stride
```

### Error

```
RuntimeError: PassManager::run failed
```

MLIR pass pipeline fails at `Inliner` / `Canonicalizer` on the kernel function — no Python-level traceback, the failure is in the MLIR optimization passes.

### Root cause

The codegen emits a `cp.call` multi-result operation for the tuple-returning inline. The tuple unpack stores each variable as a projection from that multi-result (`ssa.result(0)`, `ssa.result(1)`, etc.). These projections become while-loop carried variables.

The while-loop infrastructure in `control_flow.py` collects carried variables and emits `cp.while_loop` with `iter_args`. But:

1. **Yield emission** (`control_flow.py:1693-1700`): collects carried variable SSAs without validating they're standalone values (not multi-result projections).
2. **Result rebinding** (`control_flow.py:1708-1714`): assumes carried variables can be indexed from a single multi-result operation (`result_ssa.result(i)`).

When the MLIR inliner tries to inline the `cp.call` into the loop body, the `iter_args` type signature doesn't match the actual SSAValues being carried, and the canonicalizer fails.

**Files**:
- `python/capybara/compiler/codegen/inline.py:349-354` — multi-result projection
- `python/capybara/compiler/codegen/stmts.py:825-838` — tuple unpack stores projections
- `python/capybara/compiler/codegen/control_flow.py:1693-1714` — while-loop carry/yield

### Workaround

Don't use tuple-returning `@cp.inline` inside while loops. Write the code inline:

```python
while idx < count:
    if lock_flags & cp.int32(1):
        x = cp.float32(0.0)
    if lock_flags & cp.int32(2):
        y = cp.float32(0.0)
    if lock_flags & cp.int32(4):
        z = cp.float32(0.0)
    ...
```

### Impact

Prevents factoring out repeated patterns inside while-loop kernels. The integration kernel has 30 lines of repetitive lock-flag checks that could be 5 lines with helpers, but can't be refactored due to this limitation.

---

## Summary

| # | Limitation | Root file | Workaround cost |
|---|-----------|-----------|-----------------|
| 1 | No method chaining | `exprs.py` | +1 line per chain |
| 2 | Keyword-only constructors | `exprs.py` | +12 chars per call |
| 3 | No nested method calls | `inline.py` | Code duplication in methods |
| 4 | No tuple-return inline in while | `control_flow.py` + `inline.py` | Can't factor repeated patterns |
