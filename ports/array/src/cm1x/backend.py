"""Array-backend abstraction.

Every numerical routine in cm1x takes a ``Backend`` (``bk``) and works only
through ``bk.xp`` (an array-API-standard namespace from array-api-compat) plus
the few helpers defined here. Nothing outside this module may import numpy,
jax or torch directly for computation.

Rules for routine authors (see ports/array/README.md):
  * Never mutate arrays in place with ``x[idx] = v``; use ``x = bk.set(x, idx, v)``
    and always use the returned array (jax arrays are immutable).
  * Indexing uses basic slices only; boolean masks -> ``xp.where``.
  * No data-dependent Python control flow on array values (keeps jax.jit working).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import array_api_compat

_ALIASES = {
    "np": "numpy",
    "jnp": "jax",
    "pytorch": "torch",
    "strict": "array_api_strict",
    "array-api-strict": "array_api_strict",
}


def _torch_needs_grad(x, value) -> bool:
    """True while torch autograd records ``x`` or ``value``: an in-place
    update could then overwrite a tensor saved for backward (autograd error
    "modified by an inplace operation") or a leaf that requires grad, so
    ``Backend.set``/``add`` update a clone instead (same values)."""
    import torch

    return torch.is_grad_enabled() and (
        x.requires_grad or bool(getattr(value, "requires_grad", False))
    )


@dataclass(frozen=True)
class Backend:
    name: str
    xp: Any
    dtype: Any
    device: Any = None
    # torch only: compile the hot routines separately (``Backend.kernel``);
    # a tuple of (option, value) pairs for torch.compile, or None (off)
    kernels: Any = None

    # --- creation ---------------------------------------------------------
    def zeros(self, shape, dtype=None):
        return self.xp.zeros(shape, dtype=dtype or self.dtype, device=self.device)

    def full(self, shape, value, dtype=None):
        return self.xp.full(shape, value, dtype=dtype or self.dtype, device=self.device)

    def asarray(self, x, dtype=None):
        return self.xp.asarray(x, dtype=dtype or self.dtype, device=self.device)

    def to_numpy(self, x):
        import numpy as np

        if self.name == "torch":
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # --- functional update ------------------------------------------------
    def set(self, x, idx, value):
        """Return ``x`` with ``x[idx] = value``. In-place for numpy/torch
        (out-of-place on torch while autograd records ``x`` or ``value``)."""
        if self.name == "jax":
            return x.at[idx].set(value)
        if self.name == "torch" and _torch_needs_grad(x, value):
            x = x.clone()
        x[idx] = value
        return x

    def add(self, x, idx, value):
        """Return ``x`` with ``x[idx] += value`` (in place as ``set``)."""
        if self.name == "jax":
            return x.at[idx].add(value)
        if self.name == "torch" and _torch_needs_grad(x, value):
            x = x.clone()
        x[idx] += value
        return x

    def scalar(self, x):
        """0-d array of ``x`` (float64 when the backend has it, else
        ``self.dtype``): a traced argument for compiled steps, so that model
        time does not force a recompile every step."""
        f64 = getattr(self.xp, "float64", None)
        if self.name == "jax":
            import jax

            if not jax.config.jax_enable_x64:
                f64 = None
        return self.xp.asarray(x, dtype=f64 or self.dtype, device=self.device)

    # --- control flow / compilation ---------------------------------------
    def fori_loop(self, n, body, carry, max_n=None, until=None):
        """``for i in range(n): carry = body(i, carry)``; return ``carry``.

        ``n`` may be a 0-d integer-valued array (e.g. a sub-step count that
        depends on the data).  Eagerly it becomes a Python int and the loop
        runs in Python; under ``jax.jit`` (``n`` a tracer) it lowers to
        ``lax.while_loop`` with the same trip count, so results are identical.
        (``while_loop`` is forward-differentiable only; pass a static Python
        int ``n`` for reverse mode.)  A static ``n`` under a jax trace lowers
        to ``lax.fori_loop`` (one copy of ``body`` in the compiled program
        instead of ``n``).  ``carry`` is a tuple/list/dict tree of arrays;
        ``body`` must keep its structure, shapes and dtypes.

        ``max_n`` (static int >= n): with a traced ``n``, run ``max_n``
        masked iterations instead (``carry`` is kept where ``i >= n``), a
        fixed-trip ``lax.fori_loop`` that reverse-mode AD can differentiate;
        same result as long as ``n <= max_n``.

        ``until`` (optional ``until(carry) -> 0-d bool array``): when run
        eagerly the loop stops early once it is true.  Only for loops whose
        ``body`` returns a carry for which ``until`` holds unchanged (e.g.
        masked iterations with nothing left active), so the result is the
        same; ignored under a jax trace or torch.compile (fixed trip count)."""
        if self.name == "jax" and (_jax_tracing(n) or _jax_tracing(carry)):
            import jax

            if not isinstance(n, int) and not _jax_tracing(n):
                # concrete trip count (e.g. under eager jax.grad, where only
                # the carry is traced): a static loop keeps reverse mode
                n = int(self.to_numpy(n))
            if isinstance(n, int):
                if n <= 0:
                    return carry
                return jax.lax.fori_loop(0, n, body, carry)
            nn = self.xp.astype(n, self.xp.int32)
            if max_n is not None:
                xp = self.xp

                def masked(i, c):
                    new = body(i, c)
                    keep = i < nn
                    return jax.tree_util.tree_map(
                        lambda a, b: xp.where(keep, a, b), new, c
                    )

                return jax.lax.fori_loop(0, int(max_n), masked, carry)

            def cond(c):
                return c[0] < nn

            def step(c):
                return (c[0] + 1, body(c[0], c[1]))

            i0 = self.xp.asarray(0, dtype=self.xp.int32)
            return jax.lax.while_loop(cond, step, (i0, carry))[1]
        if not isinstance(n, int):
            n = int(self.to_numpy(n))
        if until is not None and self.name == "torch":
            import torch

            if torch.compiler.is_compiling():
                until = None
        for i in range(n):
            if until is not None and bool(until(carry)):
                break
            carry = body(i, carry)
        return carry

    def scan(self, f, init, xs, reverse=False):
        """``lax.scan`` semantics on every backend: ``carry, y = f(carry, x)``
        for each ``x`` along the leading axis of ``xs`` (an array or a tuple of
        arrays), in reverse order if ``reverse``; returns ``(carry, ys)`` with
        ``ys`` stacked along axis 0 in the order of ``xs`` (``ys`` is an
        array, a tuple of arrays, or ``None``).  A Python loop eagerly (the
        same operations as a hand-written recurrence over levels); under a jax
        trace ``lax.scan``, so a vertical recurrence is one loop in the
        compiled program instead of ``nk`` unrolled copies."""
        if self.name == "jax" and (_jax_tracing(init) or _jax_tracing(xs)):
            import jax

            return jax.lax.scan(f, init, xs, reverse=reverse)
        seq = isinstance(xs, (tuple, list))
        n = (xs[0] if seq else xs).shape[0]
        order = range(n - 1, -1, -1) if reverse else range(n)
        carry, ys = init, [None] * n
        for i in order:
            x = tuple(a[i, ...] for a in xs) if seq else xs[i, ...]
            carry, ys[i] = f(carry, x)
        y0 = ys[0] if n else None
        if y0 is None:
            return carry, None
        if isinstance(y0, tuple):
            return carry, tuple(
                self.xp.stack([y[m] for y in ys], axis=0) for m in range(len(y0))
            )
        return carry, self.xp.stack(ys, axis=0)

    def compile(self, fn, **options):
        """Compile ``fn`` for this backend: ``jax.jit`` on jax,
        ``torch.compile`` on torch, ``fn`` unchanged elsewhere.  ``options``
        go to the compiler (e.g. ``donate_argnums`` for jax, ``mode`` /
        ``dynamic`` for torch)."""
        if self.name == "jax":
            import jax

            return jax.jit(fn, **options)
        if self.name == "torch":
            import torch

            return torch.compile(fn, **options)
        return fn

    def kernel(self, fn, traced=()):
        """``fn`` as a separately compiled routine: with ``kernels`` set on
        torch (``Model(..., backend="torch", jit=True)``) a cached
        ``torch.compile`` of ``fn`` (inductor, ``dynamic=False``,
        ``fullgraph=True``); ``fn`` itself otherwise (numpy, jax, eager
        torch, or when already inside a torch.compile trace).

        ``traced`` names keyword arguments that are passed as Python floats
        (time steps, thresholds) but used only in array arithmetic: they are
        handed to the compiled routine as 0-d arrays, so that a new value is
        a new input instead of a recompile.  Every other non-array argument
        (ints, flags, ``bk``, ``g``, ``cfg``) is a compile-time constant: one
        compiled variant per distinct combination."""
        if self.kernels is None or self.name != "torch":
            return fn
        return _torch_kernel(self, fn, tuple(traced))

    def materialize(self, tree):
        """Evaluate a lazy backend's pending work on ``tree`` (a dict / list
        tree of arrays) and return it; identity for eager backends.  dask:
        ``dask.persist``, so each time step starts from computed chunks instead
        of re-running the whole history's task graph."""
        if self.name == "dask":
            import dask

            return dask.persist(tree)[0]
        return tree

    def copy(self, x):
        if self.name == "jax":
            return x
        if self.name == "torch":
            return x.clone()
        return self.xp.asarray(x, copy=True)


# torch.compile settings for Backend.kernel (``Model(..., jit_options=...)``
# entries override them).  Inductor's C++ vectorization adds ~60% compile time
# here for no measurable speed-up (the kernels are memory bound and full of
# offset slices), hence ``cpp.vec_isa_ok=False``.
TORCH_KERNEL_DEFAULTS = {
    "dynamic": False,
    "fullgraph": True,
    "options": {"cpp.vec_isa_ok": False},
}
_TORCH_KERNELS: dict = {}


def torch_kernel_options(jit_options=None) -> tuple:
    """Hashable torch.compile options for ``Backend.kernels``: the defaults
    above updated with ``jit_options`` (``options`` dicts are merged)."""
    o = {
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in TORCH_KERNEL_DEFAULTS.items()
    }
    for k, v in (jit_options or {}).items():
        if k == "options" and isinstance(v, dict):
            o["options"].update(v)
        else:
            o[k] = v
    if not o.get("options"):
        o.pop("options", None)
    return tuple(
        (k, tuple(sorted(v.items())) if isinstance(v, dict) else v)
        for k, v in sorted(o.items())
    )


def _torch_kernel(bk, fn, traced):
    key = (fn, traced, bk.kernels)
    w = _TORCH_KERNELS.get(key)
    if w is not None:
        return w
    import torch
    import torch._dynamo

    # one compiled variant per distinct set of static arguments (RK stage,
    # flags, ...): allow more than dynamo's default of 8 per routine
    cfg = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(cfg, name):
            setattr(cfg, name, max(getattr(cfg, name), 64))
    opts = {k: (dict(v) if k == "options" else v) for k, v in bk.kernels}
    compiled = torch.compile(fn, **opts)

    def w(*args, **kw):
        if torch.compiler.is_compiling():
            return fn(*args, **kw)
        for k in traced:
            v = kw.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                kw[k] = torch.tensor(float(v), dtype=torch.float64, device=bk.device)
        return compiled(*args, **kw)

    w.__wrapped__ = fn
    _TORCH_KERNELS[key] = w
    return w


def _jax_tracing(tree) -> bool:
    """True if any leaf of ``tree`` is a jax tracer (inside jit/grad/vmap)."""
    import jax

    return any(isinstance(a, jax.core.Tracer) for a in jax.tree_util.tree_leaves(tree))


def is_array(x) -> bool:
    """True for an array of any supported backend (0-d included)."""
    return array_api_compat.is_array_api_obj(x)


def get_backend(name: str = "numpy", dtype: str = "float64", device=None) -> Backend:
    name = _ALIASES.get(name, name)
    if name == "numpy":
        import numpy as mod
    elif name == "jax":
        import jax

        if dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        import jax.numpy as mod
    elif name == "torch":
        import torch as mod
    elif name == "cupy":
        import cupy as mod
    elif name == "dask":
        import dask.array as mod
    elif name == "array_api_strict":  # conformance check, not for speed
        import array_api_strict as mod
    else:
        raise ValueError(f"unknown backend {name!r}")
    xp = array_api_compat.array_namespace(mod.zeros(1))
    return Backend(name=name, xp=xp, dtype=getattr(xp, dtype), device=device)
