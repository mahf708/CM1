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

_ALIASES = {"np": "numpy", "jnp": "jax", "pytorch": "torch"}


@dataclass(frozen=True)
class Backend:
    name: str
    xp: Any
    dtype: Any
    device: Any = None

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
        """Return ``x`` with ``x[idx] = value``. In-place for numpy/torch."""
        if self.name == "jax":
            return x.at[idx].set(value)
        x[idx] = value
        return x

    def add(self, x, idx, value):
        """Return ``x`` with ``x[idx] += value``."""
        if self.name == "jax":
            return x.at[idx].add(value)
        x[idx] += value
        return x

    def copy(self, x):
        if self.name == "jax":
            return x
        if self.name == "torch":
            return x.clone()
        return self.xp.asarray(x, copy=True)


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
    else:
        raise ValueError(f"unknown backend {name!r}")
    xp = array_api_compat.array_namespace(mod.zeros(1))
    return Backend(name=name, xp=xp, dtype=getattr(xp, dtype), device=device)
