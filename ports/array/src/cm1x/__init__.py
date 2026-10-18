"""cm1x: a backend-agnostic (Python array API) port of CM1."""

from .backend import Backend, get_backend

__all__ = ["Backend", "get_backend"]
