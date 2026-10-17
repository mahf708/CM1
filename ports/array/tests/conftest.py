import importlib.util

import pytest

from cm1x import get_backend

BACKENDS = [b for b in ("numpy", "jax", "torch") if b == "numpy" or importlib.util.find_spec(b)]


@pytest.fixture(params=BACKENDS)
def bk(request):
    return get_backend(request.param, "float64")


@pytest.fixture(params=BACKENDS)
def bk32(request):
    return get_backend(request.param, "float32")
