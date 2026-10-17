import importlib.util
import os

import pytest

from cm1x import get_backend

_requested = os.environ.get("CM1X_BACKENDS", "numpy,jax,torch").split(",")
BACKENDS = [b for b in _requested if b == "numpy" or importlib.util.find_spec(b)]


@pytest.fixture(params=BACKENDS)
def bk(request):
    return get_backend(request.param, "float64")


@pytest.fixture(params=BACKENDS)
def bk32(request):
    return get_backend(request.param, "float32")
