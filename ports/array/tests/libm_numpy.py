"""A numpy backend whose transcendental functions (exp, log, atan, pow) are
glibc's, through the ``math`` module, as gfortran calls them: numpy's own
(SIMD) versions differ from libm by an ulp now and then, so this backend
gives bitwise checks of the operation order against the Fortran."""

import math
import types

import array_api_compat.numpy as cnp
import numpy as np

from cm1x.backend import Backend


def _unary(f):
    v = np.frompyfunc(f, 1, 1)

    def g(x):
        return np.asarray(v(x), dtype=np.float64)

    return g


def _pow1(x, y):
    if x < 0.0 and y != int(y):
        return math.nan
    return math.pow(x, y)


_vpow = np.frompyfunc(_pow1, 2, 1)


def _pow(a, b):
    return np.asarray(_vpow(a, b), dtype=np.float64)


def libm_backend() -> Backend:
    ns = types.SimpleNamespace(
        **{k: getattr(cnp, k) for k in dir(cnp) if not k.startswith("__")}
    )
    ns.exp = _unary(math.exp)
    ns.log = _unary(math.log)
    ns.atan = _unary(math.atan)
    ns.pow = _pow
    return Backend("numpy", ns, np.float64)
