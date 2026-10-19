"""Physical random column states for the Thompson tests (shared by
test_thompson.py and the column/driver harness comparisons)."""

from __future__ import annotations

import numpy as np


def columns(ncol=48, nk=40, seed=0, dz0=250.0):
    """``[nk, ncol]`` arrays qv qc qi qr qs qg ni nr t p w dz: a moist
    hydrostatic profile per column (surface 285-305 K, RH 40-110 %), with
    random cloud water, ice, rain, snow and graupel layers (some empty
    columns, some empty levels) spanning warm, mixed-phase and cold air."""
    rng = np.random.default_rng(seed)
    k = np.arange(nk)[:, None]
    dz = dz0 * (1.0 + 0.3 * rng.random((nk, 1))) * np.ones((1, ncol))
    z = np.cumsum(dz, axis=0) - 0.5 * dz
    tsfc = 285.0 + 20.0 * rng.random(ncol)
    lapse = 0.0055 + 0.002 * rng.random(ncol)
    t = np.maximum(tsfc - lapse * z, 200.0)
    p = 1.0e5 * np.exp(-z / 8000.0)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    qvs = 0.622 * es / (p - es)
    rh = 0.4 + 0.7 * rng.random((nk, ncol))
    qv = rh * qvs

    def layer(lo, hi, frac):
        a = 10.0 ** rng.uniform(lo, hi, (nk, ncol))
        a[rng.random((nk, ncol)) > frac] = 0.0
        a[:, rng.random(ncol) < 0.1] = 0.0
        return a

    qc = layer(-7, -2.5, 0.5) * (t > 235.0)
    qr = layer(-8, -2.5, 0.5)
    qi = layer(-8, -3.0, 0.5) * (t < 275.0)
    qs = layer(-8, -2.5, 0.5) * (t < 278.0)
    qg = layer(-8, -2.5, 0.4)
    ni = (
        10.0 ** rng.uniform(2, 6, (nk, ncol))
        * (qi > 0)
        * (rng.random((nk, ncol)) > 0.1)
    )
    nr = (
        10.0 ** rng.uniform(2, 5, (nk, ncol))
        * (qr > 0)
        * (rng.random((nk, ncol)) > 0.1)
    )
    w = rng.normal(0.0, 1.0, (nk, ncol))
    # a few fully dry columns (no_micro early return) and tiny amounts
    qv[:, :2] = 0.3 * qvs[:, :2]
    for a in (qc, qr, qi, qs, qg, ni, nr):
        a[:, :2] = 0.0
    qc[k[:, 0] % 7 == 3, 2] = 5.0e-13
    return dict(
        qv=qv, qc=qc, qi=qi, qr=qr, qs=qs, qg=qg, ni=ni, nr=nr, t=t, p=p, w=w, dz=dz
    )
