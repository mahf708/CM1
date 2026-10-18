"""sfcphys.getcecd (the neutral surface layer of sfcmodel=1) against
src/sfcphys.F on every branch: given z0 (set_znt=1), given u* (set_ust=1),
given Cd (cecd=1) and the iterated Cd(s10) over water (cecd=2, 3) with
land points (given z0) mixed in."""

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver

from cm1x.sfcphys import getcecd

pytestmark = requires_gfortran

NI, NJ, NG = 7, 5, 3
CASES = {
    "set_znt": (1, 0, 3),
    "set_ust": (0, 1, 3),
    "cecd1": (0, 0, 1),
    "cecd2": (0, 0, 2),
    "cecd3": (0, 0, 3),
}


class _Cfg(dict):
    __getattr__ = dict.__getitem__


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    set_znt, set_ust, cecd = CASES[request.param]
    rng = np.random.default_rng(3)
    shp = (NJ + 2 * NG, NI + 2 * NG)
    u1 = rng.uniform(-30, 30, shp)
    v1 = rng.uniform(-30, 30, shp)
    u1[0, 0] = v1[0, 0] = 0.0  # calm point
    ins = {
        "u1": u1,
        "v1": v1,
        "s1": np.sqrt(u1**2 + v1**2),
        "za": rng.uniform(5.0, 250.0, shp),
        "xland": np.where(rng.random(shp) < 0.7, 2.0, 1.0),
        "znt": 10.0 ** rng.uniform(-5, -1, shp),
    }
    params = [NI, NJ, set_znt, set_ust, cecd, 0.25, 1.8e-3]
    outs = {k: shp for k in ("cd", "znt", "ust", "u10", "v10", "s10")}
    ref = run_driver("sfcphys", {"params": np.array(params, float), **ins}, outs)
    cfg = _Cfg(set_znt=set_znt, set_ust=set_ust, cecd=cecd, cnst_ust=0.25)
    cfg["cnstcd"] = 1.8e-3
    return cfg, ins, ref


def test_getcecd(bk, case):
    cfg, ins, ref = case
    J, I = slice(NG, NG + NJ), slice(NG, NG + NI)
    a = {k: bk.asarray(v[J, I]) for k, v in ins.items()}
    r = getcecd(bk, cfg, **a)
    for k in ("cd", "znt", "ust", "u10", "v10", "s10"):
        got = bk.to_numpy(r[k])
        want = ref[k][J, I]
        err = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-300))
        assert err <= 1e-13, f"{k}: rel err {err:.2e}"
