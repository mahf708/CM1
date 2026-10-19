"""cm1x.testcase.simple_phys (testcase_simple_phys: simple radiation of
testcase 4, 5, the lspgrad=3 mesoscale tendencies with a nonzero gradient
wind, the testcase 14 SGP forcing and the prescribed tendencies) against
src/testcase_simple_phys.F, plus the gfortran random_number port and the
hurr_angle rounding."""

import math

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver

from cm1x.grid import Grid
from cm1x.testcase import GfortranRandom, hurr_angle, simple_phys

NI, NJ, NK, NG = 7, 5, 20, 3
DX, DY, DZ = 100.0, 120.0, 100.0

# (testcase, lspgrad, hurr_rotate, mtime)
CASES = {
    "simplerad4": (4, 0, 0.0, 60.0),
    "simplerad5_lspg3": (5, 3, 30.0, 60.0),
    "hbl_lspg3": (6, 3, 0.0, 600.0),
    "hbl_lspg3_rot": (15, 3, 17.0, 600.0),
    "sgp_early": (14, 0, 0.0, 7000.0),
    "sgp_late": (14, 0, 0.0, 47000.0),
    "tc3_prescribed": (3, 0, 0.0, 30.0),
}


class _Cfg(dict):
    __getattr__ = dict.__getitem__


def _inputs(seed):
    rng = np.random.default_rng(seed)
    S = (NK + 2, NJ + 2 * NG, NI + 2 * NG)
    U = (NK + 2, NJ + 2 * NG, NI + 2 * NG + 1)
    V = (NK + 2, NJ + 2 * NG + 1, NI + 2 * NG)
    kf = np.arange(NK + 3, dtype=float)  # Fortran k = 0..nk+2
    kh = np.arange(NK + 2, dtype=float)
    zf = np.broadcast_to(((kf - 1.0) * DZ)[:, None, None], (NK + 3,) + S[1:]).copy()
    zh = np.broadcast_to(((kh - 0.5) * DZ)[:, None, None], S).copy()
    rho0 = np.broadcast_to(
        (1.2 * np.exp(-zh[:, 0, 0] / 8000.0))[:, None, None], S
    ).copy()
    th0 = np.broadcast_to((300.0 + 0.004 * zh[:, 0, 0])[:, None, None], S).copy()
    qv = 0.013 - 0.0005 * kh[:, None, None] + rng.uniform(-2e-4, 2e-4, S)
    qc = np.where(rng.random(S) < 0.4, rng.uniform(0, 1e-3, S), 0.0)
    qr = np.where(rng.random(S) < 0.2, rng.uniform(0, 1e-4, S), 0.0)
    ins = {
        "mh": np.ones(S),
        "rho0": rho0,
        "th0": th0,
        "zh": zh,
        "zf": zf,
        "ua": rng.uniform(-15, 15, U),
        "va": rng.uniform(-15, 15, V),
        "tha": rng.uniform(-1, 1, S),
        "qa": np.stack([qv, qc, qr]),
        "uten1": rng.uniform(-1e-3, 1e-3, U),
        "vten1": rng.uniform(-1e-3, 1e-3, V),
        "thten1": rng.uniform(-1e-4, 1e-4, S),
        "qten": rng.uniform(-1e-7, 1e-7, (3,) + S),
        "ufrc": rng.uniform(-1e-4, 1e-4, NK + 2),
        "vfrc": rng.uniform(-1e-4, 1e-4, NK + 2),
        "thfrc": rng.uniform(-1e-5, 1e-5, NK + 2),
        "qvfrc": rng.uniform(-1e-8, 1e-8, NK + 2),
        "ug": np.zeros(NK + 2),
        "vg": 40.0 + rng.uniform(-5, 5, NK + 2),
        "dvdr": rng.uniform(-1e-3, 1e-3, NK + 2),
        "ruh": np.ones(NI + 2 * NG),
        "ruf": np.ones(NI + 2 * NG + 1),
        "rvh": np.ones(NJ + 2 * NG),
        "rvf": np.ones(NJ + 2 * NG + 1),
    }
    return ins


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    tc, lsp, rot, mtime = CASES[request.param]
    cfg = _Cfg(
        testcase=tc,
        lspgrad=lsp,
        hurr_rotate=rot,
        hurr_rad=40000.0,
        imoist=1,
        nqv=1,
        nql1=2,
        nql2=3,
        dx=DX,
        dy=DY,
    )
    ang = hurr_angle(cfg) if lsp == 3 else 0.0
    ins = _inputs(list(CASES).index(request.param))
    params = [NI, NJ, NK, tc, lsp, DX, DY, DZ, 40000.0, ang, mtime]
    outs = {k: ins[k].shape for k in ("uten1", "vten1", "thten1", "qten")}
    outs.update({k: (NK + 2,) for k in ("ufrc", "vfrc", "thfrc", "qvfrc")})
    S = ins["tha"].shape
    outs.update(frad=S, thraten=S, zir=S[1:])
    ref = run_driver("testcase", {"params": np.array(params, float), **ins}, outs)
    return cfg, mtime, ins, ref


@requires_gfortran
def test_simple_phys(bk, case):
    from cm1x import constants

    cfg, mtime, ins, ref = case
    constants.set_constants(cfg["testcase"])
    try:
        g = Grid(NI, NJ, NK, DX, DY, DZ)
        A = {k: bk.asarray(v.copy()) for k, v in ins.items()}
        xf = bk.asarray((np.arange(NI + 2 * NG + 1) - NG) * DX)
        yf = bk.asarray((np.arange(NJ + 2 * NG + 1) - NG) * DY)
        prof = {k: A[k] for k in ("ufrc", "vfrc", "thfrc", "qvfrc", "ug", "vg", "dvdr")}
        r = simple_phys(
            bk,
            g,
            cfg,
            mtime=mtime,
            prof=prof,
            rho0=A["rho0"],
            zh=A["zh"],
            zf=A["zf"],
            mh=A["mh"],
            th0=A["th0"],
            ua=A["ua"],
            va=A["va"],
            tha=A["tha"],
            qa=[A["qa"][n] for n in range(3)],
            ruh=A["ruh"],
            ruf=A["ruf"],
            rvh=A["rvh"],
            rvf=A["rvf"],
            xf=xf,
            yf=yf,
            uten1=A["uten1"],
            vten1=A["vten1"],
            thten1=A["thten1"],
            qten=[A["qten"][n] for n in range(3)],
        )
    finally:
        constants.set_constants(0)
    K, J, I = slice(1, NK + 1), slice(NG, NG + NJ), slice(NG, NG + NI)
    checks = [
        ("uten1", bk.to_numpy(r["uten1"]), ref["uten1"]),
        ("vten1", bk.to_numpy(r["vten1"]), ref["vten1"]),
        ("thten1", bk.to_numpy(r["thten1"]), ref["thten1"]),
        ("qv ten", bk.to_numpy(r["qten"][0]), ref["qten"][0]),
    ]
    for k in ("ufrc", "vfrc", "thfrc", "qvfrc"):
        checks.append((k, bk.to_numpy(r[k])[K], ref[k][K]))
    if cfg["testcase"] in (4, 5):
        checks.append(("frad", bk.to_numpy(r["frad"]), ref["frad"][1 : NK + 2, J, I]))
        checks.append(("zir", bk.to_numpy(r["zir"]), ref["zir"][J, I]))
        checks.append(("thraten", bk.to_numpy(r["thraten"]), ref["thraten"][K, J, I]))
    for name, got, want in checks:
        scale = max(np.abs(want).max(), 1e-300)
        err = np.abs(got - want).max() / scale
        assert err <= 1e-13, f"{name}: rel err {err:.2e}"


def test_gfortran_random_number():
    # init3d.F prints "rand-1" (the first draw after CM1's random_seed(put=))
    r = GfortranRandom()
    assert r.next() == 0.62093149829265015
    x = r.draws(1000)
    assert x.min() >= 0.0 and x.max() < 1.0


def test_hurr_angle_rounding():
    # param.F forms dble(hurr_rotate)*(pi_dp/180) in binary128 (kind dp with
    # -fdefault-real-8): for 30 degrees sin(-hurr_angle) is exactly -0.5
    a = hurr_angle({"hurr_rotate": 30.0})
    assert math.sin(-a) == -0.5
    assert hurr_angle({"hurr_rotate": 0.0}) == 0.0
