"""Thompson microphysics (thompson.F) and the ptype=3 branch of mp_driver.F
against the original Fortran (double-precision CM1 objects).

* tables: every thompson_init table/constant, bitwise (backend independent);
* mp_thompson on random physical columns vs drv_thompson_column (the real
  mp_thompson per column): to a tolerance on every backend, and bit for bit
  on numpy when numpy calls glibc's libm like the Fortran (a subprocess with
  numpy's AVX-512 SIMD exp/log/pow disabled, as test_model_dumps_supercell);
* mp_driver ptype=3 (pdefq, mp_gt_driver, the T/qv -> prs/pp3d/th3d update)
  vs drv_thompson_mp.

The Fortran thompson_init takes minutes; ``thompson_ref`` runs it once and
caches its tables and table files (``.fortran_cache/thompson_ref_ntc100``),
which the other drivers then read instead of recomputing.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from cm1x import constants as C
from cm1x import thompson as TH
from cm1x import thompson_tables as TT
from cm1x.grid import Grid
from cm1x.mp_driver import mp_driver

from fortran import requires_gfortran, run_driver
from thompson_cases import columns
from thompson_ref import DATFILES, fortran_tables, ref_dir

pytestmark = requires_gfortran

NT_C = 100.0
# tolerance relative to each field's largest magnitude: numpy with SIMD
# libm, XLA (exp/log10 and FMA contraction) and torch (SLEEF) differ from
# glibc in the last bits, amplified in cancellations (e.g. q + dt*qten
# near the R1 floor, the rain-number clamps)
RTOL = 1e-10
NPY_NO_AVX512 = (
    "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR X86_V4"
)
COL_OUT = (
    "qv",
    "qc",
    "qi",
    "qr",
    "qs",
    "qg",
    "ni",
    "nr",
    "nc",
    "t",
    "prv_rev",
    "prw_vcd",
)
PPT = ("pptrain", "pptsnow", "pptgraul", "pptice")
VT = ("vtik", "vtrk", "vtsk", "vtgk", "vtck")


def A(bk, x):
    return bk.asarray(np.asarray(x))


def close(bk, got, ref, rtol=RTOL, name=""):
    got = got if isinstance(got, np.ndarray) else bk.to_numpy(got)
    ref = np.asarray(ref)
    atol = rtol * max(np.max(np.abs(ref)), 1e-300)
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol, err_msg=name)


def _datfiles():
    fortran_tables(NT_C)  # runs drv_thompson_init once (cached)
    d = ref_dir(NT_C)
    return [d / f for f in DATFILES]


# ---------------------------------------------------------------------------
# tables


def test_tables_bitwise(bk):
    if bk.name != "numpy":
        pytest.skip("tables are built in numpy for every backend")
    ref = fortran_tables(NT_C)
    py = TT.tables(NT_C)
    bad = []
    for k, v in ref.items():
        assert k in py, f"{k} not computed"
        a, b = np.asarray(py[k], dtype=float), np.asarray(v, dtype=float)
        assert a.shape == b.shape, k
        if not np.array_equal(a, b):
            bad.append(f"{k}: {(a != b).sum()} of {a.size} differ")
    assert not bad, "; ".join(bad)


# ---------------------------------------------------------------------------
# mp_thompson on columns


def _fortran_columns(c, dt, eqtset):
    nk, ncol = c["t"].shape
    ins = {n: c[n].T for n in ("qv", "qc", "qi", "qr", "qs", "qg", "ni", "nr", "t",
                               "p", "w", "dz")}  # fmt: skip
    ins["params"] = np.array([NT_C, dt, nk, ncol, eqtset])
    outs = {n: (ncol, nk) for n in COL_OUT}
    outs["ppt"] = (ncol, 4)
    outs["vt"] = (ncol, 5, nk + 1)
    f = run_driver("thompson_column", ins, outs, files=_datfiles())
    ref = {n: f[n].T for n in COL_OUT}
    ref.update({n: f["ppt"][:, j] for j, n in enumerate(PPT)})
    ref.update({n: f["vt"][:, j, :].T for j, n in enumerate(VT)})
    return ref


def _python_columns(bk, c, dt, eqtset, jit=False):
    tb = TH.thompson_init(bk, None, nt_c_in=NT_C)
    kw = {
        f"{n}1d": A(bk, c[n]) for n in ("qv", "qc", "qi", "qr", "qs", "qg", "ni", "nr")
    }
    rho = 0.622 * c["p"] / (TT.R * c["t"] * (c["qv"] + 0.622))
    kw["nc1d"] = A(bk, NT_C * 1.0e6 / rho)
    kw.update(
        t1d=A(bk, c["t"]), p1d=A(bk, c["p"]), w1d=A(bk, c["w"]), dzq=A(bk, c["dz"])
    )
    cfg = {"eqtset": eqtset}

    def run(**k):
        return TH.mp_thompson(bk, tb, cfg, dt=dt, **k)

    if jit:
        import jax

        run = jax.jit(run)
    r = run(**kw)
    out = {n: r[n if n in ("prv_rev", "prw_vcd") else n + "1d"] for n in COL_OUT}
    out.update({n: r[n] for n in PPT + VT})
    return {k: bk.to_numpy(v) for k, v in out.items()}


CASES = [(0, 6.0, 2), (1, 150.0, 2), (2, 20.0, 1)]


@pytest.mark.parametrize("seed,dt,eqtset", CASES)
def test_mp_thompson_columns(bk, seed, dt, eqtset):
    c = columns(seed=seed)
    ref = _fortran_columns(c, dt, eqtset)
    got = _python_columns(bk, c, dt, eqtset, jit=bk.name == "jax")
    for n in ref:
        close(bk, got[n], ref[n], name=n)


_BITWISE = r"""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from cm1x import get_backend
import test_thompson as T

d = json.load(open(sys.argv[2]))
ref = {k: np.asarray(v) for k, v in d["ref"].items()}
c = {k: np.asarray(v) for k, v in d["inputs"].items()}
bk = get_backend("numpy", "float64")
dt, eqtset = float(sys.argv[3]), int(sys.argv[4])
got = T._python_columns(bk, c, dt, eqtset)
print(json.dumps({k: int((got[k] != ref[k]).sum()) for k in ref}))
"""


@pytest.mark.parametrize("seed,dt,eqtset", CASES)
def test_mp_thompson_columns_bitwise_glibc(bk, seed, dt, eqtset, tmp_path):
    if bk.name != "numpy":
        pytest.skip("glibc libm parity is only available with numpy")
    c = columns(seed=seed)
    ref = _fortran_columns(c, dt, eqtset)
    fref = tmp_path / "ref.json"
    # the inputs go along: columns() itself calls numpy's exp
    fref.write_text(
        json.dumps(
            {
                "ref": {k: v.tolist() for k, v in ref.items()},
                "inputs": {k: v.tolist() for k, v in c.items()},
            }
        )
    )
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=NPY_NO_AVX512)
    r = subprocess.run(
        [sys.executable, "-c", _BITWISE, here, str(fref), str(dt), str(eqtset)],
        env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr[-3000:]
    neq = json.loads(r.stdout.strip().splitlines()[-1])
    bad = {k: v for k, v in neq.items() if v}
    assert not bad, f"values differing from Fortran: {bad}"


# ---------------------------------------------------------------------------
# mp_driver ptype=3

G = Grid(ni=6, nj=4, nk=24, dx=1000.0, dy=1000.0, dz=400.0)
DT = 6.0
TSMALL = 1.0e-4
QSMALL = 1.0e-14


def _pad(a, nz=1, nxy=3):
    return np.pad(a, ((nz, nz), (nxy, nxy), (nxy, nxy)), mode="edge")


def make_ice_state(seed=0, g=G):
    """A mixed-phase state on ``g`` (with halos) from thompson_cases columns."""
    rng = np.random.default_rng(seed + 50)
    nk, nj, ni = g.nk, g.nj, g.ni
    c = columns(ncol=ni * nj, nk=nk, seed=seed, dz0=g.dz)
    f = {k: _pad(v.reshape(nk, nj, ni)) for k, v in c.items()}
    sh = g.s_shape
    prs = f["p"]
    t = f["t"]
    pp3d = rng.uniform(-2e-4, 2e-4, sh)
    pi0 = (prs * C.rp00) ** C.rovcp - pp3d
    th3d = rng.uniform(-1.0, 1.0, sh)
    th0 = t / (pi0 + pp3d) - th3d
    q3d = np.stack([f[n] for n in ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")])
    q3d[1] = np.where(rng.random(sh) < 0.05, -1e-7, q3d[1])  # pdefq has work to do
    rho = prs / (C.rd * t * (1.0 + q3d[0] * C.reps))
    w3d = np.concatenate([f["w"], f["w"][-1:]], axis=0)
    return dict(
        th0=th0, pi0=pi0, th3d=th3d, pp3d=pp3d, prs=prs, rho=rho, q3d=q3d, w3d=w3d,
        ruh=1.0 + 0.05 * rng.standard_normal(sh[2]),
        rvh=1.0 + 0.05 * rng.standard_normal(sh[1]),
        rmh=1.0 + 0.05 * rng.standard_normal(sh),
        mh=np.ones(sh), mf=np.ones(g.w_shape),
        zh=np.array(g.zh())[:, None, None] * np.ones(sh),
        rain=rng.random((1,) + sh[1:]), prate=rng.random(sh[1:]),
        qbudget=np.arange(1.0, 11.0), asq=np.linspace(0.1, 0.8, 8),
    )  # fmt: skip


def _mp_call(bk, s, eqtset=2, getvt=False, **kw):
    names = ("qbudget", "asq", "ruh", "rvh", "mh", "rmh", "zh", "mf", "rain", "prate",
             "pi0", "th0", "rho", "prs", "pp3d", "th3d", "q3d", "w3d")  # fmt: skip
    cfg = {
        "ptype": 3,
        "eqtset": eqtset,
        "tsmall": TSMALL,
        "qsmall": QSMALL,
        "nt_c": NT_C,
    }
    args = {n: A(bk, s[n]) for n in names}
    args.update(kw)
    return mp_driver(bk, G, cfg, dt=DT, nstep=1, getvt=getvt, **args)


@pytest.mark.parametrize("eqtset,getvt", [(2, True), (1, False)])
def test_mp_driver_ptype3(bk, eqtset, getvt):
    s = make_ice_state(3)
    ins = {n: s[n] for n in ("ruh", "rvh", "mh", "rmh", "zh", "mf", "rain", "prate",
                             "pi0", "th0", "rho", "prs", "pp3d", "th3d", "q3d", "w3d",
                             "qbudget", "asq")}  # fmt: skip
    ins["params"] = np.array([G.ni, G.nj, G.nk, G.dx, G.dy, G.dz, DT, eqtset, TSMALL,
                              QSMALL, 1, NT_C, float(getvt)])  # fmt: skip
    shp = G.s_shape
    ref = run_driver(
        "thompson_mp",
        ins,
        dict(
            rho=shp,
            prs=shp,
            pp3d=shp,
            th3d=shp,
            q3d=(8,) + shp,
            qten=(8,) + shp,
            thten=shp,
            dum1=shp,
            dum3=shp,
            rain=s["rain"].shape,
            prate=s["prate"].shape,
            qbudget=(10,),
            asq=(8,),
        ),  # fmt: skip
        files=_datfiles(),
    )
    out = _mp_call(bk, s, eqtset=eqtset, getvt=getvt)
    S = (G.K, G.J, G.I)
    for k in ("rho", "prs", "pp3d", "th3d", "rain", "prate"):
        close(bk, out[k], ref[k], name=k)
    for k in ("thten", "dum1", "dum3"):
        close(bk, bk.to_numpy(out[k])[S], ref[k][S], name=k)
    for n in range(8):
        close(bk, out["q3d"][n], ref["q3d"][n], name=f"q3d[{n}]")
    for n in range(6) if getvt else (0,):
        close(bk, bk.to_numpy(out["qten"][n])[S], ref["qten"][n][S], name=f"qten[{n}]")
    qb = bk.to_numpy(out["qbudget"])
    np.testing.assert_allclose(qb, ref["qbudget"], rtol=1e-11)
    assert np.all(qb[[0, 1, 4, 5]] != s["qbudget"][[0, 1, 4, 5]])
    w = s["ruh"][None, None, G.I] * s["rvh"][None, G.J, None] * s["rmh"][S]
    for n in range(6):
        mass = np.sum(np.abs(s["rho"][S] * s["q3d"][n][S]) * w) * G.dx * G.dy * G.dz
        np.testing.assert_allclose(
            bk.to_numpy(out["asq"])[n], ref["asq"][n], rtol=0, atol=1e-13 * mass,
            err_msg=f"asq[{n}]",
        )  # fmt: skip


def test_mp_driver_ptype3_jit():
    jax = pytest.importorskip("jax")
    from cm1x import get_backend

    bk = get_backend("jax", "float64")
    s = make_ice_state(4)
    eager = _mp_call(bk, s)

    @jax.jit
    def step(q3d, th3d):
        return _mp_call(bk, s, q3d=q3d, th3d=th3d, nstep_max=8)

    jitted = step(A(bk, s["q3d"]), A(bk, s["th3d"]))
    for k in ("q3d", "th3d", "pp3d", "prs", "rain", "prate"):
        close(bk, jitted[k], bk.to_numpy(eager[k]), rtol=1e-12, name=k)
