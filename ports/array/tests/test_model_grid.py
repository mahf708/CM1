"""Stretched vertical grids (stretch_z = 1..4) and the adaptive time step
(adapt_dt = 1) vs CM1's float64 state dumps (cm1dump, -DDP build).

Needs CM1X_GRID_REF_DIR pointing at the output directory of

    JOBS=2 cxx/scripts/make_refs.sh <dir> dry_stretch kessler_tke_stretch \
        kessler_terrain_stretch cbl_sfc_stretch pbl2_kessler_stretch \
        adapt_dry adapt_kessler_tke

(case namelists in cxx/cases; each case directory holds its namelist and
input_grid_z).  Cases missing from the directory are skipped.

The model runs in a subprocess with numpy's AVX-512 libm disabled
(NPY_DISABLE_CPU_FEATURES), as in test_model_dumps_supercell.py.  The eddy
viscosities of dump N come from state N-1 (sfc_and_turb at the top of CM1's
step N+1), while cm1x runs sfc_and_turb at the end of its step: K fields and
ksmax are compared one step behind.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REF_DIR = os.environ.get("CM1X_GRID_REF_DIR")
TOL = 1e-10
NPY_NO_AVX512 = (
    "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR X86_V4"
)

_SCRIPT = r"""
import json, sys
from pathlib import Path
import numpy as np
from cm1x.dump import read_dump, read_index
from cm1x.model import Model

ref, nmax, jit = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
backend = "numpy" if jit == "0" else "jax"
kw = {"fortran_dp": True} if backend == "numpy" else {"jit": True}
I = (slice(1, -1), slice(3, -3), slice(3, -3))
KF = ("kmh", "kmv", "khh", "khv")


def adapt_line(n):
    for line in (ref / f"step_{n:06d}" / "index.txt").read_text().splitlines():
        if line.startswith("#adapt_dt"):
            p = line.split()[1:]
            return [float(p[0]), int(p[1])] + [float(x) for x in p[2:7]] + [
                int(x) for x in p[7:]
            ]
    return None


def rel(a, b):
    a, b = np.asarray(a)[I], b[I]
    sc = np.abs(b).max()
    return float(np.abs(a - b).max() / sc) if sc > 0 else float(np.abs(a).max())


m = Model(ref / "namelist.input", backend=backend, **kw)
steps = sorted(int(p.name.split("_")[1]) for p in ref.glob("step_*"))
steps = [n for n in steps if n <= nmax]
out = {"errs": {}, "adapt": {}, "dts": [], "nsounds": []}
prevK, prevks = None, None
for n in steps:
    while m.nstep < n:
        if m.nstep == n - 1 and "kmh" in m.state:
            prevK = {k: np.asarray(m.state[k]).copy() for k in KF}
            prevks = m.adapt.ksmax if m.adapt is not None else None
        m.step()
        out["dts"].append(m.dt)
        out["nsounds"].append(int(m.cfg.nsound))
    d, t = read_dump(ref / f"step_{n:06d}")
    s = m.state
    errs = {"mtime": abs(m.mtime - t) / max(t, 1.0)}
    for k in ("ua", "va", "wa", "tha", "ppi", "prs", "rho"):
        errs[k] = rel(s[k], d[k])
    for i, k in enumerate([k for k in d if k.startswith("qa_")]):
        errs[k] = rel(s["qa"][i], d[k])
    if "tkea" in s:
        errs["tkea"] = rel(s["tkea"], d["tkea"])
        if n > 0 and prevK is not None:
            for k in KF:
                errs[k] = rel(prevK[k], d[k])
    a = adapt_line(n)
    if a is not None and m.adapt is not None:
        ad = m.adapt
        mine = [m.dt, int(m.cfg.nsound), ad.cflmax, prevks, ad.taptim,
                ad.stattim, ad.rsttim, int(ad.dowriteout), int(ad.dostat),
                int(ad.dorestart)]
        out["adapt"][n] = [mine, a]
    out["errs"][n] = errs
print(json.dumps(out))
"""


def _case(name):
    if not REF_DIR:
        pytest.skip("set CM1X_GRID_REF_DIR to the make_refs.sh output directory")
    d = Path(REF_DIR) / name
    if not (d / "step_000000").is_dir():
        pytest.skip(f"no {name} dumps in {REF_DIR}")
    return d


def _run(ref, nmax=10**9, jit=False):
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=NPY_NO_AVX512)
    if jit:
        env["JAX_ENABLE_X64"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(ref), str(nmax), "1" if jit else "0"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _check_errs(res, tol=TOL):
    bad = [
        f"{k} at step {n}: {e:.2e}"
        for n, errs in res["errs"].items()
        for k, e in errs.items()
        if not e < tol
    ]
    assert not bad, "; ".join(bad)


def _numpy_only(bk):
    if bk.name != "numpy":
        pytest.skip("whole-run dump comparisons run on numpy (subprocess)")


@pytest.mark.reference
@pytest.mark.parametrize(
    "name, nmax",
    [
        ("dry_stretch", 20),  # stretch_z = 1, dry
        ("kessler_tke_stretch", 30),  # stretch_z = 2 (ZGRID), Kessler + TKE
        ("kessler_terrain_stretch", 30),  # stretch_z = 1 + terrain
    ],
)
def test_stretched_vs_cm1_dumps(bk, name, nmax):
    _numpy_only(bk)
    _check_errs(_run(_case(name), nmax))


@pytest.mark.reference
@pytest.mark.parametrize(
    "name", ["cbl_sfc_stretch", "pbl2_kessler_stretch", "kessler_tke_stretch"]
)
def test_vertical_grid_vs_cm1_dump(bk, name):
    """stretch_z = 3 / 4 (input_grid_z) and 2: param.F's columns, bitwise."""
    _numpy_only(bk)
    from cm1x.base import vertical_grid
    from cm1x.dump import read_dump
    from cm1x.namelist import read_namelist

    ref = _case(name)
    cfg = read_namelist(ref / "namelist.input")
    nk = int(cfg.nz)
    zf, zh, mh, mf, sigmaf = vertical_grid(cfg, nk)
    d, _ = read_dump(ref / "step_000000")
    col = (slice(None), 3, 3)
    np.testing.assert_array_equal(np.asarray(zf), d["zf"][col])
    np.testing.assert_array_equal(np.asarray(sigmaf), d["sigmaf"])
    np.testing.assert_array_equal(np.asarray(mf), d["mf"][col])
    # zh / mh: param.F's ghost reset (zh(0) = zf(1), zh(nk+1) = zf(nk+1))
    # happens in param_grid
    np.testing.assert_array_equal(np.asarray(zh[1 : nk + 1]), d["zh"][col][1 : nk + 1])
    np.testing.assert_array_equal(np.asarray(mh[1 : nk + 1]), d["mh"][col][1 : nk + 1])


def _check_adapt(res, min_dts=3, min_nsound=2):
    assert len(set(res["dts"])) >= min_dts, sorted(set(res["dts"]))
    assert len(set(res["nsounds"])) >= min_nsound, sorted(set(res["nsounds"]))
    assert res["adapt"], "no #adapt_dt lines in the dumps"
    for n, (mine, a) in res["adapt"].items():
        n = int(n)
        assert mine[0] == a[0], f"dt at step {n}: {mine[0]!r} vs {a[0]!r}"
        assert mine[1] == a[1], f"nsound at step {n}"
        assert mine[4:7] == a[4:7], f"taptim/stattim/rsttim at step {n}"
        assert mine[7:] == a[7:], f"output flags at step {n}"
        if n > 0:  # the step-0 dump's cflmax is statpack's (at the new dt)
            assert abs(mine[2] - a[2]) <= 1e-12 * max(abs(a[2]), 1e-30), n
            if mine[3] is not None:
                # early on ksmax is ~1e-13 (round-off of tiny K): compare
                # on the scale of ks_limit (0.09)
                assert abs(mine[3] - a[3]) <= 1e-10 * max(abs(a[3]), 1e-3), n


@pytest.mark.reference
@pytest.mark.parametrize("name", ["adapt_dry", "adapt_kessler_tke"])
def test_adapt_dt_vs_cm1_dumps(bk, name):
    _numpy_only(bk)
    res = _run(_case(name))
    _check_errs(res)
    _check_adapt(res)


@pytest.mark.reference
def test_adapt_dt_jax_jit(bk):
    """jax.jit with adapt_dt: dt is a traced input, one compile per nsound."""
    if bk.name != "jax":
        pytest.skip("jax only")
    res = _run(_case("adapt_dry"), jit=True)
    _check_errs(res, tol=1e-11)  # XLA's own libm: round-off, not bitwise
    _check_adapt(res)


def test_adapt_dt_unsupported():
    """Options whose timing getnewdt / the output bookkeeping would get
    wrong raise instead of running something different."""
    from cm1x.adapt_dt import AdaptDt
    from cm1x.model import Model
    from cm1x.namelist import read_namelist

    nml = (
        Path(__file__).resolve().parents[2]
        / "reference"
        / "kessler_periodic"
        / "namelist.input"
    )
    ok = read_namelist(nml).replace(adapt_dt=1, rstfrq=-3600.0)
    AdaptDt.from_config(ok)
    for kw in (
        {"radopt": 2},
        {"psolver": 1},
        {"rstfrq": 0.0},
        {"ptype": 2},
        {"dodomaindiag": 1},
        {"cm1setup": 2, "sgsmodel": 0, "ipbl": 1},
    ):
        with pytest.raises(NotImplementedError):
            AdaptDt.from_config(ok.replace(**kw))
    with pytest.raises(NotImplementedError):
        Model(nml, backend="numpy", overrides={"adapt_dt": 2})
