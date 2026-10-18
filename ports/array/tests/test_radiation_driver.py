"""radiation_driver (radopt=2, RRTMG, Kessler moisture) against the original
Fortran (double-precision CM1 objects): radconst / calc_coszen, cal_cldfra1,
the level interpolation of p8w / t8w, both RRTMG schemes on every column,
the eqtset=2 temperature-tendency conversion and thraten; daytime and night
(where the clear-sky shortwave tendency keeps its previous value)."""

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver

from cm1x import _ftables as ft
from cm1x import init_surface
from cm1x import module_ra_rrtmg_lw as lw
from cm1x import radiation_driver as rd
from cm1x.model import Model

pytestmark = requires_gfortran

RTOL = 1e-12
ROOT = ft.cm1_root()
DATA = [ROOT / "run" / "RRTMG_LW_DATA", ROOT / "run" / "RRTMG_SW_DATA"]
NML = ROOT / "ports" / "reference" / "kessler_periodic" / "namelist.input"
NI, NJ, NZ = 4, 3, 40
DTRAD = 60.0


def _close(a, b, rtol=RTOL, name=""):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    scale = max(np.max(np.abs(b)), 1e-300)
    err = np.max(np.abs(a - b)) / scale
    assert err <= rtol, f"{name}: rel err {err:.3e}"


def _case(mtime):
    """Base state of the kessler_periodic sounding on a small grid, with
    random potential-temperature / pressure perturbations, cloud water
    (warm and supercooled, saturated and sub-saturated) and rain."""
    m = Model(NML, overrides={"nx": NI, "ny": NJ, "nz": NZ, "sfcmodel": 1, "radopt": 2})
    b, s, cfg = m.base, m.state, m.cfg
    rng = np.random.default_rng(7)
    shp = b["pi0"].shape
    tha = np.asarray(s["tha"]) + 0.5 * rng.standard_normal(shp)
    ppi = 1e-4 * rng.standard_normal(shp)
    prs = np.asarray(s["prs"]) * (1.0 + 1e-4 * rng.standard_normal(shp))
    qv = np.asarray(s["qa"][0]) * (1.0 + 0.05 * rng.standard_normal(shp))
    zh = np.arange(shp[0]) * float(cfg.dz)
    cloud = (rng.random(shp) < 0.4) & (zh[:, None, None] < 12000.0)
    qc = np.where(cloud, 10.0 ** rng.uniform(-6, -2.7, shp), 0.0)
    qr = np.where(rng.random(shp) < 0.3, 10.0 ** rng.uniform(-6, -3, shp), 0.0)
    qv = np.where(cloud & (rng.random(shp) < 0.5), 1.5 * qv, qv)
    qa = np.stack([qv, qc, qr])
    sf = init_surface.getlanduse(int(cfg.season), int(cfg.lu0))
    shp2 = shp[1:]
    tsk = float(cfg.tsk0) + rng.uniform(-2, 2, shp2)
    surf = {
        "tsk": tsk,
        "albd": np.full(shp2, sf["albd"]),
        "emiss": np.full(shp2, sf["emiss"]),
        "xland": np.full(shp2, sf["xland"]),
    }
    swtenc = 1e-5 * rng.standard_normal(shp)
    ins = {
        "rmh": b["rmh"], "c1": b["c1"], "c2": b["c2"], "prs0": b["prs0"],
        "pi0": b["pi0"], "th0": b["th0"], "prs": prs, "ppi": ppi, "tha": tha,
        "rho": np.asarray(s["rho"]), "qa": qa, **surf, "swtenc": swtenc,
    }  # fmt: skip
    ins = {k: np.asarray(v, dtype=float) for k, v in ins.items()}
    cg = {k: float(b[k]) for k in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")}
    params = [NI, NJ, NZ, cfg.dz, m.dt, DTRAD, mtime, cfg.year, cfg.month, cfg.day]
    params += [cfg.hour, cfg.minute, cfg.second, cfg.ctrlat, cfg.ctrlon, 2]
    params += [cg[k] for k in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")]
    ins["params"] = np.array(params, dtype=float)
    return m, ins, cg


@pytest.fixture(scope="module", params=[0.0, 7.0 * 3600.0], ids=["day", "night"])
def ref(request):
    mtime = request.param
    m, ins, cg = _case(mtime)
    s3, s2 = ins["prs"].shape, ins["tsk"].shape
    outs = {k: s3 for k in ("swten", "lwten", "swtenc", "lwtenc", "cldfra", "thraten")}
    for k in ("glw", "gsw", "lwcf", "swcf", "lwupt", "lwdnb", "swupt", "swdnb"):
        outs[k] = s2
    outs["olr"] = (NJ, NI)  # olr(ni, nj)
    for k in ("swddir", "swddif", "coszen", "hrang"):
        outs[k] = s2
    for k in ("solcon", "declin", "julian", "jday"):
        outs[k] = (1,)
    r = run_driver("radiation_driver", ins, outs, files=DATA)
    return mtime, m, ins, cg, r


def test_solar(ref):
    mtime, m, _, _, r = ref
    cfg = m.cfg
    assert rd.julday(int(cfg.year), int(cfg.month), int(cfg.day)) == r["jday"][0]
    s = rd.solar(cfg, mtime, DTRAD)
    assert abs(s["julian"] - r["julian"][0]) <= 1e-15 * s["julian"]
    assert abs(s["declin"] - r["declin"][0]) <= 1e-15
    assert abs(s["solcon"] - r["solcon"][0]) <= 1e-12
    I = slice(3, 3 + NI)
    J = slice(3, 3 + NJ)
    assert np.max(np.abs(s["coszen"] - r["coszen"][J, I])) <= 1e-14
    assert np.max(np.abs(s["hrang"] - r["hrang"][J, I])) <= 1e-13
    assert (s["coszen"] > 0) == (mtime == 0.0)


def test_radiation_driver(bk, ref):
    mtime, m, ins, cg, r = ref
    cfg = m.cfg
    g = m.grid
    a = {k: bk.asarray(v) for k, v in ins.items() if k != "params"}
    s = rd.solar(cfg, mtime, DTRAD)
    nl = lw.nlayers_lw(NZ, rd.p_top({**cg, **ins}, NZ))
    out = rd.radiation_driver(
        bk, g, cfg, coszen=s["coszen"], solcon=s["solcon"], nlayers=nl,
        c1=a["c1"], c2=a["c2"], rmh=a["rmh"], pi0=a["pi0"], th0=a["th0"],
        prs=a["prs"], ppi=a["ppi"], tha=a["tha"], qa=[a["qa"][n] for n in range(3)],
        tsk=a["tsk"], albd=a["albd"], emiss=a["emiss"], xland=a["xland"],
        xice=bk.zeros(ins["tsk"].shape), xsnow=bk.zeros(ins["tsk"].shape),
        swtenc=a["swtenc"], cg=cg,
    )  # fmt: skip
    K, J, I = g.K, g.J, g.I
    cf = r["cldfra"][K, J, I]
    assert 0.0 < np.mean((cf > 0) & (cf < 1)) and np.any(cf == 1.0)
    for k in ("cldfra", "swten", "lwten", "swtenc", "lwtenc", "thraten"):
        _close(bk.to_numpy(out[k])[K, J, I], r[k][K, J, I], name=k)
    for k in ("glw", "gsw", "lwcf", "swcf", "lwupt", "lwdnb", "swupt", "swdnb"):
        _close(bk.to_numpy(out[k]), r[k][J, I], name=k)
    _close(bk.to_numpy(out["olr"]), r["olr"], name="olr")
    for k in ("swddir", "swddif"):
        _close(bk.to_numpy(out[k]), r[k][J, I], rtol=1e-11, name=k)
    if mtime > 0:  # night: no shortwave
        assert np.all(r["swten"][K, J, I] == 0.0) and np.all(r["gsw"][J, I] == 0.0)
