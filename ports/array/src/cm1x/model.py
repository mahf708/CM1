"""Whole-model driver: the active part of CM1's time loop (src/cm1.F).

    from cm1x.model import Model
    m = Model("ports/reference/kessler_periodic/namelist.input", backend="numpy")
    m.run(10)
    f = m.fields()          # numpy arrays laid out like cm1out.nc ([k, j, i])

Per time step (cm1.F ``timeloop``):  solve1 (pre-RK tendencies) -> solve2
(RK3 + acoustic steps) -> mp_driver (Kessler microphysics) -> solve3
(BCs, diabatic tendencies for next step, equate time levels).  The
turbulence/surface driver ``sfc_and_turb`` is skipped: with sgsmodel=0,
ipbl=0, isfcflx=0 it leaves every array that the dynamics read unchanged
(zero eddy viscosities / stresses / dissipation).
"""

from __future__ import annotations

from .backend import get_backend
from .misc import dtsmall
from .mp_driver import mp_driver
from .namelist import read_namelist
from .solve1 import solve1
from .solve2 import solve2
from .solve3 import solve3
from .state import initialize

FIELDS = (
    "u",
    "v",
    "w",
    "th",
    "thpert",
    "prs",
    "prspert",
    "pi",
    "pipert",
    "qv",
    "qc",
    "qr",
    "rho",
)


class Model:
    """CM1 reference-case model on any array backend.

    Parameters: ``namelist_path`` (CM1 namelist.input), ``backend``
    ('numpy', 'jax', 'torch'), ``dtype`` ('float64' or 'float32').
    Attributes: ``cfg``, ``bk``, ``grid``, ``base``, ``state`` (dict of
    arrays keyed by Fortran name), ``nstep``, ``mtime`` (s), ``dt``.
    """

    def __init__(self, namelist_path, backend="numpy", dtype="float64"):
        self.cfg = cfg = read_namelist(namelist_path)
        self.bk = bk = get_backend(backend, dtype)
        self.grid, self.base, st = initialize(cfg, bk)
        g = self.grid
        self.dt = float(cfg.dtl)
        ds = dtsmall(
            dt=self.dt,
            dbldt=self.dt,
            psolver=int(cfg.psolver),
            nx=int(cfg.nx),
            ny=int(cfg.ny),
            min_dx=float(cfg.dx),
            min_dy=float(cfg.dy),
            min_dz=float(cfg.dz),
            csound=float(cfg.csound),
            adapt_dt=int(cfg.adapt_dt),
        )
        cfg.nsound = ds["nsound"]
        self.dt = float(ds["dt"])
        self.nstep = 0
        self.mtime = 0.0

        numq = int(cfg.numq)
        s = {
            k: st[k]
            for k in (
                "ua",
                "u3d",
                "va",
                "v3d",
                "wa",
                "w3d",
                "ppi",
                "pp3d",
                "ppx",
                "tha",
                "th3d",
                "rho",
                "prs",
                "rr",
                "rf",
            )
        }
        s["qa"] = [st["qa"][m] for m in range(numq)]
        s["q3d"] = [st["q3d"][m] for m in range(numq)]
        s["mass1"] = st["mass1"]
        for name in ("qpten", "qtten", "qvten", "qcten", "divx"):
            s[name] = bk.zeros(g.s_shape)
        # param.F: dtu0 = 1 on i=1..ni+1, j=1..nj ; dtv0 = 1 on i=1..ni, j=1..nj+1
        n = g.ngxy
        dtu0 = bk.zeros(g.s_shape[1:])
        dtu0 = bk.set(dtu0, (slice(n, n + g.nj), slice(n, n + g.ni + 1)), 1.0)
        dtv0 = bk.zeros(g.s_shape[1:])
        dtv0 = bk.set(dtv0, (slice(n, n + g.nj + 1), slice(n, n + g.ni)), 1.0)
        self.dtu0, self.dtv0 = dtu0, dtv0
        nrain = 2 if int(cfg.imove) == 1 else 1
        s["rain"] = bk.zeros((nrain,) + tuple(g.s_shape[1:]))
        s["prate"] = bk.zeros(g.s_shape[1:])
        s["qbudget"] = bk.zeros((10,))
        s["asq"] = bk.zeros((numq,))
        self.state = s

    # ------------------------------------------------------------------
    def step(self, hooks=None):
        """Advance one time step (dt = dtl)."""
        bk, g, cfg, b, s = self.bk, self.grid, self.cfg, self.base, self.state
        xp = bk.xp
        dt = self.dt

        r1 = solve1(
            bk,
            g,
            cfg,
            dt=dt,
            tauh=b["tauh"],
            taus=b["taus"],
            tauf=b["tauf"],
            rho=s["rho"],
            pi0=b["pi0"],
            th0=b["th0"],
            u0=b["u0"],
            ua=s["ua"],
            v0=b["v0"],
            va=s["va"],
            wa=s["wa"],
            ppi=s["ppi"],
            tha=s["tha"],
            qa=s["qa"],
            qpten=s["qpten"],
            qtten=s["qtten"],
            qvten=s["qvten"],
            qcten=s["qcten"],
        )

        r2 = solve2(
            bk,
            g,
            cfg,
            dt=dt,
            mass1=s["mass1"],
            uh=b["uh"],
            vh=b["vh"],
            ruh=b["ruh"],
            rvh=b["rvh"],
            uf=b["uf"],
            vf=b["vf"],
            mh=b["mh"],
            rmh=b["rmh"],
            mf=b["mf"],
            c1=b["c1"],
            c2=b["c2"],
            pi0=b["pi0"],
            rho0=b["rho0"],
            th0=b["th0"],
            rth0=b["rth0"],
            qv0=b["qv0"],
            qc0=b["qc0"],
            qi0=b["qi0"],
            rr0=b["rr0"],
            rf0=b["rf0"],
            rrf0=b["rrf0"],
            dtu0=self.dtu0,
            dtv0=self.dtv0,
            divx=s["divx"],
            rho=s["rho"],
            prs=s["prs"],
            u0=b["u0"],
            ua=s["ua"],
            u3d=s["u3d"],
            uten1=r1["uten1"],
            v0=b["v0"],
            va=s["va"],
            v3d=s["v3d"],
            vten1=r1["vten1"],
            wa=s["wa"],
            w3d=s["w3d"],
            wten1=r1["wten1"],
            ppi=s["ppi"],
            pp3d=s["pp3d"],
            ppten=r1["ppten"],
            ppx=s["ppx"],
            tha=s["tha"],
            th3d=s["th3d"],
            thten1=r1["thten1"],
            qpten=s["qpten"],
            qtten=s["qtten"],
            qvten=s["qvten"],
            qcten=s["qcten"],
            qa=s["qa"],
            q3d=s["q3d"],
            qten=r1["qten"],
            hooks=hooks,
        )
        if hooks is not None:
            hooks("post_solve2", 0, r2)

        mp = mp_driver(
            bk,
            g,
            cfg,
            dt=dt,
            qbudget=s["qbudget"],
            asq=s["asq"],
            ruh=b["ruh"],
            rvh=b["rvh"],
            mh=b["mh"],
            rmh=b["rmh"],
            zh=b["zh"],
            mf=b["mf"],
            rain=s["rain"],
            prate=s["prate"],
            pi0=b["pi0"],
            th0=b["th0"],
            rho=r2["rho"],
            prs=r2["prs"],
            pp3d=r2["pp3d"],
            th3d=r2["th3d"],
            q3d=xp.stack(r2["q3d"], axis=0),
        )
        if hooks is not None:
            hooks("post_mp", 0, mp)
        q3d = [mp["q3d"][m] for m in range(len(r2["q3d"]))]

        r3 = solve3(
            bk,
            g,
            cfg,
            dt=dt,
            c1=b["c1"],
            c2=b["c2"],
            u3d=r2["u3d"],
            v3d=r2["v3d"],
            w3d=r2["w3d"],
            pp3d=mp["pp3d"],
            th3d=mp["th3d"],
            q3d=q3d,
            rho=mp["rho"],
            ppx=r2["ppx"],
            qpten=r2["qpten"],
            qtten=r2["qtten"],
            qvten=r2["qvten"],
            qcten=r2["qcten"],
        )

        s.update(
            u3d=r2["u3d"],
            v3d=r2["v3d"],
            w3d=r2["w3d"],
            divx=r2["divx"],
            prs=mp["prs"],
            rain=mp["rain"],
            prate=mp["prate"],
            qbudget=mp["qbudget"],
            asq=mp["asq"],
        )
        for k in (
            "pp3d",
            "th3d",
            "q3d",
            "ppx",
            "rho",
            "rr",
            "rf",
            "qpten",
            "qtten",
            "qvten",
            "qcten",
            "ua",
            "va",
            "wa",
            "ppi",
            "tha",
            "qa",
        ):
            s[k] = r3[k]
        self.nstep += 1
        self.mtime += dt
        return self

    def run(self, nsteps):
        """Advance ``nsteps`` time steps."""
        for _ in range(int(nsteps)):
            self.step()
        return self

    # ------------------------------------------------------------------
    def fields(self):
        """Interior fields as numpy arrays, laid out like CM1's netCDF output
        (u: [nk, nj, ni+1], v: [nk, nj+1, ni], w: [nk+1, nj, ni], others
        [nk, nj, ni]).  Definitions follow writeout.F."""
        bk, g, b, s = self.bk, self.grid, self.base, self.state
        K, J, I = g.K, g.J, g.I
        cfg = self.cfg
        out = {
            "u": s["ua"][K, J, g.IF],
            "v": s["va"][K, g.JF, I],
            "w": s["wa"][g.KF, J, I],
            "th": b["th0"][K, J, I] + s["tha"][K, J, I],
            "thpert": s["tha"][K, J, I],
            "prs": s["prs"][K, J, I],
            "prspert": s["prs"][K, J, I] - b["prs0"][K, J, I],
            "pi": b["pi0"][K, J, I] + s["ppi"][K, J, I],
            "pipert": s["ppi"][K, J, I],
            "rho": s["rho"][K, J, I],
        }
        for name, idx in (("qv", cfg.nqv), ("qc", cfg.nqc), ("qr", cfg.nqr)):
            if idx:
                out[name] = s["qa"][idx - 1][K, J, I]
        return {k: bk.to_numpy(v) for k, v in out.items()}
