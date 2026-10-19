"""Whole-model driver: the active part of CM1's time loop (src/cm1.F).

    from cm1x.model import Model
    m = Model("ports/reference/kessler_periodic/namelist.input", backend="numpy")
    m.run(10)
    f = m.fields()          # numpy arrays laid out like cm1out.nc ([k, j, i])

Per time step (cm1.F ``timeloop``):  solve1 (pre-RK tendencies) -> solve2
(RK3 + acoustic steps) -> mp_driver (microphysics, by ptype) -> solve3
(BCs, diabatic tendencies for next step, equate time levels) ->
sfc_and_turb (sgsmodel=1 only: eddy viscosities, stresses and TKE
dissipation for the next step; it also runs once at t=0).  With sgsmodel=0,
ipbl=0, isfcflx=0 sfc_and_turb leaves every array that the dynamics read
unchanged (zero eddy viscosities / stresses / dissipation) and is skipped.

Lateral boundaries come from the namelist (periodic, open with irbc=4
radiative phase speeds, or walls); the radbc* phase speeds are persistent
state.  imove (domain translation) changes only the base-state winds
(u0 - umove, v0 - vmove; base.py) and output-only diagnostics.
"""

from __future__ import annotations

import dataclasses

from . import bc as _bc
from .backend import get_backend, is_array, torch_kernel_options
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
    ('numpy', 'jax', 'torch'; also 'dask', 'array_api_strict'), ``dtype``
    ('float64' or 'float32'), ``jit`` (jax: compile the whole time step with
    ``jax.jit``, ``jit_options`` passed to it; torch: ``torch.compile`` each
    hot routine separately (``Backend.kernel``; ``jit_options`` update
    ``backend.TORCH_KERNEL_DEFAULTS``), or the whole step with
    ``jit_options={"whole_step": True}``; a no-op elsewhere),
    ``overrides`` ({namelist name: value}, e.g. ``{"nx": 80, "ny": 80}``).
    ``step(hooks=...)`` always runs eagerly.
    Attributes: ``cfg``, ``bk``, ``grid``, ``base``, ``state`` (dict of
    arrays keyed by Fortran name), ``nstep``, ``mtime`` (s), ``dt``.
    """

    def __init__(
        self,
        namelist_path,
        backend="numpy",
        dtype="float64",
        fortran_dp=False,
        jit=False,
        jit_options=None,
        overrides=None,
    ):
        self.cfg = cfg = read_namelist(
            namelist_path, dp=fortran_dp, overrides=overrides
        )
        self.bk = bk = get_backend(backend, dtype)
        self.jit = bool(jit)
        self.jit_options = dict(jit_options or {})
        # torch: jit=True compiles the hot routines one by one (Backend.kernel)
        # and runs the step eagerly between them; whole_step=True compiles the
        # whole step instead (very slow to compile, see bench/results_cpu.md)
        self._whole_step = bool(self.jit_options.pop("whole_step", bk.name != "torch"))
        if self.jit and bk.name == "torch" and not self._whole_step:
            self.bk = bk = dataclasses.replace(
                bk, kernels=torch_kernel_options(self.jit_options)
            )
        # compiled step, shared by shallow copies (cm1x._grad_utils.clone)
        self._jit_cache = {}
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
        # param.F: dtu0 = 1 on i=1..ni+1, j=1..nj ; dtv0 = 1 on i=1..ni,
        # j=1..nj+1; zero on open / wall edges
        self.dtu0, self.dtv0 = _bc.dtu0_dtv0(bk, g, cfg)
        # cm1.F: radiative phase speeds, (jb:je,kb:ke) -> [k, j]; zero at t=0
        S = g.s_shape
        s["radbcw"] = bk.zeros((S[0], S[1]))
        s["radbce"] = bk.zeros((S[0], S[1]))
        s["radbcs"] = bk.zeros((S[0], S[2]))
        s["radbcn"] = bk.zeros((S[0], S[2]))
        self.les = int(cfg.get("sgsmodel", 0)) == 1
        if self.les:
            from .turb import turb_constants

            s["tkea"], s["tke3d"] = st["tkea"], st["tke3d"]
            tc = turb_constants(bk, g, cfg)
            self.base["cm0"] = tc["cm0"]
            for k in ("cme", "csm", "ce1", "ce2"):
                s[k] = tc[k]
        nrain = 2 if int(cfg.imove) == 1 else 1
        s["rain"] = bk.zeros((nrain,) + tuple(g.s_shape[1:]))
        s["prate"] = bk.zeros(g.s_shape[1:])
        s["qbudget"] = bk.zeros((10,))
        s["asq"] = bk.zeros((numq,))
        self.state = s
        if self.les:
            self._sfc_and_turb(
                rtime=0.0,
                u3d=s["u3d"],
                v3d=s["v3d"],
                w3d=s["w3d"],
                pp3d=s["pp3d"],
                th3d=s["th3d"],
                q3d=s["q3d"],
            )
        # lazy backends (dask): compute the initial state once
        self.base, self.state, self.dtu0, self.dtv0 = bk.materialize(
            (self.base, self.state, self.dtu0, self.dtv0)
        )

    # ------------------------------------------------------------------
    def _sfc_and_turb(self, s=None, b=None, *, rtime, u3d, v3d, w3d, pp3d, th3d, q3d):
        """cm1.F: ``sfc_and_turb`` on the new time level (LES path).  Updates
        the state dict ``s`` (default ``self.state``) in place."""
        from .turb import sfc_and_turb

        bk, g, cfg = self.bk, self.grid, self.cfg
        s = self.state if s is None else s
        b = self.base if b is None else b
        keys = (
            "kmh",
            "kmv",
            "khh",
            "khv",
            "nm",
            "defv",
            "defh",
            "lenscl",
            "dissten",
            "t11",
            "t12",
            "t13",
            "t22",
            "t23",
            "t33",
        )
        r = sfc_and_turb(
            bk,
            g,
            cfg,
            rtime=rtime,
            ruh=b["ruh"],
            rvh=b["rvh"],
            uh=b["uh"],
            uf=b["uf"],
            vh=b["vh"],
            vf=b["vf"],
            mh=b["mh"],
            c1=b["c1"],
            c2=b["c2"],
            mf=b["mf"],
            rmf=b["rmf"],
            pi0=b["pi0"],
            th0=b["th0"],
            rho=s["rho"],
            rf=s["rf"],
            prs=s["prs"],
            ua=u3d,
            va=v3d,
            wa=w3d,
            ppi=pp3d,
            tha=th3d,
            qa=q3d,
            tkea=s["tkea"],
            tke3d=s["tke3d"],
            cm0=b["cm0"],
            cme=s["cme"],
            csm=s["csm"],
            ce1=s["ce1"],
            ce2=s["ce2"],
            **{k: s.get(k) for k in keys},
        )
        for k in keys + ("tkea", "tke3d", "cme", "csm", "ce1", "ce2"):
            s[k] = r[k]

    # ------------------------------------------------------------------
    def step(self, hooks=None):
        """Advance one time step (dt = dtl).  With ``jit=True`` (and no
        ``hooks``) the compiled step is used."""
        mtime = self.mtime + self.dt
        if self.jit and self._whole_step and hooks is None:
            fn = self._compiled_step()
            barr = {k: v for k, v in self.base.items() if is_array(v)}
            self.state = fn(
                self.state,
                barr,
                self.dtu0,
                self.dtv0,
                self.bk.scalar(mtime),
            )
        else:
            self.state = self._advance(
                self.state, self.base, self.dtu0, self.dtv0, mtime, hooks=hooks
            )
        self.state = self.bk.materialize(self.state)
        self.nstep += 1
        self.mtime = mtime
        return self

    def _compiled_step(self):
        """The whole time step as one compiled function
        ``(state, base_arrays, dtu0, dtv0, mtime) -> state`` (Backend.compile:
        jax.jit, torch.compile, or the plain function elsewhere)."""
        if "step" not in self._jit_cache:
            bstatic = {k: v for k, v in self.base.items() if not is_array(v)}

            def fn(s, barr, dtu0, dtv0, mtime):
                return self._advance(s, {**bstatic, **barr}, dtu0, dtv0, mtime)

            self._jit_cache["step"] = self.bk.compile(fn, **self.jit_options)
        return self._jit_cache["step"]

    def _advance(self, s, b, dtu0, dtv0, mtime, hooks=None):
        """Pure time step: returns the state dict after one step from ``s``
        (``b`` = base dict, ``mtime`` = model time at the end of the step,
        Python float or 0-d array).  On numpy/torch arrays of ``s`` may be
        updated in place, as before."""
        bk, g, cfg = self.bk, self.grid, self.cfg
        s = dict(s)
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
            turb=self._turb_inputs(s, b) if self.les else None,
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
            dtu0=dtu0,
            dtv0=dtv0,
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
            ruf=b["ruf"],
            rvf=b["rvf"],
            radbcw=s["radbcw"],
            radbce=s["radbce"],
            radbcs=s["radbcs"],
            radbcn=s["radbcn"],
            tkea=s.get("tkea"),
            tke3d=s.get("tke3d"),
            tketen=r1.get("tketen"),
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
            w3d=r2["w3d"],
            nstep=self.nstep,
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
            ua=s["ua"],
            va=s["va"],
            wa=s["wa"],
            ppi=s["ppi"],
            tha=s["tha"],
            qa=s["qa"],
            tke3d=r2["tke3d"],
            tkea=s.get("tkea"),
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
            radbcw=r2["radbcw"],
            radbce=r2["radbce"],
            radbcs=r2["radbcs"],
            radbcn=r2["radbcn"],
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
        if self.les:
            s["tkea"], s["tke3d"] = r3["tkea"], r2["tke3d"]
            self._sfc_and_turb(
                s,
                b,
                rtime=mtime,
                u3d=r2["u3d"],
                v3d=r2["v3d"],
                w3d=r2["w3d"],
                pp3d=r3["pp3d"],
                th3d=r3["th3d"],
                q3d=r3["q3d"],
            )
        return s

    def _turb_inputs(self, s=None, b=None):
        s = self.state if s is None else s
        b = self.base if b is None else b
        d = {k: b[k] for k in ("uh", "uf", "vh", "vf", "mh", "mf")}
        for k in (
            "rr",
            "rf",
            "tkea",
            "kmh",
            "kmv",
            "khh",
            "khv",
            "nm",
            "defv",
            "defh",
            "dissten",
            "t11",
            "t12",
            "t13",
            "t22",
            "t23",
            "t33",
        ):
            d[k] = s[k]
        return d

    def run(self, nsteps):
        """Advance ``nsteps`` time steps."""
        for _ in range(int(nsteps)):
            self.step()
        return self

    # ------------------------------------------------------------------
    def fields(self):
        """Interior fields as numpy arrays, laid out like CM1's netCDF output
        (u: [nk, nj, ni+1], v: [nk, nj+1, ni], w, and with sgsmodel=1 tke, kmh,
        kmv, khh, khv: [nk+1, nj, ni], others [nk, nj, ni]).  Definitions
        follow writeout.F (u, v are grid-relative, as CM1 writes them)."""
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
        # moisture arrays by their param.F names (qv, qc, qr, qi, ..., nci, ...)
        for n, name in enumerate(cfg.get("qname", ()) or ()):
            out[name] = s["qa"][n][K, J, I]
        if self.les:  # w-level fields [nk+1, nj, ni]
            out["tke"] = s["tkea"][g.KF, J, I]
            for name in ("kmh", "kmv", "khh", "khv"):
                out[name] = s[name][g.KF, J, I]
        return {k: bk.to_numpy(v) for k, v in out.items()}
