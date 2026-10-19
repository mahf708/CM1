"""Whole-model driver: the active part of CM1's time loop (src/cm1.F).

    from cm1x.model import Model
    m = Model("ports/reference/kessler_periodic/namelist.input", backend="numpy")
    m.run(10)
    f = m.fields()          # numpy arrays laid out like cm1out.nc ([k, j, i])

Per time step (cm1.F ``timeloop``):  solve1 (pre-RK tendencies) -> solve2
(RK3 + acoustic steps) -> mp_driver (Kessler microphysics) -> solve3
(BCs, diabatic tendencies for next step, equate time levels) ->
radiation_driver (radopt=2: RRTMG on the new state when the radiation
time is reached, and once at t=0; its potential-temperature tendency
thraten enters solve1 until the next call) -> sfc_and_turb (sgsmodel=1:
eddy viscosities, stresses and TKE dissipation for the next step;
sgsmodel=0 with sfcmodel=1: the surface layer, i.e. u*, z0 and the surface
stress of the semi-slip lower boundary bbc=3 that param.F forces; both
also run once at t=0).  With sgsmodel=0, sfcmodel=0, ipbl=0, isfcflx=0
sfc_and_turb leaves every array that the dynamics read unchanged (zero eddy
viscosities / stresses / dissipation) and is skipped.

Lateral boundaries come from the namelist (periodic, open with irbc=4
radiative phase speeds, or walls); the radbc* phase speeds are persistent
state.  imove (domain translation) changes only the base-state winds
(u0 - umove, v0 - vmove; base.py) and output-only diagnostics.
"""

from __future__ import annotations

import dataclasses

from . import bc as _bc
from .backend import get_backend, is_array, torch_kernel_options
from .adapt_dt import AdaptDt, calccflquick, calcksquick
from .misc import dtsmall
from .mp_driver import mp_driver
from .namelist import read_namelist
from .solve1 import solve1
from .solve2 import solve2
from .solve3 import solve3
from .state import initialize

# terrain metrics (param.F / init_terrain.F) passed to solve2 / sound
TERRAIN_KEYS = (
    "rds",
    "sigma",
    "rdsf",
    "sigmaf",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "dzdx",
    "dzdy",
    "gx",
    "gxu",
    "gy",
    "gyv",
    "zt",
)
CG_KEYS = ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")

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
# radopt=2 output (writeout.F: output_radten=1 / output_sfcparams=1)
RAD3D = ("swten", "lwten", "cldfra")
RAD2D = (
    "lwupt",
    "lwdnt",
    "lwupb",
    "lwdnb",
    "swupt",
    "swdnt",
    "swupb",
    "swdnb",
    "lwcf",
    "swcf",
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

    ``adapt_dt = 1`` (misclibs.F getnewdt): ``dt`` and ``cfg.nsound`` change
    between steps (``adapt``: an ``adapt_dt.AdaptDt`` with the CFL numbers,
    output timers and flags; host-side Python floats).  Under jax.jit the
    step takes ``dt`` as a traced 0-d input, so a new dt is not a recompile;
    the acoustic loop length ``nsound`` is a compile-time constant, so there
    is one compiled step per distinct nsound (a recompile the first time
    getnewdt moves to a new nsound; even values 4..192).  torch's
    per-routine kernels already take dt as a traced input.
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
        # param.F: min_dz over the interior (stretched grids, terrain)
        zfn = bk.to_numpy(self.base["zf"])
        self.min_dz = float(
            (zfn[2 : g.nk + 2, g.J, g.I] - zfn[1 : g.nk + 1, g.J, g.I]).min()
        )
        ds = dtsmall(
            dt=self.dt,
            dbldt=self.dt,
            psolver=int(cfg.psolver),
            nx=int(cfg.nx),
            ny=int(cfg.ny),
            min_dx=float(cfg.dx),
            min_dy=float(cfg.dy),
            min_dz=self.min_dz,
            csound=float(cfg.csound),
            adapt_dt=int(cfg.adapt_dt),
        )
        cfg.nsound = ds["nsound"]
        self.dt = float(ds["dt"])
        self.nstep = 0
        self.mtime = 0.0
        # adapt_dt = 1: AdaptDt holds getnewdt's state (dbldt, cflmax, ksmax,
        # output timers and flags) on the host
        self.adapt = None
        if int(cfg.get("adapt_dt", 0)) not in (0, 1):
            raise NotImplementedError(f"adapt_dt={cfg.adapt_dt} is not ported")
        if int(cfg.get("adapt_dt", 0)) == 1:
            self.adapt = AdaptDt.from_config(cfg)

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
        if self.adapt is not None:
            # cm1.F before the time loop: calccflquick on the initial winds,
            # calcksquick on kmh..khv as initialized (zero), getnewdt(doinit)
            b0 = self.base
            self.adapt.dbldt = self.dt
            self.adapt.cflmax = float(
                calccflquick(
                    bk,
                    g,
                    cfg,
                    dt=self.dt,
                    uh=b0["uh"],
                    vh=b0["vh"],
                    mh=b0["mh"],
                    ua=st["ua"],
                    va=st["va"],
                    wa=st["wa"],
                )
            )
            self.adapt.ksmax = 0.0
            self.adapt.getnewdt(self.mtime, self.dt, True)
            self._apply_new_dt()
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
        # cm1setup=0 means no subgrid turbulence (idoles=.false.)
        self.les = int(cfg.get("sgsmodel", 0)) == 1 and int(cfg.get("cm1setup", 1)) in (
            1,
            4,
        )
        self.terrain = bool(cfg.get("terrain_flag", False))
        self.moist = int(cfg.imoist) == 1
        if self.les:
            from .turb import turb_constants

            s["tkea"], s["tke3d"] = st["tkea"], st["tke3d"]
            tc = turb_constants(bk, g, cfg)
            self.base["cm0"] = tc["cm0"]
            for k in ("cme", "csm", "ce1", "ce2"):
                s[k] = tc[k]
        self.sfc = self._init_surface(s)
        self.rad = self._init_radiation(s)
        nrain = 2 if int(cfg.imove) == 1 else 1
        s["rain"] = bk.zeros((nrain,) + tuple(g.s_shape[1:]))
        s["prate"] = bk.zeros(g.s_shape[1:])
        s["qbudget"] = bk.zeros((10,))
        s["asq"] = bk.zeros((numq,))
        self.state = s
        if self.sfc:
            s.update(self._sfclayer(s, self.base))
        if self.rad:
            # cm1.F startup: radiation on the initial state (dorad = .true.)
            s.update(self._radiation(s, self.base, **self._solar(0.0)))
            self._advance_radtim(0.0)
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
    def _init_surface(self, s):
        """sfcmodel>=1: param.F forces the semi-slip lower boundary (bbc=3);
        init_surface.F sets tsk, albd, emiss, xland and z0 (state ``znt``).
        Returns True when the sgsmodel=0 surface layer (surface drag) runs."""
        cfg, bk, g = self.cfg, self.bk, self.grid
        if int(cfg.get("sfcmodel", 0)) < 1:
            return False
        from .init_surface import init_surface

        cfg.bbc = 3
        sf = init_surface(bk, g, cfg)
        s["znt"] = sf.pop("znt")
        self.base.update(sf)
        if self.les:
            raise NotImplementedError(
                "sfcmodel>=1 (bbc=3) with sgsmodel=1 is not ported"
            )
        return True

    def _sfclayer(self, s, b):
        """cm1.F ``sfc_and_turb`` for sgsmodel=0 with bbc=3 on the state
        ``s``: u*, z0 and the surface stress for the next solve1."""
        from .turb import sfc_and_turb_nosgs

        return sfc_and_turb_nosgs(
            self.bk,
            self.grid,
            self.cfg,
            ua=s["ua"],
            va=s["va"],
            u0=b["u0"],
            v0=b["v0"],
            rf=s["rf"],
            zh=b["zh"],
            xland=b["xland"],
            znt=s["znt"],
        )

    def _init_radiation(self, s):
        """radopt=2: surface parameters (init_surface.F), the RRTMG layer
        count (init_physics.F p_top -> rrtmg_lwinit) and the radiation
        state.  Returns False when radiation is off."""
        cfg, bk, g = self.cfg, self.bk, self.grid
        radopt = int(cfg.get("radopt", 0))
        if radopt == 0:
            return False
        if radopt != 2:
            raise NotImplementedError(f"radopt={radopt}: only radopt=2 is ported")
        from . import module_ra_rrtmg_lw as lw
        from . import radiation_driver as rd

        rd._check(cfg)
        if not self.sfc:
            raise NotImplementedError("radopt=2 needs a surface model (sfcmodel>=1)")
        self.nlayers = lw.nlayers_lw(g.nk, rd.p_top(self.base, g.nk))
        # radiation_driver: dtrad = max(dtrad, dt)
        self.dtrad = max(float(cfg.get("dtrad", 300.0)), self.dt)
        self.radtim = 0.0  # param.F
        s["swtenc"] = bk.zeros(g.s_shape)
        return True

    def _solar(self, mtime):
        from .radiation_driver import solar

        r = solar(self.cfg, mtime, self.dtrad)
        return {"coszen": r["coszen"], "solcon": r["solcon"]}

    def _dorad(self):
        """cm1.F: radiation at the end of this step?"""
        dt = self.dt
        return self.rad and (self.mtime + dt) >= (self.radtim - 0.1 * dt)

    def _advance_radtim(self, mtime):
        while True:
            self.radtim += self.dtrad
            if self.radtim > mtime:
                break

    def _radiation(self, s, b, *, coszen, solcon):
        """cm1.F: ``radiation_driver`` on the state ``s`` (radopt=2)."""
        from .radiation_driver import radiation_driver

        cg = {k: b[k] for k in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")}
        return radiation_driver(
            self.bk,
            self.grid,
            self.cfg,
            coszen=coszen,
            solcon=solcon,
            nlayers=self.nlayers,
            c1=b["c1"],
            c2=b["c2"],
            rmh=b["rmh"],
            pi0=b["pi0"],
            th0=b["th0"],
            prs=s["prs"],
            ppi=s["ppi"],
            tha=s["tha"],
            qa=s["qa"],
            tsk=b["tsk"],
            albd=b["albd"],
            emiss=b["emiss"],
            xland=b["xland"],
            xice=b["xice"],
            xsnow=b["xsnow"],
            swtenc=s["swtenc"],
            cg=cg,
        )

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
            terrain=self._turb_terrain(b),
            **{k: s.get(k) for k in keys},
        )
        for k in keys + ("tkea", "tke3d", "cme", "csm", "ce1", "ce2"):
            s[k] = r[k]

    # ------------------------------------------------------------------
    def step(self, hooks=None):
        """Advance one time step (dt = dtl).  With ``jit=True`` (and no
        ``hooks``) the compiled step is used (with radiation: one compiled
        variant for steps with and one for steps without a radiation call)."""
        a = self.adapt
        if a is not None:
            # cm1.F after sfc_and_turb (which ran at the end of the previous
            # step here): getnewdt (not in the startup pass) with the flags of
            # the step just taken, then statpack / writeout / restart_write
            # advance their timers, then the flags of this step
            if self.nstep > 0:
                a.getnewdt(self.mtime, self.dt, False)
                self._apply_new_dt()
            a.advance_timers(self.mtime)
            a.set_flags(self.mtime)
        mtime = self.mtime + self.dt
        dorad = self._dorad()
        sol = self._solar(mtime) if dorad else {"coszen": 0.0, "solcon": 0.0}
        if self.jit and self._whole_step and hooks is None:
            fn = self._compiled_step(dorad)
            barr = {k: v for k, v in self.base.items() if is_array(v)}
            extra = (self.bk.scalar(self.dt),) if a is not None else ()
            self.state = fn(
                self.state,
                barr,
                self.dtu0,
                self.dtv0,
                self.bk.scalar(mtime),
                self.bk.scalar(sol["coszen"]),
                self.bk.scalar(sol["solcon"]),
                *extra,
            )
        else:
            self.state = self._advance(
                self.state,
                self.base,
                self.dtu0,
                self.dtv0,
                mtime,
                hooks=hooks,
                dorad=dorad,
                **sol,
            )
        self.state = self.bk.materialize(self.state)
        if a is not None:
            # solve2's calccflquick (last RK stage) and, with LES, turb.F's
            # calcksquick at the end of sfc_and_turb (both at this step's dt)
            a.cflmax = float(self.state.pop("cflmax"))
            ks = self.state.pop("ksmax", None)
            if ks is not None:
                a.ksmax = float(ks)
        self.nstep += 1
        self.mtime = mtime
        if dorad:
            self._advance_radtim(mtime)
        return self

    def _compiled_step(self, dorad=False):
        """The whole time step as one compiled function
        ``(state, base_arrays, dtu0, dtv0, mtime, coszen, solcon) -> state``
        (Backend.compile: jax.jit, torch.compile, or the plain function
        elsewhere); ``dorad`` selects the variant with a radiation call."""
        key = ("step", bool(dorad)) if dorad else "step"
        if self.adapt is not None:
            # adapt_dt: dt is a traced argument (a new dt is a new input), but
            # nsound fixes the acoustic loop's length: one compiled variant
            # per distinct nsound (a recompile when getnewdt changes it)
            key = ("step", bool(dorad), "nsound", int(self.cfg.nsound))
        if key not in self._jit_cache:
            bstatic = {k: v for k, v in self.base.items() if not is_array(v)}

            def fn(s, barr, dtu0, dtv0, mtime, coszen, solcon, dt=None):
                return self._advance(
                    s,
                    {**bstatic, **barr},
                    dtu0,
                    dtv0,
                    mtime,
                    dorad=dorad,
                    coszen=coszen,
                    solcon=solcon,
                    dt=dt,
                )

            self._jit_cache[key] = self.bk.compile(fn, **self.jit_options)
        return self._jit_cache[key]

    def _advance(
        self,
        s,
        b,
        dtu0,
        dtv0,
        mtime,
        hooks=None,
        dorad=False,
        coszen=0.0,
        solcon=0.0,
        dt=None,
    ):
        """Pure time step: returns the state dict after one step from ``s``
        (``b`` = base dict, ``mtime`` = model time at the end of the step,
        Python float or 0-d array; ``dorad``: call radiation at the end of
        the step with cos(zenith) ``coszen`` and solar constant ``solcon``).  On numpy/torch arrays of ``s`` may be
        updated in place, as before."""
        bk, g, cfg = self.bk, self.grid, self.cfg
        s = dict(s)
        dt = self.dt if dt is None else dt
        tk = {}
        if self.terrain:
            tk = {k: b[k] for k in TERRAIN_KEYS}
            tk["cg"] = {k: b[k] for k in CG_KEYS}

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
            turb=self._turb_inputs(s, b) if self.les else self._sfc_inputs(s, b),
            thraten=s.get("thraten"),
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
            **tk,
        )
        if hooks is not None:
            hooks("post_solve2", 0, r2)
        if self.adapt is not None:
            # solve2.F: calccflquick after the last RK stage's acoustic steps
            s["cflmax"] = calccflquick(
                bk,
                g,
                cfg,
                dt=dt,
                uh=b["uh"],
                vh=b["vh"],
                mh=b["mh"],
                ua=r2["u3d"],
                va=r2["v3d"],
                wa=r2["w3d"],
            )

        if not self.moist:  # cm1.F: microphysics only with imoist=1
            mp = {k: r2[k] for k in ("pp3d", "th3d", "rho", "prs")}
            mp.update({k: s[k] for k in ("rain", "prate", "qbudget", "asq")})
            mp["q3d"] = r2["q3d"]
        else:
            mp = self._mp(s, b, r2, dt=dt)
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
            **{k: b.get(k) for k in ("gz", "dzdx", "dzdy")},
        )

        s.update(
            u3d=r2["u3d"],
            v3d=r2["v3d"],
            w3d=r3["w3d"],
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
        if dorad:
            s.update(self._radiation(s, b, coszen=coszen, solcon=solcon))
            if hooks is not None:
                hooks("post_radiation", 0, s)
        if self.sfc:
            s.update(self._sfclayer(s, b))
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
            if self.adapt is not None:
                # turb.F: calcksquick at the end of sfc_and_turb
                s["ksmax"] = calcksquick(
                    bk,
                    g,
                    cfg,
                    dt=dt,
                    uh=b["uh"],
                    vh=b["vh"],
                    kmh=s["kmh"],
                    kmv=s["kmv"],
                    khh=s["khh"],
                )
        return s

    def _apply_new_dt(self):
        """misclibs.F getnewdt: ``dt = dbldt; call dtsmall(dt, dbldt)``
        (nsound for the new dt; capped at 192, shortening dt)."""
        cfg = self.cfg
        ds = dtsmall(
            dt=self.adapt.dbldt,
            dbldt=self.adapt.dbldt,
            psolver=int(cfg.psolver),
            nx=int(cfg.nx),
            ny=int(cfg.ny),
            min_dx=float(cfg.dx),
            min_dy=float(cfg.dy),
            min_dz=self.min_dz,
            csound=float(cfg.csound),
            adapt_dt=1,
        )
        cfg.nsound = ds["nsound"]
        self.adapt.dbldt = float(ds["dbldt"])
        self.dt = float(ds["dt"])

    def _mp(self, s, b, r2, dt=None):
        """cm1.F: ``mp_driver`` (imoist=1)."""
        bk, g, cfg = self.bk, self.grid, self.cfg
        xp = bk.xp
        dt = self.dt if dt is None else dt
        return mp_driver(
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

    def _sfc_inputs(self, s, b):
        if not self.sfc:
            return None
        d = {k: b[k] for k in ("uh", "uf", "vh", "vf", "mh", "mf", "zf")}
        for k in ("rf", "ust", "znt", "t13", "t23"):
            d[k] = s[k]
        return d

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
        d["terrain"] = self._turb_terrain(b)
        return d

    def _turb_terrain(self, b):
        """The flat terrain-metrics dict cm1x.turb / cm1x.turbtend take
        (None without terrain_flag)."""
        if not self.terrain:
            return None
        return {k: b[k] for k in TERRAIN_KEYS + CG_KEYS}

    def run(self, nsteps):
        """Advance ``nsteps`` time steps."""
        for _ in range(int(nsteps)):
            self.step()
        return self

    # ------------------------------------------------------------------
    def fields(self):
        """Interior fields as numpy arrays, laid out like CM1's netCDF output
        (u: [nk, nj, ni+1], v: [nk, nj+1, ni], w, and with sgsmodel=1 tke, kmh,
        kmv, khh, khv: [nk+1, nj, ni], others [nk, nj, ni]; with radopt=2
        also RAD3D [nk, nj, ni] and the radiative fluxes RAD2D [nj, ni] of
        the last radiation call).  Definitions
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
        for name, idx in (("qv", cfg.nqv), ("qc", cfg.nqc), ("qr", cfg.nqr)):
            if idx:
                out[name] = s["qa"][idx - 1][K, J, I]
        if self.les:  # w-level fields [nk+1, nj, ni]
            out["tke"] = s["tkea"][g.KF, J, I]
            for name in ("kmh", "kmv", "khh", "khv"):
                out[name] = s[name][g.KF, J, I]
        if self.rad:  # output_radten=1 [nk, nj, ni]; output_sfcparams=1 [nj, ni]
            for name in RAD3D:
                out[name] = s[name][K, J, I]
            for name in RAD2D:
                out[name] = s[name]
        return {k: bk.to_numpy(v) for k, v in out.items()}
