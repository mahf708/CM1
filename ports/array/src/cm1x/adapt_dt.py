"""Adaptive time step (``adapt_dt = 1``): misclibs.F ``calccflquick``,
``calcksquick`` and ``getnewdt`` and cm1.F's output-time bookkeeping
(``taptim``, ``stattim``, ``rsttim`` and the ``dowriteout``/``dostat``/
``dorestart`` flags of each step), which ``getnewdt`` reads to ramp the time
step down ahead of output, stat and restart times.  ``misc.dtsmall`` gives
nsound for the new dt (capping it at 192 and shortening dt).

The CFL numbers are array reductions (jit-safe); ``AdaptDt`` runs on the host
between steps on Python floats, like CM1's scalar code (double precision).
"""

from __future__ import annotations

import dataclasses

__all__ = ["calccflquick", "calcksquick", "AdaptDt"]


def calccflquick(bk, g, cfg, *, dt, uh, vh, mh, ua, va, wa):
    """misclibs.F calccflquick: max over the interior of the advective
    Courant number of (ua, va, wa) at ``dt`` (0-d array; 0 if all are 0)."""
    xp = bk.xp
    n, nk, ni, nj = g.ngxy, g.nk, g.ni, g.nj
    K, J, I = g.K, g.J, g.I
    Ip = slice(n + 1, n + ni + 1)
    Jp = slice(n + 1, n + nj + 1)
    Kp = slice(2, nk + 2)
    dtdx = 0.5 * dt * g.rdx
    dtdy = 0.5 * dt * g.rdy
    dtdz = 0.5 * dt * g.rdz
    u1 = uh[I][None, None, :]
    v1 = vh[J][None, :, None]
    a = (ua[K, J, I] + ua[K, J, Ip]) * dtdx * u1
    b = (va[K, J, I] + va[K, Jp, I]) * dtdy * v1
    c = (wa[K, J, I] + wa[Kp, J, I]) * dtdz * mh[K, J, I]
    nx, ny = int(cfg.nx), int(cfg.ny)
    if nx > 1 and ny > 1:
        wsp = xp.sqrt((a * a + b * b) + c * c)
    elif nx > 1:
        wsp = xp.sqrt(a * a + c * c)
    elif int(cfg.get("axisymm", 0)) == 0 and ny > 1:
        wsp = xp.sqrt(b * b + c * c)
    else:
        return bk.asarray(0.0)
    return xp.maximum(xp.max(wsp), bk.asarray(0.0))


def calcksquick(bk, g, cfg, *, dt, uh, vh, kmh, kmv, khh):
    """misclibs.F calcksquick: max of the horizontal diffusion numbers of
    kmh, khh and kmv on k = 2..nk (0-d array)."""
    xp = bk.xp
    nk = g.nk
    J, I = g.J, g.I
    K2 = slice(2, nk + 1)
    dtdx = dt * g.rdx * g.rdx
    dtdy = dt * g.rdy * g.rdy
    u1 = uh[I][None, None, :]
    v1 = vh[J][None, :, None]
    nx, ny = int(cfg.nx), int(cfg.ny)
    tems = []
    for k3 in (kmh, khh, kmv):
        kk = k3[K2, J, I]
        a = kk * dtdx * u1 * u1
        if nx > 1 and ny > 1:
            b = kk * dtdy * v1 * v1
            tems.append(xp.sqrt(a * a + b * b))
        elif nx > 1:
            tems.append(a)
        else:
            raise NotImplementedError("calcksquick with nx = 1 (misclibs.F: stop 1112)")
    m = xp.maximum(xp.maximum(xp.max(tems[0]), xp.max(tems[1])), xp.max(tems[2]))
    return xp.maximum(m, bk.asarray(0.0))


@dataclasses.dataclass
class AdaptDt:
    """cm1.F / misclibs.F time-step state of adapt_dt = 1."""

    cm1setup: int = 1
    nx: int = 0
    ny: int = 0
    dtl: float = 0.0
    tapfrq: float = 0.0
    statfrq: float = 0.0
    rstfrq: float = 0.0
    dbldt: float = 0.0
    cflmax: float = 0.0
    ksmax: float = 0.0
    taptim: float = 0.0
    stattim: float = 0.0
    rsttim: float = 0.0
    # flags of the last pass of the time loop (startup pass: write and stats)
    dowriteout: bool = True
    dostat: bool = True
    dorestart: bool = False

    @classmethod
    def from_config(cls, cfg):
        """Options of adapt_dt = 1; raises NotImplementedError for the
        combinations whose timing is not ported."""

        def unsupported(what):
            raise NotImplementedError(f"adapt_dt=1 with {what} is not ported")

        if int(cfg.get("psolver", 3)) == 1:
            unsupported("psolver=1 (param.F stops)")
        rstfrq = float(cfg.get("rstfrq", 0.0))
        if abs(rstfrq) < 1.0e-10:
            unsupported("rstfrq=0 (write a restart and stop)")
        if int(cfg.get("radopt", 0)) != 0:
            unsupported("radiation (radtim/dtrad timing)")
        for flag in ("dodomaindiag", "doazimavg", "dohifrq"):
            if cfg.get(flag, False):
                unsupported(flag)
        if int(cfg.get("iprcl", 0)) != 0:
            unsupported("iprcl=1")
        cm1setup = int(cfg.get("cm1setup", 1))
        sgs = int(cfg.get("sgsmodel", 0))
        if cm1setup >= 1 and not (
            (sgs == 1 and cm1setup in (1, 4))
            or (cm1setup == 1 and sgs == 0 and int(cfg.get("ipbl", 0)) == 0)
        ):
            unsupported(f"cm1setup={cm1setup}, sgsmodel={sgs} (calcksquick on PBL km)")
        if int(cfg.get("ptype", 0)) in (2, 4):
            unsupported("ptype=2, 4 (consat2/lfoice_init(dt))")
        return cls(
            cm1setup=int(cfg.get("cm1setup", 1)),
            nx=int(cfg.nx),
            ny=int(cfg.ny),
            dtl=float(cfg.dtl),
            tapfrq=float(cfg.get("tapfrq", 0.0)),
            statfrq=float(cfg.get("statfrq", 0.0)),
            rstfrq=rstfrq,
            dbldt=float(cfg.dtl),
            taptim=0.0,
            stattim=0.0,
            rsttim=rstfrq,  # param.F
        )

    def getnewdt(self, mtime, dt, doinit):
        """misclibs.F getnewdt up to ``dt = dbldt`` (the caller then calls
        dtsmall); ``dt`` is the current (last) time step."""
        cfl_limit = 1.00  # maximum CFL allowed (actually a "target" value)
        max_change = 0.10  # maximum (percentage) change in timestep
        ks_limit = 0.0
        if self.cm1setup >= 1:
            ks_limit = 0.18 / 2.0 if (self.nx > 3 and self.ny > 3) else 0.18
        self.cflmax = max(self.cflmax, 1.0e-10)
        if self.cflmax > cfl_limit:
            self.dbldt = self.dbldt * (cfl_limit / self.cflmax)
        else:
            self.dbldt = self.dbldt * min(1.0 + max_change, cfl_limit / self.cflmax)
        if self.cm1setup >= 1 and (self.ksmax * self.dbldt / dt) > ks_limit:
            self.dbldt = min(self.dbldt, dt * ks_limit / max(1.0e-10, self.ksmax))
        # don't allow dt to exceed twice initial timestep
        self.dbldt = min(self.dbldt, 2.0 * self.dtl)
        if not doinit:
            for tim, frq, flag in (
                (self.taptim, self.tapfrq, self.dowriteout),
                (self.rsttim, self.rstfrq, self.dorestart),
                (self.stattim, self.statfrq, self.dostat),
            ):
                if not tim > 0.0:
                    continue
                tout = ((tim - mtime) + frq) if flag else (tim - mtime)
                d = self.dbldt
                if 2.00001 * d < tout <= 3.0 * d:
                    self.dbldt = tout / 3.0
                elif 1.00001 * d < tout <= 2.0 * d:
                    self.dbldt = tout / 2.0
                elif tout <= d:
                    self.dbldt = tout

    def advance_timers(self, mtime):
        """cm1.F after statpack / writeout / restart_write."""
        if self.dostat and self.statfrq > 0.0:
            self.stattim = self.stattim + self.statfrq
            while not self.stattim > mtime:
                self.stattim = self.stattim + self.statfrq
        if self.dowriteout and self.tapfrq > 0.0:
            self.taptim = self.taptim + self.tapfrq
            while not self.taptim > mtime:
                self.taptim = self.taptim + self.tapfrq
        if self.dorestart and self.rstfrq > 0.0:
            self.rsttim = self.rsttim + self.rstfrq
            while not self.rsttim > mtime:
                self.rsttim = self.rsttim + self.rstfrq

    def set_flags(self, mtime):
        """cm1.F at the top of a step: this step's output flags."""
        d = self.dbldt
        self.dowriteout = (mtime + d) >= (self.taptim - 0.1 * d) or self.tapfrq < 0.0
        self.dostat = (mtime + d) >= (self.stattim - 0.1 * d) or self.statfrq < 0.0
        self.dorestart = (mtime + d) >= (self.rsttim - 0.1 * d) and self.rstfrq > 0.0
