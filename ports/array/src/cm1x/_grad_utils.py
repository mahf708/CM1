"""Helpers for differentiating cm1x model rollouts (jax / torch autodiff).

``Model.step`` works on ``model.state`` (a dict of arrays) and reads only
fixed things (grid, base state, cfg) besides it, so a rollout is a pure
function of the initial state.  These helpers build that initial state from a
(differentiable) potential-temperature perturbation and run a rollout on a
private copy of a model, leaving the template model untouched:

    from cm1x.model import Model
    from cm1x._grad_utils import initial_state, rollout, small_namelist

    nml = small_namelist(".../namelist.input", "/tmp/nml", nx=12, ny=12, nz=16)
    m = Model(nml, backend="jax")
    tha0 = m.state["tha"][m.grid.K, m.grid.J, m.grid.I]      # warm bubble

    def loss(amp):                       # d loss / d bubble amplitude
        s = rollout(m, initial_state(m, amp * tha0), nsteps=3)
        return (s["wa"] ** 2).sum()

    jax.grad(loss)(1.0)

``initial_state`` redoes init3d.F's ``getset`` (halos, prs/rho/rr/rf/ppx,
mass1) and, with sgsmodel=1, the t=0 ``sfc_and_turb`` call, exactly as
``Model.__init__`` does, so ``initial_state(m, tha0)`` reproduces
``m.state`` (to round-off).

Torch: the inputs must be leaf tensors with ``requires_grad=True`` (or
functions of them); ``Backend.set``/``add`` then switch to out-of-place
updates automatically (see cm1x.backend), so autograd sees a functional
graph.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

from .init3d import getset

__all__ = ["small_namelist", "initial_state", "clone", "rollout"]

# entries of Model.state recomputed by initial_state (getset outputs)
_GETSET_KEYS = (
    "ua",
    "va",
    "wa",
    "ppi",
    "tha",
    "u3d",
    "v3d",
    "w3d",
    "pp3d",
    "th3d",
    "rho",
    "prs",
    "rr",
    "rf",
    "ppx",
    "mass1",
)
# outputs of sfc_and_turb (Model._sfc_and_turb) that a fresh t=0 call starts
# from None
_TURB_KEYS = (
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


def small_namelist(src, dst, **overrides):
    """Copy namelist ``src`` to ``dst`` with ``name = value`` entries replaced
    (e.g. ``nx=12, ny=12, nz=16``).  Every name must already be present in
    ``src``.  Returns ``dst`` as a ``Path``."""
    text = Path(src).read_text()
    for name, val in overrides.items():
        if isinstance(val, bool):
            sval = ".true." if val else ".false."
        else:
            sval = repr(val) if isinstance(val, float) else str(val)
        pat = re.compile(rf"^(\s*{name}\s*=\s*)[^,\n]*", re.I | re.M)
        text, n = pat.subn(lambda m, s=sval: m.group(1) + s, text, count=1)
        if n != 1:
            raise KeyError(f"{name!r} not found in {src}")
    dst = Path(dst)
    dst.write_text(text)
    return dst


def clone(m):
    """Shallow copy of model ``m`` with its own state dict (and qa/q3d lists),
    so stepping the copy never rebinds ``m.state`` entries."""
    c = copy.copy(m)
    c.state = {k: (list(v) if isinstance(v, list) else v) for k, v in m.state.items()}
    c.base = dict(m.base)
    return c


def initial_state(m, tha, dqv=None):
    """Model state of ``m`` at t=0 with the interior potential-temperature
    perturbation ``tha`` (``[nk, nj, ni]``, an array of ``m.bk``) instead of
    the namelist's one, and optionally ``dqv`` (same shape) added to the
    interior water vapour.  Velocities, pressure and the rest of the moisture
    are ``m``'s initial values; ``m`` must be at step 0.  Differentiable
    w.r.t. ``tha`` and ``dqv``."""
    if m.nstep != 0:
        raise ValueError("initial_state needs a model at step 0")
    bk, g, cfg, b = m.bk, m.grid, m.cfg, m.base
    xp = bk.xp
    s0 = m.state
    tha_full = bk.set(bk.zeros(g.s_shape), (g.K, g.J, g.I), tha)
    qa = list(s0["qa"])
    if dqv is not None:
        nqv = int(cfg.nqv) - 1
        qa[nqv] = bk.add(bk.copy(qa[nqv]), (g.K, g.J, g.I), dqv)
    # same base-state / grid inputs Model.__init__ -> initialize -> getset uses
    cg = {k: b[k] for k in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")}
    gs = getset(
        bk,
        cfg,
        ua=s0["ua"],
        va=s0["va"],
        wa=s0["wa"],
        ppi=s0["ppi"],
        tha=tha_full,
        qa=xp.stack(qa, axis=0) if qa else bk.zeros((0,) + tuple(g.s_shape)),
        pi0=b["pi0"],
        th0=b["th0"],
        rho0=b["rho0"],
        rf0=b["rf0"],
        prs0=b["prs0"],
        c1=b["c1"],
        c2=b["c2"],
        rmh=b["rmh"],
        ruh=b["ruh"],
        rvh=b["rvh"],
        cg=cg,
        g=g,
    )
    s = dict(s0)
    for k in _GETSET_KEYS:
        s[k] = gs[k]
    numq = len(s0["qa"])
    s["qa"] = [gs["qa"][n] for n in range(numq)]
    s["q3d"] = [gs["q3d"][n] for n in range(numq)]
    if m.les:
        from .turb import turb_constants

        s["tkea"], s["tke3d"] = gs["tkea"], gs["tke3d"]
        tc = turb_constants(bk, g, cfg)
        for k in ("cme", "csm", "ce1", "ce2"):
            s[k] = tc[k]
        for k in _TURB_KEYS:
            s.pop(k, None)
        c = clone(m)
        c.state = s
        c._sfc_and_turb(
            rtime=0.0,
            u3d=s["u3d"],
            v3d=s["v3d"],
            w3d=s["w3d"],
            pp3d=s["pp3d"],
            th3d=s["th3d"],
            q3d=s["q3d"],
        )
        s = c.state
    return s


def rollout(m, state, nsteps):
    """Run ``nsteps`` steps of ``m`` from ``state`` on a private copy of ``m``
    (``m`` itself is not advanced).  Returns the final state dict."""
    if getattr(m, "jit", False) and hasattr(m, "_compiled_step"):
        m._compiled_step()  # build it on m so that every copy shares it
    c = clone(m)
    c.state = {k: (list(v) if isinstance(v, list) else v) for k, v in state.items()}
    for _ in range(int(nsteps)):
        c.step()
    return c.state
