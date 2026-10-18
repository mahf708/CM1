"""Small helpers needed by the time-step glue (ported from misclibs.F, param.F,
init3d.F) that no other module owns.

Only the branches active for the reference case are ported (Cartesian, no
terrain, periodic, single domain).
"""

from __future__ import annotations

import math

from . import constants as c

# CM1 sets these in cm1.F: the single-precision defaults below, or 1e-60 /
# 1e-12 / 1e-24 with -DDP. Callers read the runtime values from the config
# (namelist.DP_DEFAULTS with fortran_dp=True) and fall back to these.
SMEPS = 1.0e-30
TSMALL = 0.0001
QSMALL = 1.0e-14
NRKMAX = 3

# 2nd-order extrapolation coefficients (cm1.F / param.F, uniform dz)
CGS1, CGS2, CGS3 = 1.875, -1.25, 0.375
CGT1, CGT2, CGT3 = 1.875, -1.25, 0.375


def dtsmall(
    *,
    dt,
    dbldt,
    psolver=3,
    nx,
    ny,
    min_dx,
    min_dy,
    min_dz=None,
    csound=300.0,
    adapt_dt=0,
):
    """misclibs.F dtsmall: number of acoustic small steps.  Returns dict with
    nsound (and dt, dbldt, unchanged unless adapt_dt clamps)."""
    max_nsound = 192
    if psolver in (1, 4, 5):
        return {"nsound": 1, "dt": dt, "dbldt": dbldt}
    if psolver == 2:
        if ny == 1:
            dtsm = 0.60 * min(min_dx, min_dz) / 360.0
        elif nx == 1:
            dtsm = 0.60 * min(min_dy, min_dz) / 360.0
        else:
            dtsm = 0.50 * min(min_dx, min_dy, min_dz) / 360.0
    elif psolver == 3:
        if ny == 1:
            dtsm = 0.60 * min_dx / 360.0
        elif nx == 1:
            dtsm = 0.60 * min_dy / 360.0
        else:
            dtsm = 0.60 * min(min_dx, min_dy) / 360.0
    else:
        if ny == 1:
            dtsm = 0.60 * min(min_dx, min_dz) / csound
        elif nx == 1:
            dtsm = 0.60 * min(min_dy, min_dz) / csound
        else:
            dtsm = 0.50 * min(min_dx, min_dy, min_dz) / csound
    q = dbldt / dtsm
    # Fortran nint: round half away from zero
    nsound = max(int(math.floor(q + 0.5)), 4)
    if nsound % 2 != 0:
        nsound += 1
    if dbldt / float(nsound) > dtsm:
        nsound += 2
    if nsound > max_nsound:
        if adapt_dt == 1:
            nsound = max_nsound
            dbldt = nsound * dtsm
            dt = dbldt
        else:
            raise ValueError(f"nsound={nsound} exceeds {max_nsound}; reduce dtl")
    return {"nsound": nsound, "dt": dt, "dbldt": dbldt}


def getdiv(bk, g, *, uh, vh, mh, u, v, w, div=None, smeps=SMEPS):
    """misclibs.F getdiv (Cartesian, no terrain): 3-D divergence of (u,v,w)
    at interior scalar points.  Returns {'div': div}."""
    xp = bk.xp
    K, J, I = g.K, g.J, g.I
    n, nz = g.ngxy, g.ngz
    ni, nj, nk = g.ni, g.nj, g.nk
    if div is None:
        div = bk.zeros(g.s_shape)
    du = (u[K, J, n + 1 : n + ni + 1] - u[K, J, n : n + ni]) * g.rdx * uh[:, :, I]
    dv = (v[K, n + 1 : n + nj + 1, I] - v[K, n : n + nj, I]) * g.rdy * vh[:, J, :]
    dw = (
        (w[nz + 1 : nz + nk + 1, J, I] - w[nz : nz + nk, J, I])
        * g.rdz
        * mh[K, n : n + 1, n : n + 1]
    )
    d = (du + dv) + dw
    d = xp.where(xp.abs(d) < smeps, xp.zeros_like(d), d)
    div = bk.set(div, (K, J, I), d)
    return {"div": div}


def rdamp_coeffs(bk, g, *, zd, irdamp=1):
    """param.F: vertical Rayleigh-damping coefficients tauh, taus (scalar
    levels) and tauf (w levels), uniform grid without terrain.  Returned as
    full 3-D arrays shaped like CM1 stores them."""
    xp = bk.xp
    zh = g.zh()
    zf = g.zf()
    ztop = zf[g.nk + g.ngz]
    th = [0.0] * len(zh)
    tf = [0.0] * len(zf)
    if irdamp >= 1 and zd < ztop:
        for k in range(g.ngz, g.ngz + g.nk):
            if zh[k] > zd:
                th[k] = 0.5 * (1.0 - math.cos(c.pi * (zh[k] - zd) / (ztop - zd)))
        for k in range(g.ngz, g.ngz + g.nk + 1):
            if zf[k] > zd:
                tf[k] = 0.5 * (1.0 - math.cos(c.pi * (zf[k] - zd) / (ztop - zd)))
    tauh = bk.asarray(th)[:, None, None] * xp.ones(g.s_shape, dtype=bk.dtype)
    tauf = bk.asarray(tf)[:, None, None] * xp.ones(g.w_shape, dtype=bk.dtype)
    return {"tauh": tauh, "taus": bk.copy(tauh), "tauf": tauf}


def prsrho(bk, g, *, pi0, th0, pp3d, th3d, qv=None, prs=None, rho=None):
    """prs and rho from (pi', theta', qv) at interior points (the formula
    repeated in init3d.F getset and solve2.F).  Returns {'prs','rho'}."""
    xp = bk.xp
    K, J, I = g.K, g.J, g.I
    if prs is None:
        prs = bk.zeros(g.s_shape)
    if rho is None:
        rho = bk.zeros(g.s_shape)
    pit = pi0[K, J, I] + pp3d[K, J, I]
    p = c.p00 * (pit**c.cpdrd)
    if qv is None:
        r = p / (c.rd * (th0[K, J, I] + th3d[K, J, I]) * pit)
    else:
        r = p / (
            (th0[K, J, I] + th3d[K, J, I])
            * pit
            * (c.rd + xp.maximum(qv[K, J, I], xp.zeros_like(p)) * c.rv)
        )
    prs = bk.set(prs, (K, J, I), p)
    rho = bk.set(rho, (K, J, I), r)
    return {"prs": prs, "rho": rho}


def level_sums(bk, x):
    """Per-level sums ``dumk(k) = sum_j sum_i x(i,j,k)`` of ``x[k, j, i]``
    accumulated in CM1's loop order (i fastest, then j), as init3d.F getset
    and solve2.F apmasscon do. ``xp.sum`` sums pairwise, which differs in the
    last bits; the domain-mass correction built from these sums is added to
    every pp3d point, and with open radiative boundaries (irbc=4) such
    last-bit differences grow by orders of magnitude within a few steps."""
    xp = bk.xp
    x = xp.reshape(x, (x.shape[0], -1))
    if bk.name == "jax":  # jnp.cumsum is not a sequential sum
        import jax

        acc, _ = jax.lax.scan(lambda a, v: (a + v, None), x[:, 0], x[:, 1:].T)
        return acc
    return xp.cumsum(x, axis=1)[:, -1]  # sequential on numpy / torch (CPU)


def qp_sum(bk, v):
    """sum_k v(k) of a 1-D float64 array accumulated in quad precision and
    rounded once to double (CM1's ``real(kind=qp) :: temq`` sums), emulated
    with an error-free (TwoSum) running compensation."""
    s, e = v[0], v[0] * 0.0
    for k in range(1, v.shape[0]):
        x = v[k]
        t = s + x
        bp = t - s
        e = e + ((s - (t - bp)) + (x - bp))
        s = t
    return s + e


def drymass(bk, g, *, rho, ruh, rvh, rmh):
    """Total dry-air mass sum_k sum_ij rho*ruh*rvh*rmh (init3d.F getset /
    solve2.F apmasscon); returns a 0-d array (summed level by level like CM1)."""
    K, J, I = g.K, g.J, g.I
    m = rho[K, J, I] * ruh[:, :, I] * rvh[:, J, :] * rmh[K, J, I]
    return qp_sum(bk, level_sums(bk, m))


def rhoextra(bk, g, *, rho, c1, c2, rr=None, rf=None):
    """solve3.F end / init3d.F: rr = 1/rho and rf (rho on w levels), on
    i,j = 0..ni+1 (Python: halo width 1) -- requires rho halos filled."""
    n, nz = g.ngxy, g.ngz
    ni, nj, nk = g.ni, g.nj, g.nk
    Jh = slice(n - 1, n + nj + 1)
    Ih = slice(n - 1, n + ni + 1)
    if rr is None:
        rr = bk.zeros(g.s_shape)
    if rf is None:
        rf = bk.zeros(g.s_shape)  # CM1 dimensions rf(ib:ie,jb:je,kb:ke)
    K = slice(nz, nz + nk)
    rr = bk.set(rr, (K, Jh, Ih), 1.0 / rho[K, Jh, Ih])
    K2 = slice(nz + 1, nz + nk)  # Fortran k=2..nk
    rf = bk.set(
        rf,
        (K2, Jh, Ih),
        c1[K2, Jh, Ih] * rho[nz : nz + nk - 1, Jh, Ih]
        + c2[K2, Jh, Ih] * rho[K2, Jh, Ih],
    )
    k1 = nz
    rf = bk.set(
        rf,
        (k1, Jh, Ih),
        CGS1 * rho[k1, Jh, Ih]
        + CGS2 * rho[k1 + 1, Jh, Ih]
        + CGS3 * rho[k1 + 2, Jh, Ih],
    )
    kt = nz + nk - 1
    rf = bk.set(
        rf,
        (nz + nk, Jh, Ih),
        CGT1 * rho[kt, Jh, Ih]
        + CGT2 * rho[kt - 1, Jh, Ih]
        + CGT3 * rho[kt - 2, Jh, Ih],
    )
    return {"rr": rr, "rf": rf}
