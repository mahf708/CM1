"""The Mellor-Yamada-Janjic PBL (ipbl=6), ported from src/module_bl_myjpbl.F
(``myjpbl`` with ``mixlen``, ``prodq2``, ``difcof``, ``vdifq``, ``vdifh``,
``vdifv``) and its CM1 glue in src/turb.F (``pbl_myj``: sfc_and_turb's
domyj block).

As CM1 runs it: STEPBL = 1, LOWLYR = 1 (LMH = KTE, PSFC = PINT(1)), QCI
present (flQI: theta_e, q, qc and qi are mixed, NSPEC = 4), QCS/QCR/QCG
unused, SNOW = SICE = 0 (the RLIVWV factor is unreachable), RHO and ZNT
unused.  Every column at once: 3-D inputs are ``[k, ...]`` arrays whose
index 0 is Fortran k = 1 (``pint`` has nk+1 levels).  The scheme counts
levels downward from the model top (K = KTE+1-k); the column work arrays
are Python lists indexed by that K (entry 0 unused), loops over K are
Python loops (vectorised over the columns), data-dependent levels (the PBL
top LPBL, LMXL) are per-column float arrays used through masks.
"""

from __future__ import annotations

from .sfc_myj import (
    _max,
    _min,
    a2,
    a3,
    a4,
    cp,
    elocp,
    ep_1,
    epsq2,
    g,
    p608,
    pq0sea,
    r_d,
    xlv,
)

capa = r_d / cp
vkarman = 0.4
eps1 = 1.0e-12
eps2 = 0.0
epsl = 0.32
epsru = 1.0e-7
epsrs = 1.0e-7
epstrb = 1.0e-24
fh = 1.01
alph = 0.30
beta = 1.0 / 273.0
el0max = 1000.0
el0min = 1.0
elfc = 0.23 * 0.5
a1 = 0.659888514560862645
a2x = 0.6574209922667784586
b1 = 11.87799326209552761
b2 = 7.226971804046074028
c1 = 0.000830955950095854396
elz0 = 0.0
esq = 5.0
btg = beta * g
rb1 = 1.0 / b1
adnh = 9.0 * a1 * a2x * a2x * (12.0 * a1 + 3.0 * b2) * btg * btg
adnm = 18.0 * a1 * a1 * a2x * (b2 - 3.0 * a2x) * btg
anmh = -9.0 * a1 * a2x * a2x * btg * btg
anmm = -3.0 * a1 * a2x * (3.0 * a2x + 3.0 * b2 * c1 + 18.0 * a1 * c1 - b2) * btg
bdnh = 3.0 * a2x * (7.0 * a1 + b2) * btg
bdnm = 6.0 * a1 * a1
beqh = a2x * b1 * btg + 3.0 * a2x * (7.0 * a1 + b2) * btg
beqm = -a1 * b1 * (1.0 - 3.0 * c1) + 6.0 * a1 * a1
bnmh = -a2x * btg
bnmm = a1 * (1.0 - 3.0 * c1)
bshh = 9.0 * a1 * a2x * a2x * btg
bshm = 18.0 * a1 * a1 * a2x * c1
bsmh = -3.0 * a1 * a2x * (3.0 * a2x + 3.0 * b2 * c1 + 12.0 * a1 * c1 - b2) * btg
cesh = a2x
cesm = a1 * (1.0 - 3.0 * c1)
cnv = ep_1 * g / btg
# free term in the equilibrium equation for (L/Q)**2
aeqh = (
    9.0 * a1 * a2x * a2x * b1 * btg * btg
    + 9.0 * a1 * a2x * a2x * (12.0 * a1 + 3.0 * b2) * btg * btg
)
aeqm = (
    3.0 * a1 * a2x * b1 * (3.0 * a2x + 3.0 * b2 * c1 + 18.0 * a1 * c1 - b2) * btg
    + 18.0 * a1 * a1 * a2x * (b2 - 3.0 * a2x) * btg
)
# forbidden turbulence area
requ = -aeqh / aeqm
epsgh = 1.0e-9
epsgm = requ * epsgh
# near isotropy for shear turbulence, ww/q2 lower limit
ubryl = (
    18.0 * requ * a1 * a1 * a2x * b2 * c1 * btg + 9.0 * a1 * a2x * a2x * b2 * btg * btg
) / (requ * adnm + adnh)
ubry = (1.0 + epsrs) * ubryl
ubry3 = 3.0 * ubry
aubh = 27.0 * a1 * a2x * a2x * b2 * btg * btg - adnh * ubry3
aubm = 54.0 * a1 * a1 * a2x * b2 * c1 * btg - adnm * ubry3
bubh = (9.0 * a1 * a2x + 3.0 * a2x * b2) * btg - bdnh * ubry3
bubm = 18.0 * a1 * a1 * c1 - bdnm * ubry3
cubr = 1.0 - ubry3
rcubr = 1.0 / cubr
# PRODQ2: B1**(2./3.), the constant expression gfortran folds (correctly
# rounded; glibc's pow gives the same double)
b1p23 = 5.205894744336633
NSPEC = 4


def _pick(xp, col, idx, lo, hi, like):
    """``col(idx)`` for a per-column level index ``idx`` (float array with
    integer values in [lo, hi])."""
    out = xp.zeros_like(like)
    for k in range(lo, hi + 1):
        out = xp.where(idx == float(k), col[k], out)
    return out


def _stk(xp, col, lo, hi):
    """Column list entries K = lo..hi stacked along a new leading axis."""
    return xp.stack(col[lo : hi + 1], axis=0)


def _unstk(a, lo, n):
    """A stacked array back to a column list (entries lo.. ; length n+1)."""
    out = [None] * (n + 1)
    for m in range(a.shape[0]):
        out[lo + m] = a[m, ...]
    return out


def _levels(bk, lo, hi, like):
    """Float level numbers K = lo..hi shaped to broadcast against a stacked
    ``like`` (levels along axis 0)."""
    k = bk.asarray([float(x) for x in range(lo, hi + 1)])
    return bk.xp.reshape(k, (hi - lo + 1,) + (1,) * (like.ndim - 1))


def mixlen(bk, *, lmh, u, v, t, the, q, cwm, q2, z, ct):
    """MIXLEN: the level 2.5 mixing length.  Column lists indexed by K
    (z: 1..lmh+1).  Returns gm, gh, el (K = 1..lmh-1), pblh, lpbl (float),
    lmxl (float), ct (reset to 0), mixht.  The level-wise parts are computed
    on all levels at once (stacked along K)."""
    xp = bk.xp
    sqrt = xp.sqrt
    like = z[lmh + 1]
    m = lmh - 1  # levels K = 1..lmh-1
    q2s = _stk(xp, q2, 1, m)
    kk = _levels(bk, 1, m, q2s)
    # the height of the PBL (first K from LMH-1 upward with q2 <= epsq2*fh)
    lpbl = xp.max(xp.where(q2s <= epsq2 * fh, kk, 1.0), axis=0)
    pblh = _pick(xp, z, lpbl + 1.0, 2, lmh, like) - z[lmh + 1]

    thes = _stk(xp, the, 1, lmh)
    dth = thes[:-1] - thes[1:]  # K = 1..lmh-1
    # the countergradient term at the top of the lowest unstable layer
    hit = (dth[:-1] > 0.0) & (dth[1:] <= 0.0)  # K = 1..lmh-2
    khit = xp.max(xp.where(hit, kk[:-1], 0.0), axis=0)
    dth = xp.where(kk == khit, dth + ct, dth)
    ct = xp.zeros_like(like)

    zs = _stk(xp, z, 1, lmh + 1)
    us, vs, ts = _stk(xp, u, 1, lmh), _stk(xp, v, 1, lmh), _stk(xp, t, 1, lmh)
    qs, cs = _stk(xp, q, 1, lmh), _stk(xp, cwm, 1, lmh)
    rdz = 2.0 / (zs[:-2] - zs[2:])
    du = us[:-1] - us[1:]
    dv = vs[:-1] - vs[1:]
    gml = (du * du + dv * dv) * rdz * rdz
    gm = _max(xp, gml, epsgm)
    tem = (ts[:-1] + ts[1:]) * 0.5
    thm = (thes[:-1] + thes[1:]) * 0.5
    a = thm * p608
    b = (elocp / tem - 1.0 - p608) * thm
    q0, q1_, c0, c1_ = qs[:-1], qs[1:], cs[:-1], cs[1:]
    ghl = (
        dth * ((q0 + q1_ + c0 + c1_) * (0.5 * p608) + 1.0)
        + (q0 - q1_ + c0 - c1_) * a
        + (c0 - c1_) * b
    ) * rdz
    gh = xp.where(xp.abs(ghl) <= epsgh, epsgh, ghl)

    # maximum mixing lengths and the level of the PBL top (the branch not
    # taken gets a harmless argument: no NaN in its values or gradients)
    gml = gm
    ghl = gh
    stable = ghl >= epsgh
    forb = stable & (gml / ghl <= requ)
    aubr = (aubm * gml + aubh * ghl) * ghl
    bubr = bubm * gml + bubh * ghl
    dst = xp.where(stable, bubr * bubr * 0.25 - aubr * cubr, 1.0)
    qol2st = (-0.5 * bubr + sqrt(dst)) * rcubr
    aden = (adnm * gml + adnh * ghl) * ghl
    bden = bdnm * gml + bdnh * ghl
    dun = xp.where(stable, 1.0, bden * bden * 0.25 - aden)
    qol2un = -0.5 * bden + sqrt(dun)
    eloq2x = xp.where(stable, 1.0 / qol2st, 1.0 / (qol2un + epsru))
    eloq2x = xp.where(forb, 1.0, eloq2x)
    elm = xp.where(forb, epsl, _max(xp, sqrt(eloq2x * q2s), epsl))
    lmxl = xp.max(xp.where(forb, kk, 0.0), axis=0)
    lmxl = xp.where(lmxl == 0.0, float(lmh), lmxl)
    lmxl = xp.where(elm[m - 1] == epsl, float(lmh), lmxl)
    # the height of the mixed layer
    mixht = _pick(xp, z, lmxl, 1, lmh, like) - z[lmh + 1]
    q1 = [None] * (lmh + 1)
    for k in range(1, lmh + 1):
        q1[k] = xp.where(float(k) >= lpbl, sqrt(q2[k]), 0.0)
    szq = 0.0
    sq = 0.0
    for k in range(1, lmh):
        qdzl = (q1[k] + q1[k + 1]) * (z[k + 1] - z[k + 2])
        szq = (z[k + 1] + z[k + 2] - z[lmh + 1] - z[lmh + 1]) * qdzl + szq
        sq = qdzl + sq
    # asymptotic l in the Blackadar formula
    el0 = _min(xp, alph * szq * 0.5 / sq, el0max)
    el0 = _max(xp, el0, el0min)
    # inside the PBL (K >= LPBL) / above it (K <= max(LPBL-1, 1))
    vkrmz = (zs[1:-1] - zs[-1]) * vkarman
    e_in = _min(xp, vkrmz / (vkrmz / el0 + 1.0), elm)
    e_up = _min(xp, (zs[:-2] - zs[2:]) * elfc, elm)
    el = xp.where(kk >= lpbl, e_in, e_up)
    rel = el / elm
    # smoothing, K = LPBL+1..LMH-2
    srel = _min(xp, ((rel[:-2] + rel[2:]) * 0.5 + rel[1:-1]) * 0.5, rel[1:-1])
    els = xp.where(kk[1:-1] >= lpbl + 1.0, _max(xp, srel * elm[1:-1], epsl), el[1:-1])
    el = xp.concat([el[:1], els, el[-1:]], axis=0)
    return {
        "gm": _unstk(gm, 1, lmh),
        "gh": _unstk(gh, 1, lmh),
        "el": _unstk(el, 1, lmh),
        "pblh": pblh,
        "lpbl": lpbl,
        "lmxl": lmxl,
        "ct": ct,
        "mixht": mixht,
    }


def prodq2(bk, *, lmh, dtturbl, ustar, gm, gh, el, q2):
    """PRODQ2: level 2.5 q2 production/dissipation (all levels K =
    1..lmh-1 at once).  Returns el, q2 (new lists)."""
    xp = bk.xp
    sqrt, exp = xp.sqrt, xp.exp
    m = lmh - 1
    gml = _stk(xp, gm, 1, m)
    ghl = _stk(xp, gh, 1, m)
    els = _stk(xp, el, 1, m)
    q2s = _stk(xp, q2, 1, m)
    # coefficients of the equilibrium equation
    aequ = (aeqm * gml + aeqh * ghl) * ghl
    bequ = beqm * gml + beqh * ghl
    # equilibrium solution for l/q
    pre = (gml + ghl * ghl <= epstrb) | ((ghl >= epsgh) & (gml / ghl <= requ))
    eqol2 = -0.5 * bequ + sqrt(xp.where(pre, 1.0, bequ * bequ * 0.25 - aequ))
    # is there production/dissipation?
    noturb = pre | (eqol2 <= eps2)
    eqol2 = xp.where(noturb, 1.0, eqol2)
    # turbulence: coefficients of the numerator, denominator
    anum = (anmm * gml + anmh * ghl) * ghl
    bnum = bnmm * gml + bnmh * ghl
    aden = (adnm * gml + adnh * ghl) * ghl
    bden = bdnm * gml + bdnh * ghl
    cden = 1.0
    # coefficients of the numerator of the linearized eq.
    arhs = -(anum * bden - bnum * aden) * 2.0
    brhs = -anum * 4.0
    crhs = -bnum * 2.0
    # initial value of l/q
    dloq1 = els / sqrt(q2s)
    # first iteration for l/q, rhs=0
    eloq21 = 1.0 / eqol2
    eloq11 = sqrt(eloq21)
    eloq31 = eloq21 * eloq11
    eloq41 = eloq21 * eloq21
    eloq51 = eloq21 * eloq31
    rden1 = 1.0 / (aden * eloq41 + bden * eloq21 + cden)
    rhsp1 = (arhs * eloq51 + brhs * eloq31 + crhs * eloq11) * rden1 * rden1
    # first-guess solution
    eloq12 = eloq11 + (dloq1 - eloq11) * exp(rhsp1 * dtturbl)
    eloq12 = _max(xp, eloq12, eps1)
    # second iteration for l/q
    eloq22 = eloq12 * eloq12
    eloq32 = eloq22 * eloq12
    eloq42 = eloq22 * eloq22
    eloq52 = eloq22 * eloq32
    rden2 = 1.0 / (aden * eloq42 + bden * eloq22 + cden)
    rhs2 = -(anum * eloq42 + bnum * eloq22) * rden2 + rb1
    rhsp2 = (arhs * eloq52 + brhs * eloq32 + crhs * eloq12) * rden2 * rden2
    rhst2 = rhs2 / rhsp2
    # corrected solution
    eloq13 = eloq12 - rhst2 + (rhst2 + dloq1 - eloq12) * exp(rhsp2 * dtturbl)
    eloq13 = _max(xp, eloq13, eps1)
    # two iterations is enough in most cases ...
    eloqn = eloq13
    q2t = _max(xp, els * els / (eloqn * eloqn), epsq2)
    ok = eloqn > eps1
    q2t = xp.where(ok, q2t, epsq2)
    elt = xp.where(ok & (q2t != epsq2), els, epsl)
    q2n = _unstk(xp.where(noturb, epsq2, q2t), 1, lmh)
    eln = _unstk(xp.where(noturb, epsl, elt), 1, lmh)
    eln[lmh] = el[lmh]
    # lower boundary condition for q2
    q2n[lmh] = _max(xp, b1p23 * ustar * ustar, epsq2)
    return {"el": eln, "q2": q2n}


def difcof(bk, *, lmh, gm, gh, el, q2, z):
    """DIFCOF: level 2.5 diffusion coefficients (all levels K = 1..lmh-1 at
    once).  Returns akm, akh lists."""
    xp = bk.xp
    m = lmh - 1
    ell = _stk(xp, el, 1, m)
    q2s = _stk(xp, q2, 1, m)
    gml = _stk(xp, gm, 1, m)
    ghl = _stk(xp, gh, 1, m)
    zs = _stk(xp, z, 1, lmh + 1)
    eloq2 = ell * ell / q2s
    eloq4 = eloq2 * eloq2
    aden = (adnm * gml + adnh * ghl) * ghl
    bden = bdnm * gml + bdnh * ghl
    cden = 1.0
    besm = bsmh * ghl
    besh = bshm * gml + bshh * ghl
    rden = 1.0 / (aden * eloq4 + bden * eloq2 + cden)
    esm = (besm * eloq2 + cesm) * rden
    esh = (besh * eloq2 + cesh) * rden
    rdz = 2.0 / (zs[:-2] - zs[2:])
    q1l = xp.sqrt(q2s)
    elqdz = ell * q1l * rdz
    return {"akm": _unstk(elqdz * esm, 1, lmh), "akh": _unstk(elqdz * esh, 1, lmh)}


def vdifq(bk, *, lmh, dtdif, q2, el, z):
    """VDIFQ: implicit vertical diffusion of q2.  Returns the new q2 list
    (K = 1..lmh-1 updated)."""
    xp = bk.xp
    esqhf = 0.5 * esq
    q2 = list(q2)
    dtoz = [None] * (lmh + 1)
    akq = [None] * (lmh + 1)
    cr = [None] * (lmh + 1)
    cm = [None] * (lmh + 1)
    rsq2 = [None] * (lmh + 1)
    for k in range(1, lmh - 1):
        dtoz[k] = (dtdif + dtdif) / (z[k] - z[k + 2])
        akq[k] = (
            xp.sqrt((q2[k] + q2[k + 1]) * 0.5)
            * (el[k] + el[k + 1])
            * esqhf
            / (z[k + 1] - z[k + 2])
        )
        cr[k] = -dtoz[k] * akq[k]
    cm[1] = dtoz[1] * akq[1] + 1.0
    rsq2[1] = q2[1]
    for k in range(2, lmh - 1):
        cf = -dtoz[k] * akq[k - 1] / cm[k - 1]
        cm[k] = -cr[k - 1] * cf + (akq[k - 1] + akq[k]) * dtoz[k] + 1.0
        rsq2[k] = -rsq2[k - 1] * cf + q2[k]
    dtozs = (dtdif + dtdif) / (z[lmh - 1] - z[lmh + 1])
    akqs = (
        xp.sqrt((q2[lmh - 1] + q2[lmh]) * 0.5)
        * (el[lmh - 1] + elz0)
        * esqhf
        / (z[lmh] - z[lmh + 1])
    )
    cf = -dtozs * akq[lmh - 2] / cm[lmh - 2]
    q2[lmh - 1] = (dtozs * akqs * q2[lmh] - rsq2[lmh - 2] * cf + q2[lmh - 1]) / (
        (akq[lmh - 2] + akqs) * dtozs - cr[lmh - 2] * cf + 1.0
    )
    for k in range(lmh - 2, 0, -1):
        q2[k] = (-cr[k] * q2[k + 1] + rsq2[k]) / cm[k]
    return q2


def vdifh(bk, *, dtdif, lmh, lpbl, sz0, rkhs, clow, cts, species, rkh, zhk, rho):
    """VDIFH (MSS = 1, MSE = NSPEC): implicit vertical diffusion of the mass
    variables ``species`` (a list of NSPEC column lists) with the surface
    values sz0, the surface exchange rkhs*clow and the countergradient
    terms cts.  Returns the new species lists."""
    xp = bk.xp
    nsp = len(species)
    species = [list(s) for s in species]
    dtoz = [None] * (lmh + 1)
    cr = [None] * (lmh + 1)
    cm = [None] * (lmh + 1)
    rkct = [[None] * (lmh + 1) for _ in range(nsp)]
    rss = [[None] * (lmh + 1) for _ in range(nsp)]
    for k in range(1, lmh):
        dtoz[k] = dtdif / (zhk[k] - zhk[k + 1])
        cr[k] = -dtoz[k] * rkh[k]
        rkhz = rkh[k] * (zhk[k] - zhk[k + 2])
        below = float(k) < lpbl
        for m in range(nsp):
            rkct[m][k] = xp.where(below, 0.0, rkhz * cts[m] * 0.5)
    # top level
    rhok = rho[1]
    cm[1] = dtoz[1] * rkh[1] + rhok
    for m in range(nsp):
        rss[m][1] = -rkct[m][1] * dtoz[1] + species[m][1] * rhok
    # intermediate levels
    for k in range(2, lmh):
        dtozl = dtoz[k]
        cf = -dtozl * rkh[k - 1] / cm[k - 1]
        rhok = rho[k]
        cm[k] = -cr[k - 1] * cf + (rkh[k - 1] + rkh[k]) * dtozl + rhok
        for m in range(nsp):
            rss[m][k] = (
                -rss[m][k - 1] * cf
                + (rkct[m][k - 1] - rkct[m][k]) * dtozl
                + species[m][k] * rhok
            )
    # bottom level
    dtozs = dtdif / (zhk[lmh] - zhk[lmh + 1])
    rkhh = rkh[lmh - 1]
    cf = -dtozs * rkhh / cm[lmh - 1]
    cmb = cr[lmh - 1] * cf
    rhok = rho[lmh]
    for m in range(nsp):
        rkss = rkhs * clow[m]
        cmsb = -cmb + (rkhh + rkss) * dtozs + rhok
        rssb = -rss[m][lmh - 1] * cf + rkct[m][lmh - 1] * dtozs + species[m][lmh] * rhok
        species[m][lmh] = (dtozs * rkss * sz0[m] + rssb) / cmsb
    # backsubstitution
    for k in range(lmh - 1, 0, -1):
        rcml = 1.0 / cm[k]
        for m in range(nsp):
            species[m][k] = (-cr[k] * species[m][k + 1] + rss[m][k]) * rcml
    return species


def vdifv(bk, *, lmh, dtdif, uz0, vz0, rkms, u, v, rkm, z, rho):
    """VDIFV: implicit vertical diffusion of the velocity components.
    Returns the new (u, v) lists."""
    u = list(u)
    v = list(v)
    dtoz = [None] * (lmh + 1)
    cr = [None] * (lmh + 1)
    cm = [None] * (lmh + 1)
    rsu = [None] * (lmh + 1)
    rsv = [None] * (lmh + 1)
    for k in range(1, lmh):
        dtoz[k] = dtdif / (z[k] - z[k + 1])
        cr[k] = -dtoz[k] * rkm[k]
    rhok = rho[1]
    cm[1] = dtoz[1] * rkm[1] + rhok
    rsu[1] = u[1] * rhok
    rsv[1] = v[1] * rhok
    for k in range(2, lmh):
        dtozl = dtoz[k]
        cf = -dtozl * rkm[k - 1] / cm[k - 1]
        rhok = rho[k]
        cm[k] = -cr[k - 1] * cf + (rkm[k - 1] + rkm[k]) * dtozl + rhok
        rsu[k] = -rsu[k - 1] * cf + u[k] * rhok
        rsv[k] = -rsv[k - 1] * cf + v[k] * rhok
    dtozs = dtdif / (z[lmh] - z[lmh + 1])
    rkmh = rkm[lmh - 1]
    cf = -dtozs * rkmh / cm[lmh - 1]
    rhok = rho[lmh]
    rcmvb = 1.0 / ((rkmh + rkms) * dtozs - cr[lmh - 1] * cf + rhok)
    dtozak = dtozs * rkms
    u[lmh] = (dtozak * uz0 - rsu[lmh - 1] * cf + u[lmh] * rhok) * rcmvb
    v[lmh] = (dtozak * vz0 - rsv[lmh - 1] * cf + v[lmh] * rhok) * rcmvb
    for k in range(lmh - 1, 0, -1):
        rcml = 1.0 / cm[k]
        u[k] = (-cr[k] * u[k + 1] + rsu[k]) * rcml
        v[k] = (-cr[k] * v[k + 1] + rsv[k]) * rcml
    return u, v


def myjpbl(
    bk,
    *,
    dt,
    ht,
    dz,
    pmid,
    pint,
    th,
    t,
    exner,
    qv,
    qcw,
    qci,
    u,
    v,
    tsk,
    qsfc,
    chklowq,
    thz0,
    qz0,
    uz0,
    vz0,
    xland,
    tke_myj,
    ustar,
    ct,
    akhs,
    akms,
    elflx,
    exch_h=None,
    exch_m=None,
):
    """MYJPBL (STEPBL = 1).  3-D inputs ``[k, ...]`` with index 0 = Fortran
    k = 1 (k = 1..nk; ``pint`` 1..nk+1, only k = 1 is read); 2-D inputs on
    the columns.  ``exch_h``/``exch_m`` (INOUT, default 0) keep their
    k = nk level.  Returns (Fortran names, lower case) tke_myj, el_myj,
    exch_h, exch_m (k = 1..nk), pblh, kpbl (float), mixht, ct, qsfc, thz0,
    qz0, rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten."""
    xp = bk.xp
    kte = dz.shape[0]
    lev = list(range(kte))
    # make preparations (STEPBL = 1)
    dtturbl = dt * 1
    rdtturbl = 1.0 / dtturbl
    dtdif = dtturbl
    zint = [None] * (kte + 2)
    zint[kte + 1] = ht
    for k in range(kte, 0, -1):
        zint[k] = zint[k + 1] + dz[kte - k, ...]
    # Fortran-k (unflipped) arrays: ape, cwm, the (index k-1)
    ape = [1.0 / exner[k, ...] for k in lev]
    cwm = [qcw[k, ...] + qci[k, ...] for k in lev]
    the = [(cwm[k] * (-elocp / t[k, ...]) + 1.0) * th[k, ...] for k in lev]

    # setup_integration (LOWLYR = 1): K-indexed column lists
    lmh = kte
    psfc = pint[0, ...]

    def col(a):
        c = [None] * (kte + 1)
        for k in range(1, kte + 1):
            c[k] = a[kte - k]
        return c

    tk = col([t[k, ...] for k in lev])
    thek = col(the)
    qk = col([qv[k, ...] / (1.0 + qv[k, ...]) for k in lev])
    cwmk = col(cwm)
    uk = col([u[k, ...] for k in lev])
    vk = col([v[k, ...] for k in lev])
    # TKE=0.5*(q**2) ==> q**2=2.*TKE
    q2k = col([2.0 * tke_myj[k, ...] for k in lev])
    zhk = list(zint)
    # the mixing length
    ml = mixlen(
        bk, lmh=lmh, u=uk, v=vk, t=tk, the=thek, q=qk, cwm=cwmk, q2=q2k, z=zhk, ct=ct
    )
    lpbl = ml["lpbl"]
    ct = ml["ct"]
    # production/dissipation of TKE
    pq = prodq2(
        bk,
        lmh=lmh,
        dtturbl=dtturbl,
        ustar=ustar,
        gm=ml["gm"],
        gh=ml["gh"],
        el=ml["el"],
        q2=q2k,
    )
    el, q2k = pq["el"], pq["q2"]
    # the model layer (counting upward) containing the top of the PBL
    kpbl = float(kte + 1) - lpbl
    # exchange coefficients in the free atmosphere
    dc = difcof(bk, lmh=lmh, gm=ml["gm"], gh=ml["gh"], el=el, q2=q2k, z=zhk)
    akmk, akhk = dc["akm"], dc["akh"]
    zero = xp.zeros_like(ht)
    eh = [exch_h[k, ...] if exch_h is not None else zero for k in lev]
    em = [exch_m[k, ...] if exch_m is not None else zero for k in lev]
    for k in range(1, kte):
        kflip = kte - k
        deltaz = 0.5 * (zhk[kflip] - zhk[kflip + 2])
        eh[k - 1] = akhk[kflip] * deltaz
        em[k - 1] = akmk[kflip] * deltaz
    # vertical diffusion of TKE
    q2k = vdifq(bk, lmh=lmh, dtdif=dtdif, q2=q2k, el=el, z=zhk)
    # the new TKE and mixing length (EL is not defined at KTE)
    tke_new = []
    el_new = []
    for k in range(1, kte + 1):
        kflip = kte + 1 - k
        tke_new.append(0.5 * _max(xp, q2k[kflip], epsq2))
        el_new.append(el[kflip] if kflip < kte else zero)

    # convert surface sensible temperature to potential temperature
    thsk = tsk * (1.0e5 / psfc) ** capa

    # main_integration, first loop: theta_e, q, qc, qi
    qcwk = col([qcw[k, ...] for k in lev])
    qcik = col([qci[k, ...] for k in lev])
    rhok = [None] * (kte + 1)
    for k in range(1, kte + 1):
        kf = kte - k
        rhok[k] = pmid[kf, ...] / (r_d * t[kf, ...] * (1.0 + p608 * qk[k] - cwmk[k]))
    # exchange coefficients times the density at the bottom interface level
    akhd = [None] * (kte + 1)
    for k in range(1, kte):
        akhd[k] = akhk[k] * 0.5 * (rhok[k] + rhok[k + 1])
    seamask = xland - 1.0
    thz0 = (1.0 - seamask) * thsk + seamask * thz0
    akhs_dens = akhs * rhok[kte]
    qfc1 = xlv * chklowq * akhs_dens
    qsfc_land = xp.where(
        qfc1 > 0.0, qk[kte] + elflx / xp.where(qfc1 > 0.0, qfc1, 1.0), qsfc
    )
    exnsfc = (1.0e5 / psfc) ** capa
    qsfc_sea = pq0sea / psfc * xp.exp(a2 * (thsk - a3 * exnsfc) / (thsk - a4 * exnsfc))
    qsfc = xp.where(seamask < 0.5, qsfc_land, qsfc_sea)
    qz0 = (1.0 - seamask) * qsfc + seamask * qz0
    sz0 = [thz0, qz0, 0.0, 0.0]
    clow = [1.0, chklowq, 0.0, 0.0]
    cts = [ct, 0.0, 0.0, 0.0]
    # vertical diffusion of temperature and water
    sp = vdifh(
        bk,
        dtdif=dtdif,
        lmh=lmh,
        lpbl=lpbl,
        sz0=sz0,
        rkhs=akhs_dens,
        clow=clow,
        cts=cts,
        species=[thek, qk, qcwk, qcik],
        rkh=akhd,
        zhk=zhk,
        rho=rhok,
    )
    # primary variable tendencies
    rth, rqv, rqc, rqi = [], [], [], []
    for k in range(1, kte + 1):
        kf = kte + 1 - k
        thnew = sp[0][kf] + (sp[2][kf] + sp[3][kf]) * elocp * ape[k - 1]
        rth.append((thnew - th[k - 1, ...]) * rdtturbl)
        qold = qv[k - 1, ...] / (1.0 + qv[k - 1, ...])
        dqdt = (sp[1][kf] - qold) * rdtturbl
        omq = 1.0 - sp[1][kf]
        rqv.append(dqdt / (omq * omq))
        rqc.append((sp[2][kf] - qcw[k - 1, ...]) * rdtturbl)
        rqi.append((sp[3][kf] - qci[k - 1, ...]) * rdtturbl)

    # main_integration, second loop: velocity components
    akmd = [None] * (kte + 1)
    for k in range(1, kte):
        akmd[k] = akmk[k] * (rhok[k] + rhok[k + 1]) * 0.5
    akms_dens = akms * rhok[kte]
    un, vn = vdifv(
        bk,
        lmh=lmh,
        dtdif=dtdif,
        uz0=uz0,
        vz0=vz0,
        rkms=akms_dens,
        u=uk,
        v=vk,
        rkm=akmd,
        z=zhk,
        rho=rhok,
    )
    ru, rv = [], []
    for k in range(1, kte + 1):
        kf = kte + 1 - k
        ru.append((un[kf] - u[k - 1, ...]) * rdtturbl)
        rv.append((vn[kf] - v[k - 1, ...]) * rdtturbl)
    st = xp.stack
    return {
        "tke_myj": st(tke_new, axis=0),
        "el_myj": st(el_new, axis=0),
        "exch_h": st(eh, axis=0),
        "exch_m": st(em, axis=0),
        "pblh": ml["pblh"],
        "kpbl": kpbl,
        "mixht": ml["mixht"],
        "ct": ct,
        "qsfc": qsfc,
        "thz0": thz0,
        "qz0": qz0,
        "rublten": st(ru, axis=0),
        "rvblten": st(rv, axis=0),
        "rthblten": st(rth, axis=0),
        "rqvblten": st(rqv, axis=0),
        "rqcblten": st(rqc, axis=0),
        "rqiblten": st(rqi, axis=0),
    }


# ---------------------------------------------------------------------------
# CM1 glue (turb.F sfc_and_turb: the domyj block; cm1.F allocations)

WRF_INPUTS = True
DUMP_KEYS = ("xkzh", "xkzm", "tke_myj", "el_myj", "mixht")


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def check(cfg):
    """param.F: ipbl=6 needs sfcmodel=7; only cm1setup=2 is ported."""
    if int(_opt(cfg, "sfcmodel", 0)) != 7:
        raise NotImplementedError("ipbl=6 requires sfcmodel=7 (param.F stops)")
    if int(_opt(cfg, "cm1setup", 1)) != 2:
        raise NotImplementedError("ipbl=6: only cm1setup=2 is ported")


def init_state(ctx):
    """cm1.F: tke_myj = 0.01 and el_myj = 0 on (ib:ie, jb:je, kb:ke+1),
    xkzh = xkzm = 0, mixht = kpbl2d = 0 (and the MYJ surface arrays the PBL
    reads, if the surface layer has not made them)."""
    bk, g, s = ctx.bk, ctx.g, ctx.s
    nz, ny, nx = g.s_shape
    shp2 = (ny, nx)
    out = {
        "tke_myj": bk.zeros((nz + 1, ny, nx)) + 0.01,
        "el_myj": bk.zeros((nz + 1, ny, nx)),
        "xkzh": bk.zeros(g.s_shape),
        "xkzm": bk.zeros(g.s_shape),
        "mixht": bk.zeros(shp2),
        "kpbl2d": bk.zeros(shp2),
    }
    for k in ("chklowq", "lh", "ct", "thz0", "qz0", "uz0", "vz0", "akhs", "akms"):
        if k not in s:
            out[k] = bk.zeros(shp2)
    return out


def pbl(ctx):
    """sfc_and_turb's domyj block: MYJPBL (DT = dt, STEPBL = 1) on the
    get_vars arrays (QCI = qi), ELFLX = lh, CHKLOWQ, SICE = SNOW = 0;
    tke_myj, el_myj (k = 1..nk), xkzh(k+1) = EXCH_H(k), xkzm(k+1) =
    EXCH_M(k) (0 at k = nk+1), the u/v/theta/qv/qc/qi tendencies, hpbl,
    kpbl2d, mixht, ct, qsfc, thz0, qz0."""
    bk, g, s = ctx.bk, ctx.g, ctx.s
    n, z, nk = g.ngxy, g.ngz, g.nk
    J, I = slice(n, n + g.nj), slice(n, n + g.ni)
    K = slice(z, z + nk)
    w = ctx.w

    def i3(a):
        return a[K, J, I]

    def i2(name):
        return s[name][J, I]

    r = myjpbl(
        bk,
        dt=float(ctx.dt),
        ht=bk.zeros(i2("tsk").shape),
        dz=i3(w["dz8w"]),
        pmid=i3(ctx.prs),
        pint=w["p3di"][z : z + nk + 1, J, I],
        th=i3(w["th"]),
        t=i3(w["t"]),
        exner=i3(w["pi"]),
        qv=i3(w["qv"]),
        qcw=i3(w["qc"]),
        qci=i3(w["qi"]),
        u=i3(w["u"]),
        v=i3(w["v"]),
        tsk=i2("tsk"),
        qsfc=i2("qsfc"),
        chklowq=i2("chklowq"),
        thz0=i2("thz0"),
        qz0=i2("qz0"),
        uz0=i2("uz0"),
        vz0=i2("vz0"),
        xland=i2("xland"),
        tke_myj=i3(s["tke_myj"]),
        ustar=i2("ust"),
        ct=i2("ct"),
        akhs=i2("akhs"),
        akms=i2("akms"),
        elflx=i2("lh"),
    )

    def put3(name, val, k0=z):
        return bk.set(s[name] + 0.0, (slice(k0, k0 + val.shape[0]), J, I), val)

    def put2(name, val):
        return bk.set(s[name] + 0.0, (J, I), val)

    zero = bk.zeros((1, *r["pblh"].shape))
    out = {
        "tke_myj": put3("tke_myj", r["tke_myj"]),
        "el_myj": put3("el_myj", r["el_myj"]),
        # xkzh(k+1) = EXCH_H(k), k = 1..nk (EXCH_H(nk) is turb.F's zeroed tmp2d)
        "xkzh": put3("xkzh", bk.xp.concat([r["exch_h"][: nk - 1], zero]), z + 1),
        "xkzm": put3("xkzm", bk.xp.concat([r["exch_m"][: nk - 1], zero]), z + 1),
        "upten": put3("upten", r["rublten"]),
        "vpten": put3("vpten", r["rvblten"]),
        "thpten": put3("thpten", r["rthblten"]),
        "qvpten": put3("qvpten", r["rqvblten"]),
        "qcpten": put3("qcpten", r["rqcblten"]),
        "qipten": put3("qipten", r["rqiblten"]),
    }
    for k, m in (
        ("hpbl", "pblh"),
        ("kpbl2d", "kpbl"),
        ("mixht", "mixht"),
        ("ct", "ct"),
        ("qsfc", "qsfc"),
        ("thz0", "thz0"),
        ("qz0", "qz0"),
    ):
        out[k] = put2(k, r[m])
    return out
