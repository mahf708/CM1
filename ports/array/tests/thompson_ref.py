"""Fortran reference for the Thompson tables (drv_thompson_init), cached.

``fortran_tables(nt_c)`` runs the harness once (thompson_init takes minutes
in Fortran) and caches every constant/table in ``.fortran_cache`` together
with the table files (qr_acr_qgV2.dat, ...) that later drivers copy into
their run directory so CM1 reads the tables instead of recomputing them."""

from __future__ import annotations

import numpy as np

from fortran import CACHE, run_driver

SCALARS = [
    "Nt_c", "mu_c", "Sc3", "D0i", "xm0s", "xm0g", "oig1", "oig2", "obmi", "obmr",
    "ore1", "org1", "org2", "org3", "oams", "obms", "ocms", "oamg", "obmg", "ocmg",
    "oge1", "ogg1", "ogg2", "ogg3", "t1_qr_qc", "t1_qr_qi", "t2_qr_qi", "t1_qg_qc",
    "t1_qs_qc", "t1_qs_qi", "t1_qr_ev", "t2_qr_ev", "t1_qs_sd", "t2_qs_sd",
    "t1_qs_me", "t2_qs_me", "t1_qg_sd", "t2_qg_sd", "t1_qg_me", "t2_qg_me", "nic1",
]  # fmt: skip
DATFILES = ("qr_acr_qgV2.dat", "qr_acr_qsV2.dat", "freezeH2O.dat")


def _shapes():
    from cm1x import thompson_tables as T

    g4 = (T.ntb_g1, T.ntb_g, T.ntb_r1, T.ntb_r)
    s4 = (T.ntb_s, T.ntb_t, T.ntb_r1, T.ntb_r)
    r4 = (T.ntb_r, T.ntb_r1, 45, T.ntb_IN)
    c4 = (T.ntb_c, T.nbc, 45, T.ntb_IN)
    sh = dict(
        cce=(5, 15), ccg=(5, 15), ocg1=(15,), ocg2=(15,), cie=(7,), cig=(7,),
        cre=(13,), crg=(13,), cse=(18,), csg=(18,), cge=(12,), cgg=(12,),
        t_Efrw=(T.nbr, T.nbc), t_Efsw=(T.nbs, T.nbc),
        tpc_wev=(T.nbc, T.ntb_c, T.nbc), tnc_wev=(T.nbc, T.ntb_c, T.nbc),
        tps_iaus=(T.ntb_i, T.ntb_i1), tni_iaus=(T.ntb_i, T.ntb_i1),
        tpi_ide=(T.ntb_i, T.ntb_i1),
    )  # fmt: skip
    for b in ("Dc", "dtc", "Di", "dti", "Dr", "dtr", "Ds", "dts", "Dg", "dtg", "t_Nc"):
        sh[b] = (T.nbins,)
    for n in ("tcg_racg", "tmr_racg", "tcr_gacr", "tmg_gacr", "tnr_racg", "tnr_gacr"):
        sh[n] = g4
    for n in ("tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2", "tcr_sacr1",
              "tms_sacr1", "tcr_sacr2", "tms_sacr2", "tnr_racs1", "tnr_racs2",
              "tnr_sacr1", "tnr_sacr2"):  # fmt: skip
        sh[n] = s4
    for n in ("tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz"):
        sh[n] = r4
    for n in ("tpi_qcfz", "tni_qcfz"):
        sh[n] = c4
    return sh


def ref_dir(nt_c=100.0):
    return CACHE / f"thompson_ref_ntc{float(nt_c):g}"


def fortran_tables(nt_c=100.0) -> dict:
    d = ref_dir(nt_c)
    npz = d / "tables.npz"
    if npz.exists() and all((d / f).exists() for f in DATFILES):
        with np.load(npz) as z:
            return {k: z[k] for k in z.files}
    d.mkdir(parents=True, exist_ok=True)
    sh = _shapes()
    # Fortran arrays are column-major: read with the reversed shape and
    # transpose back to Fortran index order
    outs = {k: tuple(reversed(v)) for k, v in sh.items()}
    outs["scalars"] = (60,)
    res = run_driver(
        "thompson_init",
        {"params": np.array([nt_c])},
        outs,
        keep=[(f, d / f) for f in DATFILES],
    )
    out = {k: (v if k == "scalars" else v.T) for k, v in res.items()}
    for n, name in enumerate(SCALARS):
        out[name] = res["scalars"][n]
    del out["scalars"]
    np.savez(npz, **out)
    return out


def datfiles(nt_c=100.0):
    fortran_tables(nt_c)
    return [ref_dir(nt_c) / f for f in DATFILES]
