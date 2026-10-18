// Port of src/thompson.F mp_gt_driver / mp_thompson and of the ptype=3
// branch of src/mp_driver.F. Operation order and parenthesization follow the
// Fortran (left-to-right a*b*c, a*b/c, ...; Fortran x**3.0 -> pow(x,3.0),
// x**2.0 -> x*x, x**1.0 -> x, x**integer -> powi (libgcc __powidf2)), so the
// results match the double-precision Fortran bit for bit.
//
// One column per thread; the column arrays of mp_thompson live in a scratch
// view (Work::scr), one block of NVAR arrays of nk+2 levels per column,
// indexed with the Fortran k (1..nk, and nk+1 for the fall speeds).
#include "cm1k_thompson.hpp"

#include "physics/kessler/cm1k_kessler.hpp"   // pdefq (misclibs.F)

#include <stdexcept>
#include <string>

namespace cm1k::thompson {

using namespace par;

namespace {

KOKKOS_INLINE_FUNCTION double fmax2(double a, double b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION double fmin2(double a, double b) { return a < b ? a : b; }
KOKKOS_INLINE_FUNCTION int imax2(int a, int b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION int imin2(int a, int b) { return a < b ? a : b; }
KOKKOS_INLINE_FUNCTION int nint(double x) { return int(Kokkos::round(x)); }
// libgcc __powidf2
KOKKOS_INLINE_FUNCTION double powi(double x, int m) {
  unsigned n = m < 0 ? -(unsigned)m : (unsigned)m;
  double y = (n % 2) ? x : 1.0;
  while (n >>= 1) {
    x = x * x;
    if (n % 2) y = y * x;
  }
  return m < 0 ? 1.0 / y : y;
}

// thompson.F RSLF / RSIF
KOKKOS_INLINE_FUNCTION double rslf(double P, double T) {
  constexpr double C0 = .611583699E03, C1 = .444606896E02, C2 = .143177157E01, C3 = .264224321E-1,
                   C4 = .299291081E-3, C5 = .203154182E-5, C6 = .702620698E-8, C7 = .379534310E-11,
                   C8 = -.321582393E-13;
  const double X = fmax2(-80., T - 273.16);
  double ESL = C0 + X * (C1 + X * (C2 + X * (C3 + X * (C4 + X * (C5 + X * (C6 + X * (C7 + X * C8)))))));
  ESL = fmin2(ESL, P * 0.15);
  return .622 * ESL / (P - ESL);
}
KOKKOS_INLINE_FUNCTION double rsif(double P, double T) {
  constexpr double C0 = .609868993E03, C1 = .499320233E02, C2 = .184672631E01, C3 = .402737184E-1,
                   C4 = .565392987E-3, C5 = .521693933E-5, C6 = .307839583E-7, C7 = .105785160E-9,
                   C8 = .161444444E-12;
  const double X = fmax2(-80., T - 273.16);
  double ESI = C0 + X * (C1 + X * (C2 + X * (C3 + X * (C4 + X * (C5 + X * (C6 + X * (C7 + X * C8)))))));
  ESI = fmin2(ESI, P * 0.15);
  return .622 * ESI / (P - ESI);
}

// Lookup index of a value in the 1,2,..,9,10,20,.. tables (the "do nn =
// nic-1, nic+1 ... goto" blocks of mp_thompson): n2 = nint(log10(first entry)).
KOKKOS_INLINE_FUNCTION int decade_index(double x, int n2) {
  const int nic = nint(Kokkos::log10(x));
  int n = nic + 1;
  for (int nn = nic - 1; nn <= nic + 1; ++nn) {
    n = nn;
    if ((x / powi(10., nn)) >= 1.0 && (x / powi(10., nn)) < 10.0) break;
  }
  return int(x / powi(10., n)) + 10 * (n - n2) - (n - n2);
}

// Snow moment polynomials (sa/sb from Field et al. 2005)
struct SnowPoly {
  double sa[11] = {0, 5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
                   0.31255, 0.000204, 0.003199, 0.0, -0.015952};
  double sb[11] = {0, 0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
                   0.060366, 0.000079, 0.000594, 0.0, -0.003577};
};

// Column arrays of mp_thompson (and mp_gt_driver's 1d copies)
enum : int {
  // mp_gt_driver / mp_thompson arguments
  V_qv1d, V_qc1d, V_qi1d, V_qr1d, V_qs1d, V_qg1d, V_ni1d, V_nr1d, V_nc1d, V_t1d, V_p1d, V_w1d, V_dz1d,
  V_prv_rev, V_prw_vcd, V_vtik, V_vtrk, V_vtsk, V_vtgk, V_vtck, V_rho1d,
  // locals
  V_tten, V_qvten, V_qcten, V_qiten, V_qrten, V_qsten, V_qgten, V_niten, V_nrten, V_ncten,
  V_pnc_wcd, V_pnc_wau, V_pnc_rcw, V_pnc_scw, V_pnc_gcw,
  V_prr_wau, V_prr_rcw, V_prr_rcs, V_prr_rcg, V_prr_sml, V_prr_gml, V_prr_rci,
  V_pnr_wau, V_pnr_rcs, V_pnr_rcg, V_pnr_rci, V_pnr_sml, V_pnr_gml, V_pnr_rev, V_pnr_rcr, V_pnr_rfz,
  V_pri_inu, V_pni_inu, V_pri_ihm, V_pni_ihm, V_pri_wfz, V_pni_wfz, V_pri_rfz, V_pni_rfz,
  V_pri_ide, V_pni_ide, V_pri_rci, V_pni_rci, V_pni_sci, V_pni_iau, V_pri_iha, V_pni_iha,
  V_prs_iau, V_prs_sci, V_prs_rcs, V_prs_scw, V_prs_sde, V_prs_ihm, V_prs_ide,
  V_prg_scw, V_prg_rfz, V_prg_gde, V_prg_gcw, V_prg_rci, V_prg_rcs, V_prg_rcg, V_prg_ihm,
  V_temp, V_pres, V_qv, V_rc, V_ri, V_rr, V_rs, V_rg, V_ni, V_nr, V_nc,
  V_rho, V_rhof, V_rhof2, V_qvs, V_qvsi, V_delQvs, V_satw, V_sati, V_ssatw, V_ssati,
  V_diffu, V_visco, V_vsc2, V_tcond, V_lvap, V_ocp, V_lvt2, V_rcvm,
  V_ilamr, V_ilamg, V_N0_r, V_N0_g, V_mvd_r, V_mvd_c,
  V_smob, V_smo2, V_smo1, V_smo0, V_smoc, V_smod, V_smoe, V_smof,
  V_sed_r, V_sed_s, V_sed_g, V_sed_i, V_sed_n, V_sed_c,
  V_vtnik, V_vtnrk, V_vtnck, V_vts_boost,
  NVAR
};
enum : int { I_L_qc, I_L_qi, I_L_qr, I_L_qs, I_L_qg, NIVAR };

struct Col {
  double* base;
  int* ibase;
  int nkk;
  KOKKOS_INLINE_FUNCTION double* a(int v) const { return base + size_t(v) * nkk; }
  KOKKOS_INLINE_FUNCTION int* l(int v) const { return ibase + size_t(v) * nkk; }
};

#define TH_COL(name) double* const name = s.a(V_##name)
#define TH_LCOL(name) int* const name = s.l(I_##name)

//---------------------------------------------------------------------------
// mp_thompson: one column, levels 1..nk (kts=1, kte=nk). The 1d arrays are
// in the scratch (qv1d ... dz1d in, updated in place); ppt* accumulate.
//---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void mp_thompson(const DevTables& T, const Col& s, const int kte, const double dt,
                                        const int eqtset, double& pptrain, double& pptsnow,
                                        double& pptgraul, double& pptice) {
  const Consts& c = T.c;
  const SnowPoly sp;
  const double* sa = sp.sa;
  const double* sb = sp.sb;
  constexpr int kts = 1;

  TH_COL(qv1d); TH_COL(qc1d); TH_COL(qi1d); TH_COL(qr1d); TH_COL(qs1d); TH_COL(qg1d); TH_COL(ni1d);
  TH_COL(nr1d); TH_COL(nc1d); TH_COL(t1d); TH_COL(p1d); TH_COL(w1d); TH_COL(dz1d);
  double* const dzq = dz1d;
  TH_COL(prv_rev); TH_COL(prw_vcd); TH_COL(vtik); TH_COL(vtrk); TH_COL(vtsk); TH_COL(vtgk); TH_COL(vtck);
  TH_COL(tten); TH_COL(qvten); TH_COL(qcten); TH_COL(qiten); TH_COL(qrten); TH_COL(qsten); TH_COL(qgten);
  TH_COL(niten); TH_COL(nrten); TH_COL(ncten);
  TH_COL(pnc_wcd); TH_COL(pnc_wau); TH_COL(pnc_rcw); TH_COL(pnc_scw); TH_COL(pnc_gcw);
  TH_COL(prr_wau); TH_COL(prr_rcw); TH_COL(prr_rcs); TH_COL(prr_rcg); TH_COL(prr_sml); TH_COL(prr_gml);
  TH_COL(prr_rci);
  TH_COL(pnr_wau); TH_COL(pnr_rcs); TH_COL(pnr_rcg); TH_COL(pnr_rci); TH_COL(pnr_sml); TH_COL(pnr_gml);
  TH_COL(pnr_rev); TH_COL(pnr_rcr); TH_COL(pnr_rfz);
  TH_COL(pri_inu); TH_COL(pni_inu); TH_COL(pri_ihm); TH_COL(pni_ihm); TH_COL(pri_wfz); TH_COL(pni_wfz);
  TH_COL(pri_rfz); TH_COL(pni_rfz); TH_COL(pri_ide); TH_COL(pni_ide); TH_COL(pri_rci); TH_COL(pni_rci);
  TH_COL(pni_sci); TH_COL(pni_iau); TH_COL(pri_iha); TH_COL(pni_iha);
  TH_COL(prs_iau); TH_COL(prs_sci); TH_COL(prs_rcs); TH_COL(prs_scw); TH_COL(prs_sde); TH_COL(prs_ihm);
  TH_COL(prs_ide);
  TH_COL(prg_scw); TH_COL(prg_rfz); TH_COL(prg_gde); TH_COL(prg_gcw); TH_COL(prg_rci); TH_COL(prg_rcs);
  TH_COL(prg_rcg); TH_COL(prg_ihm);
  TH_COL(temp); TH_COL(pres); TH_COL(qv); TH_COL(rc); TH_COL(ri); TH_COL(rr); TH_COL(rs); TH_COL(rg);
  TH_COL(ni); TH_COL(nr); TH_COL(nc);
  TH_COL(rho); TH_COL(rhof); TH_COL(rhof2); TH_COL(qvs); TH_COL(qvsi); TH_COL(delQvs); TH_COL(satw);
  TH_COL(sati); TH_COL(ssatw); TH_COL(ssati); TH_COL(diffu); TH_COL(visco); TH_COL(vsc2); TH_COL(tcond);
  TH_COL(lvap); TH_COL(ocp); TH_COL(lvt2); TH_COL(rcvm);
  TH_COL(ilamr); TH_COL(ilamg); TH_COL(N0_r); TH_COL(N0_g); TH_COL(mvd_r); TH_COL(mvd_c);
  TH_COL(smob); TH_COL(smo2); TH_COL(smo1); TH_COL(smo0); TH_COL(smoc); TH_COL(smod); TH_COL(smoe);
  TH_COL(smof);
  TH_COL(sed_r); TH_COL(sed_s); TH_COL(sed_g); TH_COL(sed_i); TH_COL(sed_n); TH_COL(sed_c);
  TH_COL(vtnik); TH_COL(vtnrk); TH_COL(vtnck); TH_COL(vts_boost);
  TH_LCOL(L_qc); TH_LCOL(L_qi); TH_LCOL(L_qr); TH_LCOL(L_qs); TH_LCOL(L_qg);

  const double Nt_c = c.Nt_c;
  constexpr bool is_aerosol_aware = false;

  bool no_micro = true;
  const double dtsave = dt;
  const double odt = 1. / dt;
  const double odts = 1. / dtsave;

  double lamc = 0, lamr = 0, lamg = 0, lami = 0, ilami = 0, ilamc = 0, lam_exp = 0, N0_exp = 0;
  double xDc = 0, Dc_b = 0, Dc_g = 0, xDi = 0, xDs = 0, xDg = 0;
  double tempc = 0, tc0 = 0, loga_ = 0, a_ = 0, b_ = 0;
  double xni = 0, xnc = 0, xri = 0, xrc = 0, xrr = 0, xnr = 0, xmi = 0, oxmi = 0;
  double rate_max = 0, sump = 0, ratio = 0, otemp = 0, rvs = 0, rvs_p = 0, rvs_pp = 0, gamsc = 0,
         alphsc = 0, t1_evap = 0, t1_subl = 0, xsat = 0, orho = 0, lfus2 = 0, econd = 0, edep = 0,
         efrz = 0;
  int nu_c = 2;

  for (int k = kts; k <= kte; ++k) {
    tten[k] = 0.; qvten[k] = 0.; qcten[k] = 0.; qiten[k] = 0.; qrten[k] = 0.; qsten[k] = 0.;
    qgten[k] = 0.; niten[k] = 0.; nrten[k] = 0.; ncten[k] = 0.;
    prw_vcd[k] = 0.;
    pnc_wcd[k] = 0.; pnc_wau[k] = 0.; pnc_rcw[k] = 0.; pnc_scw[k] = 0.; pnc_gcw[k] = 0.;
    prv_rev[k] = 0.;
    prr_wau[k] = 0.; prr_rcw[k] = 0.; prr_rcs[k] = 0.; prr_rcg[k] = 0.; prr_sml[k] = 0.; prr_gml[k] = 0.;
    prr_rci[k] = 0.;
    pnr_wau[k] = 0.; pnr_rcs[k] = 0.; pnr_rcg[k] = 0.; pnr_rci[k] = 0.; pnr_sml[k] = 0.; pnr_gml[k] = 0.;
    pnr_rev[k] = 0.; pnr_rcr[k] = 0.; pnr_rfz[k] = 0.;
    pri_inu[k] = 0.; pni_inu[k] = 0.; pri_ihm[k] = 0.; pni_ihm[k] = 0.; pri_wfz[k] = 0.; pni_wfz[k] = 0.;
    pri_rfz[k] = 0.; pni_rfz[k] = 0.; pri_ide[k] = 0.; pni_ide[k] = 0.; pri_rci[k] = 0.; pni_rci[k] = 0.;
    pni_sci[k] = 0.; pni_iau[k] = 0.; pri_iha[k] = 0.; pni_iha[k] = 0.;
    prs_iau[k] = 0.; prs_sci[k] = 0.; prs_rcs[k] = 0.; prs_scw[k] = 0.; prs_sde[k] = 0.; prs_ihm[k] = 0.;
    prs_ide[k] = 0.;
    prg_scw[k] = 0.; prg_rfz[k] = 0.; prg_gde[k] = 0.; prg_gcw[k] = 0.; prg_rci[k] = 0.; prg_rcs[k] = 0.;
    prg_rcg[k] = 0.; prg_ihm[k] = 0.;
  }
  for (int k = kts; k <= kte; ++k) {
    smo0[k] = 0.; smo1[k] = 0.; smo2[k] = 0.; smob[k] = 0.; smoc[k] = 0.; smod[k] = 0.; smoe[k] = 0.;
    smof[k] = 0.; mvd_r[k] = 0.; mvd_c[k] = 0.;
  }

  //..Put column of data into local arrays.
  for (int k = kts; k <= kte; ++k) {
    temp[k] = t1d[k];
    qv[k] = fmax2(1.E-10, qv1d[k]);
    pres[k] = p1d[k];
    rho[k] = 0.622 * pres[k] / (R * temp[k] * (qv[k] + 0.622));

    if (qc1d[k] > R1) {
      no_micro = false;
      rc[k] = qc1d[k] * rho[k];
      nc[k] = fmax2(2., fmin2(nc1d[k] * rho[k], Nt_c_max));
      L_qc[k] = 1;
      nu_c = imin2(15, nint(1000.E6 / nc[k]) + 2);
      lamc = Kokkos::pow(nc[k] * am_r * c.ccg[2][nu_c] * c.ocg1[nu_c] / rc[k], c.obmr);
      xDc = (bm_r + nu_c + 1.) / lamc;
      if (xDc < D0c) {
        lamc = c.cce[2][nu_c] / D0c;
      } else if (xDc > D0r * 2.) {
        lamc = c.cce[2][nu_c] / (D0r * 2.);
      }
      nc[k] = fmin2(Nt_c_max, c.ccg[1][nu_c] * c.ocg2[nu_c] * rc[k] / am_r * Kokkos::pow(lamc, bm_r));
      if (!is_aerosol_aware) nc[k] = Nt_c;
    } else {
      qc1d[k] = 0.0;
      nc1d[k] = 0.0;
      rc[k] = R1;
      nc[k] = 2.;
      L_qc[k] = 0;
    }

    if (qi1d[k] > R1) {
      no_micro = false;
      ri[k] = qi1d[k] * rho[k];
      ni[k] = fmax2(R2, ni1d[k] * rho[k]);
      if (ni[k] <= R2) {
        lami = c.cie[2] / 5.E-6;
        ni[k] = fmin2(9999.E3, c.cig[1] * c.oig2 * ri[k] / am_i * Kokkos::pow(lami, bm_i));
      }
      L_qi[k] = 1;
      lami = Kokkos::pow(am_i * c.cig[2] * c.oig1 * ni[k] / ri[k], c.obmi);
      ilami = 1. / lami;
      xDi = (bm_i + mu_i + 1.) * ilami;
      if (xDi < 5.E-6) {
        lami = c.cie[2] / 5.E-6;
        ni[k] = fmin2(9999.E3, c.cig[1] * c.oig2 * ri[k] / am_i * Kokkos::pow(lami, bm_i));
      } else if (xDi > 300.E-6) {
        lami = c.cie[2] / 300.E-6;
        ni[k] = c.cig[1] * c.oig2 * ri[k] / am_i * Kokkos::pow(lami, bm_i);
      }
    } else {
      qi1d[k] = 0.0;
      ni1d[k] = 0.0;
      ri[k] = R1;
      ni[k] = R2;
      L_qi[k] = 0;
    }

    if (qr1d[k] > R1) {
      no_micro = false;
      rr[k] = qr1d[k] * rho[k];
      nr[k] = fmax2(R2, nr1d[k] * rho[k]);
      if (nr[k] <= R2) {
        mvd_r[k] = 1.0E-3;
        lamr = (3.0 + mu_r + 0.672) / mvd_r[k];
        nr[k] = c.crg[2] * c.org3 * rr[k] * Kokkos::pow(lamr, bm_r) / am_r;
      }
      L_qr[k] = 1;
      lamr = Kokkos::pow(am_r * c.crg[3] * c.org2 * nr[k] / rr[k], c.obmr);
      mvd_r[k] = (3.0 + mu_r + 0.672) / lamr;
      if (mvd_r[k] > 2.5E-3) {
        mvd_r[k] = 2.5E-3;
        lamr = (3.0 + mu_r + 0.672) / mvd_r[k];
        nr[k] = c.crg[2] * c.org3 * rr[k] * Kokkos::pow(lamr, bm_r) / am_r;
      } else if (mvd_r[k] < D0r * 0.75) {
        mvd_r[k] = D0r * 0.75;
        lamr = (3.0 + mu_r + 0.672) / mvd_r[k];
        nr[k] = c.crg[2] * c.org3 * rr[k] * Kokkos::pow(lamr, bm_r) / am_r;
      }
    } else {
      qr1d[k] = 0.0;
      nr1d[k] = 0.0;
      rr[k] = R1;
      nr[k] = R2;
      L_qr[k] = 0;
    }
    if (qs1d[k] > R1) {
      no_micro = false;
      rs[k] = qs1d[k] * rho[k];
      L_qs[k] = 1;
    } else {
      qs1d[k] = 0.0;
      rs[k] = R1;
      L_qs[k] = 0;
    }
    if (qg1d[k] > R1) {
      no_micro = false;
      rg[k] = qg1d[k] * rho[k];
      L_qg[k] = 1;
    } else {
      qg1d[k] = 0.0;
      rg[k] = R1;
      L_qg[k] = 0;
    }
  }

  //..Derive various thermodynamic variables frequently used.
  for (int k = kts; k <= kte; ++k) {
    tempc = temp[k] - 273.15;
    rhof[k] = Kokkos::sqrt(rho_not / rho[k]);
    rhof2[k] = Kokkos::sqrt(rhof[k]);
    qvs[k] = rslf(pres[k], temp[k]);
    delQvs[k] = fmax2(0.0, rslf(pres[k], 273.15) - qv[k]);
    if (tempc <= 0.0) {
      qvsi[k] = rsif(pres[k], temp[k]);
    } else {
      qvsi[k] = qvs[k];
    }
    satw[k] = qv[k] / qvs[k];
    sati[k] = qv[k] / qvsi[k];
    ssatw[k] = satw[k] - 1.;
    ssati[k] = sati[k] - 1.;
    if (Kokkos::fabs(ssatw[k]) < eps) ssatw[k] = 0.0;
    if (Kokkos::fabs(ssati[k]) < eps) ssati[k] = 0.0;
    if (no_micro && ssati[k] > 0.0) no_micro = false;
    diffu[k] = 2.11E-5 * Kokkos::pow(temp[k] / 273.15, 1.94) * (101325. / pres[k]);
    if (tempc >= 0.0) {
      visco[k] = (1.718 + 0.0049 * tempc) * 1.0E-5;
    } else {
      visco[k] = (1.718 + 0.0049 * tempc - 1.2E-5 * tempc * tempc) * 1.0E-5;
    }
    ocp[k] = 1. / (Cp * (1. + 0.887 * qv[k]));
    vsc2[k] = Kokkos::sqrt(rho[k] / visco[k]);
    lvap[k] = lvap0 + (2106.0 - 4218.0) * tempc;
    tcond[k] = (5.69 + 0.0168 * tempc) * 1.0E-5 * 418.936;
    if (eqtset == 2) {
      rcvm[k] = 1.0 / (cv + cvv * qv1d[k] + cpl * (qc1d[k] + qr1d[k]) + cpi * (qi1d[k] + qs1d[k] + qg1d[k]));
    }
  }

  if (no_micro) return;

  //..Calculate y-intercept, slope, and useful moments for snow.
  for (int k = kts; k <= kte; ++k) {
    if (!L_qs[k]) continue;
    tc0 = fmin2(-0.1, temp[k] - 273.15);
    smob[k] = rs[k] * c.oams;
    // bm_s = 2: the reference (2nd) moment is smob
    smo2[k] = smob[k];

    loga_ = sa[1] + sa[2] * tc0 + sa[5] * tc0 * tc0 + sa[9] * tc0 * tc0 * tc0;
    a_ = Kokkos::pow(10.0, loga_);
    b_ = sb[1] + sb[2] * tc0 + sb[5] * tc0 * tc0 + sb[9] * tc0 * tc0 * tc0;
    smo0[k] = a_ * Kokkos::pow(smo2[k], b_);

    loga_ = sa[1] + sa[2] * tc0 + sa[3] + sa[4] * tc0 + sa[5] * tc0 * tc0 + sa[6] + sa[7] * tc0 * tc0
          + sa[8] * tc0 + sa[9] * tc0 * tc0 * tc0 + sa[10];
    a_ = Kokkos::pow(10.0, loga_);
    b_ = sb[1] + sb[2] * tc0 + sb[3] + sb[4] * tc0 + sb[5] * tc0 * tc0 + sb[6] + sb[7] * tc0 * tc0
       + sb[8] * tc0 + sb[9] * tc0 * tc0 * tc0 + sb[10];
    smo1[k] = a_ * Kokkos::pow(smo2[k], b_);

    for (int which = 0; which < 3; ++which) {
      const double x = which == 0 ? c.cse[1] : (which == 1 ? c.cse[13] : c.cse[16]);
      loga_ = sa[1] + sa[2] * tc0 + sa[3] * x + sa[4] * tc0 * x + sa[5] * tc0 * tc0 + sa[6] * x * x
            + sa[7] * tc0 * tc0 * x + sa[8] * tc0 * x * x + sa[9] * tc0 * tc0 * tc0 + sa[10] * x * x * x;
      a_ = Kokkos::pow(10.0, loga_);
      b_ = sb[1] + sb[2] * tc0 + sb[3] * x + sb[4] * tc0 * x + sb[5] * tc0 * tc0 + sb[6] * x * x
         + sb[7] * tc0 * tc0 * x + sb[8] * tc0 * x * x + sb[9] * tc0 * tc0 * tc0 + sb[10] * x * x * x;
      const double mo = a_ * Kokkos::pow(smo2[k], b_);
      if (which == 0) smoc[k] = mo;
      else if (which == 1) smoe[k] = mo;
      else smof[k] = mo;
    }
  }

  //..Calculate y-intercept, slope values for graupel.
  for (int k = kte; k >= kts; --k) {
    const double ygra1 = Kokkos::log10(fmax2(1.E-9, rg[k]));
    double zans1 = 3.0 + 2. / 7. * (ygra1 + 8.);
    zans1 = fmax2(2., fmin2(zans1, 6.));
    N0_exp = Kokkos::pow(10., zans1);
    lam_exp = Kokkos::pow(N0_exp * am_g * c.cgg[1] / rg[k], c.oge1);
    lamg = lam_exp * Kokkos::pow(c.cgg[3] * c.ogg2 * c.ogg1, c.obmg);
    ilamg[k] = 1. / lamg;
    N0_g[k] = N0_exp / (c.cgg[2] * lam_exp) * Kokkos::pow(lamg, c.cge[2]);
  }

  //..Calculate y-intercept, slope values for rain.
  for (int k = kte; k >= kts; --k) {
    lamr = Kokkos::pow(am_r * c.crg[3] * c.org2 * nr[k] / rr[k], c.obmr);
    ilamr[k] = 1. / lamr;
    mvd_r[k] = (3.0 + mu_r + 0.672) / lamr;
    N0_r[k] = nr[k] * c.org2 * Kokkos::pow(lamr, c.cre[2]);
  }

  //..Compute warm-rain process terms (except evap done later).
  for (int k = kts; k <= kte; ++k) {
    if (L_qr[k] && mvd_r[k] > D0r) {
      const double Ef_rr = 1.0 - Kokkos::exp(2300.0 * (mvd_r[k] - 1950.0E-6));
      pnr_rcr[k] = Ef_rr * 2.0 * nr[k] * rr[k];
    }

    mvd_c[k] = D0c;
    if (L_qc[k]) {
      nu_c = imin2(15, nint(1000.E6 / nc[k]) + 2);
      xDc = fmax2(D0c * 1.E6, (Kokkos::pow(rc[k] / (am_r * nc[k]), c.obmr)) * 1.E6);
      lamc = Kokkos::pow(nc[k] * am_r * c.ccg[2][nu_c] * c.ocg1[nu_c] / rc[k], c.obmr);
      mvd_c[k] = (3.0 + nu_c + 0.672) / lamc;
    }

    if (rc[k] > 0.01e-3) {
      Dc_g = (Kokkos::pow(c.ccg[3][nu_c] * c.ocg2[nu_c], c.obmr) / lamc) * 1.E6;
      Dc_b = Kokkos::pow(xDc * xDc * xDc * Dc_g * Dc_g * Dc_g - xDc * xDc * xDc * xDc * xDc * xDc, 1. / 6.);
      const double zeta1 = 0.5 * ((6.25E-6 * xDc * Dc_b * Dc_b * Dc_b - 0.4)
                                  + Kokkos::fabs(6.25E-6 * xDc * Dc_b * Dc_b * Dc_b - 0.4));
      const double zeta = 0.027 * rc[k] * zeta1;
      const double taud = 0.5 * ((0.5 * Dc_b - 7.5) + Kokkos::fabs(0.5 * Dc_b - 7.5)) + R1;
      const double tau = 3.72 / (rc[k] * taud);
      prr_wau[k] = zeta / tau;
      prr_wau[k] = fmin2(rc[k] * odts, prr_wau[k]);
      pnr_wau[k] = prr_wau[k] / (am_r * nu_c * 200. * D0r * D0r * D0r);
      pnc_wau[k] = fmin2(nc[k] * odts, prr_wau[k] / (am_r * mvd_c[k] * mvd_c[k] * mvd_c[k]));
    }

    if (L_qr[k] && mvd_r[k] > D0r && mvd_c[k] > D0c) {
      lamr = 1. / ilamr[k];
      int idx = 1 + int(nbr * Kokkos::log(mvd_r[k] / c.Dr1) / Kokkos::log(c.Drn / c.Dr1));
      idx = imin2(idx, nbr);
      const double Ef_rw = T.t_Efrw(idx - 1, int(mvd_c[k] * 1.E6) - 1);
      prr_rcw[k] = rhof[k] * c.t1_qr_qc * Ef_rw * rc[k] * N0_r[k] * (Kokkos::pow(lamr + fv_r, -c.cre[9]));
      prr_rcw[k] = fmin2(rc[k] * odts, prr_rcw[k]);
      pnc_rcw[k] = rhof[k] * c.t1_qr_qc * Ef_rw * nc[k] * N0_r[k] * (Kokkos::pow(lamr + fv_r, -c.cre[9]));
      pnc_rcw[k] = fmin2(nc[k] * odts, pnc_rcw[k]);
    }
    // (rain collecting aerosols: aerosol aware only)
  }

  //..Compute all frozen hydrometeor species' process terms.
  for (int k = kts; k <= kte; ++k) {
    vts_boost[k] = 1.0;
    xDs = 0.0;
    if (L_qs[k]) xDs = smoc[k] / smob[k];

    //..Temperature lookup table indexes.
    tempc = temp[k] - 273.15;
    const int idx_tc = imax2(1, imin2(nint(-tempc), 45));
    int idx_t = int((tempc - 2.5) / 5.) - 1;
    idx_t = imax2(1, -idx_t);
    idx_t = imin2(idx_t, ntb_t);

    //..Cloud water lookup table index.
    int idx_c = 1;
    if (rc[k] > 1.e-6) idx_c = imax2(1, imin2(decade_index(rc[k], c.nic2), ntb_c));

    //..Cloud droplet number lookup table index. (unused here: idx_n)

    //..Cloud ice lookup table indexes.
    int idx_i = 1, idx_i1 = 1;
    if (ri[k] > 1.e-10) idx_i = imax2(1, imin2(decade_index(ri[k], c.nii2), ntb_i));
    if (ni[k] > 1.0) idx_i1 = imax2(1, imin2(decade_index(ni[k], c.nii3), ntb_i1));

    //..Rain lookup table indexes.
    int idx_r = 1, idx_r1 = ntb_r1;
    if (rr[k] > 1.e-6) {
      idx_r = imax2(1, imin2(decade_index(rr[k], c.nir2), ntb_r));
      lamr = 1. / ilamr[k];
      lam_exp = lamr * Kokkos::pow(c.crg[3] * c.org2 * c.org1, bm_r);
      N0_exp = c.org1 * rr[k] / am_r * Kokkos::pow(lam_exp, c.cre[1]);
      idx_r1 = imax2(1, imin2(decade_index(N0_exp, c.nir3), ntb_r1));
    }

    //..Snow lookup table index.
    int idx_s = 1;
    if (rs[k] > 1.e-6) idx_s = imax2(1, imin2(decade_index(rs[k], c.nis2), ntb_s));

    //..Graupel lookup table index.
    int idx_g = 1, idx_g1 = ntb_g1;
    if (rg[k] > 1.e-6) {
      idx_g = imax2(1, imin2(decade_index(rg[k], c.nig2), ntb_g));
      lamg = 1. / ilamg[k];
      lam_exp = lamg * Kokkos::pow(c.cgg[3] * c.ogg2 * c.ogg1, bm_g);
      N0_exp = c.ogg1 * rg[k] / am_g * Kokkos::pow(lam_exp, c.cge[1]);
      idx_g1 = imax2(1, imin2(decade_index(N0_exp, c.nig3), ntb_g1));
    }

    //..Deposition/sublimation prefactor (from Srivastava & Coen 1992).
    otemp = 1. / temp[k];
    rvs = rho[k] * qvsi[k];
    rvs_p = rvs * otemp * (lsub * otemp * oRv - 1.);
    rvs_pp = rvs * (otemp * (lsub * otemp * oRv - 1.) * otemp * (lsub * otemp * oRv - 1.)
                    + (-2. * lsub * otemp * otemp * otemp * oRv) + otemp * otemp);
    gamsc = lsub * diffu[k] / tcond[k] * rvs_p;
    alphsc = 0.5 * (gamsc / (1. + gamsc)) * (gamsc / (1. + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p;
    alphsc = fmax2(1.E-9, alphsc);
    xsat = ssati[k];
    if (Kokkos::fabs(xsat) < 1.E-9) xsat = 0.;
    t1_subl = 4. * PI * (1.0 - alphsc * xsat + 2. * alphsc * alphsc * xsat * xsat
                         - 5. * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1. + gamsc);

    //..Snow collecting cloud water.  In CE, assume Dc<<Ds and vtc=~0.
    if (L_qc[k] && mvd_c[k] > D0c) {
      if (xDs > D0s) {
        int idx = 1 + int(nbs * Kokkos::log(xDs / c.Ds1) / Kokkos::log(c.Dsn / c.Ds1));
        idx = imin2(idx, nbs);
        const double Ef_sw = T.t_Efsw(idx - 1, int(mvd_c[k] * 1.E6) - 1);
        prs_scw[k] = rhof[k] * c.t1_qs_qc * Ef_sw * rc[k] * smoe[k];
        prs_scw[k] = fmin2(rc[k] * odts, prs_scw[k]);
        pnc_scw[k] = rhof[k] * c.t1_qs_qc * Ef_sw * nc[k] * smoe[k];
        pnc_scw[k] = fmin2(nc[k] * odts, pnc_scw[k]);
      }

      //..Graupel collecting cloud water.  In CE, assume Dc<<Dg and vtc=~0.
      if (rg[k] >= 1.e-6 && mvd_c[k] > D0c) {
        xDg = (bm_g + mu_g + 1.) * ilamg[k];
        const double vtg = rhof[k] * av_g * c.cgg[6] * c.ogg3 * Kokkos::pow(ilamg[k], bv_g);
        const double stoke_g = mvd_c[k] * mvd_c[k] * vtg * rho_w / (9. * visco[k] * xDg);
        if (xDg > D0g) {
          double Ef_gw = 0.0;
          if (stoke_g >= 0.4 && stoke_g <= 10.) {
            Ef_gw = 0.55 * Kokkos::log10(2.51 * stoke_g);
          } else if (stoke_g < 0.4) {
            Ef_gw = 0.0;
          } else if (stoke_g > 10) {
            Ef_gw = 0.77;
          }
          prg_gcw[k] = rhof[k] * c.t1_qg_qc * Ef_gw * rc[k] * N0_g[k] * Kokkos::pow(ilamg[k], c.cge[9]);
          pnc_gcw[k] = rhof[k] * c.t1_qg_qc * Ef_gw * nc[k] * N0_g[k] * Kokkos::pow(ilamg[k], c.cge[9]);
          pnc_gcw[k] = fmin2(nc[k] * odts, pnc_gcw[k]);
        }
      }
    }
    // (snow and graupel collecting aerosols: aerosol aware only)

    //..Rain collecting snow.
    if (rr[k] >= 1.e-6) {
      if (rs[k] >= 1.e-6) {
        const int is = idx_s - 1, it = idx_t - 1, ir1 = idx_r1 - 1, ir = idx_r - 1;
        if (temp[k] < T_0) {
          prr_rcs[k] = -(T.tmr_racs2(is, it, ir1, ir) + T.tcr_sacr2(is, it, ir1, ir)
                         + T.tmr_racs1(is, it, ir1, ir) + T.tcr_sacr1(is, it, ir1, ir));
          prs_rcs[k] = T.tmr_racs2(is, it, ir1, ir) + T.tcr_sacr2(is, it, ir1, ir)
                     - T.tcs_racs1(is, it, ir1, ir) - T.tms_sacr1(is, it, ir1, ir);
          prg_rcs[k] = T.tmr_racs1(is, it, ir1, ir) + T.tcr_sacr1(is, it, ir1, ir)
                     + T.tcs_racs1(is, it, ir1, ir) + T.tms_sacr1(is, it, ir1, ir);
          prr_rcs[k] = fmax2(-rr[k] * odts, prr_rcs[k]);
          prs_rcs[k] = fmax2(-rs[k] * odts, prs_rcs[k]);
          prg_rcs[k] = fmin2((rr[k] + rs[k]) * odts, prg_rcs[k]);
          pnr_rcs[k] = T.tnr_racs1(is, it, ir1, ir) + T.tnr_racs2(is, it, ir1, ir)
                     + T.tnr_sacr1(is, it, ir1, ir) + T.tnr_sacr2(is, it, ir1, ir);
          pnr_rcs[k] = fmin2(nr[k] * odts, pnr_rcs[k]);
        } else {
          prs_rcs[k] = -T.tcs_racs1(is, it, ir1, ir) - T.tms_sacr1(is, it, ir1, ir)
                     + T.tmr_racs2(is, it, ir1, ir) + T.tcr_sacr2(is, it, ir1, ir);
          prs_rcs[k] = fmax2(-rs[k] * odts, prs_rcs[k]);
          prr_rcs[k] = -prs_rcs[k];
        }
      }

      //..Rain collecting graupel.
      if (rg[k] >= 1.e-6) {
        const int ig1 = idx_g1 - 1, ig = idx_g - 1, ir1 = idx_r1 - 1, ir = idx_r - 1;
        if (temp[k] < T_0) {
          prg_rcg[k] = T.tmr_racg(ig1, ig, ir1, ir) + T.tcr_gacr(ig1, ig, ir1, ir);
          prg_rcg[k] = fmin2(rr[k] * odts, prg_rcg[k]);
          prr_rcg[k] = -prg_rcg[k];
          pnr_rcg[k] = T.tnr_racg(ig1, ig, ir1, ir) + T.tnr_gacr(ig1, ig, ir1, ir);
          pnr_rcg[k] = fmin2(nr[k] * odts, pnr_rcg[k]);
        } else {
          prr_rcg[k] = T.tcg_racg(ig1, ig, ir1, ir);
          prr_rcg[k] = fmin2(rg[k] * odts, prr_rcg[k]);
          prg_rcg[k] = -prr_rcg[k];
          pnr_rcg[k] = -5. * T.tnr_gacr(ig1, ig, ir1, ir);
        }
      }
    }

    //..Next IF block handles only those processes below 0C.
    if (temp[k] < T_0) {
      vts_boost[k] = 1.0;
      rate_max = (qv[k] - qvsi[k]) * rho[k] * odts * 0.999;

      // not aerosol aware: 1 per liter
      xni = 1.0 * 1000.;
      int idx_IN = 1;
      if (xni > 1.0) idx_IN = imax2(1, imin2(decade_index(xni, c.niIN2), ntb_IN));
      // freezeH2O tables are stored for this (constant) slice only
      const bool in_ok = (idx_IN == T.idx_IN0);

      //..Freezing of water drops into graupel/cloud ice (Bigg 1953).
      if (rr[k] > 1.e-6) {
        const int ir = idx_r - 1, ir1 = idx_r1 - 1, itc = idx_tc - 1;
        prg_rfz[k] = (in_ok ? T.tpg_qrfz(ir, ir1, itc) : 0.0) * odts;
        pri_rfz[k] = (in_ok ? T.tpi_qrfz(ir, ir1, itc) : 0.0) * odts;
        pni_rfz[k] = (in_ok ? T.tni_qrfz(ir, ir1, itc) : 0.0) * odts;
        pnr_rfz[k] = (in_ok ? T.tnr_qrfz(ir, ir1, itc) : 0.0) * odts;
        pnr_rfz[k] = fmin2(nr[k] * odts, pnr_rfz[k]);
      } else if (rr[k] > R1 && temp[k] < HGFR) {
        pri_rfz[k] = rr[k] * odts;
        pni_rfz[k] = nr[k] * odts;
      }

      if (rc[k] > 1.e-6) {
        int idx_n = nint(1.0 + double(nbc) * Kokkos::log(nc[k] / c.t_Nc1) / c.nic1);
        idx_n = imax2(1, imin2(idx_n, nbc));
        const int ic = idx_c - 1, in = idx_n - 1, itc = idx_tc - 1;
        pri_wfz[k] = (in_ok ? T.tpi_qcfz(ic, in, itc) : 0.0) * odts;
        pri_wfz[k] = fmin2(rc[k] * odts, pri_wfz[k]);
        pni_wfz[k] = (in_ok ? T.tni_qcfz(ic, in, itc) : 0.0) * odts;
        pni_wfz[k] = fmin2(fmin2(nc[k] * odts, pri_wfz[k] / (2. * xm0i)), pni_wfz[k]);
      } else if (rc[k] > R1 && temp[k] < HGFR) {
        pri_wfz[k] = rc[k] * odts;
        pni_wfz[k] = nc[k] * odts;
      }

      //..Deposition nucleation of dust/mineral from DeMott et al (2010)
      if ((ssati[k] >= 0.25) || (ssatw[k] > eps && temp[k] < 253.15)) {
        xnc = fmin2(250.E3, TNO * Kokkos::exp(ATO * (T_0 - temp[k])));
        xni = ni[k] + (pni_rfz[k] + pni_wfz[k]) * dtsave;
        pni_inu[k] = 0.5 * (xnc - xni + Kokkos::fabs(xnc - xni)) * odts;
        pri_inu[k] = fmin2(rate_max, xm0i * pni_inu[k]);
        pni_inu[k] = pri_inu[k] / xm0i;
      }

      //..Freezing of aqueous aerosols (Koop et al 2001): aerosol aware only

      //..Deposition/sublimation of cloud ice (Srivastava & Coen 1992).
      if (L_qi[k]) {
        lami = Kokkos::pow(am_i * c.cig[2] * c.oig1 * ni[k] / ri[k], c.obmi);
        ilami = 1. / lami;
        xDi = fmax2(c.D0i, (bm_i + mu_i + 1.) * ilami);
        xmi = am_i * Kokkos::pow(xDi, bm_i);
        oxmi = 1. / xmi;
        pri_ide[k] = C_cube * t1_subl * diffu[k] * ssati[k] * rvs * c.oig1 * c.cig[5] * ni[k] * ilami;

        if (pri_ide[k] < 0.0) {
          pri_ide[k] = fmax2(fmax2(-ri[k] * odts, pri_ide[k]), rate_max);
          pni_ide[k] = pri_ide[k] * oxmi;
          pni_ide[k] = fmax2(-ni[k] * odts, pni_ide[k]);
        } else {
          pri_ide[k] = fmin2(pri_ide[k], rate_max);
          prs_ide[k] = (1.0 - T.tpi_ide(idx_i - 1, idx_i1 - 1)) * pri_ide[k];
          pri_ide[k] = T.tpi_ide(idx_i - 1, idx_i1 - 1) * pri_ide[k];
        }

        //..Some cloud ice needs to move into the snow category.
        if ((idx_i == ntb_i) || (xDi > 5.0 * D0s)) {
          prs_iau[k] = ri[k] * .99 * odts;
          pni_iau[k] = ni[k] * .95 * odts;
        } else if (xDi < 0.1 * D0s) {
          prs_iau[k] = 0.;
          pni_iau[k] = 0.;
        } else {
          prs_iau[k] = T.tps_iaus(idx_i - 1, idx_i1 - 1) * odts;
          prs_iau[k] = fmin2(ri[k] * .99 * odts, prs_iau[k]);
          pni_iau[k] = T.tni_iaus(idx_i - 1, idx_i1 - 1) * odts;
          pni_iau[k] = fmin2(ni[k] * .95 * odts, pni_iau[k]);
        }
      }

      //..Deposition/sublimation of snow/graupel follows Srivastava & Coen (1992).
      if (L_qs[k]) {
        double C_snow = C_sqrd + (tempc + 1.5) * (C_cube - C_sqrd) / (-30. + 1.5);
        C_snow = fmax2(C_sqrd, fmin2(C_snow, C_cube));
        prs_sde[k] = C_snow * t1_subl * diffu[k] * ssati[k] * rvs
                   * (c.t1_qs_sd * smo1[k] + c.t2_qs_sd * rhof2[k] * vsc2[k] * smof[k]);
        if (prs_sde[k] < 0.) {
          prs_sde[k] = fmax2(fmax2(-rs[k] * odts, prs_sde[k]), rate_max);
        } else {
          prs_sde[k] = fmin2(prs_sde[k], rate_max);
        }
      }

      if (L_qg[k] && ssati[k] < -eps) {
        prg_gde[k] = C_cube * t1_subl * diffu[k] * ssati[k] * rvs * N0_g[k]
                   * (c.t1_qg_sd * Kokkos::pow(ilamg[k], c.cge[10])
                      + c.t2_qg_sd * vsc2[k] * rhof2[k] * Kokkos::pow(ilamg[k], c.cge[11]));
        if (prg_gde[k] < 0.) {
          prg_gde[k] = fmax2(fmax2(-rg[k] * odts, prg_gde[k]), rate_max);
        } else {
          prg_gde[k] = fmin2(prg_gde[k], rate_max);
        }
      }

      //..Snow collecting cloud ice.  In CE, assume Di<<Ds and vti=~0.
      if (L_qi[k]) {
        lami = Kokkos::pow(am_i * c.cig[2] * c.oig1 * ni[k] / ri[k], c.obmi);
        ilami = 1. / lami;
        xDi = fmax2(c.D0i, (bm_i + mu_i + 1.) * ilami);
        xmi = am_i * Kokkos::pow(xDi, bm_i);
        oxmi = 1. / xmi;
        if (rs[k] >= 1.e-6) {
          prs_sci[k] = c.t1_qs_qi * rhof[k] * Ef_si * ri[k] * smoe[k];
          pni_sci[k] = prs_sci[k] * oxmi;
        }

        //..Rain collecting cloud ice.  In CE, assume Di<<Dr and vti=~0.
        if (rr[k] >= 1.e-6 && mvd_r[k] > 4. * xDi) {
          lamr = 1. / ilamr[k];
          pri_rci[k] = rhof[k] * c.t1_qr_qi * Ef_ri * ri[k] * N0_r[k] * (Kokkos::pow(lamr + fv_r, -c.cre[9]));
          pnr_rci[k] = rhof[k] * c.t1_qr_qi * Ef_ri * ni[k] * N0_r[k] * (Kokkos::pow(lamr + fv_r, -c.cre[9]));
          pni_rci[k] = pri_rci[k] * oxmi;
          prr_rci[k] = rhof[k] * c.t2_qr_qi * Ef_ri * ni[k] * N0_r[k] * (Kokkos::pow(lamr + fv_r, -c.cre[8]));
          prr_rci[k] = fmin2(rr[k] * odts, prr_rci[k]);
          prg_rci[k] = pri_rci[k] + prr_rci[k];
        }
      }

      //..Ice multiplication from rime-splinters (Hallet & Mossop 1974).
      if (prg_gcw[k] > eps && tempc > -8.0) {
        double tf = 0.;
        if (tempc >= -5.0 && tempc < -3.0) {
          tf = 0.5 * (-3.0 - tempc);
        } else if (tempc > -8.0 && tempc < -5.0) {
          tf = 0.33333333 * (8.0 + tempc);
        }
        pni_ihm[k] = 3.5E8 * tf * prg_gcw[k];
        pri_ihm[k] = xm0i * pni_ihm[k];
        prs_ihm[k] = prs_scw[k] / (prs_scw[k] + prg_gcw[k]) * pri_ihm[k];
        prg_ihm[k] = prg_gcw[k] / (prs_scw[k] + prg_gcw[k]) * pri_ihm[k];
      }

      //..A portion of rimed snow converts to graupel but some remains snow.
      if (prs_scw[k] > 2.0 * prs_sde[k] && prs_sde[k] > eps) {
        const double r_frac = fmin2(30.0, prs_scw[k] / prs_sde[k]);
        const double g_frac = fmin2(0.95, 0.15 + (r_frac - 2.) * .028);
        vts_boost[k] = fmin2(1.5, 1.1 + (r_frac - 2.) * .014);
        prg_scw[k] = g_frac * prs_scw[k];
        prs_scw[k] = (1. - g_frac) * prs_scw[k];
      }

    } else {

      //..Melt snow and graupel and enhance from collisions with liquid.
      if (L_qs[k]) {
        prr_sml[k] = (tempc * tcond[k] - lvap0 * diffu[k] * delQvs[k])
                   * (c.t1_qs_me * smo1[k] + c.t2_qs_me * rhof2[k] * vsc2[k] * smof[k]);
        if (prr_sml[k] > 0.) {
          prr_sml[k] = prr_sml[k] + 4218. * olfus * tempc * (prr_rcs[k] + prs_scw[k]);
        }
        prr_sml[k] = fmin2(rs[k] * odts, fmax2(0., prr_sml[k]));
        pnr_sml[k] = smo0[k] / rs[k] * prr_sml[k] * Kokkos::pow(10.0, -0.25 * tempc);
        pnr_sml[k] = fmin2(smo0[k] * odts, pnr_sml[k]);

        if (ssati[k] < 0.) {
          prs_sde[k] = C_cube * t1_subl * diffu[k] * ssati[k] * rvs
                     * (c.t1_qs_sd * smo1[k] + c.t2_qs_sd * rhof2[k] * vsc2[k] * smof[k]);
          prs_sde[k] = fmax2(-rs[k] * odts, prs_sde[k]);
        }
      }

      if (L_qg[k]) {
        prr_gml[k] = (tempc * tcond[k] - lvap0 * diffu[k] * delQvs[k]) * N0_g[k]
                   * (c.t1_qg_me * Kokkos::pow(ilamg[k], c.cge[10])
                      + c.t2_qg_me * rhof2[k] * vsc2[k] * Kokkos::pow(ilamg[k], c.cge[11]));
        prr_gml[k] = fmin2(rg[k] * odts, fmax2(0., prr_gml[k]));
        pnr_gml[k] = N0_g[k] * c.cgg[2] * Kokkos::pow(ilamg[k], c.cge[2]) / rg[k] * prr_gml[k]
                   * Kokkos::pow(10.0, -0.5 * tempc);

        if (ssati[k] < 0.) {
          prg_gde[k] = C_cube * t1_subl * diffu[k] * ssati[k] * rvs * N0_g[k]
                     * (c.t1_qg_sd * Kokkos::pow(ilamg[k], c.cge[10])
                        + c.t2_qg_sd * vsc2[k] * rhof2[k] * Kokkos::pow(ilamg[k], c.cge[11]));
          prg_gde[k] = fmax2(-rg[k] * odts, prg_gde[k]);
        }
      }

      if (dt > 120.) {
        prr_rcw[k] = prr_rcw[k] + prs_scw[k] + prg_gcw[k];
        prs_scw[k] = 0.;
        prg_gcw[k] = 0.;
      }
    }
  }

  //..Ensure we do not deplete more hydrometeor species than exists.
  for (int k = kts; k <= kte; ++k) {
    sump = pri_inu[k] + pri_ide[k] + prs_ide[k] + prs_sde[k] + prg_gde[k] + pri_iha[k];
    rate_max = (qv[k] - qvsi[k]) * rho[k] * odts * 0.999;
    if ((sump > eps && sump > rate_max) || (sump < -eps && sump < rate_max)) {
      ratio = rate_max / sump;
      pri_inu[k] = pri_inu[k] * ratio;
      pri_ide[k] = pri_ide[k] * ratio;
      pni_ide[k] = pni_ide[k] * ratio;
      prs_ide[k] = prs_ide[k] * ratio;
      prs_sde[k] = prs_sde[k] * ratio;
      prg_gde[k] = prg_gde[k] * ratio;
      pri_iha[k] = pri_iha[k] * ratio;
    }

    sump = -prr_wau[k] - pri_wfz[k] - prr_rcw[k] - prs_scw[k] - prg_scw[k] - prg_gcw[k];
    rate_max = -rc[k] * odts;
    if (sump < rate_max && L_qc[k]) {
      ratio = rate_max / sump;
      prr_wau[k] = prr_wau[k] * ratio;
      pri_wfz[k] = pri_wfz[k] * ratio;
      prr_rcw[k] = prr_rcw[k] * ratio;
      prs_scw[k] = prs_scw[k] * ratio;
      prg_scw[k] = prg_scw[k] * ratio;
      prg_gcw[k] = prg_gcw[k] * ratio;
    }

    sump = pri_ide[k] - prs_iau[k] - prs_sci[k] - pri_rci[k];
    rate_max = -ri[k] * odts;
    if (sump < rate_max && L_qi[k]) {
      ratio = rate_max / sump;
      pri_ide[k] = pri_ide[k] * ratio;
      prs_iau[k] = prs_iau[k] * ratio;
      prs_sci[k] = prs_sci[k] * ratio;
      pri_rci[k] = pri_rci[k] * ratio;
    }

    sump = -prg_rfz[k] - pri_rfz[k] - prr_rci[k] + prr_rcs[k] + prr_rcg[k];
    rate_max = -rr[k] * odts;
    if (sump < rate_max && L_qr[k]) {
      ratio = rate_max / sump;
      prg_rfz[k] = prg_rfz[k] * ratio;
      pri_rfz[k] = pri_rfz[k] * ratio;
      prr_rci[k] = prr_rci[k] * ratio;
      prr_rcs[k] = prr_rcs[k] * ratio;
      prr_rcg[k] = prr_rcg[k] * ratio;
    }

    sump = prs_sde[k] - prs_ihm[k] - prr_sml[k] + prs_rcs[k];
    rate_max = -rs[k] * odts;
    if (sump < rate_max && L_qs[k]) {
      ratio = rate_max / sump;
      prs_sde[k] = prs_sde[k] * ratio;
      prs_ihm[k] = prs_ihm[k] * ratio;
      prr_sml[k] = prr_sml[k] * ratio;
      prs_rcs[k] = prs_rcs[k] * ratio;
    }

    sump = prg_gde[k] - prg_ihm[k] - prr_gml[k] + prg_rcg[k];
    rate_max = -rg[k] * odts;
    if (sump < rate_max && L_qg[k]) {
      ratio = rate_max / sump;
      prg_gde[k] = prg_gde[k] * ratio;
      prg_ihm[k] = prg_ihm[k] * ratio;
      prr_gml[k] = prr_gml[k] * ratio;
      prg_rcg[k] = prg_rcg[k] * ratio;
    }

    pri_ihm[k] = prs_ihm[k] + prg_ihm[k];
    ratio = fmin2(Kokkos::fabs(prr_rcg[k]), Kokkos::fabs(prg_rcg[k]));
    prr_rcg[k] = ratio * Kokkos::copysign(1.0, prr_rcg[k]);
    prg_rcg[k] = -prr_rcg[k];
    if (temp[k] > T_0) {
      ratio = fmin2(Kokkos::fabs(prr_rcs[k]), Kokkos::fabs(prs_rcs[k]));
      prr_rcs[k] = ratio * Kokkos::copysign(1.0, prr_rcs[k]);
      prs_rcs[k] = -prr_rcs[k];
    }
  }

  //..Calculate tendencies of all species but constrain the number of ice
  //.. to reasonable values.
  for (int k = kts; k <= kte; ++k) {
    orho = 1. / rho[k];
    lfus2 = lsub - lvap[k];

    qvten[k] = qvten[k] + (-pri_inu[k] - pri_iha[k] - pri_ide[k] - prs_ide[k] - prs_sde[k] - prg_gde[k]) * orho;
    qcten[k] = qcten[k] + (-prr_wau[k] - pri_wfz[k] - prr_rcw[k] - prs_scw[k] - prg_scw[k] - prg_gcw[k]) * orho;
    ncten[k] = ncten[k] + (-pnc_wau[k] - pnc_rcw[k] - pni_wfz[k] - pnc_scw[k] - pnc_gcw[k]) * orho;

    xrc = fmax2(R1, (qc1d[k] + qcten[k] * dtsave) * rho[k]);
    xnc = fmax2(2., (nc1d[k] + ncten[k] * dtsave) * rho[k]);
    if (xrc > R1) {
      nu_c = imin2(15, nint(1000.E6 / xnc) + 2);
      lamc = Kokkos::pow(xnc * am_r * c.ccg[2][nu_c] * c.ocg1[nu_c] / rc[k], c.obmr);
      xDc = (bm_r + nu_c + 1.) / lamc;
      if (xDc < D0c) {
        lamc = c.cce[2][nu_c] / D0c;
        xnc = c.ccg[1][nu_c] * c.ocg2[nu_c] * xrc / am_r * Kokkos::pow(lamc, bm_r);
        ncten[k] = (xnc - nc1d[k] * rho[k]) * odts * orho;
      } else if (xDc > D0r * 2.) {
        lamc = c.cce[2][nu_c] / (D0r * 2.);
        xnc = c.ccg[1][nu_c] * c.ocg2[nu_c] * xrc / am_r * Kokkos::pow(lamc, bm_r);
        ncten[k] = (xnc - nc1d[k] * rho[k]) * odts * orho;
      }
    } else {
      ncten[k] = -nc1d[k] * odts;
    }
    xnc = fmax2(0., (nc1d[k] + ncten[k] * dtsave) * rho[k]);
    if (xnc > Nt_c_max) ncten[k] = (Nt_c_max - nc1d[k] * rho[k]) * odts * orho;

    qiten[k] = qiten[k] + (pri_inu[k] + pri_iha[k] + pri_ihm[k] + pri_wfz[k] + pri_rfz[k] + pri_ide[k]
                           - prs_iau[k] - prs_sci[k] - pri_rci[k]) * orho;
    niten[k] = niten[k] + (pni_inu[k] + pni_iha[k] + pni_ihm[k] + pni_wfz[k] + pni_rfz[k] + pni_ide[k]
                           - pni_iau[k] - pni_sci[k] - pni_rci[k]) * orho;

    xri = fmax2(R1, (qi1d[k] + qiten[k] * dtsave) * rho[k]);
    xni = fmax2(R2, (ni1d[k] + niten[k] * dtsave) * rho[k]);
    if (xri > R1) {
      lami = Kokkos::pow(am_i * c.cig[2] * c.oig1 * xni / xri, c.obmi);
      ilami = 1. / lami;
      xDi = (bm_i + mu_i + 1.) * ilami;
      if (xDi < 5.E-6) {
        lami = c.cie[2] / 5.E-6;
        xni = fmin2(9999.E3, c.cig[1] * c.oig2 * xri / am_i * Kokkos::pow(lami, bm_i));
        niten[k] = (xni - ni1d[k] * rho[k]) * odts * orho;
      } else if (xDi > 300.E-6) {
        lami = c.cie[2] / 300.E-6;
        xni = c.cig[1] * c.oig2 * xri / am_i * Kokkos::pow(lami, bm_i);
        niten[k] = (xni - ni1d[k] * rho[k]) * odts * orho;
      }
    } else {
      niten[k] = -ni1d[k] * odts;
    }
    xni = fmax2(0., (ni1d[k] + niten[k] * dtsave) * rho[k]);
    if (xni > 9999.E3) niten[k] = (9999.E3 - ni1d[k] * rho[k]) * odts * orho;

    qrten[k] = qrten[k] + (prr_wau[k] + prr_rcw[k] + prr_sml[k] + prr_gml[k] + prr_rcs[k] + prr_rcg[k]
                           - prg_rfz[k] - pri_rfz[k] - prr_rci[k]) * orho;
    nrten[k] = nrten[k] + (pnr_wau[k] + pnr_sml[k] + pnr_gml[k]
                           - (pnr_rfz[k] + pnr_rcr[k] + pnr_rcg[k] + pnr_rcs[k] + pnr_rci[k] + pni_rfz[k])) * orho;

    xrr = fmax2(R1, (qr1d[k] + qrten[k] * dtsave) * rho[k]);
    xnr = fmax2(R2, (nr1d[k] + nrten[k] * dtsave) * rho[k]);
    if (xrr > R1) {
      lamr = Kokkos::pow(am_r * c.crg[3] * c.org2 * xnr / xrr, c.obmr);
      mvd_r[k] = (3.0 + mu_r + 0.672) / lamr;
      if (mvd_r[k] > 2.5E-3) {
        mvd_r[k] = 2.5E-3;
        lamr = (3.0 + mu_r + 0.672) / mvd_r[k];
        xnr = c.crg[2] * c.org3 * xrr * Kokkos::pow(lamr, bm_r) / am_r;
        nrten[k] = (xnr - nr1d[k] * rho[k]) * odts * orho;
      } else if (mvd_r[k] < D0r * 0.75) {
        mvd_r[k] = D0r * 0.75;
        lamr = (3.0 + mu_r + 0.672) / mvd_r[k];
        xnr = c.crg[2] * c.org3 * xrr * Kokkos::pow(lamr, bm_r) / am_r;
        nrten[k] = (xnr - nr1d[k] * rho[k]) * odts * orho;
      }
    } else {
      qrten[k] = -qr1d[k] * odts;
      nrten[k] = -nr1d[k] * odts;
    }

    qsten[k] = qsten[k] + (prs_iau[k] + prs_sde[k] + prs_sci[k] + prs_scw[k] + prs_rcs[k] + prs_ide[k]
                           - prs_ihm[k] - prr_sml[k]) * orho;
    qgten[k] = qgten[k] + (prg_scw[k] + prg_rfz[k] + prg_gde[k] + prg_rcg[k] + prg_gcw[k] + prg_rci[k]
                           + prg_rcs[k] - prg_ihm[k] - prr_gml[k]) * orho;

    //..Temperature tendency
    const double dry = double(1 - IFDRY);
    if (temp[k] < T_0) {
      if (eqtset == 1) {
        tten[k] = tten[k] + (lsub * ocp[k] * (pri_inu[k] + pri_ide[k] + prs_ide[k] + prs_sde[k] + prg_gde[k] + pri_iha[k])
                             + lfus2 * ocp[k] * (pri_wfz[k] + pri_rfz[k] + prg_rfz[k] + prs_scw[k] + prg_scw[k]
                                                 + prg_gcw[k] + prg_rcs[k] + prs_rcs[k] + prr_rci[k] + prg_rcg[k]))
                            * orho * dry;
      } else if (eqtset == 2) {
        edep = (lsub - Rv * temp[k]) * rcvm[k];
        efrz = lfus2 * rcvm[k];
        tten[k] = tten[k] + (edep * (pri_inu[k] + pri_ide[k] + prs_ide[k] + prs_sde[k] + prg_gde[k] + pri_iha[k])
                             + efrz * (pri_wfz[k] + pri_rfz[k] + prg_rfz[k] + prs_scw[k] + prg_scw[k]
                                       + prg_gcw[k] + prg_rcs[k] + prs_rcs[k] + prr_rci[k] + prg_rcg[k]))
                            * orho * dry;
      }
    } else {
      if (eqtset == 1) {
        tten[k] = tten[k] + (lfus * ocp[k] * (-prr_sml[k] - prr_gml[k] - prr_rcg[k] - prr_rcs[k])
                             + lsub * ocp[k] * (prs_sde[k] + prg_gde[k])) * orho * dry;
      } else if (eqtset == 2) {
        efrz = lfus * rcvm[k];
        edep = (lsub - Rv * temp[k]) * rcvm[k];
        tten[k] = tten[k] + (efrz * (-prr_sml[k] - prr_gml[k] - prr_rcg[k] - prr_rcs[k])
                             + edep * (prs_sde[k] + prg_gde[k])) * orho * dry;
      }
    }
  }

  //..Update variables for TAU+1 before condensation & sedimention.
  for (int k = kts; k <= kte; ++k) {
    temp[k] = t1d[k] + dt * tten[k];
    otemp = 1. / temp[k];
    tempc = temp[k] - 273.15;
    qv[k] = fmax2(1.E-10, qv1d[k] + dt * qvten[k]);
    if (eqtset == 1) {
      rho[k] = 0.622 * pres[k] / (R * temp[k] * (qv[k] + 0.622));
    } else if (eqtset == 2) {
      pres[k] = rho[k] * (R + Rv * qv[k]) * temp[k];
    }
    rhof[k] = Kokkos::sqrt(rho_not / rho[k]);
    rhof2[k] = Kokkos::sqrt(rhof[k]);
    qvs[k] = rslf(pres[k], temp[k]);
    ssatw[k] = qv[k] / qvs[k] - 1.;
    if (Kokkos::fabs(ssatw[k]) < eps) ssatw[k] = 0.0;
    diffu[k] = 2.11E-5 * Kokkos::pow(temp[k] / 273.15, 1.94) * (101325. / pres[k]);
    if (tempc >= 0.0) {
      visco[k] = (1.718 + 0.0049 * tempc) * 1.0E-5;
    } else {
      visco[k] = (1.718 + 0.0049 * tempc - 1.2E-5 * tempc * tempc) * 1.0E-5;
    }
    vsc2[k] = Kokkos::sqrt(rho[k] / visco[k]);
    lvap[k] = lvap0 + (2106.0 - 4218.0) * tempc;
    tcond[k] = (5.69 + 0.0168 * tempc) * 1.0E-5 * 418.936;
    ocp[k] = 1. / (Cp * (1. + 0.887 * qv[k]));
    lvt2[k] = lvap[k] * lvap[k] * ocp[k] * oRv * otemp * otemp;
  }

  for (int k = kts; k <= kte; ++k) {
    if ((qc1d[k] + qcten[k] * dt) > R1) {
      rc[k] = (qc1d[k] + qcten[k] * dt) * rho[k];
      nc[k] = fmax2(2., fmin2((nc1d[k] + ncten[k] * dt) * rho[k], Nt_c_max));
      if (!is_aerosol_aware) nc[k] = Nt_c;
      L_qc[k] = 1;
    } else {
      rc[k] = R1;
      nc[k] = 2.;
      L_qc[k] = 0;
    }

    if ((qi1d[k] + qiten[k] * dt) > R1) {
      ri[k] = (qi1d[k] + qiten[k] * dt) * rho[k];
      ni[k] = fmax2(R2, (ni1d[k] + niten[k] * dt) * rho[k]);
      L_qi[k] = 1;
    } else {
      ri[k] = R1;
      ni[k] = R2;
      L_qi[k] = 0;
    }

    if ((qr1d[k] + qrten[k] * dt) > R1) {
      rr[k] = (qr1d[k] + qrten[k] * dt) * rho[k];
      nr[k] = fmax2(R2, (nr1d[k] + nrten[k] * dt) * rho[k]);
      L_qr[k] = 1;
      lamr = Kokkos::pow(am_r * c.crg[3] * c.org2 * nr[k] / rr[k], c.obmr);
      mvd_r[k] = (3.0 + mu_r + 0.672) / lamr;
      if (mvd_r[k] > 2.5E-3) {
        mvd_r[k] = 2.5E-3;
        lamr = (3.0 + mu_r + 0.672) / mvd_r[k];
        nr[k] = c.crg[2] * c.org3 * rr[k] * Kokkos::pow(lamr, bm_r) / am_r;
      } else if (mvd_r[k] < D0r * 0.75) {
        mvd_r[k] = D0r * 0.75;
        lamr = (3.0 + mu_r + 0.672) / mvd_r[k];
        nr[k] = c.crg[2] * c.org3 * rr[k] * Kokkos::pow(lamr, bm_r) / am_r;
      }
    } else {
      rr[k] = R1;
      nr[k] = R2;
      L_qr[k] = 0;
    }

    if ((qs1d[k] + qsten[k] * dt) > R1) {
      rs[k] = (qs1d[k] + qsten[k] * dt) * rho[k];
      L_qs[k] = 1;
    } else {
      rs[k] = R1;
      L_qs[k] = 0;
    }

    if ((qg1d[k] + qgten[k] * dt) > R1) {
      rg[k] = (qg1d[k] + qgten[k] * dt) * rho[k];
      L_qg[k] = 1;
    } else {
      rg[k] = R1;
      L_qg[k] = 0;
    }
  }

  //..With tendency-updated mixing ratios, recalculate snow moments and
  //.. intercepts/slopes of graupel and rain.
  for (int k = kts; k <= kte; ++k) {
    smo2[k] = 0.; smob[k] = 0.; smoc[k] = 0.; smod[k] = 0.;
  }
  for (int k = kts; k <= kte; ++k) {
    if (!L_qs[k]) continue;
    tc0 = fmin2(-0.1, temp[k] - 273.15);
    smob[k] = rs[k] * c.oams;
    smo2[k] = smob[k];
    for (int which = 0; which < 2; ++which) {
      const double x = which == 0 ? c.cse[1] : c.cse[14];
      loga_ = sa[1] + sa[2] * tc0 + sa[3] * x + sa[4] * tc0 * x + sa[5] * tc0 * tc0 + sa[6] * x * x
            + sa[7] * tc0 * tc0 * x + sa[8] * tc0 * x * x + sa[9] * tc0 * tc0 * tc0 + sa[10] * x * x * x;
      a_ = Kokkos::pow(10.0, loga_);
      b_ = sb[1] + sb[2] * tc0 + sb[3] * x + sb[4] * tc0 * x + sb[5] * tc0 * tc0 + sb[6] * x * x
         + sb[7] * tc0 * tc0 * x + sb[8] * tc0 * x * x + sb[9] * tc0 * tc0 * tc0 + sb[10] * x * x * x;
      const double mo = a_ * Kokkos::pow(smo2[k], b_);
      if (which == 0) smoc[k] = mo;
      else smod[k] = mo;
    }
  }

  for (int k = kte; k >= kts; --k) {
    const double ygra1 = Kokkos::log10(fmax2(1.E-9, rg[k]));
    double zans1 = 3.0 + 2. / 7. * (ygra1 + 8.);
    zans1 = fmax2(2., fmin2(zans1, 6.));
    N0_exp = Kokkos::pow(10., zans1);
    lam_exp = Kokkos::pow(N0_exp * am_g * c.cgg[1] / rg[k], c.oge1);
    lamg = lam_exp * Kokkos::pow(c.cgg[3] * c.ogg2 * c.ogg1, c.obmg);
    ilamg[k] = 1. / lamg;
    N0_g[k] = N0_exp / (c.cgg[2] * lam_exp) * Kokkos::pow(lamg, c.cge[2]);
  }

  for (int k = kte; k >= kts; --k) {
    lamr = Kokkos::pow(am_r * c.crg[3] * c.org2 * nr[k] / rr[k], c.obmr);
    ilamr[k] = 1. / lamr;
    mvd_r[k] = (3.0 + mu_r + 0.672) / lamr;
    N0_r[k] = nr[k] * c.org2 * Kokkos::pow(lamr, c.cre[2]);
  }

  //..Cloud water condensation and evaporation.
  for (int k = kts; k <= kte; ++k) {
    orho = 1. / rho[k];
    if ((ssatw[k] > eps) || (ssatw[k] < -eps && L_qc[k])) {
      double clap = (qv[k] - qvs[k]) / (1. + lvt2[k] * qvs[k]);
      for (int n = 1; n <= 3; ++n) {
        const double fcd = qvs[k] * Kokkos::exp(lvt2[k] * clap) - qv[k] + clap;
        const double dfcd = qvs[k] * lvt2[k] * Kokkos::exp(lvt2[k] * clap) + 1.;
        clap = clap - fcd / dfcd;
      }
      xrc = rc[k] + clap * rho[k];
      xnc = 0.;
      if (xrc > R1) {
        prw_vcd[k] = clap * odt;
        if (clap > eps) {
          xnc = Nt_c;
          pnc_wcd[k] = 0.5 * (xnc - nc[k] + Kokkos::fabs(xnc - nc[k])) * odts * orho;
        }
        // (evaporation branch with droplet number: aerosol aware only)
      } else {
        prw_vcd[k] = -rc[k] * orho * odt;
        pnc_wcd[k] = -nc[k] * orho * odt;
      }

      qvten[k] = qvten[k] - prw_vcd[k];
      qcten[k] = qcten[k] + prw_vcd[k];
      ncten[k] = ncten[k] + pnc_wcd[k];
      if (eqtset == 1) {
        tten[k] = tten[k] + lvap[k] * ocp[k] * prw_vcd[k] * double(1 - IFDRY);
      } else if (eqtset == 2) {
        econd = (lvap[k] - Rv * temp[k]) * rcvm[k];
        tten[k] = tten[k] + econd * prw_vcd[k] * double(1 - IFDRY);
      }
      rc[k] = fmax2(R1, (qc1d[k] + dt * qcten[k]) * rho[k]);
      if (rc[k] == R1) L_qc[k] = 0;
      nc[k] = fmax2(2., fmin2((nc1d[k] + ncten[k] * dt) * rho[k], Nt_c_max));
      if (!is_aerosol_aware) nc[k] = Nt_c;
      qv[k] = fmax2(1.E-10, qv1d[k] + dt * qvten[k]);
      temp[k] = t1d[k] + dt * tten[k];
      if (eqtset == 1) {
        rho[k] = 0.622 * pres[k] / (R * temp[k] * (qv[k] + 0.622));
      } else if (eqtset == 2) {
        pres[k] = rho[k] * (R + Rv * qv[k]) * temp[k];
      }
      qvs[k] = rslf(pres[k], temp[k]);
      ssatw[k] = qv[k] / qvs[k] - 1.;
    }
  }

  //.. If still subsaturated, allow rain to evaporate.
  for (int k = kts; k <= kte; ++k) {
    if ((ssatw[k] < -eps) && L_qr[k] && (!(prw_vcd[k] > 0.))) {
      tempc = temp[k] - 273.15;
      otemp = 1. / temp[k];
      orho = 1. / rho[k];
      rhof[k] = Kokkos::sqrt(rho_not * orho);
      rhof2[k] = Kokkos::sqrt(rhof[k]);
      diffu[k] = 2.11E-5 * Kokkos::pow(temp[k] / 273.15, 1.94) * (101325. / pres[k]);
      if (tempc >= 0.0) {
        visco[k] = (1.718 + 0.0049 * tempc) * 1.0E-5;
      } else {
        visco[k] = (1.718 + 0.0049 * tempc - 1.2E-5 * tempc * tempc) * 1.0E-5;
      }
      vsc2[k] = Kokkos::sqrt(rho[k] / visco[k]);
      lvap[k] = lvap0 + (2106.0 - 4218.0) * tempc;
      tcond[k] = (5.69 + 0.0168 * tempc) * 1.0E-5 * 418.936;
      ocp[k] = 1. / (Cp * (1. + 0.887 * qv[k]));

      rvs = rho[k] * qvs[k];
      rvs_p = rvs * otemp * (lvap[k] * otemp * oRv - 1.);
      rvs_pp = rvs * (otemp * (lvap[k] * otemp * oRv - 1.) * otemp * (lvap[k] * otemp * oRv - 1.)
                      + (-2. * lvap[k] * otemp * otemp * otemp * oRv) + otemp * otemp);
      gamsc = lvap[k] * diffu[k] / tcond[k] * rvs_p;
      alphsc = 0.5 * (gamsc / (1. + gamsc)) * (gamsc / (1. + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p;
      alphsc = fmax2(1.E-9, alphsc);
      xsat = fmin2(-1.E-9, ssatw[k]);
      t1_evap = 2. * PI * (1.0 - alphsc * xsat + 2. * alphsc * alphsc * xsat * xsat
                           - 5. * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1. + gamsc);

      lamr = 1. / ilamr[k];
      if (qv[k] / qvs[k] < 0.95 && rr[k] * orho <= 1.E-8) {
        prv_rev[k] = rr[k] * orho * odts;
      } else {
        prv_rev[k] = t1_evap * diffu[k] * (-ssatw[k]) * N0_r[k] * rvs
                   * (c.t1_qr_ev * Kokkos::pow(ilamr[k], c.cre[10])
                      + c.t2_qr_ev * vsc2[k] * rhof2[k] * (Kokkos::pow(lamr + 0.5 * fv_r, -c.cre[11])));
        rate_max = fmin2((rr[k] * orho * odts), (qvs[k] - qv[k]) * odts);
        prv_rev[k] = fmin2(rate_max, prv_rev[k] * orho);

        if (prr_gml[k] > 0.0) {
          const double eva_factor = fmin2(1.0, 0.01 + (0.99 - 0.01) * (tempc / 20.0));
          prv_rev[k] = prv_rev[k] * eva_factor;
        }
      }

      pnr_rev[k] = fmin2(nr[k] * 0.99 * orho * odts, prv_rev[k] * nr[k] / rr[k]);

      qrten[k] = qrten[k] - prv_rev[k];
      qvten[k] = qvten[k] + prv_rev[k];
      nrten[k] = nrten[k] - pnr_rev[k];
      if (eqtset == 1) {
        tten[k] = tten[k] - lvap[k] * ocp[k] * prv_rev[k] * double(1 - IFDRY);
      } else if (eqtset == 2) {
        econd = (lvap[k] - Rv * temp[k]) * rcvm[k];
        tten[k] = tten[k] - econd * prv_rev[k] * double(1 - IFDRY);
      }

      rr[k] = fmax2(R1, (qr1d[k] + dt * qrten[k]) * rho[k]);
      qv[k] = fmax2(1.E-10, qv1d[k] + dt * qvten[k]);
      nr[k] = fmax2(R2, (nr1d[k] + dt * nrten[k]) * rho[k]);
      temp[k] = t1d[k] + dt * tten[k];
      if (eqtset == 1) {
        rho[k] = 0.622 * pres[k] / (R * temp[k] * (qv[k] + 0.622));
      } else if (eqtset == 2) {
        pres[k] = rho[k] * (R + Rv * qv[k]) * temp[k];
      }
    }
  }

  //..Find max terminal fallspeed and the sedimentation substeps.
  int nstep = 0;
  double onstep[6] = {0, 1.0, 1.0, 1.0, 1.0, 1.0};
  int ksed1[6] = {0, 1, 1, 1, 1, 1};
  for (int k = kte + 1; k >= kts; --k) {
    vtrk[k] = 0.; vtnrk[k] = 0.; vtik[k] = 0.; vtnik[k] = 0.; vtsk[k] = 0.; vtgk[k] = 0.;
    vtck[k] = 0.; vtnck[k] = 0.;
  }
  auto any = [&](const int* L) {
    for (int k = kts; k <= kte; ++k)
      if (L[k]) return true;
    return false;
  };

  if (any(L_qr)) {
    for (int k = kte; k >= kts; --k) {
      double vtr = 0.;
      rhof[k] = Kokkos::sqrt(rho_not / rho[k]);
      if (rr[k] > R1) {
        lamr = Kokkos::pow(am_r * c.crg[3] * c.org2 * nr[k] / rr[k], c.obmr);
        vtr = rhof[k] * av_r * c.crg[6] * c.org3 * Kokkos::pow(lamr, c.cre[3])
            * (Kokkos::pow(lamr + fv_r, -c.cre[6]));
        vtrk[k] = vtr;
        vtr = rhof[k] * av_r * c.crg[7] / c.crg[12] * Kokkos::pow(lamr, c.cre[12])
            * (Kokkos::pow(lamr + fv_r, -c.cre[7]));
        vtnrk[k] = vtr;
      } else {
        vtrk[k] = vtrk[k + 1];
        vtnrk[k] = vtnrk[k + 1];
      }
      if (fmax2(vtrk[k], vtnrk[k]) > 1.E-3) {
        ksed1[1] = imax2(ksed1[1], k);
        const double delta_tp = dzq[k] / (fmax2(vtrk[k], vtnrk[k]));
        nstep = imax2(nstep, int(dt / delta_tp + 1.));
      }
    }
    if (ksed1[1] == kte) ksed1[1] = kte - 1;
    if (nstep > 0) onstep[1] = 1. / double(nstep);
  }

  if (any(L_qc)) {
    double hgt_agl = 0.;
    for (int k = kts; k <= kte - 1; ++k) {
      if (rc[k] > R2) ksed1[5] = k;
      hgt_agl = hgt_agl + dzq[k];
      if (hgt_agl > 500.0) break;
    }
    for (int k = ksed1[5]; k >= kts; --k) {
      double vtc = 0.;
      if (rc[k] > R1 && w1d[k] < 1.E-1) {
        nu_c = imin2(15, nint(1000.E6 / nc[k]) + 2);
        lamc = Kokkos::pow(nc[k] * am_r * c.ccg[2][nu_c] * c.ocg1[nu_c] / rc[k], c.obmr);
        ilamc = 1. / lamc;
        vtc = rhof[k] * av_c * c.ccg[5][nu_c] * c.ocg2[nu_c] * (ilamc * ilamc);
        vtck[k] = vtc;
        vtc = rhof[k] * av_c * c.ccg[4][nu_c] * c.ocg1[nu_c] * (ilamc * ilamc);
        vtnck[k] = vtc;
      }
    }
  }

  if (any(L_qi)) {
    nstep = 0;
    for (int k = kte; k >= kts; --k) {
      double vti = 0.;
      if (ri[k] > R1) {
        lami = Kokkos::pow(am_i * c.cig[2] * c.oig1 * ni[k] / ri[k], c.obmi);
        ilami = 1. / lami;
        vti = rhof[k] * av_i * c.cig[3] * c.oig2 * ilami;
        vtik[k] = vti;
        vti = rhof[k] * av_i * c.cig[6] / c.cig[7] * ilami;
        vtnik[k] = vti;
      } else {
        vtik[k] = vtik[k + 1];
        vtnik[k] = vtnik[k + 1];
      }
      if (vtik[k] > 1.E-3) {
        ksed1[2] = imax2(ksed1[2], k);
        const double delta_tp = dzq[k] / vtik[k];
        nstep = imax2(nstep, int(dt / delta_tp + 1.));
      }
    }
    if (ksed1[2] == kte) ksed1[2] = kte - 1;
    if (nstep > 0) onstep[2] = 1. / double(nstep);
  }

  if (any(L_qs)) {
    nstep = 0;
    for (int k = kte; k >= kts; --k) {
      double vts = 0.;
      if (rs[k] > R1) {
        xDs = smoc[k] / smob[k];
        const double Mrat = 1. / xDs;
        double ils1 = 1. / (Mrat * Lam0 + fv_s);
        double ils2 = 1. / (Mrat * Lam1 + fv_s);
        const double t1_vts = Kap0 * c.csg[4] * Kokkos::pow(ils1, c.cse[4]);
        const double t2_vts = Kap1 * Kokkos::pow(Mrat, mu_s) * c.csg[10] * Kokkos::pow(ils2, c.cse[10]);
        ils1 = 1. / (Mrat * Lam0);
        ils2 = 1. / (Mrat * Lam1);
        const double t3_vts = Kap0 * c.csg[1] * Kokkos::pow(ils1, c.cse[1]);
        const double t4_vts = Kap1 * Kokkos::pow(Mrat, mu_s) * c.csg[7] * Kokkos::pow(ils2, c.cse[7]);
        vts = rhof[k] * av_s * (t1_vts + t2_vts) / (t3_vts + t4_vts);
        if (temp[k] > (T_0 + 0.1)) {
          const double SR = rs[k] / (rs[k] + rr[k]);
          vtsk[k] = vts * SR + (1. - SR) * vtrk[k];
        } else {
          vtsk[k] = vts * vts_boost[k];
        }
      } else {
        vtsk[k] = vtsk[k + 1];
      }
      if (vtsk[k] > 1.E-3) {
        ksed1[3] = imax2(ksed1[3], k);
        const double delta_tp = dzq[k] / vtsk[k];
        nstep = imax2(nstep, int(dt / delta_tp + 1.));
      }
    }
    if (ksed1[3] == kte) ksed1[3] = kte - 1;
    if (nstep > 0) onstep[3] = 1. / double(nstep);
  }

  if (any(L_qg)) {
    nstep = 0;
    for (int k = kte; k >= kts; --k) {
      double vtg = 0.;
      if (rg[k] > R1) {
        vtg = rhof[k] * av_g * c.cgg[6] * c.ogg3 * Kokkos::pow(ilamg[k], bv_g);
        if (temp[k] > T_0) {
          vtgk[k] = fmax2(vtg, vtrk[k]);
        } else {
          vtgk[k] = vtg;
        }
      } else {
        vtgk[k] = vtgk[k + 1];
      }
      if (vtgk[k] > 1.E-3) {
        ksed1[4] = imax2(ksed1[4], k);
        const double delta_tp = dzq[k] / vtgk[k];
        nstep = imax2(nstep, int(dt / delta_tp + 1.));
      }
    }
    if (ksed1[4] == kte) ksed1[4] = kte - 1;
    if (nstep > 0) onstep[4] = 1. / double(nstep);
  }

  //..Sedimentation
  if (any(L_qr)) {
    nstep = nint(1. / onstep[1]);
    for (int n = 1; n <= nstep; ++n) {
      for (int k = kte; k >= kts; --k) {
        sed_r[k] = vtrk[k] * rr[k];
        sed_n[k] = vtnrk[k] * nr[k];
      }
      int k = kte;
      double odzq = 1. / dzq[k];
      orho = 1. / rho[k];
      qrten[k] = qrten[k] - sed_r[k] * odzq * onstep[1] * orho;
      nrten[k] = nrten[k] - sed_n[k] * odzq * onstep[1] * orho;
      rr[k] = fmax2(R1, rr[k] - sed_r[k] * odzq * dt * onstep[1]);
      nr[k] = fmax2(R2, nr[k] - sed_n[k] * odzq * dt * onstep[1]);
      for (k = ksed1[1]; k >= kts; --k) {
        odzq = 1. / dzq[k];
        orho = 1. / rho[k];
        qrten[k] = qrten[k] + (sed_r[k + 1] - sed_r[k]) * odzq * onstep[1] * orho;
        nrten[k] = nrten[k] + (sed_n[k + 1] - sed_n[k]) * odzq * onstep[1] * orho;
        rr[k] = fmax2(R1, rr[k] + (sed_r[k + 1] - sed_r[k]) * odzq * dt * onstep[1]);
        nr[k] = fmax2(R2, nr[k] + (sed_n[k + 1] - sed_n[k]) * odzq * dt * onstep[1]);
      }
      if (rr[kts] > R1 * 1000.) pptrain = pptrain + sed_r[kts] * dt * onstep[1];
    }
  }

  if (any(L_qc)) {
    for (int k = kte; k >= kts; --k) {
      sed_c[k] = vtck[k] * rc[k];
      sed_n[k] = vtnck[k] * nc[k];
    }
    for (int k = ksed1[5]; k >= kts; --k) {
      const double odzq = 1. / dzq[k];
      orho = 1. / rho[k];
      qcten[k] = qcten[k] + (sed_c[k + 1] - sed_c[k]) * odzq * orho;
      ncten[k] = ncten[k] + (sed_n[k + 1] - sed_n[k]) * odzq * orho;
      rc[k] = fmax2(R1, rc[k] + (sed_c[k + 1] - sed_c[k]) * odzq * dt);
      nc[k] = fmax2(10., nc[k] + (sed_n[k + 1] - sed_n[k]) * odzq * dt);
    }
  }

  if (any(L_qi)) {
    nstep = nint(1. / onstep[2]);
    for (int n = 1; n <= nstep; ++n) {
      for (int k = kte; k >= kts; --k) {
        sed_i[k] = vtik[k] * ri[k];
        sed_n[k] = vtnik[k] * ni[k];
      }
      int k = kte;
      double odzq = 1. / dzq[k];
      orho = 1. / rho[k];
      qiten[k] = qiten[k] - sed_i[k] * odzq * onstep[2] * orho;
      niten[k] = niten[k] - sed_n[k] * odzq * onstep[2] * orho;
      ri[k] = fmax2(R1, ri[k] - sed_i[k] * odzq * dt * onstep[2]);
      ni[k] = fmax2(R2, ni[k] - sed_n[k] * odzq * dt * onstep[2]);
      for (k = ksed1[2]; k >= kts; --k) {
        odzq = 1. / dzq[k];
        orho = 1. / rho[k];
        qiten[k] = qiten[k] + (sed_i[k + 1] - sed_i[k]) * odzq * onstep[2] * orho;
        niten[k] = niten[k] + (sed_n[k + 1] - sed_n[k]) * odzq * onstep[2] * orho;
        ri[k] = fmax2(R1, ri[k] + (sed_i[k + 1] - sed_i[k]) * odzq * dt * onstep[2]);
        ni[k] = fmax2(R2, ni[k] + (sed_n[k + 1] - sed_n[k]) * odzq * dt * onstep[2]);
      }
      if (ri[kts] > R1 * 1000.) pptice = pptice + sed_i[kts] * dt * onstep[2];
    }
  }

  if (any(L_qs)) {
    nstep = nint(1. / onstep[3]);
    for (int n = 1; n <= nstep; ++n) {
      for (int k = kte; k >= kts; --k) sed_s[k] = vtsk[k] * rs[k];
      int k = kte;
      double odzq = 1. / dzq[k];
      orho = 1. / rho[k];
      qsten[k] = qsten[k] - sed_s[k] * odzq * onstep[3] * orho;
      rs[k] = fmax2(R1, rs[k] - sed_s[k] * odzq * dt * onstep[3]);
      for (k = ksed1[3]; k >= kts; --k) {
        odzq = 1. / dzq[k];
        orho = 1. / rho[k];
        qsten[k] = qsten[k] + (sed_s[k + 1] - sed_s[k]) * odzq * onstep[3] * orho;
        rs[k] = fmax2(R1, rs[k] + (sed_s[k + 1] - sed_s[k]) * odzq * dt * onstep[3]);
      }
      if (rs[kts] > R1 * 1000.) pptsnow = pptsnow + sed_s[kts] * dt * onstep[3];
    }
  }

  if (any(L_qg)) {
    nstep = nint(1. / onstep[4]);
    for (int n = 1; n <= nstep; ++n) {
      for (int k = kte; k >= kts; --k) sed_g[k] = vtgk[k] * rg[k];
      int k = kte;
      double odzq = 1. / dzq[k];
      orho = 1. / rho[k];
      qgten[k] = qgten[k] - sed_g[k] * odzq * onstep[4] * orho;
      rg[k] = fmax2(R1, rg[k] - sed_g[k] * odzq * dt * onstep[4]);
      for (k = ksed1[4]; k >= kts; --k) {
        odzq = 1. / dzq[k];
        orho = 1. / rho[k];
        qgten[k] = qgten[k] + (sed_g[k + 1] - sed_g[k]) * odzq * onstep[4] * orho;
        rg[k] = fmax2(R1, rg[k] + (sed_g[k + 1] - sed_g[k]) * odzq * dt * onstep[4]);
      }
      if (rg[kts] > R1 * 1000.) pptgraul = pptgraul + sed_g[kts] * dt * onstep[4];
    }
  }

  //.. Instantly melt any cloud ice into cloud water if above 0C and
  //.. instantly freeze any cloud water found below HGFR.
  for (int k = kts; k <= kte; ++k) {
    xri = fmax2(0.0, qi1d[k] + qiten[k] * dt);
    if ((temp[k] > T_0) && (xri > 0.0)) {
      qcten[k] = qcten[k] + xri * odt;
      ncten[k] = ncten[k] + ni1d[k] * odt;
      qiten[k] = qiten[k] - xri * odt;
      niten[k] = -ni1d[k] * odt;
      if (eqtset == 1) {
        tten[k] = tten[k] - lfus * ocp[k] * xri * odt * double(1 - IFDRY);
      } else if (eqtset == 2) {
        efrz = lfus * rcvm[k];
        tten[k] = tten[k] - efrz * xri * odt * double(1 - IFDRY);
      }
    }

    xrc = fmax2(0.0, qc1d[k] + qcten[k] * dt);
    if ((temp[k] < HGFR) && (xrc > 0.0)) {
      lfus2 = lsub - lvap[k];
      xnc = nc1d[k] + ncten[k] * dt;
      qiten[k] = qiten[k] + xrc * odt;
      niten[k] = niten[k] + xnc * odt;
      qcten[k] = qcten[k] - xrc * odt;
      ncten[k] = ncten[k] - xnc * odt;
      if (eqtset == 1) {
        tten[k] = tten[k] + lfus2 * ocp[k] * xrc * odt * double(1 - IFDRY);
      } else if (eqtset == 2) {
        efrz = lfus * rcvm[k];
        tten[k] = tten[k] + efrz * xrc * odt * double(1 - IFDRY);
      }
    }
  }

  //.. All tendencies computed, apply and pass back final values to parent.
  for (int k = kts; k <= kte; ++k) {
    t1d[k] = t1d[k] + tten[k] * dt;
    qv1d[k] = fmax2(1.E-10, qv1d[k] + qvten[k] * dt);
    qc1d[k] = qc1d[k] + qcten[k] * dt;
    nc1d[k] = fmax2(2. / rho[k], fmin2(nc1d[k] + ncten[k] * dt, Nt_c_max));

    if (qc1d[k] <= R1) {
      qc1d[k] = 0.0;
      nc1d[k] = 0.0;
    } else {
      nu_c = imin2(15, nint(1000.E6 / (nc1d[k] * rho[k])) + 2);
      lamc = Kokkos::pow(am_r * c.ccg[2][nu_c] * c.ocg1[nu_c] * nc1d[k] / qc1d[k], c.obmr);
      xDc = (bm_r + nu_c + 1.) / lamc;
      if (xDc < D0c) {
        lamc = c.cce[2][nu_c] / D0c;
      } else if (xDc > D0r * 2.) {
        lamc = c.cce[2][nu_c] / (D0r * 2.);
      }
      nc1d[k] = fmin2(c.ccg[1][nu_c] * c.ocg2[nu_c] * qc1d[k] / am_r * Kokkos::pow(lamc, bm_r),
                      Nt_c_max / rho[k]);
    }

    qi1d[k] = qi1d[k] + qiten[k] * dt;
    ni1d[k] = fmax2(R2 / rho[k], ni1d[k] + niten[k] * dt);
    if (qi1d[k] <= R1) {
      qi1d[k] = 0.0;
      ni1d[k] = 0.0;
    } else {
      lami = Kokkos::pow(am_i * c.cig[2] * c.oig1 * ni1d[k] / qi1d[k], c.obmi);
      ilami = 1. / lami;
      xDi = (bm_i + mu_i + 1.) * ilami;
      if (xDi < 5.E-6) {
        lami = c.cie[2] / 5.E-6;
      } else if (xDi > 300.E-6) {
        lami = c.cie[2] / 300.E-6;
      }
      ni1d[k] = fmin2(c.cig[1] * c.oig2 * qi1d[k] / am_i * Kokkos::pow(lami, bm_i), 9999.E3 / rho[k]);
    }
    qr1d[k] = qr1d[k] + qrten[k] * dt;
    nr1d[k] = fmax2(R2 / rho[k], nr1d[k] + nrten[k] * dt);
    if (qr1d[k] <= R1) {
      qr1d[k] = 0.0;
      nr1d[k] = 0.0;
    } else {
      lamr = Kokkos::pow(am_r * c.crg[3] * c.org2 * nr1d[k] / qr1d[k], c.obmr);
      mvd_r[k] = (3.0 + mu_r + 0.672) / lamr;
      if (mvd_r[k] > 2.5E-3) {
        mvd_r[k] = 2.5E-3;
      } else if (mvd_r[k] < D0r * 0.75) {
        mvd_r[k] = D0r * 0.75;
      }
      lamr = (3.0 + mu_r + 0.672) / mvd_r[k];
      nr1d[k] = c.crg[2] * c.org3 * qr1d[k] * Kokkos::pow(lamr, bm_r) / am_r;
    }
    qs1d[k] = qs1d[k] + qsten[k] * dt;
    if (qs1d[k] <= R1) qs1d[k] = 0.0;
    qg1d[k] = qg1d[k] + qgten[k] * dt;
    if (qg1d[k] <= R1) qg1d[k] = 0.0;
  }
}

#undef TH_COL
#undef TH_LCOL

} // namespace

//---------------------------------------------------------------------------
Options Options::from_namelist(const Namelist& nl) {
  if (nl.geti("imoist", 0) != 1 || nl.geti("ptype", 0) != 3)
    throw std::runtime_error("thompson::Options: requires imoist=1 and ptype=3");
  if (nl.geti("radopt", 0) == 2)
    throw std::runtime_error("thompson: radopt=2 (effective radii, calc_effectRad) is not ported");
  Options o;
  o.eqtset = std::min(std::max(nl.geti("eqtset", 2), 1), 2);
  const int psolver = nl.geti("psolver", 3);
  if (psolver >= 4 && psolver <= 7 && o.eqtset == 2) o.eqtset = 1;
  o.nt_c = nl.getr("nt_c", 100.0);
  if (o.nt_c <= 1.0e-6) throw std::runtime_error("thompson: nt_c is too small (param.F)");
  if (nl.geti("efall", 0) >= 1) throw std::runtime_error("thompson: efall is not used with ptype=3");
  return o;
}

void Work::allocate(const Bounds& B, const Options& opt) {
  auto s3 = [&](const char* n) { return FArray3(n, {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke}); };
  tmp = s3("thompson_tmp");
  thten = s3("thompson_thten");
  qvold = s3("thompson_qvold");
  dzq = s3("thompson_dzq");
  vtc = s3("thompson_vtc");
  vtr = s3("thompson_vtr");
  vti = s3("thompson_vti");
  vts = s3("thompson_vts");
  vtg = s3("thompson_vtg");
  prate = FArray2("prate", {B.ib, B.jb}, {B.ie, B.je});
  const size_t ncol = size_t(B.ni) * B.nj, nkk = size_t(B.nk) + 2;
  scr = Kokkos::View<double*, Device>("thompson_scr", ncol * NVAR * nkk);
  iscr = Kokkos::View<int*, Device>("thompson_iscr", ncol * NIVAR * nkk);
  tab = DevTables::upload(*get(opt.nt_c));
}

//---------------------------------------------------------------------------
void mp_gt_driver(const Options& opt, const Bounds& b, Real dt, const FArray3& t3d, const FArray3& p,
                  const FArray3& w, const FArray3& dz, const FArray4& q3d, const FArray3& rain,
                  const Work& wk) {
  const int ni = b.ni, nk = b.nk, nrain = b.nrain, eqtset = opt.eqtset;
  const int nkk = nk + 2;
  const DevTables T = wk.tab;
  const auto scr = wk.scr;
  const auto iscr = wk.iscr;
  const auto prate = wk.prate;
  const auto vtc = wk.vtc, vtr = wk.vtr, vti = wk.vti, vts = wk.vts, vtg = wk.vtg;
  const double rdt = 1.0 / dt;
  Kokkos::parallel_for("thompson_mp_gt_driver", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
    const size_t col = size_t(i - 1) + size_t(ni) * (j - 1);
    const Col s{scr.data() + col * NVAR * nkk, iscr.data() + col * NIVAR * nkk, nkk};
    double* const qv1d = s.a(V_qv1d);
    double* const qc1d = s.a(V_qc1d);
    double* const qi1d = s.a(V_qi1d);
    double* const qr1d = s.a(V_qr1d);
    double* const qs1d = s.a(V_qs1d);
    double* const qg1d = s.a(V_qg1d);
    double* const ni1d = s.a(V_ni1d);
    double* const nr1d = s.a(V_nr1d);
    double* const nc1d = s.a(V_nc1d);
    double* const t1d = s.a(V_t1d);
    double* const p1d = s.a(V_p1d);
    double* const w1d = s.a(V_w1d);
    double* const dz1d = s.a(V_dz1d);
    double* const rho = s.a(V_rho1d);
    double* const prv_rev = s.a(V_prv_rev);
    double* const prw_vcd = s.a(V_prw_vcd);
    double* const vtik = s.a(V_vtik);
    double* const vtrk = s.a(V_vtrk);
    double* const vtsk = s.a(V_vtsk);
    double* const vtgk = s.a(V_vtgk);
    double* const vtck = s.a(V_vtck);

    for (int k = 1; k <= nk; ++k) { prv_rev[k] = 0.0; prw_vcd[k] = 0.0; }
    for (int k = 1; k <= nk + 1; ++k) { vtik[k] = 0.0; vtrk[k] = 0.0; vtsk[k] = 0.0; vtgk[k] = 0.0; vtck[k] = 0.0; }
    double pptrain = 0., pptsnow = 0., pptgraul = 0., pptice = 0.;

    for (int k = 1; k <= nk; ++k) {
      t1d[k] = t3d(i, j, k);
      p1d[k] = p(i, j, k);
      w1d[k] = w(i, j, k);
      dz1d[k] = dz(i, j, k);
      qv1d[k] = q3d(i, j, k, nqv);
      qc1d[k] = q3d(i, j, k, nqc);
      qi1d[k] = q3d(i, j, k, nqi);
      qr1d[k] = q3d(i, j, k, nqr);
      qs1d[k] = q3d(i, j, k, nqs);
      qg1d[k] = q3d(i, j, k, nqg);
      ni1d[k] = q3d(i, j, k, nnci);
      nr1d[k] = q3d(i, j, k, nncr);
      rho[k] = 0.622 * p1d[k] / (R * t1d[k] * (qv1d[k] + 0.622));
    }
    for (int k = 1; k <= nk; ++k) nc1d[k] = T.c.Nt_c / rho[k];

    mp_thompson(T, s, nk, dt, eqtset, pptrain, pptsnow, pptgraul, pptice);

    for (int n = 1; n <= nrain; ++n)
      rain(i, j, n) = rain(i, j, n) + 0.1 * (pptrain + pptsnow + pptgraul + pptice);
    prate(i, j) = rdt * (pptrain + pptsnow + pptgraul + pptice);

    for (int k = 1; k <= nk; ++k) {
      vtc(i, j, k) = vtck[k];
      vtr(i, j, k) = vtrk[k];
      vti(i, j, k) = vtik[k];
      vts(i, j, k) = vtsk[k];
      vtg(i, j, k) = vtgk[k];
    }

    for (int k = 1; k <= nk; ++k) {
      q3d(i, j, k, nqv) = qv1d[k];
      q3d(i, j, k, nqc) = qc1d[k];
      q3d(i, j, k, nqi) = qi1d[k];
      q3d(i, j, k, nqr) = qr1d[k];
      q3d(i, j, k, nqs) = qs1d[k];
      q3d(i, j, k, nqg) = qg1d[k];
      q3d(i, j, k, nnci) = ni1d[k];
      q3d(i, j, k, nncr) = nr1d[k];
      t3d(i, j, k) = t1d[k];
      if (qv1d[k] < 0.0) {
        if (k < nk - 2 && k > 1) {
          q3d(i, j, k, nqv) = fmax2(1.E-7, 0.5 * (q3d(i, j, k - 1, nqv) + q3d(i, j, k + 1, nqv)));
        } else {
          q3d(i, j, k, nqv) = 1.E-7;
        }
      }
    }
  });
}

//---------------------------------------------------------------------------
void mp_driver(const Options& opt, const Constants& c, const Bounds& b, Real dt, Real dz,
               const FArray3& rmh, const FArray3& pi0, const FArray3& th0, const FArray3& w3d,
               const FArray3& rho, const FArray3& prs, const FArray3& pp3d, const FArray3& th3d,
               const FArray4& q3d, const FArray3& rain, const Work& w) {
  if (b.numq < 8) throw std::runtime_error("thompson::mp_driver: numq must be 8 for ptype=3");
  const auto tmp = w.tmp, thten = w.thten, qvold = w.qvold, dzq = w.dzq;
  const auto all = box3(1, b.ni, 1, b.nj, 1, b.nk);

  Kokkos::parallel_for("thompson_mp_driver_T", all, KOKKOS_LAMBDA(int i, int j, int k) {
    tmp(i, j, k) = (th0(i, j, k) + th3d(i, j, k)) * (pi0(i, j, k) + pp3d(i, j, k));
    thten(i, j, k) = tmp(i, j, k);
    qvold(i, j, k) = q3d(i, j, k, nqv);
  });

  kessler::pdefq(b, 0.0, rho, q3d, 1);
  for (int n = 2; n <= 6; ++n) kessler::pdefq(b, opt.qsmall, rho, q3d, n);

  Kokkos::parallel_for("thompson_mp_driver_dz", all, KOKKOS_LAMBDA(int i, int j, int k) {
    dzq(i, j, k) = dz * rmh(i, j, k);
  });

  mp_gt_driver(opt, b, dt, tmp, prs, w3d, dzq, q3d, rain, w);

  const Constants C = c;
  const double tsmall = opt.tsmall, qsmall = opt.qsmall;
  if (opt.eqtset == 2) {
    Kokkos::parallel_for("thompson_mp_driver_eq2", all, KOKKOS_LAMBDA(int i, int j, int k) {
      if (Kokkos::fabs(tmp(i, j, k) - thten(i, j, k)) >= tsmall
          || Kokkos::fabs(q3d(i, j, k, nqv) - qvold(i, j, k)) >= qsmall) {
        prs(i, j, k) = rho(i, j, k) * (C.rd + C.rv * q3d(i, j, k, nqv)) * tmp(i, j, k);
        pp3d(i, j, k) = Kokkos::pow(prs(i, j, k) * phys::rp00, C.rovcp) - pi0(i, j, k);
        th3d(i, j, k) = tmp(i, j, k) / (pi0(i, j, k) + pp3d(i, j, k)) - th0(i, j, k);
      }
    });
  } else {
    Kokkos::parallel_for("thompson_mp_driver_eq1", all, KOKKOS_LAMBDA(int i, int j, int k) {
      if (Kokkos::fabs(tmp(i, j, k) - thten(i, j, k)) >= tsmall
          || Kokkos::fabs(q3d(i, j, k, nqv) - qvold(i, j, k)) >= qsmall) {
        th3d(i, j, k) = tmp(i, j, k) / (pi0(i, j, k) + pp3d(i, j, k)) - th0(i, j, k);
        rho(i, j, k) = prs(i, j, k) / (C.rd * tmp(i, j, k) * (1.0 + q3d(i, j, k, nqv) * C.reps));
      }
    });
  }
}

void mp_driver(const Options& opt, const Constants& c, const Grid& g, const BaseState& base, Real dt,
               const FArray3& w3d, const FArray3& rho, const FArray3& prs, const FArray3& pp3d,
               const FArray3& th3d, const FArray4& q3d, const FArray3& rain, const Work& w) {
  mp_driver(opt, c, g.b, dt, g.dz, g.rmh, base.pi0, base.th0, w3d, rho, prs, pp3d, th3d, q3d, rain, w);
}

} // namespace cm1k::thompson
