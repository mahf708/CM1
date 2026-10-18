// RRTMG long-wave radiation (radopt = 2): RRTMG_LWRAD of src/module_ra_rrtmg_lw.F
// as src/radiation_driver.F calls it, with rrtmg_lw, inatm, cldprmc, setcoef,
// taumol (taugb1..16), rtrnmc, mcica_subcol_lw / generate_stochastic_clouds
// (KISS generator, irng = 0), INIRAD/O3DATA, relcalc and reicalc.
//
// Every (i,j) column is independent: one kernel thread runs RRTMG_LWRAD's
// i/j loop body for its column, with the Fortran's routines as member
// functions of LwCol (which derives from LwTables, so module data such as
// absa, totplnk, ngb, fluxfac resolve by their Fortran names) and every
// array of those routines a member FA bound to a per-column slice of a
// scratch pool. Columns are processed in chunks to bound the scratch memory.
#include "physics/rrtmg/cm1k_rrtmg_lw.hpp"
#include "physics/rrtmg/cm1k_rrtmg_common.hpp"

#include <stdexcept>
#include <string>

namespace cm1k::rrtmg {

namespace {

using namespace lw;

// Error codes of the Fortran's stop / wrf_error_fatal calls.
enum LwErr {
  LW_OK = 0,
  LW_INFLAG1 = 1,
  LW_ICE_SMALL,
  LW_ICE_BOUNDS,
  LW_ICE_GEN_BOUNDS,
  LW_SNOW_GEN_BOUNDS,
  LW_LIQ_BOUNDS,
  LW_ICLD,
  LW_KISS_PMID,
  LW_OVERLAP5,
  LW_O3FRAC,
};
const char* lw_err_message(int e) {
  switch (e) {
  case LW_INFLAG1: return "INFLAG = 1 OPTION NOT AVAILABLE WITH MCICA";
  case LW_ICE_SMALL: return "ICE RADIUS TOO SMALL";
  case LW_ICE_BOUNDS: return "ICE RADIUS OUT OF BOUNDS";
  case LW_ICE_GEN_BOUNDS: return "ERROR: ICE GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS";
  case LW_SNOW_GEN_BOUNDS: return "ERROR: SNOW GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS";
  case LW_LIQ_BOUNDS: return "LIQUID EFFECTIVE RADIUS OUT OF BOUNDS";
  case LW_ICLD: return "MCICA_SUBCOL: INVALID ICLD";
  case LW_KISS_PMID: return "MCICA_SUBCOL: KISSVEC SEED GENERATOR REQUIRES PMID FROM BOTTOM FOUR LAYERS.";
  case LW_OVERLAP5: return "Cloud Overlap case 5: ER has not yet been implemented. Stopping...";
  case LW_O3FRAC: return "O3DATA: frac out of range (stop 1231)";
  default: return "unknown error";
  }
}
#define LW_FAIL(code) \
  do {                \
    ierr = (code);    \
    return;           \
  } while (0)

// RRTMG_LWRAD's inputs for one call (radiation_driver.F's actual arguments).
struct LwIn {
  FArray3 rthratenlw, lwtenc, lwupt, lwuptc, lwdnt, lwdntc, lwupb, lwupbc, lwdnb, lwdnbc, glw, olr, lwcf;
  FArray3 emiss, tsk, p8w, p3d, dz8w, t3d, t8w, cldfra3d, xland, xice, snow, qv3d, qc3d, qr3d, qi3d, qs3d,
      qg3d, o33d;
  Real g, co2;
  int icloud, cldovrlp, o3input;
  bool warm_rain, f_qc, f_qr, f_qi, f_qs, f_qg, has_o33d, has_lwupt;
  int nk;
};

struct LwCol : LwTables {
  static constexpr int ncol = 1, iplon = 1, kts = 1;
  int kte = 0, nlayers = 0, nlay = 0, ierr = 0;
  // rrtmg_lw scalars
  int istart = 1, iend = 16, iout = 0, iaer = 10, icld = 0;
  int inflag = 0, iceflag = 0, liqflag = 0, inflglw = 0, iceflglw = 0, liqflglw = 0;
  int ncbands = 0, laytrop = 0;
  Real tbound = 0, pwvcm = 0, tsfc1 = 0;

  // RRTMG_LWRAD
  FA<1> Pw1D, Tw1D, TTEN1D, CLDFRA1D, DZ1D, P1D, T1D, QV1D, QC1D, QR1D, QI1D, RHO1D, QS1D, QG1D, O31D;
  FA<2> plev, tlev, play, tlay, h2ovmr, o3vmr, co2vmr, o2vmr, ch4vmr, n2ovmr, cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr;
  FA<1> o3mmr, varint;
  FA<2> hgt, emis, clwpth, ciwpth, cswpth, rel, rei, res, cldfrac, relqmcl, reicmcl, resnmcl;
  FA<3> taucld, cldfmcl, clwpmcl, ciwpmcl, cswpmcl, taucmcl, tauaer;
  FA<2> uflx, dflx, uflxc, dflxc, uflxcln, dflxcln, hr, hrc;
  FA<2> pdel, cicewp, cliqwp, csnowp, reliq, reice, recloud1d, reice1d, resnow1d;
  // generate_stochastic_clouds
  FA<2> pmid, cldf, alpha;
  FA<3> CDF, CDF2;
  // rrtmg_lw
  FA<1> pavel, tavel, pz, tz, coldry, wbrodl, semiss;
  FA<2> wkl, wx, fracs, taug, taut, taua, planklay, planklev;
  FA<1> plankbnd;
  IA<1> jp, jt, jt1, indself, indfor, indminor;
  FA<1> colh2o, colco2, colo3, coln2o, colco, colch4, colo2, colbrd, selffac, selffrac, forfac, forfrac, minorfrac,
      scaleminor, scaleminorn2, fac00, fac01, fac10, fac11, rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1,
      rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1;
  FA<2> cldfmc, ciwpmc, clwpmc, cswpmc, taucmc;
  FA<1> relqmc, reicmc, resnmc;
  FA<1> totuflux, totdflux, fnet, htr, totuclfl, totdclfl, fnetc, htrc, totuclnlfl, totdclnlfl, fnetcln, htrcln;
  // cldprmc
  FA<1> abscoice, abscoliq, abscosno;
  // rtrnmc
  FA<2> abscld, efclfrac, odcld;
  FA<1> atot, atrans, bbugas, bbutot, clrurad, clrdrad, uflux, dflux, urad, drad, uclfl, dclfl, secdiff;
  IA<1> icldlyr;

  template <class B>
  KOKKOS_INLINE_FUNCTION void layout(B& b) {
    const int nk = kte, nl = nlay;
    b.arr(Pw1D, {1}, {nk + 1});
    b.arr(Tw1D, {1}, {nk + 1});
    FA<1>* k1[] = {&TTEN1D, &CLDFRA1D, &DZ1D, &P1D, &T1D, &QV1D, &QC1D, &QR1D, &QI1D, &RHO1D, &QS1D, &QG1D, &O31D};
    for (auto* a : k1) b.arr(*a, {1}, {nk});
    b.arr(plev, {1, 1}, {1, nl + 1});
    b.arr(tlev, {1, 1}, {1, nl + 1});
    FA<2>* l2[] = {&play,     &tlay,     &h2ovmr,   &o3vmr,   &co2vmr,  &o2vmr,   &ch4vmr,  &n2ovmr,  &cfc11vmr,
                   &cfc12vmr, &cfc22vmr, &ccl4vmr,  &hgt,     &clwpth,  &ciwpth,  &cswpth,  &rel,     &rei,
                   &res,      &cldfrac,  &relqmcl,  &reicmcl, &resnmcl, &hr,      &hrc,     &pmid,    &cldf,
                   &alpha};
    for (auto* a : l2) b.arr(*a, {1, 1}, {1, nl});
    b.arr(o3mmr, {1}, {nl});
    b.arr(varint, {1}, {nl + 1});
    b.arr(emis, {1, 1}, {1, nbndlw});
    b.arr(taucld, {1, 1, 1}, {nbndlw, 1, nl});
    FA<3>* g3[] = {&cldfmcl, &clwpmcl, &ciwpmcl, &cswpmcl, &taucmcl, &CDF};
    for (auto* a : g3) b.arr(*a, {1, 1, 1}, {ngptlw, 1, nl});
    if (icld == 4) b.arr(CDF2, {1, 1, 1}, {ngptlw, 1, nl});
    b.arr(tauaer, {1, 1, 1}, {1, nl, nbndlw});
    FA<2>* f2[] = {&uflx, &dflx, &uflxc, &dflxc, &uflxcln, &dflxcln};
    for (auto* a : f2) b.arr(*a, {1, 1}, {1, nl + 1});
    FA<2>* m2[] = {&pdel, &cicewp, &cliqwp, &csnowp, &reliq, &reice, &recloud1d, &reice1d, &resnow1d};
    for (auto* a : m2) b.arr(*a, {1, 1}, {1, nk});
    // rrtmg_lw
    FA<1>* r1[] = {&pavel, &tavel, &coldry, &wbrodl, &colh2o, &colco2, &colo3, &coln2o, &colco, &colch4,
                   &colo2, &colbrd, &selffac, &selffrac, &forfac, &forfrac, &minorfrac, &scaleminor,
                   &scaleminorn2, &fac00, &fac01, &fac10, &fac11, &rat_h2oco2, &rat_h2oco2_1, &rat_h2oo3,
                   &rat_h2oo3_1, &rat_h2on2o, &rat_h2on2o_1, &rat_h2och4, &rat_h2och4_1, &rat_n2oco2,
                   &rat_n2oco2_1, &rat_o3co2, &rat_o3co2_1, &relqmc, &reicmc, &resnmc};
    for (auto* a : r1) b.arr(*a, {1}, {nl + 1});
    FA<1>* z1[] = {&pz,       &tz,       &totuflux,   &totdflux,   &fnet,    &htr,   &totuclfl,
                   &totdclfl, &fnetc,    &htrc,       &totuclnlfl, &totdclnlfl, &fnetcln, &htrcln};
    for (auto* a : z1) b.arr(*a, {0}, {nl + 1});
    b.arr(semiss, {1}, {nbndlw});
    b.arr(wkl, {1, 1}, {mxmol, nl + 1});
    b.arr(wx, {1, 1}, {maxxsec, nl + 1});
    b.arr(fracs, {1, 1}, {nl + 1, ngptlw});
    b.arr(taug, {1, 1}, {nl + 1, ngptlw});
    b.arr(taut, {1, 1}, {nl + 1, ngptlw});
    b.arr(taua, {1, 1}, {nl + 1, nbndlw});
    b.arr(planklay, {1, 1}, {nl + 1, nbndlw});
    b.arr(planklev, {0, 1}, {nl + 1, nbndlw});
    b.arr(plankbnd, {1}, {nbndlw});
    IA<1>* i1[] = {&jp, &jt, &jt1, &indself, &indfor, &indminor};
    for (auto* a : i1) b.arr(*a, {1}, {nl + 1});
    FA<2>* c2[] = {&cldfmc, &ciwpmc, &clwpmc, &cswpmc, &taucmc};
    for (auto* a : c2) b.arr(*a, {1, 1}, {ngptlw, nl + 1});
    b.arr(abscoice, {1}, {ngptlw});
    b.arr(abscoliq, {1}, {ngptlw});
    b.arr(abscosno, {1}, {ngptlw});
    // rtrnmc (nlayers = nlay)
    b.arr(abscld, {1, 1}, {nl, ngptlw});
    b.arr(efclfrac, {1, 1}, {nl, ngptlw});
    b.arr(odcld, {1, 1}, {nl, ngptlw});
    FA<1>* t1[] = {&atot, &atrans, &bbugas, &bbutot};
    for (auto* a : t1) b.arr(*a, {1}, {nl});
    FA<1>* t0[] = {&clrurad, &clrdrad, &uflux, &dflux, &urad, &drad, &uclfl, &dclfl};
    for (auto* a : t0) b.arr(*a, {0}, {nl});
    b.arr(secdiff, {1}, {nbndlw});
    b.arr(icldlyr, {1}, {nl});
  }

  KOKKOS_INLINE_FUNCTION void inatm() {
    constexpr Real amd = 28.9660;
    constexpr Real amw = 18.0160;
    constexpr Real amdw = 1.607793;
    constexpr Real amdc = 0.658114;
    constexpr Real amdo = 0.603428;
    constexpr Real amdm = 1.805423;
    constexpr Real amdn = 0.658090;
    constexpr Real amdo2 = 0.905140;
    constexpr Real amdc1 = 0.210852;
    constexpr Real amdc2 = 0.239546;
    int isp;
    int l;
    int ix;
    int n;
    int imol;
    int ib;
    int ig;
    Real amm;
    Real amttl;
    Real wvttl;
    Real wvsh;
    Real summol;
    for (l = 1; l <= nlay + 1; ++l) {
      for (imol = 1; imol <= mxmol; ++imol) wkl(imol, l) = 0.0;
      for (ix = 1; ix <= maxxsec; ++ix) wx(ix, l) = 0.0;
    }
    for (l = 1; l <= nlay + 1; ++l)
      for (ig = 1; ig <= ngptlw; ++ig) cldfmc(ig, l) = 0.0;
    for (l = 1; l <= nlay + 1; ++l)
      for (ig = 1; ig <= ngptlw; ++ig) taucmc(ig, l) = 0.0;
    for (l = 1; l <= nlay + 1; ++l)
      for (ig = 1; ig <= ngptlw; ++ig) ciwpmc(ig, l) = 0.0;
    for (l = 1; l <= nlay + 1; ++l)
      for (ig = 1; ig <= ngptlw; ++ig) clwpmc(ig, l) = 0.0;
    for (l = 1; l <= nlay + 1; ++l)
      for (ig = 1; ig <= ngptlw; ++ig) cswpmc(ig, l) = 0.0;
    for (l = 1; l <= nlay + 1; ++l) reicmc(l) = 0.0;
    for (l = 1; l <= nlay + 1; ++l) relqmc(l) = 0.0;
    for (l = 1; l <= nlay + 1; ++l) resnmc(l) = 0.0;
    for (ib = 1; ib <= nbndlw; ++ib)
      for (l = 1; l <= nlay + 1; ++l) taua(l, ib) = 0.0;
    amttl = 0.0;
    wvttl = 0.0;
    tbound = tsfc1;
    pz(0) = plev(iplon, 1);
    tz(0) = tlev(iplon, 1);
    for (l = 1; l <= nlayers; ++l) {
      pavel(l) = play(iplon, l);
      tavel(l) = tlay(iplon, l);
      pz(l) = plev(iplon, l + 1);
      tz(l) = tlev(iplon, l + 1);
      wkl(1, l) = h2ovmr(iplon, l);
      wkl(2, l) = co2vmr(iplon, l);
      wkl(3, l) = o3vmr(iplon, l);
      wkl(4, l) = n2ovmr(iplon, l);
      wkl(6, l) = ch4vmr(iplon, l);
      wkl(7, l) = o2vmr(iplon, l);
      amm = (1.0 - wkl(1, l)) * amd + wkl(1, l) * amw;
      coldry(l) = (pz(l - 1) - pz(l)) * 1.0e3 * avogad / (1.0e2 * grav * amm * (1.0 + wkl(1, l)));
    }
    for (l = 1; l <= nlayers; ++l) {
      wx(1, l) = ccl4vmr(iplon, l);
      wx(2, l) = cfc11vmr(iplon, l);
      wx(3, l) = cfc12vmr(iplon, l);
      wx(4, l) = cfc22vmr(iplon, l);
    }
    for (l = 1; l <= nlayers; ++l) {
      summol = 0.0;
      for (imol = 2; imol <= nmol; ++imol) {
        summol = summol + wkl(imol, l);
      }
      wbrodl(l) = coldry(l) * (1.0 - summol);
      for (imol = 1; imol <= nmol; ++imol) {
        wkl(imol, l) = coldry(l) * wkl(imol, l);
      }
      amttl = amttl + coldry(l) + wkl(1, l);
      wvttl = wvttl + wkl(1, l);
      for (ix = 1; ix <= maxxsec; ++ix) {
        if (ixindx(ix) != 0) {
          wx(ixindx(ix), l) = coldry(l) * wx(ix, l) * 1.0e-20;
        }
      }
    }
    wvsh = (amw * wvttl) / (amd * amttl);
    pwvcm = wvsh * (1.0e3 * pz(0)) / (1.0e2 * grav);
    for (n = 1; n <= nbndlw; ++n) {
      semiss(n) = emis(iplon, n);
    }
    if (iaer >= 1) {
      for (l = 1; l <= nlayers; ++l) {
        for (ib = 1; ib <= nbndlw; ++ib) {
          taua(l, ib) = tauaer(iplon, l, ib);
        }
      }
    }
    if (icld >= 1) {
      inflag = inflglw;
      iceflag = iceflglw;
      liqflag = liqflglw;
      for (l = 1; l <= nlayers; ++l) {
        for (ig = 1; ig <= ngptlw; ++ig) {
          cldfmc(ig, l) = cldfmcl(ig, iplon, l);
          taucmc(ig, l) = taucmcl(ig, iplon, l);
          ciwpmc(ig, l) = ciwpmcl(ig, iplon, l);
          clwpmc(ig, l) = clwpmcl(ig, iplon, l);
          cswpmc(ig, l) = cswpmcl(ig, iplon, l);
        }
        reicmc(l) = reicmcl(iplon, l);
        relqmc(l) = relqmcl(iplon, l);
        resnmc(l) = resnmcl(iplon, l);
      }
    }
  
  }

  KOKKOS_INLINE_FUNCTION void cldprmc() {
    int lay;
    int ib;
    int ig;
    int index;
    Real cwp;
    Real radice;
    Real factor;
    Real fint;
    Real radliq;
    Real radsno;
    constexpr Real eps = 1.0e-6;
    constexpr Real cldmin = 1.0e-20;
    const int icb[17] = {0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5};
    ncbands = 1;
    for (lay = 1; lay <= nlayers; ++lay) {
      for (ig = 1; ig <= ngptlw; ++ig) {
        cwp = ciwpmc(ig, lay) + clwpmc(ig, lay) + cswpmc(ig, lay);
        if (cldfmc(ig, lay) >= cldmin && (cwp >= cldmin || taucmc(ig, lay) >= cldmin)) {
          if (inflag == 0) {
            return;
          } else if (inflag == 1) {
            LW_FAIL(LW_INFLAG1);
          } else if (inflag >= 2) {
            radice = reicmc(lay);
            if ((ciwpmc(ig, lay) + cswpmc(ig, lay)) == 0.0) {
              abscoice(ig) = 0.0;
              abscosno(ig) = 0.0;
            } else if (iceflag == 0) {
              if (radice < 10.0) LW_FAIL(LW_ICE_SMALL);
              abscoice(ig) = absice0(1) + absice0(2) / radice;
              abscosno(ig) = 0.0;
            } else if (iceflag == 1) {
              if (radice < 13.0 || radice > 130.0) LW_FAIL(LW_ICE_BOUNDS);
              ncbands = 5;
              ib = icb[ngb(ig)];
              abscoice(ig) = absice1(1, ib) + absice1(2, ib) / radice;
              abscosno(ig) = 0.0;
            } else if (iceflag == 2) {
              if (radice < 5.0 || radice > 131.0) LW_FAIL(LW_ICE_BOUNDS);
              ncbands = 16;
              factor = (radice - 2.0) / 3.0;
              index = int(factor);
              if (index == 43) index = 42;
              fint = factor - Real(index);
              ib = ngb(ig);
              abscoice(ig) = absice2(index, ib) + fint * (absice2(index + 1, ib) - (absice2(index, ib)));
              abscosno(ig) = 0.0;
            } else if (iceflag >= 3) {
              if (radice < 5.0 || radice > 140.0) {
                LW_FAIL(LW_ICE_GEN_BOUNDS);
              }
              ncbands = 16;
              factor = (radice - 2.0) / 3.0;
              index = int(factor);
              if (index == 46) index = 45;
              fint = factor - Real(index);
              ib = ngb(ig);
              abscoice(ig) = absice3(index, ib) + fint * (absice3(index + 1, ib) - (absice3(index, ib)));
              abscosno(ig) = 0.0;
            }
            if (cswpmc(ig, lay) > 0.0 && iceflag == 5) {
              radsno = resnmc(lay);
              if (radsno < 5.0 || radsno > 140.0) {
                LW_FAIL(LW_SNOW_GEN_BOUNDS);
              }
              ncbands = 16;
              factor = (radsno - 2.0) / 3.0;
              index = int(factor);
              if (index == 46) index = 45;
              fint = factor - Real(index);
              ib = ngb(ig);
              abscosno(ig) = absice3(index, ib) + fint * (absice3(index + 1, ib) - (absice3(index, ib)));
            }
            if (clwpmc(ig, lay) == 0.0) {
              abscoliq(ig) = 0.0;
            } else if (liqflag == 0) {
              abscoliq(ig) = absliq0;
            } else if (liqflag == 1) {
              radliq = relqmc(lay);
              if (radliq < 2.5 || radliq > 60.0) LW_FAIL(LW_LIQ_BOUNDS);
              index = int(radliq - 1.5);
              if (index == 0) index = 1;
              if (index == 58) index = 57;
              fint = radliq - 1.5 - Real(index);
              ib = ngb(ig);
              abscoliq(ig) = absliq1(index, ib) + fint * (absliq1(index + 1, ib) - (absliq1(index, ib)));
            }
            taucmc(ig, lay) = ciwpmc(ig, lay) * abscoice(ig) + clwpmc(ig, lay) * abscoliq(ig) + cswpmc(ig, lay) * abscosno(ig);
          }
        }
      }
    }
  
  }

  KOKKOS_INLINE_FUNCTION void setcoef() {
    int indbound;
    int indlev0;
    int lay;
    int indlay;
    int indlev;
    int iband;
    int jp1;
    Real stpfac;
    Real tbndfrac;
    Real t0frac;
    Real tlayfrac;
    Real tlevfrac;
    Real dbdtlev;
    Real dbdtlay;
    Real plog;
    Real fp;
    Real ft;
    Real ft1;
    Real water;
    Real scalefac;
    Real factor;
    Real compfp;
    stpfac = 296.0 / 1013.0;
    indbound = tbound - 159.0;
    if (indbound < 1) {
      indbound = 1;
    } else if (indbound > 180) {
      indbound = 180;
    }
    tbndfrac = tbound - 159.0 - Real(indbound);
    indlev0 = tz(0) - 159.0;
    if (indlev0 < 1) {
      indlev0 = 1;
    } else if (indlev0 > 180) {
      indlev0 = 180;
    }
    t0frac = tz(0) - 159.0 - Real(indlev0);
    laytrop = 0;
    for (lay = 1; lay <= nlayers; ++lay) {
      indlay = tavel(lay) - 159.0;
      if (indlay < 1) {
        indlay = 1;
      } else if (indlay > 180) {
        indlay = 180;
      }
      tlayfrac = tavel(lay) - 159.0 - Real(indlay);
      indlev = tz(lay) - 159.0;
      if (indlev < 1) {
        indlev = 1;
      } else if (indlev > 180) {
        indlev = 180;
      }
      tlevfrac = tz(lay) - 159.0 - Real(indlev);
      for (iband = 1; iband <= 15; ++iband) {
        if (lay == 1) {
          dbdtlev = totplnk(indbound + 1, iband) - totplnk(indbound, iband);
          plankbnd(iband) = semiss(iband) * (totplnk(indbound, iband) + tbndfrac * dbdtlev);
          dbdtlev = totplnk(indlev0 + 1, iband) - totplnk(indlev0, iband);
          planklev(0, iband) = totplnk(indlev0, iband) + t0frac * dbdtlev;
        }
        dbdtlev = totplnk(indlev + 1, iband) - totplnk(indlev, iband);
        dbdtlay = totplnk(indlay + 1, iband) - totplnk(indlay, iband);
        planklay(lay, iband) = totplnk(indlay, iband) + tlayfrac * dbdtlay;
        planklev(lay, iband) = totplnk(indlev, iband) + tlevfrac * dbdtlev;
      }
      iband = 16;
      if (istart == 16) {
        if (lay == 1) {
          dbdtlev = totplk16(indbound + 1) - totplk16(indbound);
          plankbnd(iband) = semiss(iband) * (totplk16(indbound) + tbndfrac * dbdtlev);
          dbdtlev = totplnk(indlev0 + 1, iband) - totplnk(indlev0, iband);
          planklev(0, iband) = totplk16(indlev0) + t0frac * dbdtlev;
        }
        dbdtlev = totplk16(indlev + 1) - totplk16(indlev);
        dbdtlay = totplk16(indlay + 1) - totplk16(indlay);
        planklay(lay, iband) = totplk16(indlay) + tlayfrac * dbdtlay;
        planklev(lay, iband) = totplk16(indlev) + tlevfrac * dbdtlev;
      } else {
        if (lay == 1) {
          dbdtlev = totplnk(indbound + 1, iband) - totplnk(indbound, iband);
          plankbnd(iband) = semiss(iband) * (totplnk(indbound, iband) + tbndfrac * dbdtlev);
          dbdtlev = totplnk(indlev0 + 1, iband) - totplnk(indlev0, iband);
          planklev(0, iband) = totplnk(indlev0, iband) + t0frac * dbdtlev;
        }
        dbdtlev = totplnk(indlev + 1, iband) - totplnk(indlev, iband);
        dbdtlay = totplnk(indlay + 1, iband) - totplnk(indlay, iband);
        planklay(lay, iband) = totplnk(indlay, iband) + tlayfrac * dbdtlay;
        planklev(lay, iband) = totplnk(indlev, iband) + tlevfrac * dbdtlev;
      }
      plog = Kokkos::log(pavel(lay));
      jp(lay) = int(36.0 - 5 * (plog + 0.04));
      if (jp(lay) < 1) {
        jp(lay) = 1;
      } else if (jp(lay) > 58) {
        jp(lay) = 58;
      }
      jp1 = jp(lay) + 1;
      fp = 5.0 * (preflog(jp(lay)) - plog);
      jt(lay) = int(3.0 + (tavel(lay) - tref(jp(lay))) / 15.0);
      if (jt(lay) < 1) {
        jt(lay) = 1;
      } else if (jt(lay) > 4) {
        jt(lay) = 4;
      }
      ft = ((tavel(lay) - tref(jp(lay))) / 15.0) - Real(jt(lay) - 3);
      jt1(lay) = int(3.0 + (tavel(lay) - tref(jp1)) / 15.0);
      if (jt1(lay) < 1) {
        jt1(lay) = 1;
      } else if (jt1(lay) > 4) {
        jt1(lay) = 4;
      }
      ft1 = ((tavel(lay) - tref(jp1)) / 15.0) - Real(jt1(lay) - 3);
      water = wkl(1, lay) / coldry(lay);
      scalefac = pavel(lay) * stpfac / tavel(lay);
      if (!(plog <= 4.56)) {  // (Fortran: if (plog .le. 4.56) go to 5300)
        laytrop = laytrop + 1;
        forfac(lay) = scalefac / (1.0 + water);
        factor = (332.0 - tavel(lay)) / 36.0;
        indfor(lay) = Kokkos::min(2, Kokkos::max(1, int(factor)));
        forfrac(lay) = factor - Real(indfor(lay));
        selffac(lay) = water * forfac(lay);
        factor = (tavel(lay) - 188.0) / 7.2;
        indself(lay) = Kokkos::min(9, Kokkos::max(1, int(factor) - 7));
        selffrac(lay) = factor - Real(indself(lay) + 7);
        scaleminor(lay) = pavel(lay) / tavel(lay);
        scaleminorn2(lay) = (pavel(lay) / tavel(lay)) * (wbrodl(lay) / (coldry(lay) + wkl(1, lay)));
        factor = (tavel(lay) - 180.8) / 7.2;
        indminor(lay) = Kokkos::min(18, Kokkos::max(1, int(factor)));
        minorfrac(lay) = factor - Real(indminor(lay));
        rat_h2oco2(lay) = chi_mls(1, jp(lay)) / chi_mls(2, jp(lay));
        rat_h2oco2_1(lay) = chi_mls(1, jp(lay) + 1) / chi_mls(2, jp(lay) + 1);
        rat_h2oo3(lay) = chi_mls(1, jp(lay)) / chi_mls(3, jp(lay));
        rat_h2oo3_1(lay) = chi_mls(1, jp(lay) + 1) / chi_mls(3, jp(lay) + 1);
        rat_h2on2o(lay) = chi_mls(1, jp(lay)) / chi_mls(4, jp(lay));
        rat_h2on2o_1(lay) = chi_mls(1, jp(lay) + 1) / chi_mls(4, jp(lay) + 1);
        rat_h2och4(lay) = chi_mls(1, jp(lay)) / chi_mls(6, jp(lay));
        rat_h2och4_1(lay) = chi_mls(1, jp(lay) + 1) / chi_mls(6, jp(lay) + 1);
        rat_n2oco2(lay) = chi_mls(4, jp(lay)) / chi_mls(2, jp(lay));
        rat_n2oco2_1(lay) = chi_mls(4, jp(lay) + 1) / chi_mls(2, jp(lay) + 1);
        colh2o(lay) = 1.0e-20 * wkl(1, lay);
        colco2(lay) = 1.0e-20 * wkl(2, lay);
        colo3(lay) = 1.0e-20 * wkl(3, lay);
        coln2o(lay) = 1.0e-20 * wkl(4, lay);
        colco(lay) = 1.0e-20 * wkl(5, lay);
        colch4(lay) = 1.0e-20 * wkl(6, lay);
        colo2(lay) = 1.0e-20 * wkl(7, lay);
        if (colco2(lay) == 0.0) colco2(lay) = 1.0e-32 * coldry(lay);
        if (colo3(lay) == 0.0) colo3(lay) = 1.0e-32 * coldry(lay);
        if (coln2o(lay) == 0.0) coln2o(lay) = 1.0e-32 * coldry(lay);
        if (colco(lay) == 0.0) colco(lay) = 1.0e-32 * coldry(lay);
        if (colch4(lay) == 0.0) colch4(lay) = 1.0e-32 * coldry(lay);
        colbrd(lay) = 1.0e-20 * wbrodl(lay);
      } else {  // 5300
        forfac(lay) = scalefac / (1.0 + water);
        factor = (tavel(lay) - 188.0) / 36.0;
        indfor(lay) = 3;
        forfrac(lay) = factor - 1.0;
        selffac(lay) = water * forfac(lay);
        scaleminor(lay) = pavel(lay) / tavel(lay);
        scaleminorn2(lay) = (pavel(lay) / tavel(lay)) * (wbrodl(lay) / (coldry(lay) + wkl(1, lay)));
        factor = (tavel(lay) - 180.8) / 7.2;
        indminor(lay) = Kokkos::min(18, Kokkos::max(1, int(factor)));
        minorfrac(lay) = factor - Real(indminor(lay));
        rat_h2oco2(lay) = chi_mls(1, jp(lay)) / chi_mls(2, jp(lay));
        rat_h2oco2_1(lay) = chi_mls(1, jp(lay) + 1) / chi_mls(2, jp(lay) + 1);
        rat_o3co2(lay) = chi_mls(3, jp(lay)) / chi_mls(2, jp(lay));
        rat_o3co2_1(lay) = chi_mls(3, jp(lay) + 1) / chi_mls(2, jp(lay) + 1);
        colh2o(lay) = 1.0e-20 * wkl(1, lay);
        colco2(lay) = 1.0e-20 * wkl(2, lay);
        colo3(lay) = 1.0e-20 * wkl(3, lay);
        coln2o(lay) = 1.0e-20 * wkl(4, lay);
        colco(lay) = 1.0e-20 * wkl(5, lay);
        colch4(lay) = 1.0e-20 * wkl(6, lay);
        colo2(lay) = 1.0e-20 * wkl(7, lay);
        if (colco2(lay) == 0.0) colco2(lay) = 1.0e-32 * coldry(lay);
        if (colo3(lay) == 0.0) colo3(lay) = 1.0e-32 * coldry(lay);
        if (coln2o(lay) == 0.0) coln2o(lay) = 1.0e-32 * coldry(lay);
        if (colco(lay) == 0.0) colco(lay) = 1.0e-32 * coldry(lay);
        if (colch4(lay) == 0.0) colch4(lay) = 1.0e-32 * coldry(lay);
        colbrd(lay) = 1.0e-20 * wbrodl(lay);
      }  // 5400
      compfp = 1.0 - fp;
      fac10(lay) = compfp * ft;
      fac00(lay) = compfp * (1.0 - ft);
      fac11(lay) = fp * ft1;
      fac01(lay) = fp * (1.0 - ft1);
      selffac(lay) = colh2o(lay) * selffac(lay);
      forfac(lay) = colh2o(lay) * forfac(lay);
    }
  
  }

  // taumol: gaseous optical depths and Planck fractions of the 16 bands
  KOKKOS_INLINE_FUNCTION void taumol() {
    taugb1();
    taugb2();
    taugb3();
    taugb4();
    taugb5();
    taugb6();
    taugb7();
    taugb8();
    taugb9();
    taugb10();
    taugb11();
    taugb12();
    taugb13();
    taugb14();
    taugb15();
    taugb16();
  }

  KOKKOS_INLINE_FUNCTION void taugb1() {
    const auto& fracrefa = kg01.fracrefa;
    const auto& fracrefb = kg01.fracrefb;
    const auto& absa = kg01.absa;
    const auto& absb = kg01.absb;
    const auto& ka_mn2 = kg01.ka_mn2;
    const auto& kb_mn2 = kg01.kb_mn2;
    const auto& selfref = kg01.selfref;
    const auto& forref = kg01.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int indm;
    int ig;
    Real pp;
    Real corradj;
    Real scalen2;
    Real tauself;
    Real taufor;
    Real taun2;
    for (lay = 1; lay <= laytrop; ++lay) {
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(1) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(1) + 1;
      inds = indself(lay);
      indf = indfor(lay);
      indm = indminor(lay);
      pp = pavel(lay);
      corradj = 1.0;
      if (pp < 250.0) {
        corradj = 1.0 - 0.15 * (250.0 - pp) / 154.4;
      }
      scalen2 = colbrd(lay) * scaleminorn2(lay);
      for (ig = 1; ig <= ng1; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        taun2 = scalen2 * (ka_mn2(indm, ig) + minorfrac(lay) * (ka_mn2(indm + 1, ig) - ka_mn2(indm, ig)));
        taug(lay, ig) = corradj * (colh2o(lay) * (fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + tauself + taufor + taun2);
        fracs(lay, ig) = fracrefa(ig);
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(1) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(1) + 1;
      indf = indfor(lay);
      indm = indminor(lay);
      pp = pavel(lay);
      corradj = 1.0 - 0.15 * (pp / 95.6);
      scalen2 = colbrd(lay) * scaleminorn2(lay);
      for (ig = 1; ig <= ng1; ++ig) {
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        taun2 = scalen2 * (kb_mn2(indm, ig) + minorfrac(lay) * (kb_mn2(indm + 1, ig) - kb_mn2(indm, ig)));
        taug(lay, ig) = corradj * (colh2o(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig)) + taufor + taun2);
        fracs(lay, ig) = fracrefb(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb2() {
    const auto& fracrefa = kg02.fracrefa;
    const auto& fracrefb = kg02.fracrefb;
    const auto& absa = kg02.absa;
    const auto& absb = kg02.absb;
    const auto& selfref = kg02.selfref;
    const auto& forref = kg02.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int ig;
    Real pp;
    Real corradj;
    Real tauself;
    Real taufor;
    for (lay = 1; lay <= laytrop; ++lay) {
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(2) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(2) + 1;
      inds = indself(lay);
      indf = indfor(lay);
      pp = pavel(lay);
      corradj = 1.0 - 0.05 * (pp - 100.0) / 900.0;
      for (ig = 1; ig <= ng2; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        taug(lay, ngs1 + ig) = corradj * (colh2o(lay) * (fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + tauself + taufor);
        fracs(lay, ngs1 + ig) = fracrefa(ig);
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(2) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(2) + 1;
      indf = indfor(lay);
      for (ig = 1; ig <= ng2; ++ig) {
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        taug(lay, ngs1 + ig) = colh2o(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig)) + taufor;
        fracs(lay, ngs1 + ig) = fracrefb(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb3() {
    const auto& fracrefa = kg03.fracrefa;
    const auto& fracrefb = kg03.fracrefb;
    const auto& absa = kg03.absa;
    const auto& absb = kg03.absb;
    const auto& ka_mn2o = kg03.ka_mn2o;
    const auto& kb_mn2o = kg03.kb_mn2o;
    const auto& selfref = kg03.selfref;
    const auto& forref = kg03.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int indm;
    int ig;
    int js;
    int js1;
    int jmn2o;
    int jpl;
    Real speccomb;
    Real specparm;
    Real specmult;
    Real fs;
    Real speccomb1;
    Real specparm1;
    Real specmult1;
    Real fs1;
    Real speccomb_mn2o;
    Real specparm_mn2o;
    Real specmult_mn2o;
    Real fmn2o;
    Real fmn2omf;
    Real chi_n2o;
    Real ratn2o;
    Real adjfac;
    Real adjcoln2o;
    Real speccomb_planck;
    Real specparm_planck;
    Real specmult_planck;
    Real fpl;
    Real p;
    Real p4;
    Real fk0;
    Real fk1;
    Real fk2;
    Real fac000;
    Real fac100;
    Real fac200;
    Real fac010;
    Real fac110;
    Real fac210;
    Real fac001;
    Real fac101;
    Real fac201;
    Real fac011;
    Real fac111;
    Real fac211;
    Real tauself;
    Real taufor;
    Real n2om1;
    Real n2om2;
    Real absn2o;
    Real refrat_planck_a;
    Real refrat_planck_b;
    Real refrat_m_a;
    Real refrat_m_b;
    Real tau_major;
    Real tau_major1;
    refrat_planck_a = chi_mls(1, 9) / chi_mls(2, 9);
    refrat_planck_b = chi_mls(1, 13) / chi_mls(2, 13);
    refrat_m_a = chi_mls(1, 3) / chi_mls(2, 3);
    refrat_m_b = chi_mls(1, 13) / chi_mls(2, 13);
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colh2o(lay) + rat_h2oco2(lay) * colco2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colh2o(lay) + rat_h2oco2_1(lay) * colco2(lay);
      specparm1 = colh2o(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 8.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      speccomb_mn2o = colh2o(lay) + refrat_m_a * colco2(lay);
      specparm_mn2o = colh2o(lay) / speccomb_mn2o;
      if (specparm_mn2o >= oneminus) specparm_mn2o = oneminus;
      specmult_mn2o = 8.0 * specparm_mn2o;
      jmn2o = 1 + int(specmult_mn2o);
      fmn2o = Kokkos::fmod(specmult_mn2o, 1.0);
      fmn2omf = minorfrac(lay) * fmn2o;
      chi_n2o = coln2o(lay) / coldry(lay);
      ratn2o = 1.0e20 * chi_n2o / chi_mls(4, jp(lay) + 1);
      if (ratn2o > 1.5) {
        adjfac = 0.5 + Kokkos::pow((ratn2o - 0.5), 0.65);
        adjcoln2o = adjfac * chi_mls(4, jp(lay) + 1) * coldry(lay) * 1.0e-20;
      } else {
        adjcoln2o = coln2o(lay);
      }
      speccomb_planck = colh2o(lay) + refrat_planck_a * colco2(lay);
      specparm_planck = colh2o(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 8.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(3) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(3) + js1;
      inds = indself(lay);
      indf = indfor(lay);
      indm = indminor(lay);
      if (specparm < 0.125) {
        p = fs - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else if (specparm > 0.875) {
        p = -fs;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else {
        fac000 = (1.0 - fs) * fac00(lay);
        fac010 = (1.0 - fs) * fac10(lay);
        fac100 = fs * fac00(lay);
        fac110 = fs * fac10(lay);
      }
      if (specparm1 < 0.125) {
        p = fs1 - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else if (specparm1 > 0.875) {
        p = -fs1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else {
        fac001 = (1.0 - fs1) * fac01(lay);
        fac011 = (1.0 - fs1) * fac11(lay);
        fac101 = fs1 * fac01(lay);
        fac111 = fs1 * fac11(lay);
      }
      for (ig = 1; ig <= ng3; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        n2om1 = ka_mn2o(jmn2o, indm, ig) + fmn2o * (ka_mn2o(jmn2o + 1, indm, ig) - ka_mn2o(jmn2o, indm, ig));
        n2om2 = ka_mn2o(jmn2o, indm + 1, ig) + fmn2o * (ka_mn2o(jmn2o + 1, indm + 1, ig) - ka_mn2o(jmn2o, indm + 1, ig));
        absn2o = n2om1 + minorfrac(lay) * (n2om2 - n2om1);
        if (specparm < 0.125) {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac200 * absa(ind0 + 2, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac210 * absa(ind0 + 11, ig));
        } else if (specparm > 0.875) {
          tau_major = speccomb * (fac200 * absa(ind0 - 1, ig) + fac100 * absa(ind0, ig) + fac000 * absa(ind0 + 1, ig) + fac210 * absa(ind0 + 8, ig) + fac110 * absa(ind0 + 9, ig) + fac010 * absa(ind0 + 10, ig));
        } else {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig));
        }
        if (specparm1 < 0.125) {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac201 * absa(ind1 + 2, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig) + fac211 * absa(ind1 + 11, ig));
        } else if (specparm1 > 0.875) {
          tau_major1 = speccomb1 * (fac201 * absa(ind1 - 1, ig) + fac101 * absa(ind1, ig) + fac001 * absa(ind1 + 1, ig) + fac211 * absa(ind1 + 8, ig) + fac111 * absa(ind1 + 9, ig) + fac011 * absa(ind1 + 10, ig));
        } else {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig));
        }
        taug(lay, ngs2 + ig) = tau_major + tau_major1 + tauself + taufor + adjcoln2o * absn2o;
        fracs(lay, ngs2 + ig) = fracrefa(ig, jpl) + fpl * (fracrefa(ig, jpl + 1) - fracrefa(ig, jpl));
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      speccomb = colh2o(lay) + rat_h2oco2(lay) * colco2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 4.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colh2o(lay) + rat_h2oco2_1(lay) * colco2(lay);
      specparm1 = colh2o(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 4.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs1) * fac01(lay);
      fac011 = (1.0 - fs1) * fac11(lay);
      fac101 = fs1 * fac01(lay);
      fac111 = fs1 * fac11(lay);
      speccomb_mn2o = colh2o(lay) + refrat_m_b * colco2(lay);
      specparm_mn2o = colh2o(lay) / speccomb_mn2o;
      if (specparm_mn2o >= oneminus) specparm_mn2o = oneminus;
      specmult_mn2o = 4.0 * specparm_mn2o;
      jmn2o = 1 + int(specmult_mn2o);
      fmn2o = Kokkos::fmod(specmult_mn2o, 1.0);
      fmn2omf = minorfrac(lay) * fmn2o;
      chi_n2o = coln2o(lay) / coldry(lay);
      ratn2o = 1.0e20 * chi_n2o / chi_mls(4, jp(lay) + 1);
      if (ratn2o > 1.5) {
        adjfac = 0.5 + Kokkos::pow((ratn2o - 0.5), 0.65);
        adjcoln2o = adjfac * chi_mls(4, jp(lay) + 1) * coldry(lay) * 1.0e-20;
      } else {
        adjcoln2o = coln2o(lay);
      }
      speccomb_planck = colh2o(lay) + refrat_planck_b * colco2(lay);
      specparm_planck = colh2o(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 4.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(3) + js;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(3) + js1;
      indf = indfor(lay);
      indm = indminor(lay);
      for (ig = 1; ig <= ng3; ++ig) {
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        n2om1 = kb_mn2o(jmn2o, indm, ig) + fmn2o * (kb_mn2o(jmn2o + 1, indm, ig) - kb_mn2o(jmn2o, indm, ig));
        n2om2 = kb_mn2o(jmn2o, indm + 1, ig) + fmn2o * (kb_mn2o(jmn2o + 1, indm + 1, ig) - kb_mn2o(jmn2o, indm + 1, ig));
        absn2o = n2om1 + minorfrac(lay) * (n2om2 - n2om1);
        taug(lay, ngs2 + ig) = speccomb * (fac000 * absb(ind0, ig) + fac100 * absb(ind0 + 1, ig) + fac010 * absb(ind0 + 5, ig) + fac110 * absb(ind0 + 6, ig)) + speccomb1 * (fac001 * absb(ind1, ig) + fac101 * absb(ind1 + 1, ig) + fac011 * absb(ind1 + 5, ig) + fac111 * absb(ind1 + 6, ig)) + taufor + adjcoln2o * absn2o;
        fracs(lay, ngs2 + ig) = fracrefb(ig, jpl) + fpl * (fracrefb(ig, jpl + 1) - fracrefb(ig, jpl));
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb4() {
    const auto& fracrefa = kg04.fracrefa;
    const auto& fracrefb = kg04.fracrefb;
    const auto& absa = kg04.absa;
    const auto& absb = kg04.absb;
    const auto& selfref = kg04.selfref;
    const auto& forref = kg04.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int ig;
    int js;
    int js1;
    int jpl;
    Real speccomb;
    Real specparm;
    Real specmult;
    Real fs;
    Real speccomb1;
    Real specparm1;
    Real specmult1;
    Real fs1;
    Real speccomb_planck;
    Real specparm_planck;
    Real specmult_planck;
    Real fpl;
    Real p;
    Real p4;
    Real fk0;
    Real fk1;
    Real fk2;
    Real fac000;
    Real fac100;
    Real fac200;
    Real fac010;
    Real fac110;
    Real fac210;
    Real fac001;
    Real fac101;
    Real fac201;
    Real fac011;
    Real fac111;
    Real fac211;
    Real tauself;
    Real taufor;
    Real refrat_planck_a;
    Real refrat_planck_b;
    Real tau_major;
    Real tau_major1;
    refrat_planck_a = chi_mls(1, 11) / chi_mls(2, 11);
    refrat_planck_b = chi_mls(3, 13) / chi_mls(2, 13);
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colh2o(lay) + rat_h2oco2(lay) * colco2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colh2o(lay) + rat_h2oco2_1(lay) * colco2(lay);
      specparm1 = colh2o(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 8.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      speccomb_planck = colh2o(lay) + refrat_planck_a * colco2(lay);
      specparm_planck = colh2o(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 8.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(4) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(4) + js1;
      inds = indself(lay);
      indf = indfor(lay);
      if (specparm < 0.125) {
        p = fs - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else if (specparm > 0.875) {
        p = -fs;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else {
        fac000 = (1.0 - fs) * fac00(lay);
        fac010 = (1.0 - fs) * fac10(lay);
        fac100 = fs * fac00(lay);
        fac110 = fs * fac10(lay);
      }
      if (specparm1 < 0.125) {
        p = fs1 - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else if (specparm1 > 0.875) {
        p = -fs1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else {
        fac001 = (1.0 - fs1) * fac01(lay);
        fac011 = (1.0 - fs1) * fac11(lay);
        fac101 = fs1 * fac01(lay);
        fac111 = fs1 * fac11(lay);
      }
      for (ig = 1; ig <= ng4; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        if (specparm < 0.125) {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac200 * absa(ind0 + 2, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac210 * absa(ind0 + 11, ig));
        } else if (specparm > 0.875) {
          tau_major = speccomb * (fac200 * absa(ind0 - 1, ig) + fac100 * absa(ind0, ig) + fac000 * absa(ind0 + 1, ig) + fac210 * absa(ind0 + 8, ig) + fac110 * absa(ind0 + 9, ig) + fac010 * absa(ind0 + 10, ig));
        } else {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig));
        }
        if (specparm1 < 0.125) {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac201 * absa(ind1 + 2, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig) + fac211 * absa(ind1 + 11, ig));
        } else if (specparm1 > 0.875) {
          tau_major1 = speccomb1 * (fac201 * absa(ind1 - 1, ig) + fac101 * absa(ind1, ig) + fac001 * absa(ind1 + 1, ig) + fac211 * absa(ind1 + 8, ig) + fac111 * absa(ind1 + 9, ig) + fac011 * absa(ind1 + 10, ig));
        } else {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig));
        }
        taug(lay, ngs3 + ig) = tau_major + tau_major1 + tauself + taufor;
        fracs(lay, ngs3 + ig) = fracrefa(ig, jpl) + fpl * (fracrefa(ig, jpl + 1) - fracrefa(ig, jpl));
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      speccomb = colo3(lay) + rat_o3co2(lay) * colco2(lay);
      specparm = colo3(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 4.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colo3(lay) + rat_o3co2_1(lay) * colco2(lay);
      specparm1 = colo3(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 4.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs1) * fac01(lay);
      fac011 = (1.0 - fs1) * fac11(lay);
      fac101 = fs1 * fac01(lay);
      fac111 = fs1 * fac11(lay);
      speccomb_planck = colo3(lay) + refrat_planck_b * colco2(lay);
      specparm_planck = colo3(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 4.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(4) + js;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(4) + js1;
      for (ig = 1; ig <= ng4; ++ig) {
        taug(lay, ngs3 + ig) = speccomb * (fac000 * absb(ind0, ig) + fac100 * absb(ind0 + 1, ig) + fac010 * absb(ind0 + 5, ig) + fac110 * absb(ind0 + 6, ig)) + speccomb1 * (fac001 * absb(ind1, ig) + fac101 * absb(ind1 + 1, ig) + fac011 * absb(ind1 + 5, ig) + fac111 * absb(ind1 + 6, ig));
        fracs(lay, ngs3 + ig) = fracrefb(ig, jpl) + fpl * (fracrefb(ig, jpl + 1) - fracrefb(ig, jpl));
      }
      taug(lay, ngs3 + 8) = taug(lay, ngs3 + 8) * 0.92;
      taug(lay, ngs3 + 9) = taug(lay, ngs3 + 9) * 0.88;
      taug(lay, ngs3 + 10) = taug(lay, ngs3 + 10) * 1.07;
      taug(lay, ngs3 + 11) = taug(lay, ngs3 + 11) * 1.1;
      taug(lay, ngs3 + 12) = taug(lay, ngs3 + 12) * 0.99;
      taug(lay, ngs3 + 13) = taug(lay, ngs3 + 13) * 0.88;
      taug(lay, ngs3 + 14) = taug(lay, ngs3 + 14) * 0.943;
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb5() {
    const auto& fracrefa = kg05.fracrefa;
    const auto& fracrefb = kg05.fracrefb;
    const auto& absa = kg05.absa;
    const auto& absb = kg05.absb;
    const auto& ka_mo3 = kg05.ka_mo3;
    const auto& selfref = kg05.selfref;
    const auto& forref = kg05.forref;
    const auto& ccl4 = kg05.ccl4;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int indm;
    int ig;
    int js;
    int js1;
    int jmo3;
    int jpl;
    Real speccomb;
    Real specparm;
    Real specmult;
    Real fs;
    Real speccomb1;
    Real specparm1;
    Real specmult1;
    Real fs1;
    Real speccomb_mo3;
    Real specparm_mo3;
    Real specmult_mo3;
    Real fmo3;
    Real speccomb_planck;
    Real specparm_planck;
    Real specmult_planck;
    Real fpl;
    Real p;
    Real p4;
    Real fk0;
    Real fk1;
    Real fk2;
    Real fac000;
    Real fac100;
    Real fac200;
    Real fac010;
    Real fac110;
    Real fac210;
    Real fac001;
    Real fac101;
    Real fac201;
    Real fac011;
    Real fac111;
    Real fac211;
    Real tauself;
    Real taufor;
    Real o3m1;
    Real o3m2;
    Real abso3;
    Real refrat_planck_a;
    Real refrat_planck_b;
    Real refrat_m_a;
    Real tau_major;
    Real tau_major1;
    refrat_planck_a = chi_mls(1, 5) / chi_mls(2, 5);
    refrat_planck_b = chi_mls(3, 43) / chi_mls(2, 43);
    refrat_m_a = chi_mls(1, 7) / chi_mls(2, 7);
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colh2o(lay) + rat_h2oco2(lay) * colco2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colh2o(lay) + rat_h2oco2_1(lay) * colco2(lay);
      specparm1 = colh2o(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 8.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      speccomb_mo3 = colh2o(lay) + refrat_m_a * colco2(lay);
      specparm_mo3 = colh2o(lay) / speccomb_mo3;
      if (specparm_mo3 >= oneminus) specparm_mo3 = oneminus;
      specmult_mo3 = 8.0 * specparm_mo3;
      jmo3 = 1 + int(specmult_mo3);
      fmo3 = Kokkos::fmod(specmult_mo3, 1.0);
      speccomb_planck = colh2o(lay) + refrat_planck_a * colco2(lay);
      specparm_planck = colh2o(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 8.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(5) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(5) + js1;
      inds = indself(lay);
      indf = indfor(lay);
      indm = indminor(lay);
      if (specparm < 0.125) {
        p = fs - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else if (specparm > 0.875) {
        p = -fs;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else {
        fac000 = (1.0 - fs) * fac00(lay);
        fac010 = (1.0 - fs) * fac10(lay);
        fac100 = fs * fac00(lay);
        fac110 = fs * fac10(lay);
      }
      if (specparm1 < 0.125) {
        p = fs1 - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else if (specparm1 > 0.875) {
        p = -fs1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else {
        fac001 = (1.0 - fs1) * fac01(lay);
        fac011 = (1.0 - fs1) * fac11(lay);
        fac101 = fs1 * fac01(lay);
        fac111 = fs1 * fac11(lay);
      }
      for (ig = 1; ig <= ng5; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        o3m1 = ka_mo3(jmo3, indm, ig) + fmo3 * (ka_mo3(jmo3 + 1, indm, ig) - ka_mo3(jmo3, indm, ig));
        o3m2 = ka_mo3(jmo3, indm + 1, ig) + fmo3 * (ka_mo3(jmo3 + 1, indm + 1, ig) - ka_mo3(jmo3, indm + 1, ig));
        abso3 = o3m1 + minorfrac(lay) * (o3m2 - o3m1);
        if (specparm < 0.125) {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac200 * absa(ind0 + 2, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac210 * absa(ind0 + 11, ig));
        } else if (specparm > 0.875) {
          tau_major = speccomb * (fac200 * absa(ind0 - 1, ig) + fac100 * absa(ind0, ig) + fac000 * absa(ind0 + 1, ig) + fac210 * absa(ind0 + 8, ig) + fac110 * absa(ind0 + 9, ig) + fac010 * absa(ind0 + 10, ig));
        } else {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig));
        }
        if (specparm1 < 0.125) {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac201 * absa(ind1 + 2, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig) + fac211 * absa(ind1 + 11, ig));
        } else if (specparm1 > 0.875) {
          tau_major1 = speccomb1 * (fac201 * absa(ind1 - 1, ig) + fac101 * absa(ind1, ig) + fac001 * absa(ind1 + 1, ig) + fac211 * absa(ind1 + 8, ig) + fac111 * absa(ind1 + 9, ig) + fac011 * absa(ind1 + 10, ig));
        } else {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig));
        }
        taug(lay, ngs4 + ig) = tau_major + tau_major1 + tauself + taufor + abso3 * colo3(lay) + wx(1, lay) * ccl4(ig);
        fracs(lay, ngs4 + ig) = fracrefa(ig, jpl) + fpl * (fracrefa(ig, jpl + 1) - fracrefa(ig, jpl));
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      speccomb = colo3(lay) + rat_o3co2(lay) * colco2(lay);
      specparm = colo3(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 4.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colo3(lay) + rat_o3co2_1(lay) * colco2(lay);
      specparm1 = colo3(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 4.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs1) * fac01(lay);
      fac011 = (1.0 - fs1) * fac11(lay);
      fac101 = fs1 * fac01(lay);
      fac111 = fs1 * fac11(lay);
      speccomb_planck = colo3(lay) + refrat_planck_b * colco2(lay);
      specparm_planck = colo3(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 4.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(5) + js;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(5) + js1;
      for (ig = 1; ig <= ng5; ++ig) {
        taug(lay, ngs4 + ig) = speccomb * (fac000 * absb(ind0, ig) + fac100 * absb(ind0 + 1, ig) + fac010 * absb(ind0 + 5, ig) + fac110 * absb(ind0 + 6, ig)) + speccomb1 * (fac001 * absb(ind1, ig) + fac101 * absb(ind1 + 1, ig) + fac011 * absb(ind1 + 5, ig) + fac111 * absb(ind1 + 6, ig)) + wx(1, lay) * ccl4(ig);
        fracs(lay, ngs4 + ig) = fracrefb(ig, jpl) + fpl * (fracrefb(ig, jpl + 1) - fracrefb(ig, jpl));
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb6() {
    const auto& fracrefa = kg06.fracrefa;
    const auto& absa = kg06.absa;
    const auto& ka_mco2 = kg06.ka_mco2;
    const auto& selfref = kg06.selfref;
    const auto& forref = kg06.forref;
    const auto& cfc11adj = kg06.cfc11adj;
    const auto& cfc12 = kg06.cfc12;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int indm;
    int ig;
    Real chi_co2;
    Real ratco2;
    Real adjfac;
    Real adjcolco2;
    Real tauself;
    Real taufor;
    Real absco2;
    for (lay = 1; lay <= laytrop; ++lay) {
      chi_co2 = colco2(lay) / (coldry(lay));
      ratco2 = 1.0e20 * chi_co2 / chi_mls(2, jp(lay) + 1);
      if (ratco2 > 3.0) {
        adjfac = 2.0 + Kokkos::pow((ratco2 - 2.0), 0.77);
        adjcolco2 = adjfac * chi_mls(2, jp(lay) + 1) * coldry(lay) * 1.0e-20;
      } else {
        adjcolco2 = colco2(lay);
      }
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(6) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(6) + 1;
      inds = indself(lay);
      indf = indfor(lay);
      indm = indminor(lay);
      for (ig = 1; ig <= ng6; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        absco2 = (ka_mco2(indm, ig) + minorfrac(lay) * (ka_mco2(indm + 1, ig) - ka_mco2(indm, ig)));
        taug(lay, ngs5 + ig) = colh2o(lay) * (fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + tauself + taufor + adjcolco2 * absco2 + wx(2, lay) * cfc11adj(ig) + wx(3, lay) * cfc12(ig);
        fracs(lay, ngs5 + ig) = fracrefa(ig);
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      for (ig = 1; ig <= ng6; ++ig) {
        taug(lay, ngs5 + ig) = 0.0 + wx(2, lay) * cfc11adj(ig) + wx(3, lay) * cfc12(ig);
        fracs(lay, ngs5 + ig) = fracrefa(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb7() {
    const auto& fracrefa = kg07.fracrefa;
    const auto& fracrefb = kg07.fracrefb;
    const auto& absa = kg07.absa;
    const auto& absb = kg07.absb;
    const auto& ka_mco2 = kg07.ka_mco2;
    const auto& kb_mco2 = kg07.kb_mco2;
    const auto& selfref = kg07.selfref;
    const auto& forref = kg07.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int indm;
    int ig;
    int js;
    int js1;
    int jmco2;
    int jpl;
    Real speccomb;
    Real specparm;
    Real specmult;
    Real fs;
    Real speccomb1;
    Real specparm1;
    Real specmult1;
    Real fs1;
    Real speccomb_mco2;
    Real specparm_mco2;
    Real specmult_mco2;
    Real fmco2;
    Real speccomb_planck;
    Real specparm_planck;
    Real specmult_planck;
    Real fpl;
    Real p;
    Real p4;
    Real fk0;
    Real fk1;
    Real fk2;
    Real fac000;
    Real fac100;
    Real fac200;
    Real fac010;
    Real fac110;
    Real fac210;
    Real fac001;
    Real fac101;
    Real fac201;
    Real fac011;
    Real fac111;
    Real fac211;
    Real tauself;
    Real taufor;
    Real co2m1;
    Real co2m2;
    Real absco2;
    Real chi_co2;
    Real ratco2;
    Real adjfac;
    Real adjcolco2;
    Real refrat_planck_a;
    Real refrat_m_a;
    Real tau_major;
    Real tau_major1;
    refrat_planck_a = chi_mls(1, 3) / chi_mls(3, 3);
    refrat_m_a = chi_mls(1, 3) / chi_mls(3, 3);
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colh2o(lay) + rat_h2oo3(lay) * colo3(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colh2o(lay) + rat_h2oo3_1(lay) * colo3(lay);
      specparm1 = colh2o(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 8.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      speccomb_mco2 = colh2o(lay) + refrat_m_a * colo3(lay);
      specparm_mco2 = colh2o(lay) / speccomb_mco2;
      if (specparm_mco2 >= oneminus) specparm_mco2 = oneminus;
      specmult_mco2 = 8.0 * specparm_mco2;
      jmco2 = 1 + int(specmult_mco2);
      fmco2 = Kokkos::fmod(specmult_mco2, 1.0);
      chi_co2 = colco2(lay) / (coldry(lay));
      ratco2 = 1.0e20 * chi_co2 / chi_mls(2, jp(lay) + 1);
      if (ratco2 > 3.0) {
        adjfac = 3.0 + Kokkos::pow((ratco2 - 3.0), 0.79);
        adjcolco2 = adjfac * chi_mls(2, jp(lay) + 1) * coldry(lay) * 1.0e-20;
      } else {
        adjcolco2 = colco2(lay);
      }
      speccomb_planck = colh2o(lay) + refrat_planck_a * colo3(lay);
      specparm_planck = colh2o(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 8.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(7) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(7) + js1;
      inds = indself(lay);
      indf = indfor(lay);
      indm = indminor(lay);
      if (specparm < 0.125) {
        p = fs - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else if (specparm > 0.875) {
        p = -fs;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else {
        fac000 = (1.0 - fs) * fac00(lay);
        fac010 = (1.0 - fs) * fac10(lay);
        fac100 = fs * fac00(lay);
        fac110 = fs * fac10(lay);
      }
      if (specparm1 < 0.125) {
        p = fs1 - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else if (specparm1 > 0.875) {
        p = -fs1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else {
        fac001 = (1.0 - fs1) * fac01(lay);
        fac011 = (1.0 - fs1) * fac11(lay);
        fac101 = fs1 * fac01(lay);
        fac111 = fs1 * fac11(lay);
      }
      for (ig = 1; ig <= ng7; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        co2m1 = ka_mco2(jmco2, indm, ig) + fmco2 * (ka_mco2(jmco2 + 1, indm, ig) - ka_mco2(jmco2, indm, ig));
        co2m2 = ka_mco2(jmco2, indm + 1, ig) + fmco2 * (ka_mco2(jmco2 + 1, indm + 1, ig) - ka_mco2(jmco2, indm + 1, ig));
        absco2 = co2m1 + minorfrac(lay) * (co2m2 - co2m1);
        if (specparm < 0.125) {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac200 * absa(ind0 + 2, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac210 * absa(ind0 + 11, ig));
        } else if (specparm > 0.875) {
          tau_major = speccomb * (fac200 * absa(ind0 - 1, ig) + fac100 * absa(ind0, ig) + fac000 * absa(ind0 + 1, ig) + fac210 * absa(ind0 + 8, ig) + fac110 * absa(ind0 + 9, ig) + fac010 * absa(ind0 + 10, ig));
        } else {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig));
        }
        if (specparm1 < 0.125) {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac201 * absa(ind1 + 2, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig) + fac211 * absa(ind1 + 11, ig));
        } else if (specparm1 > 0.875) {
          tau_major1 = speccomb1 * (fac201 * absa(ind1 - 1, ig) + fac101 * absa(ind1, ig) + fac001 * absa(ind1 + 1, ig) + fac211 * absa(ind1 + 8, ig) + fac111 * absa(ind1 + 9, ig) + fac011 * absa(ind1 + 10, ig));
        } else {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig));
        }
        taug(lay, ngs6 + ig) = tau_major + tau_major1 + tauself + taufor + adjcolco2 * absco2;
        fracs(lay, ngs6 + ig) = fracrefa(ig, jpl) + fpl * (fracrefa(ig, jpl + 1) - fracrefa(ig, jpl));
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      chi_co2 = colco2(lay) / (coldry(lay));
      ratco2 = 1.0e20 * chi_co2 / chi_mls(2, jp(lay) + 1);
      if (ratco2 > 3.0) {
        adjfac = 2.0 + Kokkos::pow((ratco2 - 2.0), 0.79);
        adjcolco2 = adjfac * chi_mls(2, jp(lay) + 1) * coldry(lay) * 1.0e-20;
      } else {
        adjcolco2 = colco2(lay);
      }
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(7) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(7) + 1;
      indm = indminor(lay);
      for (ig = 1; ig <= ng7; ++ig) {
        absco2 = kb_mco2(indm, ig) + minorfrac(lay) * (kb_mco2(indm + 1, ig) - kb_mco2(indm, ig));
        taug(lay, ngs6 + ig) = colo3(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig)) + adjcolco2 * absco2;
        fracs(lay, ngs6 + ig) = fracrefb(ig);
      }
      taug(lay, ngs6 + 6) = taug(lay, ngs6 + 6) * 0.92;
      taug(lay, ngs6 + 7) = taug(lay, ngs6 + 7) * 0.88;
      taug(lay, ngs6 + 8) = taug(lay, ngs6 + 8) * 1.07;
      taug(lay, ngs6 + 9) = taug(lay, ngs6 + 9) * 1.1;
      taug(lay, ngs6 + 10) = taug(lay, ngs6 + 10) * 0.99;
      taug(lay, ngs6 + 11) = taug(lay, ngs6 + 11) * 0.855;
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb8() {
    const auto& fracrefa = kg08.fracrefa;
    const auto& fracrefb = kg08.fracrefb;
    const auto& absa = kg08.absa;
    const auto& absb = kg08.absb;
    const auto& ka_mco2 = kg08.ka_mco2;
    const auto& ka_mn2o = kg08.ka_mn2o;
    const auto& ka_mo3 = kg08.ka_mo3;
    const auto& kb_mco2 = kg08.kb_mco2;
    const auto& kb_mn2o = kg08.kb_mn2o;
    const auto& selfref = kg08.selfref;
    const auto& forref = kg08.forref;
    const auto& cfc12 = kg08.cfc12;
    const auto& cfc22adj = kg08.cfc22adj;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int indm;
    int ig;
    Real tauself;
    Real taufor;
    Real absco2;
    Real abso3;
    Real absn2o;
    Real chi_co2;
    Real ratco2;
    Real adjfac;
    Real adjcolco2;
    for (lay = 1; lay <= laytrop; ++lay) {
      chi_co2 = colco2(lay) / (coldry(lay));
      ratco2 = 1.0e20 * chi_co2 / chi_mls(2, jp(lay) + 1);
      if (ratco2 > 3.0) {
        adjfac = 2.0 + Kokkos::pow((ratco2 - 2.0), 0.65);
        adjcolco2 = adjfac * chi_mls(2, jp(lay) + 1) * coldry(lay) * 1.0e-20;
      } else {
        adjcolco2 = colco2(lay);
      }
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(8) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(8) + 1;
      inds = indself(lay);
      indf = indfor(lay);
      indm = indminor(lay);
      for (ig = 1; ig <= ng8; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        absco2 = (ka_mco2(indm, ig) + minorfrac(lay) * (ka_mco2(indm + 1, ig) - ka_mco2(indm, ig)));
        abso3 = (ka_mo3(indm, ig) + minorfrac(lay) * (ka_mo3(indm + 1, ig) - ka_mo3(indm, ig)));
        absn2o = (ka_mn2o(indm, ig) + minorfrac(lay) * (ka_mn2o(indm + 1, ig) - ka_mn2o(indm, ig)));
        taug(lay, ngs7 + ig) = colh2o(lay) * (fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + tauself + taufor + adjcolco2 * absco2 + colo3(lay) * abso3 + coln2o(lay) * absn2o + wx(3, lay) * cfc12(ig) + wx(4, lay) * cfc22adj(ig);
        fracs(lay, ngs7 + ig) = fracrefa(ig);
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      chi_co2 = colco2(lay) / coldry(lay);
      ratco2 = 1.0e20 * chi_co2 / chi_mls(2, jp(lay) + 1);
      if (ratco2 > 3.0) {
        adjfac = 2.0 + Kokkos::pow((ratco2 - 2.0), 0.65);
        adjcolco2 = adjfac * chi_mls(2, jp(lay) + 1) * coldry(lay) * 1.0e-20;
      } else {
        adjcolco2 = colco2(lay);
      }
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(8) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(8) + 1;
      indm = indminor(lay);
      for (ig = 1; ig <= ng8; ++ig) {
        absco2 = (kb_mco2(indm, ig) + minorfrac(lay) * (kb_mco2(indm + 1, ig) - kb_mco2(indm, ig)));
        absn2o = (kb_mn2o(indm, ig) + minorfrac(lay) * (kb_mn2o(indm + 1, ig) - kb_mn2o(indm, ig)));
        taug(lay, ngs7 + ig) = colo3(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig)) + adjcolco2 * absco2 + coln2o(lay) * absn2o + wx(3, lay) * cfc12(ig) + wx(4, lay) * cfc22adj(ig);
        fracs(lay, ngs7 + ig) = fracrefb(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb9() {
    const auto& fracrefa = kg09.fracrefa;
    const auto& fracrefb = kg09.fracrefb;
    const auto& absa = kg09.absa;
    const auto& absb = kg09.absb;
    const auto& ka_mn2o = kg09.ka_mn2o;
    const auto& kb_mn2o = kg09.kb_mn2o;
    const auto& selfref = kg09.selfref;
    const auto& forref = kg09.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int indm;
    int ig;
    int js;
    int js1;
    int jmn2o;
    int jpl;
    Real speccomb;
    Real specparm;
    Real specmult;
    Real fs;
    Real speccomb1;
    Real specparm1;
    Real specmult1;
    Real fs1;
    Real speccomb_mn2o;
    Real specparm_mn2o;
    Real specmult_mn2o;
    Real fmn2o;
    Real speccomb_planck;
    Real specparm_planck;
    Real specmult_planck;
    Real fpl;
    Real p;
    Real p4;
    Real fk0;
    Real fk1;
    Real fk2;
    Real fac000;
    Real fac100;
    Real fac200;
    Real fac010;
    Real fac110;
    Real fac210;
    Real fac001;
    Real fac101;
    Real fac201;
    Real fac011;
    Real fac111;
    Real fac211;
    Real tauself;
    Real taufor;
    Real n2om1;
    Real n2om2;
    Real absn2o;
    Real chi_n2o;
    Real ratn2o;
    Real adjfac;
    Real adjcoln2o;
    Real refrat_planck_a;
    Real refrat_m_a;
    Real tau_major;
    Real tau_major1;
    refrat_planck_a = chi_mls(1, 9) / chi_mls(6, 9);
    refrat_m_a = chi_mls(1, 3) / chi_mls(6, 3);
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colh2o(lay) + rat_h2och4(lay) * colch4(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colh2o(lay) + rat_h2och4_1(lay) * colch4(lay);
      specparm1 = colh2o(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 8.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      speccomb_mn2o = colh2o(lay) + refrat_m_a * colch4(lay);
      specparm_mn2o = colh2o(lay) / speccomb_mn2o;
      if (specparm_mn2o >= oneminus) specparm_mn2o = oneminus;
      specmult_mn2o = 8.0 * specparm_mn2o;
      jmn2o = 1 + int(specmult_mn2o);
      fmn2o = Kokkos::fmod(specmult_mn2o, 1.0);
      chi_n2o = coln2o(lay) / (coldry(lay));
      ratn2o = 1.0e20 * chi_n2o / chi_mls(4, jp(lay) + 1);
      if (ratn2o > 1.5) {
        adjfac = 0.5 + Kokkos::pow((ratn2o - 0.5), 0.65);
        adjcoln2o = adjfac * chi_mls(4, jp(lay) + 1) * coldry(lay) * 1.0e-20;
      } else {
        adjcoln2o = coln2o(lay);
      }
      speccomb_planck = colh2o(lay) + refrat_planck_a * colch4(lay);
      specparm_planck = colh2o(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 8.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(9) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(9) + js1;
      inds = indself(lay);
      indf = indfor(lay);
      indm = indminor(lay);
      if (specparm < 0.125) {
        p = fs - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else if (specparm > 0.875) {
        p = -fs;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else {
        fac000 = (1.0 - fs) * fac00(lay);
        fac010 = (1.0 - fs) * fac10(lay);
        fac100 = fs * fac00(lay);
        fac110 = fs * fac10(lay);
      }
      if (specparm1 < 0.125) {
        p = fs1 - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else if (specparm1 > 0.875) {
        p = -fs1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else {
        fac001 = (1.0 - fs1) * fac01(lay);
        fac011 = (1.0 - fs1) * fac11(lay);
        fac101 = fs1 * fac01(lay);
        fac111 = fs1 * fac11(lay);
      }
      for (ig = 1; ig <= ng9; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        n2om1 = ka_mn2o(jmn2o, indm, ig) + fmn2o * (ka_mn2o(jmn2o + 1, indm, ig) - ka_mn2o(jmn2o, indm, ig));
        n2om2 = ka_mn2o(jmn2o, indm + 1, ig) + fmn2o * (ka_mn2o(jmn2o + 1, indm + 1, ig) - ka_mn2o(jmn2o, indm + 1, ig));
        absn2o = n2om1 + minorfrac(lay) * (n2om2 - n2om1);
        if (specparm < 0.125) {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac200 * absa(ind0 + 2, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac210 * absa(ind0 + 11, ig));
        } else if (specparm > 0.875) {
          tau_major = speccomb * (fac200 * absa(ind0 - 1, ig) + fac100 * absa(ind0, ig) + fac000 * absa(ind0 + 1, ig) + fac210 * absa(ind0 + 8, ig) + fac110 * absa(ind0 + 9, ig) + fac010 * absa(ind0 + 10, ig));
        } else {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig));
        }
        if (specparm1 < 0.125) {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac201 * absa(ind1 + 2, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig) + fac211 * absa(ind1 + 11, ig));
        } else if (specparm1 > 0.875) {
          tau_major1 = speccomb1 * (fac201 * absa(ind1 - 1, ig) + fac101 * absa(ind1, ig) + fac001 * absa(ind1 + 1, ig) + fac211 * absa(ind1 + 8, ig) + fac111 * absa(ind1 + 9, ig) + fac011 * absa(ind1 + 10, ig));
        } else {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig));
        }
        taug(lay, ngs8 + ig) = tau_major + tau_major1 + tauself + taufor + adjcoln2o * absn2o;
        fracs(lay, ngs8 + ig) = fracrefa(ig, jpl) + fpl * (fracrefa(ig, jpl + 1) - fracrefa(ig, jpl));
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      chi_n2o = coln2o(lay) / (coldry(lay));
      ratn2o = 1.0e20 * chi_n2o / chi_mls(4, jp(lay) + 1);
      if (ratn2o > 1.5) {
        adjfac = 0.5 + Kokkos::pow((ratn2o - 0.5), 0.65);
        adjcoln2o = adjfac * chi_mls(4, jp(lay) + 1) * coldry(lay) * 1.0e-20;
      } else {
        adjcoln2o = coln2o(lay);
      }
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(9) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(9) + 1;
      indm = indminor(lay);
      for (ig = 1; ig <= ng9; ++ig) {
        absn2o = kb_mn2o(indm, ig) + minorfrac(lay) * (kb_mn2o(indm + 1, ig) - kb_mn2o(indm, ig));
        taug(lay, ngs8 + ig) = colch4(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig)) + adjcoln2o * absn2o;
        fracs(lay, ngs8 + ig) = fracrefb(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb10() {
    const auto& fracrefa = kg10.fracrefa;
    const auto& fracrefb = kg10.fracrefb;
    const auto& absa = kg10.absa;
    const auto& absb = kg10.absb;
    const auto& selfref = kg10.selfref;
    const auto& forref = kg10.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int ig;
    Real tauself;
    Real taufor;
    for (lay = 1; lay <= laytrop; ++lay) {
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(10) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(10) + 1;
      inds = indself(lay);
      indf = indfor(lay);
      for (ig = 1; ig <= ng10; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        taug(lay, ngs9 + ig) = colh2o(lay) * (fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + tauself + taufor;
        fracs(lay, ngs9 + ig) = fracrefa(ig);
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(10) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(10) + 1;
      indf = indfor(lay);
      for (ig = 1; ig <= ng10; ++ig) {
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        taug(lay, ngs9 + ig) = colh2o(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig)) + taufor;
        fracs(lay, ngs9 + ig) = fracrefb(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb11() {
    const auto& fracrefa = kg11.fracrefa;
    const auto& fracrefb = kg11.fracrefb;
    const auto& absa = kg11.absa;
    const auto& absb = kg11.absb;
    const auto& ka_mo2 = kg11.ka_mo2;
    const auto& kb_mo2 = kg11.kb_mo2;
    const auto& selfref = kg11.selfref;
    const auto& forref = kg11.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int indm;
    int ig;
    Real scaleo2;
    Real tauself;
    Real taufor;
    Real tauo2;
    for (lay = 1; lay <= laytrop; ++lay) {
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(11) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(11) + 1;
      inds = indself(lay);
      indf = indfor(lay);
      indm = indminor(lay);
      scaleo2 = colo2(lay) * scaleminor(lay);
      for (ig = 1; ig <= ng11; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        tauo2 = scaleo2 * (ka_mo2(indm, ig) + minorfrac(lay) * (ka_mo2(indm + 1, ig) - ka_mo2(indm, ig)));
        taug(lay, ngs10 + ig) = colh2o(lay) * (fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + tauself + taufor + tauo2;
        fracs(lay, ngs10 + ig) = fracrefa(ig);
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(11) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(11) + 1;
      indf = indfor(lay);
      indm = indminor(lay);
      scaleo2 = colo2(lay) * scaleminor(lay);
      for (ig = 1; ig <= ng11; ++ig) {
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        tauo2 = scaleo2 * (kb_mo2(indm, ig) + minorfrac(lay) * (kb_mo2(indm + 1, ig) - kb_mo2(indm, ig)));
        taug(lay, ngs10 + ig) = colh2o(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig)) + taufor + tauo2;
        fracs(lay, ngs10 + ig) = fracrefb(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb12() {
    const auto& fracrefa = kg12.fracrefa;
    const auto& absa = kg12.absa;
    const auto& selfref = kg12.selfref;
    const auto& forref = kg12.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int ig;
    int js;
    int js1;
    int jpl;
    Real speccomb;
    Real specparm;
    Real specmult;
    Real fs;
    Real speccomb1;
    Real specparm1;
    Real specmult1;
    Real fs1;
    Real speccomb_planck;
    Real specparm_planck;
    Real specmult_planck;
    Real fpl;
    Real p;
    Real p4;
    Real fk0;
    Real fk1;
    Real fk2;
    Real fac000;
    Real fac100;
    Real fac200;
    Real fac010;
    Real fac110;
    Real fac210;
    Real fac001;
    Real fac101;
    Real fac201;
    Real fac011;
    Real fac111;
    Real fac211;
    Real tauself;
    Real taufor;
    Real refrat_planck_a;
    Real tau_major;
    Real tau_major1;
    refrat_planck_a = chi_mls(1, 10) / chi_mls(2, 10);
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colh2o(lay) + rat_h2oco2(lay) * colco2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colh2o(lay) + rat_h2oco2_1(lay) * colco2(lay);
      specparm1 = colh2o(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 8.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      speccomb_planck = colh2o(lay) + refrat_planck_a * colco2(lay);
      specparm_planck = colh2o(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 8.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(12) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(12) + js1;
      inds = indself(lay);
      indf = indfor(lay);
      if (specparm < 0.125) {
        p = fs - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else if (specparm > 0.875) {
        p = -fs;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else {
        fac000 = (1.0 - fs) * fac00(lay);
        fac010 = (1.0 - fs) * fac10(lay);
        fac100 = fs * fac00(lay);
        fac110 = fs * fac10(lay);
      }
      if (specparm1 < 0.125) {
        p = fs1 - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else if (specparm1 > 0.875) {
        p = -fs1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else {
        fac001 = (1.0 - fs1) * fac01(lay);
        fac011 = (1.0 - fs1) * fac11(lay);
        fac101 = fs1 * fac01(lay);
        fac111 = fs1 * fac11(lay);
      }
      for (ig = 1; ig <= ng12; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        if (specparm < 0.125) {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac200 * absa(ind0 + 2, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac210 * absa(ind0 + 11, ig));
        } else if (specparm > 0.875) {
          tau_major = speccomb * (fac200 * absa(ind0 - 1, ig) + fac100 * absa(ind0, ig) + fac000 * absa(ind0 + 1, ig) + fac210 * absa(ind0 + 8, ig) + fac110 * absa(ind0 + 9, ig) + fac010 * absa(ind0 + 10, ig));
        } else {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig));
        }
        if (specparm1 < 0.125) {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac201 * absa(ind1 + 2, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig) + fac211 * absa(ind1 + 11, ig));
        } else if (specparm1 > 0.875) {
          tau_major1 = speccomb1 * (fac201 * absa(ind1 - 1, ig) + fac101 * absa(ind1, ig) + fac001 * absa(ind1 + 1, ig) + fac211 * absa(ind1 + 8, ig) + fac111 * absa(ind1 + 9, ig) + fac011 * absa(ind1 + 10, ig));
        } else {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig));
        }
        taug(lay, ngs11 + ig) = tau_major + tau_major1 + tauself + taufor;
        fracs(lay, ngs11 + ig) = fracrefa(ig, jpl) + fpl * (fracrefa(ig, jpl + 1) - fracrefa(ig, jpl));
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      for (ig = 1; ig <= ng12; ++ig) {
        taug(lay, ngs11 + ig) = 0.0;
        fracs(lay, ngs11 + ig) = 0.0;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb13() {
    const auto& fracrefa = kg13.fracrefa;
    const auto& fracrefb = kg13.fracrefb;
    const auto& absa = kg13.absa;
    const auto& ka_mco2 = kg13.ka_mco2;
    const auto& ka_mco = kg13.ka_mco;
    const auto& kb_mo3 = kg13.kb_mo3;
    const auto& selfref = kg13.selfref;
    const auto& forref = kg13.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int indm;
    int ig;
    int js;
    int js1;
    int jmco2;
    int jmco;
    int jpl;
    Real speccomb;
    Real specparm;
    Real specmult;
    Real fs;
    Real speccomb1;
    Real specparm1;
    Real specmult1;
    Real fs1;
    Real speccomb_mco2;
    Real specparm_mco2;
    Real specmult_mco2;
    Real fmco2;
    Real speccomb_mco;
    Real specparm_mco;
    Real specmult_mco;
    Real fmco;
    Real speccomb_planck;
    Real specparm_planck;
    Real specmult_planck;
    Real fpl;
    Real p;
    Real p4;
    Real fk0;
    Real fk1;
    Real fk2;
    Real fac000;
    Real fac100;
    Real fac200;
    Real fac010;
    Real fac110;
    Real fac210;
    Real fac001;
    Real fac101;
    Real fac201;
    Real fac011;
    Real fac111;
    Real fac211;
    Real tauself;
    Real taufor;
    Real co2m1;
    Real co2m2;
    Real absco2;
    Real com1;
    Real com2;
    Real absco;
    Real abso3;
    Real chi_co2;
    Real ratco2;
    Real adjfac;
    Real adjcolco2;
    Real refrat_planck_a;
    Real refrat_m_a;
    Real refrat_m_a3;
    Real tau_major;
    Real tau_major1;
    refrat_planck_a = chi_mls(1, 5) / chi_mls(4, 5);
    refrat_m_a = chi_mls(1, 1) / chi_mls(4, 1);
    refrat_m_a3 = chi_mls(1, 3) / chi_mls(4, 3);
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colh2o(lay) + rat_h2on2o(lay) * coln2o(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colh2o(lay) + rat_h2on2o_1(lay) * coln2o(lay);
      specparm1 = colh2o(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 8.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      speccomb_mco2 = colh2o(lay) + refrat_m_a * coln2o(lay);
      specparm_mco2 = colh2o(lay) / speccomb_mco2;
      if (specparm_mco2 >= oneminus) specparm_mco2 = oneminus;
      specmult_mco2 = 8.0 * specparm_mco2;
      jmco2 = 1 + int(specmult_mco2);
      fmco2 = Kokkos::fmod(specmult_mco2, 1.0);
      chi_co2 = colco2(lay) / (coldry(lay));
      ratco2 = 1.0e20 * chi_co2 / 3.55e-4;
      if (ratco2 > 3.0) {
        adjfac = 2.0 + Kokkos::pow((ratco2 - 2.0), 0.68);
        adjcolco2 = adjfac * 3.55e-4 * coldry(lay) * 1.0e-20;
      } else {
        adjcolco2 = colco2(lay);
      }
      speccomb_mco = colh2o(lay) + refrat_m_a3 * coln2o(lay);
      specparm_mco = colh2o(lay) / speccomb_mco;
      if (specparm_mco >= oneminus) specparm_mco = oneminus;
      specmult_mco = 8.0 * specparm_mco;
      jmco = 1 + int(specmult_mco);
      fmco = Kokkos::fmod(specmult_mco, 1.0);
      speccomb_planck = colh2o(lay) + refrat_planck_a * coln2o(lay);
      specparm_planck = colh2o(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 8.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(13) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(13) + js1;
      inds = indself(lay);
      indf = indfor(lay);
      indm = indminor(lay);
      if (specparm < 0.125) {
        p = fs - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else if (specparm > 0.875) {
        p = -fs;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else {
        fac000 = (1.0 - fs) * fac00(lay);
        fac010 = (1.0 - fs) * fac10(lay);
        fac100 = fs * fac00(lay);
        fac110 = fs * fac10(lay);
      }
      if (specparm1 < 0.125) {
        p = fs1 - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else if (specparm1 > 0.875) {
        p = -fs1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else {
        fac001 = (1.0 - fs1) * fac01(lay);
        fac011 = (1.0 - fs1) * fac11(lay);
        fac101 = fs1 * fac01(lay);
        fac111 = fs1 * fac11(lay);
      }
      for (ig = 1; ig <= ng13; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        co2m1 = ka_mco2(jmco2, indm, ig) + fmco2 * (ka_mco2(jmco2 + 1, indm, ig) - ka_mco2(jmco2, indm, ig));
        co2m2 = ka_mco2(jmco2, indm + 1, ig) + fmco2 * (ka_mco2(jmco2 + 1, indm + 1, ig) - ka_mco2(jmco2, indm + 1, ig));
        absco2 = co2m1 + minorfrac(lay) * (co2m2 - co2m1);
        com1 = ka_mco(jmco, indm, ig) + fmco * (ka_mco(jmco + 1, indm, ig) - ka_mco(jmco, indm, ig));
        com2 = ka_mco(jmco, indm + 1, ig) + fmco * (ka_mco(jmco + 1, indm + 1, ig) - ka_mco(jmco, indm + 1, ig));
        absco = com1 + minorfrac(lay) * (com2 - com1);
        if (specparm < 0.125) {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac200 * absa(ind0 + 2, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac210 * absa(ind0 + 11, ig));
        } else if (specparm > 0.875) {
          tau_major = speccomb * (fac200 * absa(ind0 - 1, ig) + fac100 * absa(ind0, ig) + fac000 * absa(ind0 + 1, ig) + fac210 * absa(ind0 + 8, ig) + fac110 * absa(ind0 + 9, ig) + fac010 * absa(ind0 + 10, ig));
        } else {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig));
        }
        if (specparm1 < 0.125) {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac201 * absa(ind1 + 2, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig) + fac211 * absa(ind1 + 11, ig));
        } else if (specparm1 > 0.875) {
          tau_major1 = speccomb1 * (fac201 * absa(ind1 - 1, ig) + fac101 * absa(ind1, ig) + fac001 * absa(ind1 + 1, ig) + fac211 * absa(ind1 + 8, ig) + fac111 * absa(ind1 + 9, ig) + fac011 * absa(ind1 + 10, ig));
        } else {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig));
        }
        taug(lay, ngs12 + ig) = tau_major + tau_major1 + tauself + taufor + adjcolco2 * absco2 + colco(lay) * absco;
        fracs(lay, ngs12 + ig) = fracrefa(ig, jpl) + fpl * (fracrefa(ig, jpl + 1) - fracrefa(ig, jpl));
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      indm = indminor(lay);
      for (ig = 1; ig <= ng13; ++ig) {
        abso3 = kb_mo3(indm, ig) + minorfrac(lay) * (kb_mo3(indm + 1, ig) - kb_mo3(indm, ig));
        taug(lay, ngs12 + ig) = colo3(lay) * abso3;
        fracs(lay, ngs12 + ig) = fracrefb(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb14() {
    const auto& fracrefa = kg14.fracrefa;
    const auto& fracrefb = kg14.fracrefb;
    const auto& absa = kg14.absa;
    const auto& absb = kg14.absb;
    const auto& selfref = kg14.selfref;
    const auto& forref = kg14.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int ig;
    Real tauself;
    Real taufor;
    for (lay = 1; lay <= laytrop; ++lay) {
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(14) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(14) + 1;
      inds = indself(lay);
      indf = indfor(lay);
      for (ig = 1; ig <= ng14; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        taug(lay, ngs13 + ig) = colco2(lay) * (fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + tauself + taufor;
        fracs(lay, ngs13 + ig) = fracrefa(ig);
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(14) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(14) + 1;
      for (ig = 1; ig <= ng14; ++ig) {
        taug(lay, ngs13 + ig) = colco2(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig));
        fracs(lay, ngs13 + ig) = fracrefb(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb15() {
    const auto& fracrefa = kg15.fracrefa;
    const auto& absa = kg15.absa;
    const auto& ka_mn2 = kg15.ka_mn2;
    const auto& selfref = kg15.selfref;
    const auto& forref = kg15.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int indm;
    int ig;
    int js;
    int js1;
    int jmn2;
    int jpl;
    Real speccomb;
    Real specparm;
    Real specmult;
    Real fs;
    Real speccomb1;
    Real specparm1;
    Real specmult1;
    Real fs1;
    Real speccomb_mn2;
    Real specparm_mn2;
    Real specmult_mn2;
    Real fmn2;
    Real speccomb_planck;
    Real specparm_planck;
    Real specmult_planck;
    Real fpl;
    Real p;
    Real p4;
    Real fk0;
    Real fk1;
    Real fk2;
    Real fac000;
    Real fac100;
    Real fac200;
    Real fac010;
    Real fac110;
    Real fac210;
    Real fac001;
    Real fac101;
    Real fac201;
    Real fac011;
    Real fac111;
    Real fac211;
    Real scalen2;
    Real tauself;
    Real taufor;
    Real n2m1;
    Real n2m2;
    Real taun2;
    Real refrat_planck_a;
    Real refrat_m_a;
    Real tau_major;
    Real tau_major1;
    refrat_planck_a = chi_mls(4, 1) / chi_mls(2, 1);
    refrat_m_a = chi_mls(4, 1) / chi_mls(2, 1);
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = coln2o(lay) + rat_n2oco2(lay) * colco2(lay);
      specparm = coln2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = coln2o(lay) + rat_n2oco2_1(lay) * colco2(lay);
      specparm1 = coln2o(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 8.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      speccomb_mn2 = coln2o(lay) + refrat_m_a * colco2(lay);
      specparm_mn2 = coln2o(lay) / speccomb_mn2;
      if (specparm_mn2 >= oneminus) specparm_mn2 = oneminus;
      specmult_mn2 = 8.0 * specparm_mn2;
      jmn2 = 1 + int(specmult_mn2);
      fmn2 = Kokkos::fmod(specmult_mn2, 1.0);
      speccomb_planck = coln2o(lay) + refrat_planck_a * colco2(lay);
      specparm_planck = coln2o(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 8.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(15) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(15) + js1;
      inds = indself(lay);
      indf = indfor(lay);
      indm = indminor(lay);
      scalen2 = colbrd(lay) * scaleminor(lay);
      if (specparm < 0.125) {
        p = fs - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else if (specparm > 0.875) {
        p = -fs;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else {
        fac000 = (1.0 - fs) * fac00(lay);
        fac010 = (1.0 - fs) * fac10(lay);
        fac100 = fs * fac00(lay);
        fac110 = fs * fac10(lay);
      }
      if (specparm1 < 0.125) {
        p = fs1 - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else if (specparm1 > 0.875) {
        p = -fs1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else {
        fac001 = (1.0 - fs1) * fac01(lay);
        fac011 = (1.0 - fs1) * fac11(lay);
        fac101 = fs1 * fac01(lay);
        fac111 = fs1 * fac11(lay);
      }
      for (ig = 1; ig <= ng15; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        n2m1 = ka_mn2(jmn2, indm, ig) + fmn2 * (ka_mn2(jmn2 + 1, indm, ig) - ka_mn2(jmn2, indm, ig));
        n2m2 = ka_mn2(jmn2, indm + 1, ig) + fmn2 * (ka_mn2(jmn2 + 1, indm + 1, ig) - ka_mn2(jmn2, indm + 1, ig));
        taun2 = scalen2 * (n2m1 + minorfrac(lay) * (n2m2 - n2m1));
        if (specparm < 0.125) {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac200 * absa(ind0 + 2, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac210 * absa(ind0 + 11, ig));
        } else if (specparm > 0.875) {
          tau_major = speccomb * (fac200 * absa(ind0 - 1, ig) + fac100 * absa(ind0, ig) + fac000 * absa(ind0 + 1, ig) + fac210 * absa(ind0 + 8, ig) + fac110 * absa(ind0 + 9, ig) + fac010 * absa(ind0 + 10, ig));
        } else {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig));
        }
        if (specparm1 < 0.125) {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac201 * absa(ind1 + 2, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig) + fac211 * absa(ind1 + 11, ig));
        } else if (specparm1 > 0.875) {
          tau_major1 = speccomb1 * (fac201 * absa(ind1 - 1, ig) + fac101 * absa(ind1, ig) + fac001 * absa(ind1 + 1, ig) + fac211 * absa(ind1 + 8, ig) + fac111 * absa(ind1 + 9, ig) + fac011 * absa(ind1 + 10, ig));
        } else {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig));
        }
        taug(lay, ngs14 + ig) = tau_major + tau_major1 + tauself + taufor + taun2;
        fracs(lay, ngs14 + ig) = fracrefa(ig, jpl) + fpl * (fracrefa(ig, jpl + 1) - fracrefa(ig, jpl));
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      for (ig = 1; ig <= ng15; ++ig) {
        taug(lay, ngs14 + ig) = 0.0;
        fracs(lay, ngs14 + ig) = 0.0;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taugb16() {
    const auto& fracrefa = kg16.fracrefa;
    const auto& fracrefb = kg16.fracrefb;
    const auto& absa = kg16.absa;
    const auto& absb = kg16.absb;
    const auto& selfref = kg16.selfref;
    const auto& forref = kg16.forref;
    int lay;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int ig;
    int js;
    int js1;
    int jpl;
    Real speccomb;
    Real specparm;
    Real specmult;
    Real fs;
    Real speccomb1;
    Real specparm1;
    Real specmult1;
    Real fs1;
    Real speccomb_planck;
    Real specparm_planck;
    Real specmult_planck;
    Real fpl;
    Real p;
    Real p4;
    Real fk0;
    Real fk1;
    Real fk2;
    Real fac000;
    Real fac100;
    Real fac200;
    Real fac010;
    Real fac110;
    Real fac210;
    Real fac001;
    Real fac101;
    Real fac201;
    Real fac011;
    Real fac111;
    Real fac211;
    Real tauself;
    Real taufor;
    Real refrat_planck_a;
    Real tau_major;
    Real tau_major1;
    refrat_planck_a = chi_mls(1, 6) / chi_mls(6, 6);
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colh2o(lay) + rat_h2och4(lay) * colch4(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      speccomb1 = colh2o(lay) + rat_h2och4_1(lay) * colch4(lay);
      specparm1 = colh2o(lay) / speccomb1;
      if (specparm1 >= oneminus) specparm1 = oneminus;
      specmult1 = 8.0 * (specparm1);
      js1 = 1 + int(specmult1);
      fs1 = Kokkos::fmod(specmult1, 1.0);
      speccomb_planck = colh2o(lay) + refrat_planck_a * colch4(lay);
      specparm_planck = colh2o(lay) / speccomb_planck;
      if (specparm_planck >= oneminus) specparm_planck = oneminus;
      specmult_planck = 8.0 * specparm_planck;
      jpl = 1 + int(specmult_planck);
      fpl = Kokkos::fmod(specmult_planck, 1.0);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(16) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(16) + js1;
      inds = indself(lay);
      indf = indfor(lay);
      if (specparm < 0.125) {
        p = fs - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else if (specparm > 0.875) {
        p = -fs;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac000 = fk0 * fac00(lay);
        fac100 = fk1 * fac00(lay);
        fac200 = fk2 * fac00(lay);
        fac010 = fk0 * fac10(lay);
        fac110 = fk1 * fac10(lay);
        fac210 = fk2 * fac10(lay);
      } else {
        fac000 = (1.0 - fs) * fac00(lay);
        fac010 = (1.0 - fs) * fac10(lay);
        fac100 = fs * fac00(lay);
        fac110 = fs * fac10(lay);
      }
      if (specparm1 < 0.125) {
        p = fs1 - 1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else if (specparm1 > 0.875) {
        p = -fs1;
        p4 = POWI<4>(p);
        fk0 = p4;
        fk1 = 1 - p - 2.0 * p4;
        fk2 = p + p4;
        fac001 = fk0 * fac01(lay);
        fac101 = fk1 * fac01(lay);
        fac201 = fk2 * fac01(lay);
        fac011 = fk0 * fac11(lay);
        fac111 = fk1 * fac11(lay);
        fac211 = fk2 * fac11(lay);
      } else {
        fac001 = (1.0 - fs1) * fac01(lay);
        fac011 = (1.0 - fs1) * fac11(lay);
        fac101 = fs1 * fac01(lay);
        fac111 = fs1 * fac11(lay);
      }
      for (ig = 1; ig <= ng16; ++ig) {
        tauself = selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig)));
        taufor = forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        if (specparm < 0.125) {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac200 * absa(ind0 + 2, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac210 * absa(ind0 + 11, ig));
        } else if (specparm > 0.875) {
          tau_major = speccomb * (fac200 * absa(ind0 - 1, ig) + fac100 * absa(ind0, ig) + fac000 * absa(ind0 + 1, ig) + fac210 * absa(ind0 + 8, ig) + fac110 * absa(ind0 + 9, ig) + fac010 * absa(ind0 + 10, ig));
        } else {
          tau_major = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig));
        }
        if (specparm1 < 0.125) {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac201 * absa(ind1 + 2, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig) + fac211 * absa(ind1 + 11, ig));
        } else if (specparm1 > 0.875) {
          tau_major1 = speccomb1 * (fac201 * absa(ind1 - 1, ig) + fac101 * absa(ind1, ig) + fac001 * absa(ind1 + 1, ig) + fac211 * absa(ind1 + 8, ig) + fac111 * absa(ind1 + 9, ig) + fac011 * absa(ind1 + 10, ig));
        } else {
          tau_major1 = speccomb1 * (fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig));
        }
        taug(lay, ngs15 + ig) = tau_major + tau_major1 + tauself + taufor;
        fracs(lay, ngs15 + ig) = fracrefa(ig, jpl) + fpl * (fracrefa(ig, jpl + 1) - fracrefa(ig, jpl));
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(16) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(16) + 1;
      for (ig = 1; ig <= ng16; ++ig) {
        taug(lay, ngs15 + ig) = colch4(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig));
        fracs(lay, ngs15 + ig) = fracrefb(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void rtrnmc() {
    Real transcld;
    Real radld;
    Real radclrd;
    Real plfrac;
    Real blay;
    Real dplankup;
    Real dplankdn;
    Real odepth;
    Real odtot;
    Real odepth_rec;
    Real odtot_rec;
    Real gassrc;
    Real tblind;
    Real tfactot;
    Real bbd;
    Real bbdtot;
    Real tfacgas;
    Real transc;
    Real tausfac;
    Real rad0;
    Real reflect;
    Real radlu;
    Real radclru;
    int ibnd;
    int ib;
    int iband;
    int lay;
    int lev;
    int l;
    int ig;
    int igc;
    int iclddn;
    int ittot;
    int itgas;
    int itr;
    for (ibnd = 1; ibnd <= nbndlw; ++ibnd) {
      if (ibnd == 1 || ibnd == 4 || ibnd >= 10) {
        secdiff(ibnd) = 1.66;
      } else {
        secdiff(ibnd) = a0(ibnd) + a1(ibnd) * Kokkos::exp(a2(ibnd) * pwvcm);
        if (secdiff(ibnd) > 1.80) secdiff(ibnd) = 1.80;
        if (secdiff(ibnd) < 1.50) secdiff(ibnd) = 1.50;
      }
    }
    urad(0) = 0.0;
    drad(0) = 0.0;
    totuflux(0) = 0.0;
    totdflux(0) = 0.0;
    clrurad(0) = 0.0;
    clrdrad(0) = 0.0;
    totuclfl(0) = 0.0;
    totdclfl(0) = 0.0;
    for (lay = 1; lay <= nlayers; ++lay) {
      urad(lay) = 0.0;
      drad(lay) = 0.0;
      totuflux(lay) = 0.0;
      totdflux(lay) = 0.0;
      clrurad(lay) = 0.0;
      clrdrad(lay) = 0.0;
      totuclfl(lay) = 0.0;
      totdclfl(lay) = 0.0;
      icldlyr(lay) = 0;
      for (ig = 1; ig <= ngptlw; ++ig) {
        if (cldfmc(ig, lay) == 1.0) {
          ib = ngb(ig);
          odcld(lay, ig) = secdiff(ib) * taucmc(ig, lay);
          transcld = Kokkos::exp(-odcld(lay, ig));
          abscld(lay, ig) = 1.0 - transcld;
          efclfrac(lay, ig) = abscld(lay, ig) * cldfmc(ig, lay);
          icldlyr(lay) = 1;
        } else {
          odcld(lay, ig) = 0.0;
          abscld(lay, ig) = 0.0;
          efclfrac(lay, ig) = 0.0;
        }
      }
    }
    igc = 1;
    for (iband = istart; iband <= iend; ++iband) {
      if (iout > 0 && iband >= 2) igc = ngs(iband - 1) + 1;
      do {  // 1000 continue
        radld = 0.0;
        radclrd = 0.0;
        iclddn = 0;
        for (lev = nlayers; lev >= 1; --lev) {
          plfrac = fracs(lev, igc);
          blay = planklay(lev, iband);
          dplankup = planklev(lev, iband) - blay;
          dplankdn = planklev(lev - 1, iband) - blay;
          odepth = secdiff(iband) * taut(lev, igc);
          if (odepth < 0.0) odepth = 0.0;
          if (icldlyr(lev) == 1) {
            iclddn = 1;
            odtot = odepth + odcld(lev, igc);
            if (odtot < 0.06) {
              atrans(lev) = odepth - 0.5 * odepth * odepth;
              odepth_rec = rec_6 * odepth;
              gassrc = plfrac * (blay + dplankdn * odepth_rec) * atrans(lev);
              atot(lev) = odtot - 0.5 * odtot * odtot;
              odtot_rec = rec_6 * odtot;
              bbdtot = plfrac * (blay + dplankdn * odtot_rec);
              bbd = plfrac * (blay + dplankdn * odepth_rec);
              radld = radld - radld * (atrans(lev) + efclfrac(lev, igc) * (1.0 - atrans(lev))) + gassrc + cldfmc(igc, lev) * (bbdtot * atot(lev) - gassrc);
              drad(lev - 1) = drad(lev - 1) + radld;
              bbugas(lev) = plfrac * (blay + dplankup * odepth_rec);
              bbutot(lev) = plfrac * (blay + dplankup * odtot_rec);
            } else if (odepth <= 0.06) {
              atrans(lev) = odepth - 0.5 * odepth * odepth;
              odepth_rec = rec_6 * odepth;
              gassrc = plfrac * (blay + dplankdn * odepth_rec) * atrans(lev);
              odtot = odepth + odcld(lev, igc);
              tblind = odtot / (bpade + odtot);
              ittot = tblint * tblind + 0.5;
              tfactot = tfn_tbl(ittot);
              bbdtot = plfrac * (blay + tfactot * dplankdn);
              bbd = plfrac * (blay + dplankdn * odepth_rec);
              atot(lev) = 1.0 - exp_tbl(ittot);
              radld = radld - radld * (atrans(lev) + efclfrac(lev, igc) * (1.0 - atrans(lev))) + gassrc + cldfmc(igc, lev) * (bbdtot * atot(lev) - gassrc);
              drad(lev - 1) = drad(lev - 1) + radld;
              bbugas(lev) = plfrac * (blay + dplankup * odepth_rec);
              bbutot(lev) = plfrac * (blay + tfactot * dplankup);
            } else {
              tblind = odepth / (bpade + odepth);
              itgas = tblint * tblind + 0.5;
              odepth = tau_tbl(itgas);
              atrans(lev) = 1.0 - exp_tbl(itgas);
              tfacgas = tfn_tbl(itgas);
              gassrc = atrans(lev) * plfrac * (blay + tfacgas * dplankdn);
              odtot = odepth + odcld(lev, igc);
              tblind = odtot / (bpade + odtot);
              ittot = tblint * tblind + 0.5;
              tfactot = tfn_tbl(ittot);
              bbdtot = plfrac * (blay + tfactot * dplankdn);
              bbd = plfrac * (blay + tfacgas * dplankdn);
              atot(lev) = 1.0 - exp_tbl(ittot);
              radld = radld - radld * (atrans(lev) + efclfrac(lev, igc) * (1.0 - atrans(lev))) + gassrc + cldfmc(igc, lev) * (bbdtot * atot(lev) - gassrc);
              drad(lev - 1) = drad(lev - 1) + radld;
              bbugas(lev) = plfrac * (blay + tfacgas * dplankup);
              bbutot(lev) = plfrac * (blay + tfactot * dplankup);
            }
          } else {
            if (odepth <= 0.06) {
              atrans(lev) = odepth - 0.5 * odepth * odepth;
              odepth = rec_6 * odepth;
              bbd = plfrac * (blay + dplankdn * odepth);
              bbugas(lev) = plfrac * (blay + dplankup * odepth);
            } else {
              tblind = odepth / (bpade + odepth);
              itr = tblint * tblind + 0.5;
              transc = exp_tbl(itr);
              atrans(lev) = 1.0 - transc;
              tausfac = tfn_tbl(itr);
              bbd = plfrac * (blay + tausfac * dplankdn);
              bbugas(lev) = plfrac * (blay + tausfac * dplankup);
            }
            radld = radld + (bbd - radld) * atrans(lev);
            drad(lev - 1) = drad(lev - 1) + radld;
          }
          if (iclddn == 1) {
            radclrd = radclrd + (bbd - radclrd) * atrans(lev);
            clrdrad(lev - 1) = clrdrad(lev - 1) + radclrd;
          } else {
            radclrd = radld;
            clrdrad(lev - 1) = drad(lev - 1);
          }
        }
        rad0 = fracs(1, igc) * plankbnd(iband);
        reflect = 1.0 - semiss(iband);
        radlu = rad0 + reflect * radld;
        radclru = rad0 + reflect * radclrd;
        urad(0) = urad(0) + radlu;
        clrurad(0) = clrurad(0) + radclru;
        for (lev = 1; lev <= nlayers; ++lev) {
          if (icldlyr(lev) == 1) {
            gassrc = bbugas(lev) * atrans(lev);
            radlu = radlu - radlu * (atrans(lev) + efclfrac(lev, igc) * (1.0 - atrans(lev))) + gassrc + cldfmc(igc, lev) * (bbutot(lev) * atot(lev) - gassrc);
            urad(lev) = urad(lev) + radlu;
          } else {
            radlu = radlu + (bbugas(lev) - radlu) * atrans(lev);
            urad(lev) = urad(lev) + radlu;
          }
          if (iclddn == 1) {
            radclru = radclru + (bbugas(lev) - radclru) * atrans(lev);
            clrurad(lev) = clrurad(lev) + radclru;
          } else {
            radclru = radlu;
            clrurad(lev) = urad(lev);
          }
        }
        igc = igc + 1;
      } while (igc <= ngs(iband));  // if (igc .le. ngs(iband)) go to 1000
      for (lev = nlayers; lev >= 0; --lev) {
        uflux(lev) = urad(lev) * wtdiff;
        dflux(lev) = drad(lev) * wtdiff;
        urad(lev) = 0.0;
        drad(lev) = 0.0;
        totuflux(lev) = totuflux(lev) + uflux(lev) * delwave(iband);
        totdflux(lev) = totdflux(lev) + dflux(lev) * delwave(iband);
        uclfl(lev) = clrurad(lev) * wtdiff;
        dclfl(lev) = clrdrad(lev) * wtdiff;
        clrurad(lev) = 0.0;
        clrdrad(lev) = 0.0;
        totuclfl(lev) = totuclfl(lev) + uclfl(lev) * delwave(iband);
        totdclfl(lev) = totdclfl(lev) + dclfl(lev) * delwave(iband);
      }
    }
    totuflux(0) = totuflux(0) * fluxfac;
    totdflux(0) = totdflux(0) * fluxfac;
    fnet(0) = totuflux(0) - totdflux(0);
    totuclfl(0) = totuclfl(0) * fluxfac;
    totdclfl(0) = totdclfl(0) * fluxfac;
    fnetc(0) = totuclfl(0) - totdclfl(0);
    for (lev = 1; lev <= nlayers; ++lev) {
      totuflux(lev) = totuflux(lev) * fluxfac;
      totdflux(lev) = totdflux(lev) * fluxfac;
      fnet(lev) = totuflux(lev) - totdflux(lev);
      totuclfl(lev) = totuclfl(lev) * fluxfac;
      totdclfl(lev) = totdclfl(lev) * fluxfac;
      fnetc(lev) = totuclfl(lev) - totdclfl(lev);
      l = lev - 1;
      htr(l) = heatfac * (fnet(l) - fnet(lev)) / (pz(l) - pz(lev));
      htrc(l) = heatfac * (fnetc(l) - fnetc(lev)) / (pz(l) - pz(lev));
    }
    htr(nlayers) = 0.0;
    htrc(nlayers) = 0.0;
  
  }

  // module_ra_rrtmg_lw's INIRAD, relcalc and reicalc (cm1k_rrtmg_common.hpp)
  KOKKOS_INLINE_FUNCTION void inirad(const FA<1>& O3PROF, const FA<2>& Plev2, int kte_) {
    if (!rrtmg::inirad(O3PROF, Plev2, kts, kte_, O3SUM, PPSUM, O3WIN, PPWIN)) LW_FAIL(LW_O3FRAC);
  }
  KOKKOS_INLINE_FUNCTION void relcalc(int pver, Real landfrac, Real landm, Real icefrac, Real snowh) {
    rrtmg::relcalc(pver, tlay, landfrac, landm, icefrac, reliq, snowh);
  }
  KOKKOS_INLINE_FUNCTION void reicalc(int pver) { rrtmg::reicalc(pver, tlay, reice, retab); }

  // mcica_subcol_lw + generate_stochastic_clouds (irng = 0: KISS)
  KOKKOS_INLINE_FUNCTION void mcica_subcol_lw(int permuteseed) {
    if (icld == 0) return;
    if (icld < 0 || icld > 5) LW_FAIL(LW_ICLD);
    const int nsubcol = ngptlw;
    for (int k = 1; k <= nlay; ++k) {
      reicmcl(1, k) = rei(1, k);
      relqmcl(1, k) = rel(1, k);
      resnmcl(1, k) = res(1, k);
      pmid(1, k) = play(1, k) * 1.e2;
    }
    // generate_stochastic_clouds(ncol, nlay, nsubclw, icld, irng, pmid, hgt, cldfrac, clwpth, ciwpth,
    //   cswpth, taucld, cldfmcl, clwpmcl, ciwpmcl, cswpmcl, taucmcl, permuteseed)
    const Real Zo = 2500.;
    const Real cldmin = 1.0e-20;
    const int overlap = icld;
    int ilev, isubcol;
    const int i = 1;
    for (ilev = 1; ilev <= nlay; ++ilev) {
      cldf(i, ilev) = cldfrac(i, ilev);
      if (cldf(i, ilev) < cldmin) cldf(i, ilev) = 0.;
    }
    if (pmid(i, 1) < pmid(i, 2)) LW_FAIL(LW_KISS_PMID);
    int seed1 = int((pmid(i, 1) - Real(int(pmid(i, 1)))) * 1000000000);
    int seed2 = int((pmid(i, 2) - Real(int(pmid(i, 2)))) * 1000000000);
    int seed3 = int((pmid(i, 3) - Real(int(pmid(i, 3)))) * 1000000000);
    int seed4 = int((pmid(i, 4) - Real(int(pmid(i, 4)))) * 1000000000);
    Real rand_num;
    for (int n = 1; n <= permuteseed; ++n) kissvec(seed1, seed2, seed3, seed4, rand_num);
    if (overlap == 1 || overlap == 2) {
      for (isubcol = 1; isubcol <= nsubcol; ++isubcol)
        for (ilev = 1; ilev <= nlay; ++ilev) {
          kissvec(seed1, seed2, seed3, seed4, rand_num);
          CDF(isubcol, i, ilev) = rand_num;
        }
      if (overlap == 2) {
        for (ilev = 2; ilev <= nlay; ++ilev)
          for (isubcol = 1; isubcol <= nsubcol; ++isubcol) {
            if (CDF(isubcol, i, ilev - 1) > 1. - cldf(i, ilev - 1)) CDF(isubcol, i, ilev) = CDF(isubcol, i, ilev - 1);
            else CDF(isubcol, i, ilev) = CDF(isubcol, i, ilev) * (1. - cldf(i, ilev - 1));
          }
      }
    } else if (overlap == 3) {
      for (isubcol = 1; isubcol <= nsubcol; ++isubcol) {
        kissvec(seed1, seed2, seed3, seed4, rand_num);
        for (ilev = 1; ilev <= nlay; ++ilev) CDF(isubcol, i, ilev) = rand_num;
      }
    } else if (overlap == 4) {
      alpha(i, 1) = 0.;
      for (ilev = 2; ilev <= nlay; ++ilev) alpha(i, ilev) = Kokkos::exp(-(hgt(i, ilev) - hgt(i, ilev - 1)) / Zo);
      for (isubcol = 1; isubcol <= nsubcol; ++isubcol)
        for (ilev = 1; ilev <= nlay; ++ilev) {
          kissvec(seed1, seed2, seed3, seed4, rand_num);
          CDF(isubcol, i, ilev) = rand_num;
          kissvec(seed1, seed2, seed3, seed4, rand_num);
          CDF2(isubcol, i, ilev) = rand_num;
        }
      for (ilev = 2; ilev <= nlay; ++ilev)
        for (isubcol = 1; isubcol <= nsubcol; ++isubcol)
          if (CDF2(isubcol, i, ilev) < alpha(i, ilev)) CDF(isubcol, i, ilev) = CDF(isubcol, i, ilev - 1);
    } else {
      LW_FAIL(LW_OVERLAP5);
    }
    for (ilev = 1; ilev <= nlay; ++ilev)
      for (isubcol = 1; isubcol <= nsubcol; ++isubcol) {
        const bool iscloudy = CDF(isubcol, i, ilev) >= 1. - cldf(i, ilev);
        if (iscloudy) {
          cldfmcl(isubcol, i, ilev) = 1.;
          clwpmcl(isubcol, i, ilev) = clwpth(i, ilev);
          ciwpmcl(isubcol, i, ilev) = ciwpth(i, ilev);
          cswpmcl(isubcol, i, ilev) = cswpth(i, ilev);
          const int n = ngb(isubcol);
          taucmcl(isubcol, i, ilev) = taucld(n, i, ilev);
        } else {
          cldfmcl(isubcol, i, ilev) = 0.;
          clwpmcl(isubcol, i, ilev) = 0.;
          ciwpmcl(isubcol, i, ilev) = 0.;
          cswpmcl(isubcol, i, ilev) = 0.;
          taucmcl(isubcol, i, ilev) = 0.;
        }
      }
  }

  // rrtmg_lw (ncol = 1; calc_clean_atm_diag = 0)
  KOKKOS_INLINE_FUNCTION void rrtmg_lw() {
    istart = 1;
    iend = 16;
    iout = 0;
    iaer = 10;
    inatm();
    cldprmc();
    if (ierr) return;
    setcoef();
    taumol();
    for (int k = 1; k <= nlayers; ++k)
      for (int ig = 1; ig <= ngptlw; ++ig) taut(k, ig) = taug(k, ig) + taua(k, ngb(ig));
    for (int k = 0; k <= nlayers; ++k) {
      totuclnlfl(k) = 0.0;
      totdclnlfl(k) = 0.0;
    }
    rtrnmc();
    for (int k = 0; k <= nlayers; ++k) {
      uflx(iplon, k + 1) = totuflux(k);
      dflx(iplon, k + 1) = totdflux(k);
      uflxc(iplon, k + 1) = totuclfl(k);
      dflxc(iplon, k + 1) = totdclfl(k);
      uflxcln(iplon, k + 1) = totuclnlfl(k);
      dflxcln(iplon, k + 1) = totdclnlfl(k);
    }
    for (int k = 0; k <= nlayers - 1; ++k) {
      hr(iplon, k + 1) = htr(k);
      hrc(iplon, k + 1) = htrc(k);
    }
  }

  // RRTMG_LWRAD's i/j loop body for column (i,j)
  KOKKOS_INLINE_FUNCTION void column(const LwIn& a, int i, int j) {
    constexpr Real deltap = 4.;
    constexpr int nproflevs = 60;
    const Real ch4 = 1774.e-9, n2o = 319.e-9, cfc11 = 0.251e-9, cfc12 = 0.538e-9, cfc22 = 0.169e-9,
               ccl4 = 0.093e-9, o2 = 0.209488;
    const Real amdw = 1.607793, amdo = 0.603461;
    const Real co2 = a.co2;
    int k, nb, L, LL, klev = 0;
    Real dz = 0, dzsum, wght, vark, vark1;

    for (k = kts; k <= kte + 1; ++k) {
      Pw1D(k) = a.p8w(i, j, k) * 0.01;
      Tw1D(k) = a.t8w(i, j, k);
    }
    for (k = kts; k <= kte; ++k) {
      QV1D(k) = 0.;
      QC1D(k) = 0.;
      QR1D(k) = 0.;
      QI1D(k) = 0.;
      QS1D(k) = 0.;
      CLDFRA1D(k) = 0.;
    }
    for (k = kts; k <= kte; ++k) {
      QV1D(k) = a.qv3d(i, j, k);
      QV1D(k) = Kokkos::max(0., QV1D(k));
    }
    for (k = kts; k <= kte; ++k) O31D(k) = a.has_o33d ? a.o33d(i, j, k) : 0.0;
    for (k = kts; k <= kte; ++k) {
      TTEN1D(k) = 0.;
      T1D(k) = a.t3d(i, j, k);
      P1D(k) = a.p3d(i, j, k) * 0.01;
      DZ1D(k) = a.dz8w(i, j, k);
    }
    if (a.icloud != 0) {
      for (k = kts; k <= kte; ++k) CLDFRA1D(k) = a.cldfra3d(i, j, k);
      if (a.f_qc)
        for (k = kts; k <= kte; ++k) {
          QC1D(k) = a.qc3d(i, j, k);
          QC1D(k) = Kokkos::max(0., QC1D(k));
        }
      if (a.f_qr)
        for (k = kts; k <= kte; ++k) {
          QR1D(k) = a.qr3d(i, j, k);
          QR1D(k) = Kokkos::max(0., QR1D(k));
        }
      if (!a.f_qi && !a.warm_rain)
        for (k = kts; k <= kte; ++k)
          if (T1D(k) < 273.15) {
            QI1D(k) = QC1D(k);
            QS1D(k) = QR1D(k);
            QC1D(k) = 0.;
            QR1D(k) = 0.;
          }
      if (a.f_qi)
        for (k = kts; k <= kte; ++k) {
          QI1D(k) = a.qi3d(i, j, k);
          QI1D(k) = Kokkos::max(0., QI1D(k));
        }
      if (a.f_qs)
        for (k = kts; k <= kte; ++k) {
          QS1D(k) = a.qs3d(i, j, k);
          QS1D(k) = Kokkos::max(0., QS1D(k));
        }
      if (a.f_qg)
        for (k = kts; k <= kte; ++k) {
          QG1D(k) = a.qg3d(i, j, k);
          QG1D(k) = Kokkos::max(0., QG1D(k));
        }
    }
    for (k = kts; k <= kte; ++k) QV1D(k) = Kokkos::max(QV1D(k), 1.e-12);

    inflglw = 2;
    iceflglw = 3;
    liqflglw = 1;
    if (a.icloud != 0) {
      // has_reqc = has_reqi = has_reqs = 0
      for (k = kts; k <= kte; ++k) recloud1d(ncol, k) = 5.0;
      for (k = kts; k <= kte; ++k) reice1d(ncol, k) = 10.0;
      for (k = kts; k <= kte; ++k) resnow1d(ncol, k) = 10.0;
    }

    plev(ncol, 1) = Pw1D(1);
    tlev(ncol, 1) = Tw1D(1);
    const Real tsfc = a.tsk(i, j, 1);
    for (k = kts; k <= kte; ++k) {
      play(ncol, k) = P1D(k);
      plev(ncol, k + 1) = Pw1D(k + 1);
      pdel(ncol, k) = plev(ncol, k) - plev(ncol, k + 1);
      tlay(ncol, k) = T1D(k);
      tlev(ncol, k + 1) = Tw1D(k + 1);
      h2ovmr(ncol, k) = QV1D(k) * amdw;
      co2vmr(ncol, k) = co2;
      o2vmr(ncol, k) = o2;
      ch4vmr(ncol, k) = ch4;
      n2ovmr(ncol, k) = n2o;
      cfc11vmr(ncol, k) = cfc11;
      cfc12vmr(ncol, k) = cfc12;
      cfc22vmr(ncol, k) = cfc22;
      ccl4vmr(ncol, k) = ccl4;
    }
    dzsum = 0.0;
    for (k = kts; k <= kte; ++k) {
      dz = DZ1D(k);
      hgt(ncol, k) = dzsum + 0.5 * dz;
      dzsum = dzsum + dz;
    }
    // buffer layers above the model top (Cavallo)
    for (L = kte + 1; L <= nlayers; ++L) {
      plev(ncol, L + 1) = plev(ncol, L) - deltap;
      play(ncol, L) = 0.5 * (plev(ncol, L) + plev(ncol, L + 1));
      hgt(ncol, L) = dzsum + 0.5 * dz;
      dzsum = dzsum + dz;
    }
    plev(ncol, nlayers + 1) = 0.00;
    play(ncol, nlayers) = 0.5 * (plev(ncol, nlayers) + plev(ncol, nlayers + 1));
    for (L = 1; L <= nlayers + 1; ++L) {
      if (PPROF(nproflevs) < plev(ncol, L)) {
        for (LL = 2; LL <= nproflevs; ++LL)
          if (PPROF(LL) < plev(ncol, L)) {
            klev = LL - 1;
            break;
          }
      } else {
        klev = nproflevs;
      }
      if (klev != nproflevs) {
        vark = TPROF(klev);
        vark1 = TPROF(klev + 1);
        wght = (plev(ncol, L) - PPROF(klev)) / (PPROF(klev + 1) - PPROF(klev));
      } else {
        vark = TPROF(klev);
        vark1 = TPROF(klev);
        wght = 0.0;
      }
      varint(L) = wght * (vark1 - vark) + vark;
    }
    for (L = kte + 1; L <= nlayers + 1; ++L) {
      tlev(ncol, L) = varint(L) + (tlev(ncol, kte) - varint(kte));
      tlay(ncol, L - 1) = 0.5 * (tlev(ncol, L) + tlev(ncol, L - 1));
    }
    for (L = kte + 1; L <= nlayers; ++L) {
      h2ovmr(ncol, L) = h2ovmr(ncol, kte);
      co2vmr(ncol, L) = co2vmr(ncol, kte);
      o2vmr(ncol, L) = o2vmr(ncol, kte);
      ch4vmr(ncol, L) = ch4vmr(ncol, kte);
      n2ovmr(ncol, L) = n2ovmr(ncol, kte);
      cfc11vmr(ncol, L) = cfc11vmr(ncol, kte);
      cfc12vmr(ncol, L) = cfc12vmr(ncol, kte);
      cfc22vmr(ncol, L) = cfc22vmr(ncol, kte);
      ccl4vmr(ncol, L) = ccl4vmr(ncol, kte);
    }
    inirad(o3mmr, plev, nlay - 1);
    if (ierr) return;
    for (k = kts; k <= nlayers; ++k) {
      o3vmr(ncol, k) = o3mmr(k) * amdo;
      if (a.has_o33d && a.o3input == 2) {
        if (k <= kte) {
          o3vmr(ncol, k) = O31D(k);
        } else {
          o3vmr(ncol, k) = O31D(kte) - o3mmr(kte) * amdo + o3mmr(k) * amdo;
          if (o3vmr(ncol, k) <= 0.) o3vmr(ncol, k) = o3mmr(k) * amdo;
        }
      }
    }
    if (a.o3input == 0 && a.has_o33d)
      for (k = kts; k <= kte; ++k) a.o33d(i, j, k) = o3vmr(ncol, k);
    for (nb = 1; nb <= nbndlw; ++nb) emis(ncol, nb) = a.emiss(i, j, 1);

    // inflglw = 2 (> 0)
    {
      for (k = kts; k <= kte; ++k) cldfrac(ncol, k) = CLDFRA1D(k);
      const int pver = kte - kts + 1;
      const Real gravmks = a.g;
      const Real landfrac = 2. - a.xland(i, j, 1);
      const Real landm = landfrac;
      const Real snowh = 0.001 * a.snow(i, j, 1);
      const Real icefrac = a.xice(i, j, 1);
      for (k = kts; k <= kte; ++k) {
        const Real gicewp = (QI1D(k) + QS1D(k)) * pdel(ncol, k) * 100.0 / gravmks * 1000.0;
        const Real gliqwp = QC1D(k) * pdel(ncol, k) * 100.0 / gravmks * 1000.0;
        cicewp(ncol, k) = gicewp / Kokkos::max(0.01, cldfrac(ncol, k));
        cliqwp(ncol, k) = gliqwp / Kokkos::max(0.01, cldfrac(ncol, k));
      }
      relcalc(pver, landfrac, landm, icefrac, snowh);
      reicalc(pver);
      // iceflglw = 3
      for (k = kts; k <= kte; ++k) {
        reice(ncol, k) = reice(ncol, k) * 1.0315;
        reice(ncol, k) = Kokkos::min(140.0, reice(ncol, k));
      }
      for (k = kts; k <= kte; ++k) {
        clwpth(ncol, k) = cliqwp(ncol, k);
        ciwpth(ncol, k) = cicewp(ncol, k);
        rel(ncol, k) = reliq(ncol, k);
        rei(ncol, k) = reice(ncol, k);
      }
      for (k = kts; k <= kte; ++k) {
        cswpth(ncol, k) = 0.;
        res(ncol, k) = 10.;
      }
      for (k = kts; k <= kte; ++k)
        for (nb = 1; nb <= nbndlw; ++nb) taucld(nb, ncol, k) = 0.0;
    }
    for (k = kte + 1; k <= nlayers; ++k) {
      clwpth(ncol, k) = 0.;
      ciwpth(ncol, k) = 0.;
      cswpth(ncol, k) = 0.;
      rel(ncol, k) = 10.;
      rei(ncol, k) = 10.;
      res(ncol, k) = 10.;
      cldfrac(ncol, k) = 0.;
      for (nb = 1; nb <= nbndlw; ++nb) taucld(nb, ncol, k) = 0.;
    }
    mcica_subcol_lw(150);
    if (ierr) return;
    for (nb = 1; nb <= nbndlw; ++nb)
      for (k = kts; k <= nlayers; ++k) tauaer(ncol, k, nb) = 0.;

    rrtmg_lw();
    if (ierr) return;

    a.glw(i, j, 1) = dflx(1, 1);
    a.olr(i, j, 1) = uflx(1, nlayers + 1);
    a.lwcf(i, j, 1) = uflxc(1, nlayers + 1) - uflx(1, nlayers + 1);
    if (a.has_lwupt) {
      a.lwupt(i, j, 1) = uflx(1, nlayers + 1);
      a.lwuptc(i, j, 1) = uflxc(1, nlayers + 1);
      a.lwdnt(i, j, 1) = dflx(1, nlayers + 1);
      a.lwdntc(i, j, 1) = dflxc(1, nlayers + 1);
      a.lwupb(i, j, 1) = uflx(1, 1);
      a.lwupbc(i, j, 1) = uflxc(1, 1);
      a.lwdnb(i, j, 1) = dflx(1, 1);
      a.lwdnbc(i, j, 1) = dflxc(1, 1);
    }
    for (k = kts; k <= kte; ++k) {
      TTEN1D(k) = hr(ncol, k) / 86400.;
      a.rthratenlw(i, j, k) = TTEN1D(k);
      a.lwtenc(i, j, k) = hrc(ncol, k) / 86400.0;
    }
  }
};

} // namespace

int lw_nlayers(int kme, Real p_top) {
  const Real deltap = 4.;
  return kme + NINT(p_top * 0.01 / deltap) - 1;
}

Real rrtmg_lwrad(const LwTableData& tab, const LwRadArgs& a) {
  if (a.has_reqc != 0 || a.has_reqi != 0 || a.has_reqs != 0)
    throw std::runtime_error("cm1k rrtmg_lwrad: has_reqc/has_reqi/has_reqs /= 0 is not ported");
  if (a.is_cammgmp_used) throw std::runtime_error("cm1k rrtmg_lwrad: is_CAMMGMP_used is not ported");
  if (a.calc_clean_atm_diag > 0) throw std::runtime_error("cm1k rrtmg_lwrad: calc_clean_atm_diag > 0 is not ported");
  if (a.mp_physics == 5 || a.mp_physics == 95)
    throw std::runtime_error("cm1k rrtmg_lwrad: FER_MP_HIRES microphysics is not ported");
  if (a.cldovrlp < 0 || a.cldovrlp > 4) throw std::runtime_error("cm1k rrtmg_lwrad: cldovrlp must be 0..4");
  if (a.kts != 1) throw std::runtime_error("cm1k rrtmg_lwrad: kts must be 1");
  if (a.nlayers < a.kte + 1) throw std::runtime_error("cm1k rrtmg_lwrad: nlayers must exceed kte");

  // co2 = (280. + 90.*exp(0.02*(yr-2000)))*1.e-6
  const Real co2 = (280. + 90. * std::exp(0.02 * Real(a.yr - 2000))) * 1.e-6;

  LwIn in;
  in.rthratenlw = a.rthratenlw; in.lwtenc = a.lwtenc;
  in.lwupt = a.lwupt; in.lwuptc = a.lwuptc; in.lwdnt = a.lwdnt; in.lwdntc = a.lwdntc;
  in.lwupb = a.lwupb; in.lwupbc = a.lwupbc; in.lwdnb = a.lwdnb; in.lwdnbc = a.lwdnbc;
  in.glw = a.glw; in.olr = a.olr; in.lwcf = a.lwcf;
  in.emiss = a.emiss; in.tsk = a.tsk; in.p8w = a.p8w; in.p3d = a.p3d; in.dz8w = a.dz8w; in.t3d = a.t3d;
  in.t8w = a.t8w; in.cldfra3d = a.cldfra3d; in.xland = a.xland; in.xice = a.xice; in.snow = a.snow;
  in.qv3d = a.qv3d; in.qc3d = a.qc3d; in.qr3d = a.qr3d; in.qi3d = a.qi3d; in.qs3d = a.qs3d; in.qg3d = a.qg3d;
  in.o33d = a.o33d;
  in.g = a.g; in.co2 = co2;
  in.icloud = a.icloud; in.cldovrlp = a.cldovrlp; in.o3input = a.o3input;
  in.warm_rain = a.warm_rain;
  in.f_qc = a.f_qc; in.f_qr = a.f_qr; in.f_qi = a.f_qi; in.f_qs = a.f_qs; in.f_qg = a.f_qg;
  in.has_o33d = a.o33d.is_allocated();
  in.has_lwupt = a.lwupt.is_allocated();
  in.nk = a.kte;

  LwCol proto;
  static_cast<LwTables&>(proto) = tab.dev;
  proto.kte = a.kte;
  proto.nlayers = a.nlayers;
  proto.nlay = a.nlayers;
  proto.icld = a.cldovrlp;
  Binder cnt;
  proto.layout(cnt);
  const long nr = cnt.nr, ni = cnt.ni;

  const int ni_ = a.ite - a.its + 1, nj_ = a.jte - a.jts + 1;
  const int ncols = ni_ * nj_;
  if (ncols <= 0) return co2;
  int nc = a.chunk;
  if (nc <= 0) {
    const long budget = 512L * 1024 * 1024 / long(sizeof(Real));   // ~512 MB of scratch
    const long conc = Kokkos::DefaultExecutionSpace().concurrency();
    nc = int(std::min<long>({long(ncols), std::max<long>(1, budget / (nr + ni)), std::max<long>(1, 2 * conc)}));
  }
  nc = std::min(nc, ncols);
  Kokkos::View<Real*, Device> rpool("rrtmg_lw_scratch_r", size_t(nr) * size_t(nc));
  Kokkos::View<int*, Device> ipool("rrtmg_lw_scratch_i", size_t(std::max<long>(ni, 1)) * size_t(nc));
  Kokkos::View<int*, Device> errs("rrtmg_lw_err", size_t(ncols));
  const int its = a.its, jts = a.jts;
  for (int c0 = 0; c0 < ncols; c0 += nc) {
    const int n = std::min(nc, ncols - c0);
    Real* rp = rpool.data();
    int* ip = ipool.data();
    Kokkos::parallel_for(
        "rrtmg_lwrad", Kokkos::RangePolicy<Kokkos::IndexType<int>>(0, n), KOKKOS_LAMBDA(int c) {
          LwCol col = proto;
          Binder b(rp, ip, nc, c);
          col.layout(b);
          const int ic = c0 + c;
          const int i = its + ic % ni_, j = jts + ic / ni_;
          col.tsfc1 = in.tsk(i, j, 1);
          col.column(in, i, j);
          errs(ic) = col.ierr;
        });
  }
  auto he = Kokkos::create_mirror_view_and_copy(HostSpace(), errs);
  for (int ic = 0; ic < ncols; ++ic)
    if (he(ic) != 0)
      throw std::runtime_error(std::string("cm1k rrtmg_lwrad: ") + lw_err_message(he(ic)) + " (column i=" +
                               std::to_string(a.its + ic % ni_) + ", j=" + std::to_string(a.jts + ic / ni_) + ")");
  return co2;
}

} // namespace cm1k::rrtmg
