// RRTMG short-wave radiation (radopt = 2): RRTMG_SWRAD of src/module_ra_rrtmg_sw.F
// as src/radiation_driver.F calls it, with rrtmg_sw, inatm_sw, cldprmc_sw,
// setcoef_sw, taumol_sw (taumol16..29), spcvmc_sw, reftra_sw, vrtqdr_sw,
// mcica_subcol_sw / generate_stochastic_clouds_sw (KISS generator, irng = 0,
// permuteseed = 1) and module_ra_rrtmg_lw's INIRAD/O3DATA, relcalc, reicalc.
//
// Organized as the LW port (cm1k_rrtmg_lw.cpp): one kernel thread runs
// RRTMG_SWRAD's i/j loop body for one column, the Fortran's routines are
// member functions of SwCol (derived from SwTables, so module data resolve
// by their Fortran names) and every array is a member FA bound to the
// column's slice of a scratch pool. reftra_sw and vrtqdr_sw, which
// spcvmc_sw calls with different actual arrays, take them as arguments.
#include "physics/rrtmg/cm1k_rrtmg_sw.hpp"
#include "physics/rrtmg/cm1k_rrtmg_common.hpp"

#include <stdexcept>
#include <string>

namespace cm1k::rrtmg {

namespace {

using namespace sw;

// Error codes of the Fortran's stop / wrf_error_fatal calls.
enum SwErr {
  SW_OK = 0,
  SW_INFLAG1 = 1,
  SW_ICE_EXT,
  SW_ICE_SSA_GT,
  SW_ICE_SSA_LT,
  SW_ICE_ASY_GT,
  SW_ICE_ASY_LT,
  SW_ICE_BOUNDS,
  SW_ICE_GEN_BOUNDS,
  SW_SNOW_GEN_BOUNDS,
  SW_FDELTA_LT,
  SW_FDELTA_GT,
  SW_SNOW_EXT,
  SW_SNOW_SSA_GT,
  SW_SNOW_SSA_LT,
  SW_SNOW_ASY_GT,
  SW_SNOW_ASY_LT,
  SW_LIQ_BOUNDS,
  SW_LIQ_EXT,
  SW_LIQ_SSA_GT,
  SW_LIQ_SSA_LT,
  SW_LIQ_ASY_GT,
  SW_LIQ_ASY_LT,
  SW_KISS_PMID,
  SW_OVERLAP5,
  SW_O3FRAC,
};
const char* sw_err_message(int e) {
  switch (e) {
  case SW_INFLAG1: return "INFLAG = 1 OPTION NOT AVAILABLE WITH MCICA";
  case SW_ICE_EXT: return "ICE EXTINCTION LESS THAN 0.0";
  case SW_ICE_SSA_GT: return "ICE SSA GRTR THAN 1.0";
  case SW_ICE_SSA_LT: return "ICE SSA LESS THAN 0.0";
  case SW_ICE_ASY_GT: return "ICE ASYM GRTR THAN 1.0";
  case SW_ICE_ASY_LT: return "ICE ASYM LESS THAN 0.0";
  case SW_ICE_BOUNDS: return "ICE RADIUS OUT OF BOUNDS";
  case SW_ICE_GEN_BOUNDS: return "ERROR: ICE GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS";
  case SW_SNOW_GEN_BOUNDS: return "ERROR: SNOW GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS";
  case SW_FDELTA_LT: return "FDELTA LESS THAN 0.0";
  case SW_FDELTA_GT: return "FDELTA GT THAN 1.0";
  case SW_SNOW_EXT: return "SNOW EXTINCTION LESS THAN 0.0";
  case SW_SNOW_SSA_GT: return "SNOW SSA GRTR THAN 1.0";
  case SW_SNOW_SSA_LT: return "SNOW SSA LESS THAN 0.0";
  case SW_SNOW_ASY_GT: return "SNOW ASYM GRTR THAN 1.0";
  case SW_SNOW_ASY_LT: return "SNOW ASYM LESS THAN 0.0";
  case SW_LIQ_BOUNDS: return "liquid effective radius out of bounds";
  case SW_LIQ_EXT: return "LIQUID EXTINCTION LESS THAN 0.0";
  case SW_LIQ_SSA_GT: return "LIQUID SSA GRTR THAN 1.0";
  case SW_LIQ_SSA_LT: return "LIQUID SSA LESS THAN 0.0";
  case SW_LIQ_ASY_GT: return "LIQUID ASYM GRTR THAN 1.0";
  case SW_LIQ_ASY_LT: return "LIQUID ASYM LESS THAN 0.0";
  case SW_KISS_PMID: return "MCICA_SUBCOL: KISSVEC SEED GENERATOR REQUIRES PMID FROM BOTTOM FOUR LAYERS.";
  case SW_OVERLAP5: return "Cloud Overlap case 5: ER has not yet been implemented. Stopping...";
  case SW_O3FRAC: return "O3DATA: frac out of range (stop 1231)";
  default: return "unknown error";
  }
}
#define SW_FAIL(code) \
  do {                \
    ierr = (code);    \
    return;           \
  } while (0)

// RRTMG_SWRAD's inputs for one call (radiation_driver.F's actual arguments).
struct SwIn {
  FArray3 rthratensw, swtenc;
  FArray3 swupt, swuptc, swdnt, swdntc, swupb, swupbc, swdnb, swdnbc, swcf, gsw, coszr, swddir, swddni, swddif,
      swdownc, swddnic, swddirc;
  FArray3 xcoszen, albedo, tsk, xland, xice, snow;
  FArray3 t3d, t8w, p3d, p8w, rho3d, dz8w, cldfra3d, qv3d, qc3d, qr3d, qi3d, qs3d, qg3d, o33d;
  Real g, co2, solcon;
  int icloud, o3input;
  bool warm_rain, f_qc, f_qr, f_qi, f_qs, f_qg, has_o33d, has_swupt;
};

struct SwCol : SwTables {
  static constexpr int ncol = 1, iplon = 1, kts = 1;
  int kte = 0, nlay = 0, nlayers = 0, ierr = 0;
  int icld = 0, inflgsw = 0, iceflgsw = 0, liqflgsw = 0, inflag = 0, iceflag = 0, liqflag = 0;
  // rrtmg_sw scalars
  int istart = 0, iend = 0, icpr = 0, iout = 0, iaer = 0, dyofyr = 0;
  int laytrop = 0, layswtch = 0, laylow = 0;
  Real tbound = 0, cossza = 0, coszen = 0, adjes = 0, scon = 0, tsfc = 0;
  Real asdir = 0, asdif = 0, aldir = 0, aldif = 0;

  // RRTMG_SWRAD
  FA<1> Pw1D, Tw1D, TTEN1D, CLDFRA1D, DZ1D, P1D, T1D, QV1D, QC1D, QR1D, QI1D, RHO1D, QS1D, QG1D, O31D;
  FA<2> plev, tlev, play, tlay, h2ovmr, o3vmr, co2vmr, o2vmr, ch4vmr, n2ovmr;
  FA<1> o3mmr;
  FA<2> hgt, clwpth, ciwpth, cswpth, rel, rei, res, cldfrac, relqmcl, reicmcl, resnmcl;
  FA<3> taucld, ssacld, asmcld, fsfcld, cldfmcl, clwpmcl, ciwpmcl, cswpmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl;
  FA<3> tauaer, ssaaer, asmaer;
  FA<2> swuflx, swdflx, swuflxc, swdflxc, swuflxcln, swdflxcln, sibvisdir, sibvisdif, sibnirdir, sibnirdif, swdkdir,
      swdkdif, swdkdirc, swhr, swhrc;
  FA<2> pdel, cicewp, cliqwp, csnowp, reliq, reice, recloud1d, reice1d, resnow1d;
  // generate_stochastic_clouds_sw
  FA<2> pmid, cldf, alpha;
  FA<3> CDF, CDF2;
  // rrtmg_sw
  FA<1> pavel, tavel, pz, tz, pdp, coldry, adjflux, solvar, albdir, albdif;
  FA<2> wkl, taua, ssaa, asma;
  IA<1> jp, jt, jt1, indself, indfor;
  FA<1> colh2o, colco2, colo3, coln2o, colch4, colo2, colmol, co2mult, selffac, selffrac, forfac, forfrac, fac00,
      fac01, fac10, fac11;
  FA<2> cldfmc, ciwpmc, clwpmc, cswpmc, taucmc, taormc, ssacmc, asmcmc, fsfcmc;
  FA<1> relqmc, reicmc, resnmc;
  FA<2> ztaua, ztauacln, zasya, zomga, zcldfmc, ztaucmc, ztaormc, zasycmc, zomgcmc;
  FA<1> zbbfu, zbbfd, zbbcu, zbbcd, zbbfddir, zbbcddir, zuvfd, zuvcd, zuvfddir, zuvcddir, znifd, znicd, znifddir,
      znicddir, swnflx, swnflxc, dirdflux, difdflux, uvdflx, nidflx, dirdnuv, difdnuv, dirdnir, difdnir;
  // cldprmc_sw
  FA<1> fdelta, extcoice, gice, ssacoice, forwice, extcoliq, gliq, ssacoliq, forwliq, extcosno, gsno, ssacosno,
      forwsno;
  // spcvmc_sw
  IA<1> lrtchkclr, lrtchkcld;
  FA<1> zdbt, zdbt_nodel, zgc, zgcc, zgco, zomc, zomcc, zomco, zrdnd, zrdndc, zref, zrefc, zrefo, zrefd, zrefdc,
      zrefdo, zrup, zrupd, zrupc, zrupdc, zs1, ztauc, ztauo, ztdn, ztdnd, ztdbt, ztoc, ztor, ztra, ztrac, ztrao,
      ztrad, ztradc, ztrado, zdbtc, ztdbtc, zincflx, zdbtc_nodel, ztdbt_nodel, ztdbtc_nodel, zsflxzen;
  FA<2> ztaug, ztaur, zcd, zcu, zfd, zfu;
  // vrtqdr_sw's ztdn
  FA<1> vq_ztdn;

  template <class B>
  KOKKOS_INLINE_FUNCTION void layout(B& b) {
    const int nk = kte, nl = nlay;   // nl = nlay = nlayers = kte + 1
    b.arr(Pw1D, {kts}, {nk + 1});
    b.arr(Tw1D, {kts}, {nk + 1});
    FA<1>* k1[] = {&TTEN1D, &CLDFRA1D, &DZ1D, &P1D, &T1D, &QV1D, &QC1D, &QR1D, &QI1D, &RHO1D, &QS1D, &QG1D, &O31D};
    for (auto* a : k1) b.arr(*a, {kts}, {nk});
    b.arr(plev, {1, kts}, {1, nk + 2});
    b.arr(tlev, {1, kts}, {1, nk + 2});
    FA<2>* l2[] = {&play,   &tlay,   &h2ovmr, &o3vmr, &co2vmr,  &o2vmr,   &ch4vmr,  &n2ovmr, &hgt,
                   &clwpth, &ciwpth, &cswpth, &rel,   &rei,     &res,     &cldfrac, &relqmcl, &reicmcl,
                   &resnmcl, &swhr, &swhrc};
    for (auto* a : l2) b.arr(*a, {1, kts}, {1, nk + 1});
    b.arr(o3mmr, {kts}, {nk + 1});
    FA<3>* c3[] = {&taucld, &ssacld, &asmcld, &fsfcld};
    for (auto* a : c3) b.arr(*a, {1, 1, kts}, {nbndsw, 1, nk + 1});
    FA<3>* g3[] = {&cldfmcl, &clwpmcl, &ciwpmcl, &cswpmcl, &taucmcl, &ssacmcl, &asmcmcl, &fsfcmcl};
    for (auto* a : g3) b.arr(*a, {1, 1, kts}, {ngptsw, 1, nk + 1});
    FA<3>* a3[] = {&tauaer, &ssaaer, &asmaer};
    for (auto* a : a3) b.arr(*a, {1, kts, 1}, {1, nk + 1, nbndsw});
    FA<2>* f2[] = {&swuflx,    &swdflx,    &swuflxc,   &swdflxc,   &swuflxcln, &swdflxcln, &sibvisdir,
                   &sibvisdif, &sibnirdir, &sibnirdif, &swdkdir,   &swdkdif,   &swdkdirc};
    for (auto* a : f2) b.arr(*a, {1, kts}, {1, nk + 2});
    FA<2>* m2[] = {&pdel, &cicewp, &cliqwp, &csnowp, &reliq, &reice, &recloud1d, &reice1d, &resnow1d};
    for (auto* a : m2) b.arr(*a, {1, 1}, {1, nk - kts + 1});
    // generate_stochastic_clouds_sw (nlay = nl)
    FA<2>* s2[] = {&pmid, &cldf, &alpha};
    for (auto* a : s2) b.arr(*a, {1, 1}, {1, nl});
    b.arr(CDF, {1, 1, 1}, {ngptsw, 1, nl});
    if (icld == 4) b.arr(CDF2, {1, 1, 1}, {ngptsw, 1, nl});
    // rrtmg_sw
    FA<1>* r1[] = {&pavel,   &tavel,   &pdp,    &coldry, &colh2o,   &colco2, &colo3,  &coln2o, &colch4,
                   &colo2,   &colmol,  &co2mult, &selffac, &selffrac, &forfac, &forfrac, &fac00,  &fac01,
                   &fac10,   &fac11,   &relqmc, &reicmc, &resnmc};
    for (auto* a : r1) b.arr(*a, {1}, {nl + 1});
    b.arr(pz, {0}, {nl + 1});
    b.arr(tz, {0}, {nl + 1});
    b.arr(wkl, {1, 1}, {mxmol, nl + 1});
    b.arr(adjflux, {1}, {jpband});
    b.arr(solvar, {1}, {jpband});
    b.arr(albdir, {1}, {nbndsw});
    b.arr(albdif, {1}, {nbndsw});
    FA<2>* b2[] = {&taua, &ssaa, &asma, &ztaua, &ztauacln, &zasya, &zomga};
    for (auto* a : b2) b.arr(*a, {1, 1}, {nl + 1, nbndsw});
    IA<1>* i1[] = {&jp, &jt, &jt1, &indself, &indfor};
    for (auto* a : i1) b.arr(*a, {1}, {nl + 1});
    FA<2>* g2[] = {&cldfmc, &ciwpmc, &clwpmc, &cswpmc, &taucmc, &taormc, &ssacmc, &asmcmc, &fsfcmc};
    for (auto* a : g2) b.arr(*a, {1, 1}, {ngptsw, nl + 1});
    FA<2>* z2[] = {&zcldfmc, &ztaucmc, &ztaormc, &zasycmc, &zomgcmc};
    for (auto* a : z2) b.arr(*a, {1, 1}, {nl + 1, ngptsw});
    FA<1>* p1[] = {&zbbfu,   &zbbfd,    &zbbcu,    &zbbcd,  &zbbfddir, &zbbcddir, &zuvfd,   &zuvcd,
                   &zuvfddir, &zuvcddir, &znifd,    &znicd,  &znifddir, &znicddir, &swnflx,  &swnflxc,
                   &dirdflux, &difdflux, &uvdflx,   &nidflx, &dirdnuv,  &difdnuv,  &dirdnir, &difdnir};
    for (auto* a : p1) b.arr(*a, {1}, {nl + 2});
    // cldprmc_sw
    FA<1>* q1[] = {&fdelta,   &extcoice, &gice, &ssacoice, &forwice, &extcoliq, &gliq,
                   &ssacoliq, &forwliq,  &extcosno, &gsno, &ssacosno, &forwsno};
    for (auto* a : q1) b.arr(*a, {1}, {ngptsw});
    // spcvmc_sw (nlayers = nl)
    b.arr(lrtchkclr, {1}, {nl});
    b.arr(lrtchkcld, {1}, {nl});
    FA<1>* v0[] = {&zgc, &zgcc, &zgco, &zomc, &zomcc, &zomco, &ztauc, &ztauo, &ztoc, &ztor};
    for (auto* a : v0) b.arr(*a, {1}, {nl});
    FA<1>* v1[] = {&zdbt,   &zdbt_nodel, &zrdnd,  &zrdndc, &zref,   &zrefc,  &zrefo,       &zrefd,       &zrefdc,
                   &zrefdo, &zrup,       &zrupd,  &zrupc,  &zrupdc, &zs1,    &ztdn,        &ztdnd,       &ztdbt,
                   &ztra,   &ztrac,      &ztrao,  &ztrad,  &ztradc, &ztrado, &zdbtc,       &ztdbtc,      &zdbtc_nodel,
                   &ztdbt_nodel, &ztdbtc_nodel, &vq_ztdn};
    for (auto* a : v1) b.arr(*a, {1}, {nl + 1});
    b.arr(zincflx, {1}, {ngptsw});
    b.arr(zsflxzen, {1}, {ngptsw});
    b.arr(ztaug, {1, 1}, {nl, ngptsw});
    b.arr(ztaur, {1, 1}, {nl, ngptsw});
    FA<2>* w2[] = {&zcd, &zcu, &zfd, &zfu};
    for (auto* a : w2) b.arr(*a, {1, 1}, {nl + 1, ngptsw});
  }

  // cldprmc_sw: cloud optical properties of the McICA subcolumns
  KOKKOS_INLINE_FUNCTION void cldprmc_sw() {
    int ib;
    int lay;
    int istr;
    int index;
    int icx;
    int ig;
    constexpr Real eps = 1.0e-06;
    constexpr Real cldmin = 1.0e-20;
    Real cwp;
    Real radliq;
    Real radice;
    Real radsno;
    Real factor;
    Real fint;
    Real taucldorig_a;
    Real taucloud_a;
    Real ssacloud_a;
    Real ffp;
    Real ffp1;
    Real ffpssa;
    Real tauiceorig;
    Real scatice;
    Real ssaice;
    Real tauice;
    Real tauliqorig;
    Real scatliq;
    Real ssaliq;
    Real tauliq;
    Real tausnoorig;
    Real scatsno;
    Real ssasno;
    Real tausno;
    for (lay = 1; lay <= nlayers; ++lay) {
      for (ig = 1; ig <= ngptsw; ++ig) {
        taormc(ig, lay) = taucmc(ig, lay);
      }
    }
    for (lay = 1; lay <= nlayers; ++lay) {
      for (ig = 1; ig <= ngptsw; ++ig) {
        cwp = ciwpmc(ig, lay) + clwpmc(ig, lay) + cswpmc(ig, lay);
        if (cldfmc(ig, lay) >= cldmin && (cwp >= cldmin || taucmc(ig, lay) >= cldmin)) {
          if (inflag == 0) {
            taucldorig_a = taucmc(ig, lay);
            ffp = fsfcmc(ig, lay);
            ffp1 = 1.0 - ffp;
            ffpssa = 1.0 - ffp * ssacmc(ig, lay);
            ssacloud_a = ffp1 * ssacmc(ig, lay) / ffpssa;
            taucloud_a = ffpssa * taucldorig_a;
            taormc(ig, lay) = taucldorig_a;
            ssacmc(ig, lay) = ssacloud_a;
            taucmc(ig, lay) = taucloud_a;
            asmcmc(ig, lay) = (asmcmc(ig, lay) - ffp) / (ffp1);
          } else if (inflag == 1) {
            SW_FAIL(SW_INFLAG1);
          } else if (inflag >= 2) {
            radice = reicmc(lay);
            if ((ciwpmc(ig, lay) + cswpmc(ig, lay)) == 0.0) {
              extcoice(ig) = 0.0;
              ssacoice(ig) = 0.0;
              gice(ig) = 0.0;
              forwice(ig) = 0.0;
              extcosno(ig) = 0.0;
              ssacosno(ig) = 0.0;
              gsno(ig) = 0.0;
              forwsno(ig) = 0.0;
            } else if (iceflag == 1) {
              ib = ngb(ig);
              if (wavenum2(ib) > 1.43e04) {
                icx = 1;
              } else if (wavenum2(ib) > 7.7e03) {
                icx = 2;
              } else if (wavenum2(ib) > 5.3e03) {
                icx = 3;
              } else if (wavenum2(ib) > 4.0e03) {
                icx = 4;
              } else if (wavenum2(ib) >= 2.5e03) {
                icx = 5;
              }
              extcoice(ig) = (abari(icx) + bbari(icx) / radice);
              ssacoice(ig) = 1.0 - cbari(icx) - dbari(icx) * radice;
              gice(ig) = ebari(icx) + fbari(icx) * radice;
              if (gice(ig) >= 1.0) gice(ig) = 1.0 - eps;
              forwice(ig) = gice(ig) * gice(ig);
              if (extcoice(ig) < 0.0) SW_FAIL(SW_ICE_EXT);
              if (ssacoice(ig) > 1.0) SW_FAIL(SW_ICE_SSA_GT);
              if (ssacoice(ig) < 0.0) SW_FAIL(SW_ICE_SSA_LT);
              if (gice(ig) > 1.0) SW_FAIL(SW_ICE_ASY_GT);
              if (gice(ig) < 0.0) SW_FAIL(SW_ICE_ASY_LT);
            } else if (iceflag == 2) {
              if (radice < 5.0 || radice > 131.0) SW_FAIL(SW_ICE_BOUNDS);
              factor = (radice - 2.0) / 3.0;
              index = int(factor);
              if (index == 43) index = 42;
              fint = factor - Real(index);
              ib = ngb(ig);
              extcoice(ig) = extice2(index, ib) + fint * (extice2(index + 1, ib) - extice2(index, ib));
              ssacoice(ig) = ssaice2(index, ib) + fint * (ssaice2(index + 1, ib) - ssaice2(index, ib));
              gice(ig) = asyice2(index, ib) + fint * (asyice2(index + 1, ib) - asyice2(index, ib));
              forwice(ig) = gice(ig) * gice(ig);
              if (extcoice(ig) < 0.0) SW_FAIL(SW_ICE_EXT);
              if (ssacoice(ig) > 1.0) SW_FAIL(SW_ICE_SSA_GT);
              if (ssacoice(ig) < 0.0) SW_FAIL(SW_ICE_SSA_LT);
              if (gice(ig) > 1.0) SW_FAIL(SW_ICE_ASY_GT);
              if (gice(ig) < 0.0) SW_FAIL(SW_ICE_ASY_LT);
            } else if (iceflag >= 3) {
              if (radice < 5.0 || radice > 140.0) {
                SW_FAIL(SW_ICE_GEN_BOUNDS);
              }
              factor = (radice - 2.0) / 3.0;
              index = int(factor);
              if (index == 46) index = 45;
              fint = factor - Real(index);
              ib = ngb(ig);
              extcoice(ig) = extice3(index, ib) + fint * (extice3(index + 1, ib) - extice3(index, ib));
              ssacoice(ig) = ssaice3(index, ib) + fint * (ssaice3(index + 1, ib) - ssaice3(index, ib));
              gice(ig) = asyice3(index, ib) + fint * (asyice3(index + 1, ib) - asyice3(index, ib));
              fdelta(ig) = fdlice3(index, ib) + fint * (fdlice3(index + 1, ib) - fdlice3(index, ib));
              if (fdelta(ig) < 0.0) {
                SW_FAIL(SW_FDELTA_LT);
              }
              if (fdelta(ig) > 1.0) {
                SW_FAIL(SW_FDELTA_GT);
              }
              forwice(ig) = fdelta(ig) + 0.5 / ssacoice(ig);
              if (forwice(ig) > gice(ig)) forwice(ig) = gice(ig);
              if (extcoice(ig) < 0.0) SW_FAIL(SW_ICE_EXT);
              if (ssacoice(ig) > 1.0) SW_FAIL(SW_ICE_SSA_GT);
              if (ssacoice(ig) < 0.0) SW_FAIL(SW_ICE_SSA_LT);
              if (gice(ig) > 1.0) SW_FAIL(SW_ICE_ASY_GT);
              if (gice(ig) < 0.0) SW_FAIL(SW_ICE_ASY_LT);
            }
            if (cswpmc(ig, lay) > 0.0 && iceflag == 5) {
              radsno = resnmc(lay);
              if (radsno < 5.0 || radsno > 140.0) {
                SW_FAIL(SW_SNOW_GEN_BOUNDS);
              }
              factor = (radsno - 2.0) / 3.0;
              index = int(factor);
              if (index == 46) index = 45;
              fint = factor - Real(index);
              ib = ngb(ig);
              extcosno(ig) = extice3(index, ib) + fint * (extice3(index + 1, ib) - extice3(index, ib));
              ssacosno(ig) = ssaice3(index, ib) + fint * (ssaice3(index + 1, ib) - ssaice3(index, ib));
              gsno(ig) = asyice3(index, ib) + fint * (asyice3(index + 1, ib) - asyice3(index, ib));
              fdelta(ig) = fdlice3(index, ib) + fint * (fdlice3(index + 1, ib) - fdlice3(index, ib));
              if (fdelta(ig) < 0.0) {
                SW_FAIL(SW_FDELTA_LT);
              }
              if (fdelta(ig) > 1.0) {
                SW_FAIL(SW_FDELTA_GT);
              }
              forwsno(ig) = fdelta(ig) + 0.5 / ssacosno(ig);
              if (forwsno(ig) > gsno(ig)) forwsno(ig) = gsno(ig);
              if (extcosno(ig) < 0.0) {
                SW_FAIL(SW_SNOW_EXT);
              }
              if (ssacosno(ig) > 1.0) {
                SW_FAIL(SW_SNOW_SSA_GT);
              }
              if (ssacosno(ig) < 0.0) {
                SW_FAIL(SW_SNOW_SSA_LT);
              }
              if (gsno(ig) > 1.0) {
                SW_FAIL(SW_SNOW_ASY_GT);
              }
              if (gsno(ig) < 0.0) {
                SW_FAIL(SW_SNOW_ASY_LT);
              }
            } else {
              extcosno(ig) = 0.0;
              ssacosno(ig) = 0.0;
              gsno(ig) = 0.0;
              forwsno(ig) = 0.0;
            }
            if (clwpmc(ig, lay) == 0.0) {
              extcoliq(ig) = 0.0;
              ssacoliq(ig) = 0.0;
              gliq(ig) = 0.0;
              forwliq(ig) = 0.0;
            } else if (liqflag == 1) {
              radliq = relqmc(lay);
              if (radliq < 1.5 || radliq > 60.0) SW_FAIL(SW_LIQ_BOUNDS);
              index = int(radliq - 1.5);
              if (index == 0) index = 1;
              if (index == 58) index = 57;
              fint = radliq - 1.5 - Real(index);
              ib = ngb(ig);
              extcoliq(ig) = extliq1(index, ib) + fint * (extliq1(index + 1, ib) - extliq1(index, ib));
              ssacoliq(ig) = ssaliq1(index, ib) + fint * (ssaliq1(index + 1, ib) - ssaliq1(index, ib));
              if (fint < 0.0 && ssacoliq(ig) > 1.0) ssacoliq(ig) = ssaliq1(index, ib);
              gliq(ig) = asyliq1(index, ib) + fint * (asyliq1(index + 1, ib) - asyliq1(index, ib));
              forwliq(ig) = gliq(ig) * gliq(ig);
              if (extcoliq(ig) < 0.0) SW_FAIL(SW_LIQ_EXT);
              if (ssacoliq(ig) > 1.0) SW_FAIL(SW_LIQ_SSA_GT);
              if (ssacoliq(ig) < 0.0) SW_FAIL(SW_LIQ_SSA_LT);
              if (gliq(ig) > 1.0) SW_FAIL(SW_LIQ_ASY_GT);
              if (gliq(ig) < 0.0) SW_FAIL(SW_LIQ_ASY_LT);
            }
            if (iceflag < 5) {
              tauliqorig = clwpmc(ig, lay) * extcoliq(ig);
              tauiceorig = ciwpmc(ig, lay) * extcoice(ig);
              taormc(ig, lay) = tauliqorig + tauiceorig;
              ssaliq = ssacoliq(ig) * (1.0 - forwliq(ig)) / (1.0 - forwliq(ig) * ssacoliq(ig));
              tauliq = (1.0 - forwliq(ig) * ssacoliq(ig)) * tauliqorig;
              ssaice = ssacoice(ig) * (1.0 - forwice(ig)) / (1.0 - forwice(ig) * ssacoice(ig));
              tauice = (1.0 - forwice(ig) * ssacoice(ig)) * tauiceorig;
              scatliq = ssaliq * tauliq;
              scatice = ssaice * tauice;
              scatsno = 0.0;
              taucmc(ig, lay) = tauliq + tauice;
            } else {
              tauliqorig = clwpmc(ig, lay) * extcoliq(ig);
              tauiceorig = ciwpmc(ig, lay) * extcoice(ig);
              tausnoorig = cswpmc(ig, lay) * extcosno(ig);
              taormc(ig, lay) = tauliqorig + tauiceorig + tausnoorig;
              ssaliq = ssacoliq(ig) * (1.0 - forwliq(ig)) / (1.0 - forwliq(ig) * ssacoliq(ig));
              tauliq = (1.0 - forwliq(ig) * ssacoliq(ig)) * tauliqorig;
              ssaice = ssacoice(ig) * (1.0 - forwice(ig)) / (1.0 - forwice(ig) * ssacoice(ig));
              tauice = (1.0 - forwice(ig) * ssacoice(ig)) * tauiceorig;
              ssasno = ssacosno(ig) * (1.0 - forwsno(ig)) / (1.0 - forwsno(ig) * ssacosno(ig));
              tausno = (1.0 - forwsno(ig) * ssacosno(ig)) * tausnoorig;
              scatliq = ssaliq * tauliq;
              scatice = ssaice * tauice;
              scatsno = ssasno * tausno;
              taucmc(ig, lay) = tauliq + tauice + tausno;
            }
            if (taucmc(ig, lay) == 0.0) taucmc(ig, lay) = cldmin;
            if (scatice == 0.0) scatice = cldmin;
            if (scatsno == 0.0) scatsno = cldmin;
            if (iceflag < 5) {
              ssacmc(ig, lay) = (scatliq + scatice) / taucmc(ig, lay);
            } else {
              ssacmc(ig, lay) = (scatliq + scatice + scatsno) / taucmc(ig, lay);
            }
            if (iceflag == 3 || iceflag == 4) {
              istr = 1;
              asmcmc(ig, lay) = (1.0 / (scatliq + scatice)) * (scatliq * (gliq(ig) - forwliq(ig)) / (1.0 - forwliq(ig)) + scatice * ((gice(ig) - forwice(ig)) / (1.0 - forwice(ig))));
            } else if (iceflag == 5) {
              istr = 1;
              asmcmc(ig, lay) = (1.0 / (scatliq + scatice + scatsno)) * (scatliq * (gliq(ig) - forwliq(ig)) / (1.0 - forwliq(ig)) + scatice * ((gice(ig) - forwice(ig)) / (1.0 - forwice(ig))) + scatsno * ((gsno(ig) - forwsno(ig)) / (1.0 - forwsno(ig))));
            } else {
              istr = 1;
              asmcmc(ig, lay) = (scatliq * (gliq(ig) - forwliq(ig)) / (1.0 - forwliq(ig)) + scatice * (gice(ig) - forwice(ig)) / (1.0 - forwice(ig))) / (scatliq + scatice);
            }
          }
        }
      }
    }

  }

  // reftra_sw: layer reflectivity and transmissivity (two-stream, kmodts = 2)
  KOKKOS_INLINE_FUNCTION void reftra_sw(int nlayers, const IA<1>& lrtchk, const FA<1>& pgg, Real prmuz, const FA<1>& ptau, const FA<1>& pw,
                                           const FA<1>& pref, const FA<1>& prefd, const FA<1>& ptra, const FA<1>& ptrad) const {
    int jk;
    int jl;
    int kmodts;
    int itind;
    Real tblind;
    Real za;
    Real za1;
    Real za2;
    Real zbeta;
    Real zdend;
    Real zdenr;
    Real zdent;
    Real ze1;
    Real ze2;
    Real zem1;
    Real zem2;
    Real zemm;
    Real zep1;
    Real zep2;
    Real zg;
    Real zg3;
    Real zgamma1;
    Real zgamma2;
    Real zgamma3;
    Real zgamma4;
    Real zgt;
    Real zr1;
    Real zr2;
    Real zr3;
    Real zr4;
    Real zr5;
    Real zrk;
    Real zrk2;
    Real zrkg;
    Real zrm1;
    Real zrp;
    Real zrp1;
    Real zrpp;
    Real zsr3;
    Real zt1;
    Real zt2;
    Real zt3;
    Real zt4;
    Real zt5;
    Real zto1;
    Real zw;
    Real zwcrit;
    Real zwo;
    Real denom;
    constexpr Real eps = 1.0e-08;
    zsr3 = Kokkos::sqrt(3.0);
    zwcrit = 0.9999995;
    kmodts = 2;
    for (jk = 1; jk <= nlayers; ++jk) {
      if (!lrtchk(jk)) {
        pref(jk) = 0.0;
        ptra(jk) = 1.0;
        prefd(jk) = 0.0;
        ptrad(jk) = 1.0;
      } else {
        zto1 = ptau(jk);
        zw = pw(jk);
        zg = pgg(jk);
        zg3 = 3.0 * zg;
        if (kmodts == 1) {
          zgamma1 = (7.0 - zw * (4.0 + zg3)) * 0.25;
          zgamma2 = -((1.0 - zw * (4.0 - zg3)) * 0.25);
          zgamma3 = (2.0 - zg3 * prmuz) * 0.25;
        } else if (kmodts == 2) {
          zgamma1 = (8.0 - zw * (5.0 + zg3)) * 0.25;
          zgamma2 = 3.0 * (zw * (1.0 - zg)) * 0.25;
          zgamma3 = (2.0 - zg3 * prmuz) * 0.25;
        } else if (kmodts == 3) {
          zgamma1 = zsr3 * (2.0 - zw * (1.0 + zg)) * 0.5;
          zgamma2 = zsr3 * zw * (1.0 - zg) * 0.5;
          zgamma3 = (1.0 - zsr3 * zg * prmuz) * 0.5;
        }
        zgamma4 = 1.0 - zgamma3;
        zwo = 0.0;
        denom = 1.0;
        if (zg != 1.0) denom = (1.0 - (1.0 - zw) * ((zg / (1.0 - zg))*(zg / (1.0 - zg))));
        if (zw > 0.0 && denom != 0.0) zwo = zw / denom;
        if (zwo >= zwcrit) {
          za = zgamma1 * prmuz;
          za1 = za - zgamma3;
          zgt = zgamma1 * zto1;
          ze1 = Kokkos::min(zto1 / prmuz, 500.0);
          if (ze1 <= od_lo) {
            ze2 = 1.0 - ze1 + 0.5 * ze1 * ze1;
          } else {
            tblind = ze1 / (bpade + ze1);
            itind = int(tblint * tblind + 0.5);
            ze2 = exp_tbl(itind);
          }
          pref(jk) = (zgt - za1 * (1.0 - ze2)) / (1.0 + zgt);
          ptra(jk) = 1.0 - pref(jk);
          prefd(jk) = zgt / (1.0 + zgt);
          ptrad(jk) = 1.0 - prefd(jk);
          if (ze2 == 1.0) {
            pref(jk) = 0.0;
            ptra(jk) = 1.0;
            prefd(jk) = 0.0;
            ptrad(jk) = 1.0;
          }
        } else {
          za1 = zgamma1 * zgamma4 + zgamma2 * zgamma3;
          za2 = zgamma1 * zgamma3 + zgamma2 * zgamma4;
          zrk = Kokkos::sqrt((zgamma1*zgamma1) - (zgamma2*zgamma2));
          zrp = zrk * prmuz;
          zrp1 = 1.0 + zrp;
          zrm1 = 1.0 - zrp;
          zrk2 = 2.0 * zrk;
          zrpp = 1.0 - zrp * zrp;
          zrkg = zrk + zgamma1;
          zr1 = zrm1 * (za2 + zrk * zgamma3);
          zr2 = zrp1 * (za2 - zrk * zgamma3);
          zr3 = zrk2 * (zgamma3 - za2 * prmuz);
          zr4 = zrpp * zrkg;
          zr5 = zrpp * (zrk - zgamma1);
          zt1 = zrp1 * (za1 + zrk * zgamma4);
          zt2 = zrm1 * (za1 - zrk * zgamma4);
          zt3 = zrk2 * (zgamma4 + za1 * prmuz);
          zt4 = zr4;
          zt5 = zr5;
          zbeta = (zgamma1 - zrk) / zrkg;
          ze1 = Kokkos::min(zrk * zto1, 500.0);
          ze2 = Kokkos::min(zto1 / prmuz, 500.0);
          if (ze1 <= od_lo) {
            zem1 = 1.0 - ze1 + 0.5 * ze1 * ze1;
            zep1 = 1.0 / zem1;
          } else {
            tblind = ze1 / (bpade + ze1);
            itind = int(tblint * tblind + 0.5);
            zem1 = exp_tbl(itind);
            zep1 = 1.0 / zem1;
          }
          if (ze2 <= od_lo) {
            zem2 = 1.0 - ze2 + 0.5 * ze2 * ze2;
            zep2 = 1.0 / zem2;
          } else {
            tblind = ze2 / (bpade + ze2);
            itind = int(tblint * tblind + 0.5);
            zem2 = exp_tbl(itind);
            zep2 = 1.0 / zem2;
          }
          zdenr = zr4 * zep1 + zr5 * zem1;
          zdent = zt4 * zep1 + zt5 * zem1;
          if (zdenr >= -eps && zdenr <= eps) {
            pref(jk) = eps;
            ptra(jk) = zem2;
          } else {
            pref(jk) = zw * (zr1 * zep1 - zr2 * zem1 - zr3 * zem2) / zdenr;
            ptra(jk) = zem2 - zem2 * zw * (zt1 * zep1 - zt2 * zem1 - zt3 * zep2) / zdent;
          }
          zemm = zem1 * zem1;
          zdend = 1.0 / ((1.0 - zbeta * zemm) * zrkg);
          prefd(jk) = zgamma2 * (1.0 - zemm) * zdend;
          ptrad(jk) = zrk2 * zem1 * zdend;
        }
      }
    }

  }

  // setcoef_sw
  KOKKOS_INLINE_FUNCTION void setcoef_sw() {
    int lay;
    int jp1;
    Real stpfac;
    Real plog;
    Real fp;
    Real ft;
    Real ft1;
    Real water;
    Real scalefac;
    Real factor;
    Real co2reg;
    Real compfp;
    stpfac = 296.0 / 1013.0;
    laytrop = 0;
    layswtch = 0;
    laylow = 0;
    for (lay = 1; lay <= nlayers; ++lay) {
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
      if (!(plog <= 4.56)) {  // (goto 5300 when plog <= 4.56)
        laytrop = laytrop + 1;
        if (plog >= 6.62) laylow = laylow + 1;
        forfac(lay) = scalefac / (1.0 + water);
        factor = (332.0 - tavel(lay)) / 36.0;
        indfor(lay) = Kokkos::min(2, Kokkos::max(1, int(factor)));
        forfrac(lay) = factor - Real(indfor(lay));
        selffac(lay) = water * forfac(lay);
        factor = (tavel(lay) - 188.0) / 7.2;
        indself(lay) = Kokkos::min(9, Kokkos::max(1, int(factor) - 7));
        selffrac(lay) = factor - Real(indself(lay) + 7);
        colh2o(lay) = 1.0e-20 * wkl(1, lay);
        colco2(lay) = 1.0e-20 * wkl(2, lay);
        colo3(lay) = 1.0e-20 * wkl(3, lay);
        coln2o(lay) = 1.0e-20 * wkl(4, lay);
        colch4(lay) = 1.0e-20 * wkl(6, lay);
        colo2(lay) = 1.0e-20 * wkl(7, lay);
        colmol(lay) = 1.0e-20 * coldry(lay) + colh2o(lay);
        if (colco2(lay) == 0.0) colco2(lay) = 1.0e-32 * coldry(lay);
        if (coln2o(lay) == 0.0) coln2o(lay) = 1.0e-32 * coldry(lay);
        if (colch4(lay) == 0.0) colch4(lay) = 1.0e-32 * coldry(lay);
        if (colo2(lay) == 0.0) colo2(lay) = 1.0e-32 * coldry(lay);
        co2reg = 3.55e-24 * coldry(lay);
        co2mult(lay) = (colco2(lay) - co2reg) * 272.63 * Kokkos::exp(-(1919.4 / tavel(lay))) / (8.7604e-4 * tavel(lay));
      } else {  // 5300
        forfac(lay) = scalefac / (1.0 + water);
        factor = (tavel(lay) - 188.0) / 36.0;
        indfor(lay) = 3;
        forfrac(lay) = factor - 1.0;
        colh2o(lay) = 1.0e-20 * wkl(1, lay);
        colco2(lay) = 1.0e-20 * wkl(2, lay);
        colo3(lay) = 1.0e-20 * wkl(3, lay);
        coln2o(lay) = 1.0e-20 * wkl(4, lay);
        colch4(lay) = 1.0e-20 * wkl(6, lay);
        colo2(lay) = 1.0e-20 * wkl(7, lay);
        colmol(lay) = 1.0e-20 * coldry(lay) + colh2o(lay);
        if (colco2(lay) == 0.0) colco2(lay) = 1.0e-32 * coldry(lay);
        if (coln2o(lay) == 0.0) coln2o(lay) = 1.0e-32 * coldry(lay);
        if (colch4(lay) == 0.0) colch4(lay) = 1.0e-32 * coldry(lay);
        if (colo2(lay) == 0.0) colo2(lay) = 1.0e-32 * coldry(lay);
        co2reg = 3.55e-24 * coldry(lay);
        co2mult(lay) = (colco2(lay) - co2reg) * 272.63 * Kokkos::exp(-(1919.4 / tavel(lay))) / (8.7604e-4 * tavel(lay));
        selffac(lay) = 0.0;
        selffrac(lay) = 0.0;
        indself(lay) = 0;
      }  // 5400
      compfp = 1.0 - fp;
      fac10(lay) = compfp * ft;
      fac00(lay) = compfp * (1.0 - ft);
      fac11(lay) = fp * ft1;
      fac01(lay) = fp * (1.0 - ft1);
    }

  }

  // taumol_sw: gaseous and Rayleigh optical depths and the TOA solar source of the 14 bands
  // (taug = ztaug, taur = ztaur, sfluxzen = zsflxzen of spcvmc_sw)
  KOKKOS_INLINE_FUNCTION void taumol_sw() {
    for (int ig = 1; ig <= ngptsw; ++ig) zsflxzen(ig) = 0.0;
    taumol16();
    taumol17();
    taumol18();
    taumol19();
    taumol20();
    taumol21();
    taumol22();
    taumol23();
    taumol24();
    taumol25();
    taumol26();
    taumol27();
    taumol28();
    taumol29();
  }

  KOKKOS_INLINE_FUNCTION void taumol16() {
    const auto& absa = kg16.absa;
    const auto& absb = kg16.absb;
    const auto& forref = kg16.forref;
    const auto& selfref = kg16.selfref;
    const auto& sfluxref = kg16.sfluxref;
    const auto& rayl = kg16.rayl;
    const auto& layreffr = kg16.layreffr;
    const auto& strrat1 = kg16.strrat1;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colh2o(lay) + strrat1 * colch4(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(16) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(16) + js;
      inds = indself(lay);
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng16; ++ig) {
        ztaug(lay, ig) = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig)) + colh2o(lay) * (selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig))) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig))));
        ztaur(lay, ig) = tauray;
      }
    }
    laysolfr = nlayers;
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      if (jp(lay - 1) < layreffr && jp(lay) >= layreffr) laysolfr = lay;
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(16) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(16) + 1;
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng16; ++ig) {
        ztaug(lay, ig) = colch4(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig));
        if (lay == laysolfr) zsflxzen(ig) = sfluxref(ig);
        ztaur(lay, ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol17() {
    const auto& absa = kg17.absa;
    const auto& absb = kg17.absb;
    const auto& forref = kg17.forref;
    const auto& selfref = kg17.selfref;
    const auto& sfluxref = kg17.sfluxref;
    const auto& rayl = kg17.rayl;
    const auto& layreffr = kg17.layreffr;
    const auto& strrat = kg17.strrat;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colh2o(lay) + strrat * colco2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(17) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(17) + js;
      inds = indself(lay);
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng17; ++ig) {
        ztaug(lay, ngs16 + ig) = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig)) + colh2o(lay) * (selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig))) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig))));
        ztaur(lay, ngs16 + ig) = tauray;
      }
    }
    laysolfr = nlayers;
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      if (jp(lay - 1) < layreffr && jp(lay) >= layreffr) laysolfr = lay;
      speccomb = colh2o(lay) + strrat * colco2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 4.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(17) + js;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(17) + js;
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng17; ++ig) {
        ztaug(lay, ngs16 + ig) = speccomb * (fac000 * absb(ind0, ig) + fac100 * absb(ind0 + 1, ig) + fac010 * absb(ind0 + 5, ig) + fac110 * absb(ind0 + 6, ig) + fac001 * absb(ind1, ig) + fac101 * absb(ind1 + 1, ig) + fac011 * absb(ind1 + 5, ig) + fac111 * absb(ind1 + 6, ig)) + colh2o(lay) * forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        if (lay == laysolfr) zsflxzen(ngs16 + ig) = sfluxref(ig, js) + fs * (sfluxref(ig, js + 1) - sfluxref(ig, js));
        ztaur(lay, ngs16 + ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol18() {
    const auto& absa = kg18.absa;
    const auto& absb = kg18.absb;
    const auto& forref = kg18.forref;
    const auto& selfref = kg18.selfref;
    const auto& sfluxref = kg18.sfluxref;
    const auto& rayl = kg18.rayl;
    const auto& layreffr = kg18.layreffr;
    const auto& strrat = kg18.strrat;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    laysolfr = laytrop;
    for (lay = 1; lay <= laytrop; ++lay) {
      if (jp(lay) < layreffr && jp(lay + 1) >= layreffr) laysolfr = Kokkos::min(lay + 1, laytrop);
      speccomb = colh2o(lay) + strrat * colch4(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(18) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(18) + js;
      inds = indself(lay);
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng18; ++ig) {
        ztaug(lay, ngs17 + ig) = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig)) + colh2o(lay) * (selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig))) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig))));
        if (lay == laysolfr) zsflxzen(ngs17 + ig) = sfluxref(ig, js) + fs * (sfluxref(ig, js + 1) - sfluxref(ig, js));
        ztaur(lay, ngs17 + ig) = tauray;
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(18) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(18) + 1;
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng18; ++ig) {
        ztaug(lay, ngs17 + ig) = colch4(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig));
        ztaur(lay, ngs17 + ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol19() {
    const auto& absa = kg19.absa;
    const auto& absb = kg19.absb;
    const auto& forref = kg19.forref;
    const auto& selfref = kg19.selfref;
    const auto& sfluxref = kg19.sfluxref;
    const auto& rayl = kg19.rayl;
    const auto& layreffr = kg19.layreffr;
    const auto& strrat = kg19.strrat;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    laysolfr = laytrop;
    for (lay = 1; lay <= laytrop; ++lay) {
      if (jp(lay) < layreffr && jp(lay + 1) >= layreffr) laysolfr = Kokkos::min(lay + 1, laytrop);
      speccomb = colh2o(lay) + strrat * colco2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(19) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(19) + js;
      inds = indself(lay);
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng19; ++ig) {
        ztaug(lay, ngs18 + ig) = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig)) + colh2o(lay) * (selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig))) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig))));
        if (lay == laysolfr) zsflxzen(ngs18 + ig) = sfluxref(ig, js) + fs * (sfluxref(ig, js + 1) - sfluxref(ig, js));
        ztaur(lay, ngs18 + ig) = tauray;
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(19) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(19) + 1;
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng19; ++ig) {
        ztaug(lay, ngs18 + ig) = colco2(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig));
        ztaur(lay, ngs18 + ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol20() {
    const auto& absa = kg20.absa;
    const auto& absb = kg20.absb;
    const auto& forref = kg20.forref;
    const auto& selfref = kg20.selfref;
    const auto& sfluxref = kg20.sfluxref;
    const auto& absch4 = kg20.absch4;
    const auto& rayl = kg20.rayl;
    const auto& layreffr = kg20.layreffr;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    laysolfr = laytrop;
    for (lay = 1; lay <= laytrop; ++lay) {
      if (jp(lay) < layreffr && jp(lay + 1) >= layreffr) laysolfr = Kokkos::min(lay + 1, laytrop);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(20) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(20) + 1;
      inds = indself(lay);
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng20; ++ig) {
        ztaug(lay, ngs19 + ig) = colh2o(lay) * ((fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig))) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)))) + colch4(lay) * absch4(ig);
        ztaur(lay, ngs19 + ig) = tauray;
        if (lay == laysolfr) zsflxzen(ngs19 + ig) = sfluxref(ig);
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(20) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(20) + 1;
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng20; ++ig) {
        ztaug(lay, ngs19 + ig) = colh2o(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)))) + colch4(lay) * absch4(ig);
        ztaur(lay, ngs19 + ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol21() {
    const auto& absa = kg21.absa;
    const auto& absb = kg21.absb;
    const auto& forref = kg21.forref;
    const auto& selfref = kg21.selfref;
    const auto& sfluxref = kg21.sfluxref;
    const auto& rayl = kg21.rayl;
    const auto& layreffr = kg21.layreffr;
    const auto& strrat = kg21.strrat;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    laysolfr = laytrop;
    for (lay = 1; lay <= laytrop; ++lay) {
      if (jp(lay) < layreffr && jp(lay + 1) >= layreffr) laysolfr = Kokkos::min(lay + 1, laytrop);
      speccomb = colh2o(lay) + strrat * colco2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(21) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(21) + js;
      inds = indself(lay);
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng21; ++ig) {
        ztaug(lay, ngs20 + ig) = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig)) + colh2o(lay) * (selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig))) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig))));
        if (lay == laysolfr) zsflxzen(ngs20 + ig) = sfluxref(ig, js) + fs * (sfluxref(ig, js + 1) - sfluxref(ig, js));
        ztaur(lay, ngs20 + ig) = tauray;
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      speccomb = colh2o(lay) + strrat * colco2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 4.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(21) + js;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(21) + js;
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng21; ++ig) {
        ztaug(lay, ngs20 + ig) = speccomb * (fac000 * absb(ind0, ig) + fac100 * absb(ind0 + 1, ig) + fac010 * absb(ind0 + 5, ig) + fac110 * absb(ind0 + 6, ig) + fac001 * absb(ind1, ig) + fac101 * absb(ind1 + 1, ig) + fac011 * absb(ind1 + 5, ig) + fac111 * absb(ind1 + 6, ig)) + colh2o(lay) * forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)));
        ztaur(lay, ngs20 + ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol22() {
    const auto& absa = kg22.absa;
    const auto& absb = kg22.absb;
    const auto& forref = kg22.forref;
    const auto& selfref = kg22.selfref;
    const auto& sfluxref = kg22.sfluxref;
    const auto& rayl = kg22.rayl;
    const auto& layreffr = kg22.layreffr;
    const auto& strrat = kg22.strrat;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    Real o2adj;
    Real o2cont;
    o2adj = 1.6;
    laysolfr = laytrop;
    for (lay = 1; lay <= laytrop; ++lay) {
      if (jp(lay) < layreffr && jp(lay + 1) >= layreffr) laysolfr = Kokkos::min(lay + 1, laytrop);
      o2cont = 4.35e-4 * colo2(lay) / (350.0 * 2.0);
      speccomb = colh2o(lay) + o2adj * strrat * colo2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(22) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(22) + js;
      inds = indself(lay);
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng22; ++ig) {
        ztaug(lay, ngs21 + ig) = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig)) + colh2o(lay) * (selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig))) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)))) + o2cont;
        if (lay == laysolfr) zsflxzen(ngs21 + ig) = sfluxref(ig, js) + fs * (sfluxref(ig, js + 1) - sfluxref(ig, js));
        ztaur(lay, ngs21 + ig) = tauray;
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      o2cont = 4.35e-4 * colo2(lay) / (350.0 * 2.0);
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(22) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(22) + 1;
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng22; ++ig) {
        ztaug(lay, ngs21 + ig) = colo2(lay) * o2adj * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig)) + o2cont;
        ztaur(lay, ngs21 + ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol23() {
    const auto& absa = kg23.absa;
    const auto& forref = kg23.forref;
    const auto& selfref = kg23.selfref;
    const auto& sfluxref = kg23.sfluxref;
    const auto& rayl = kg23.rayl;
    const auto& layreffr = kg23.layreffr;
    const auto& givfac = kg23.givfac;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    laysolfr = laytrop;
    for (lay = 1; lay <= laytrop; ++lay) {
      if (jp(lay) < layreffr && jp(lay + 1) >= layreffr) laysolfr = Kokkos::min(lay + 1, laytrop);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(23) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(23) + 1;
      inds = indself(lay);
      indf = indfor(lay);
      for (ig = 1; ig <= ng23; ++ig) {
        tauray = colmol(lay) * rayl(ig);
        ztaug(lay, ngs22 + ig) = colh2o(lay) * (givfac * (fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig))) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig))));
        if (lay == laysolfr) zsflxzen(ngs22 + ig) = sfluxref(ig);
        ztaur(lay, ngs22 + ig) = tauray;
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      for (ig = 1; ig <= ng23; ++ig) {
        ztaug(lay, ngs22 + ig) = 0.0;
        ztaur(lay, ngs22 + ig) = colmol(lay) * rayl(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol24() {
    const auto& absa = kg24.absa;
    const auto& absb = kg24.absb;
    const auto& forref = kg24.forref;
    const auto& selfref = kg24.selfref;
    const auto& sfluxref = kg24.sfluxref;
    const auto& abso3a = kg24.abso3a;
    const auto& abso3b = kg24.abso3b;
    const auto& rayla = kg24.rayla;
    const auto& raylb = kg24.raylb;
    const auto& layreffr = kg24.layreffr;
    const auto& strrat = kg24.strrat;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    laysolfr = laytrop;
    for (lay = 1; lay <= laytrop; ++lay) {
      if (jp(lay) < layreffr && jp(lay + 1) >= layreffr) laysolfr = Kokkos::min(lay + 1, laytrop);
      speccomb = colh2o(lay) + strrat * colo2(lay);
      specparm = colh2o(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(24) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(24) + js;
      inds = indself(lay);
      indf = indfor(lay);
      for (ig = 1; ig <= ng24; ++ig) {
        tauray = colmol(lay) * (rayla(ig, js) + fs * (rayla(ig, js + 1) - rayla(ig, js)));
        ztaug(lay, ngs23 + ig) = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig)) + colo3(lay) * abso3a(ig) + colh2o(lay) * (selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig))) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig))));
        if (lay == laysolfr) zsflxzen(ngs23 + ig) = sfluxref(ig, js) + fs * (sfluxref(ig, js + 1) - sfluxref(ig, js));
        ztaur(lay, ngs23 + ig) = tauray;
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(24) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(24) + 1;
      for (ig = 1; ig <= ng24; ++ig) {
        tauray = colmol(lay) * raylb(ig);
        ztaug(lay, ngs23 + ig) = colo2(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig)) + colo3(lay) * abso3b(ig);
        ztaur(lay, ngs23 + ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol25() {
    const auto& absa = kg25.absa;
    const auto& sfluxref = kg25.sfluxref;
    const auto& abso3a = kg25.abso3a;
    const auto& abso3b = kg25.abso3b;
    const auto& rayl = kg25.rayl;
    const auto& layreffr = kg25.layreffr;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    laysolfr = laytrop;
    for (lay = 1; lay <= laytrop; ++lay) {
      if (jp(lay) < layreffr && jp(lay + 1) >= layreffr) laysolfr = Kokkos::min(lay + 1, laytrop);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(25) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(25) + 1;
      for (ig = 1; ig <= ng25; ++ig) {
        tauray = colmol(lay) * rayl(ig);
        ztaug(lay, ngs24 + ig) = colh2o(lay) * (fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + colo3(lay) * abso3a(ig);
        if (lay == laysolfr) zsflxzen(ngs24 + ig) = sfluxref(ig);
        ztaur(lay, ngs24 + ig) = tauray;
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      for (ig = 1; ig <= ng25; ++ig) {
        tauray = colmol(lay) * rayl(ig);
        ztaug(lay, ngs24 + ig) = colo3(lay) * abso3b(ig);
        ztaur(lay, ngs24 + ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol26() {
    const auto& sfluxref = kg26.sfluxref;
    const auto& rayl = kg26.rayl;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    laysolfr = laytrop;
    for (lay = 1; lay <= laytrop; ++lay) {
      for (ig = 1; ig <= ng26; ++ig) {
        if (lay == laysolfr) zsflxzen(ngs25 + ig) = sfluxref(ig);
        ztaug(lay, ngs25 + ig) = 0.0;
        ztaur(lay, ngs25 + ig) = colmol(lay) * rayl(ig);
      }
    }
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      for (ig = 1; ig <= ng26; ++ig) {
        ztaug(lay, ngs25 + ig) = 0.0;
        ztaur(lay, ngs25 + ig) = colmol(lay) * rayl(ig);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol27() {
    const auto& absa = kg27.absa;
    const auto& absb = kg27.absb;
    const auto& sfluxref = kg27.sfluxref;
    const auto& rayl = kg27.rayl;
    const auto& layreffr = kg27.layreffr;
    const auto& scalekur = kg27.scalekur;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    for (lay = 1; lay <= laytrop; ++lay) {
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(27) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(27) + 1;
      for (ig = 1; ig <= ng27; ++ig) {
        tauray = colmol(lay) * rayl(ig);
        ztaug(lay, ngs26 + ig) = colo3(lay) * (fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig));
        ztaur(lay, ngs26 + ig) = tauray;
      }
    }
    laysolfr = nlayers;
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      if (jp(lay - 1) < layreffr && jp(lay) >= layreffr) laysolfr = lay;
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(27) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(27) + 1;
      for (ig = 1; ig <= ng27; ++ig) {
        tauray = colmol(lay) * rayl(ig);
        ztaug(lay, ngs26 + ig) = colo3(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig));
        if (lay == laysolfr) zsflxzen(ngs26 + ig) = scalekur * sfluxref(ig);
        ztaur(lay, ngs26 + ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol28() {
    const auto& absa = kg28.absa;
    const auto& absb = kg28.absb;
    const auto& sfluxref = kg28.sfluxref;
    const auto& rayl = kg28.rayl;
    const auto& layreffr = kg28.layreffr;
    const auto& strrat = kg28.strrat;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    for (lay = 1; lay <= laytrop; ++lay) {
      speccomb = colo3(lay) + strrat * colo2(lay);
      specparm = colo3(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 8.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(28) + js;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(28) + js;
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng28; ++ig) {
        ztaug(lay, ngs27 + ig) = speccomb * (fac000 * absa(ind0, ig) + fac100 * absa(ind0 + 1, ig) + fac010 * absa(ind0 + 9, ig) + fac110 * absa(ind0 + 10, ig) + fac001 * absa(ind1, ig) + fac101 * absa(ind1 + 1, ig) + fac011 * absa(ind1 + 9, ig) + fac111 * absa(ind1 + 10, ig));
        ztaur(lay, ngs27 + ig) = tauray;
      }
    }
    laysolfr = nlayers;
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      if (jp(lay - 1) < layreffr && jp(lay) >= layreffr) laysolfr = lay;
      speccomb = colo3(lay) + strrat * colo2(lay);
      specparm = colo3(lay) / speccomb;
      if (specparm >= oneminus) specparm = oneminus;
      specmult = 4.0 * (specparm);
      js = 1 + int(specmult);
      fs = Kokkos::fmod(specmult, 1.0);
      fac000 = (1.0 - fs) * fac00(lay);
      fac010 = (1.0 - fs) * fac10(lay);
      fac100 = fs * fac00(lay);
      fac110 = fs * fac10(lay);
      fac001 = (1.0 - fs) * fac01(lay);
      fac011 = (1.0 - fs) * fac11(lay);
      fac101 = fs * fac01(lay);
      fac111 = fs * fac11(lay);
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(28) + js;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(28) + js;
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng28; ++ig) {
        ztaug(lay, ngs27 + ig) = speccomb * (fac000 * absb(ind0, ig) + fac100 * absb(ind0 + 1, ig) + fac010 * absb(ind0 + 5, ig) + fac110 * absb(ind0 + 6, ig) + fac001 * absb(ind1, ig) + fac101 * absb(ind1 + 1, ig) + fac011 * absb(ind1 + 5, ig) + fac111 * absb(ind1 + 6, ig));
        if (lay == laysolfr) zsflxzen(ngs27 + ig) = sfluxref(ig, js) + fs * (sfluxref(ig, js + 1) - sfluxref(ig, js));
        ztaur(lay, ngs27 + ig) = tauray;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void taumol29() {
    const auto& absa = kg29.absa;
    const auto& absb = kg29.absb;
    const auto& forref = kg29.forref;
    const auto& selfref = kg29.selfref;
    const auto& sfluxref = kg29.sfluxref;
    const auto& absh2o = kg29.absh2o;
    const auto& absco2 = kg29.absco2;
    const auto& rayl = kg29.rayl;
    const auto& layreffr = kg29.layreffr;
    int ig;
    int ind0;
    int ind1;
    int inds;
    int indf;
    int js;
    int lay;
    int laysolfr;
    Real fac000;
    Real fac001;
    Real fac010;
    Real fac011;
    Real fac100;
    Real fac101;
    Real fac110;
    Real fac111;
    Real fs;
    Real speccomb;
    Real specmult;
    Real specparm;
    Real tauray;
    for (lay = 1; lay <= laytrop; ++lay) {
      ind0 = ((jp(lay) - 1) * 5 + (jt(lay) - 1)) * nspa(29) + 1;
      ind1 = (jp(lay) * 5 + (jt1(lay) - 1)) * nspa(29) + 1;
      inds = indself(lay);
      indf = indfor(lay);
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng29; ++ig) {
        ztaug(lay, ngs28 + ig) = colh2o(lay) * ((fac00(lay) * absa(ind0, ig) + fac10(lay) * absa(ind0 + 1, ig) + fac01(lay) * absa(ind1, ig) + fac11(lay) * absa(ind1 + 1, ig)) + selffac(lay) * (selfref(inds, ig) + selffrac(lay) * (selfref(inds + 1, ig) - selfref(inds, ig))) + forfac(lay) * (forref(indf, ig) + forfrac(lay) * (forref(indf + 1, ig) - forref(indf, ig)))) + colco2(lay) * absco2(ig);
        ztaur(lay, ngs28 + ig) = tauray;
      }
    }
    laysolfr = nlayers;
    for (lay = laytrop + 1; lay <= nlayers; ++lay) {
      if (jp(lay - 1) < layreffr && jp(lay) >= layreffr) laysolfr = lay;
      ind0 = ((jp(lay) - 13) * 5 + (jt(lay) - 1)) * nspb(29) + 1;
      ind1 = ((jp(lay) - 12) * 5 + (jt1(lay) - 1)) * nspb(29) + 1;
      tauray = colmol(lay) * rayl;
      for (ig = 1; ig <= ng29; ++ig) {
        ztaug(lay, ngs28 + ig) = colco2(lay) * (fac00(lay) * absb(ind0, ig) + fac10(lay) * absb(ind0 + 1, ig) + fac01(lay) * absb(ind1, ig) + fac11(lay) * absb(ind1 + 1, ig)) + colh2o(lay) * absh2o(ig);
        if (lay == laysolfr) zsflxzen(ngs28 + ig) = sfluxref(ig);
        ztaur(lay, ngs28 + ig) = tauray;
      }
    }
  }

  // vrtqdr_sw: vertical quadrature (adding method) for g-point kw
  KOKKOS_INLINE_FUNCTION void vrtqdr_sw(int klev, int kw, const FA<1>& pref, const FA<1>& prefd, const FA<1>& ptra, const FA<1>& ptrad,
                                           const FA<1>& pdbt, const FA<1>& prdnd, const FA<1>& prup, const FA<1>& prupd,
                                           const FA<1>& ptdbt, const FA<2>& pfd, const FA<2>& pfu) const {
    int ikp;
    int ikx;
    int jk;
    Real zreflect;
    zreflect = 1.0 / (1.0 - prefd(klev + 1) * prefd(klev));
    prup(klev) = pref(klev) + (ptrad(klev) * ((ptra(klev) - pdbt(klev)) * prefd(klev + 1) + pdbt(klev) * pref(klev + 1))) * zreflect;
    prupd(klev) = prefd(klev) + ptrad(klev) * ptrad(klev) * prefd(klev + 1) * zreflect;
    for (jk = 1; jk <= klev - 1; ++jk) {
      ikp = klev + 1 - jk;
      ikx = ikp - 1;
      zreflect = 1.0 / (1.0 - prupd(ikp) * prefd(ikx));
      prup(ikx) = pref(ikx) + (ptrad(ikx) * ((ptra(ikx) - pdbt(ikx)) * prupd(ikp) + pdbt(ikx) * prup(ikp))) * zreflect;
      prupd(ikx) = prefd(ikx) + ptrad(ikx) * ptrad(ikx) * prupd(ikp) * zreflect;
    }
    vq_ztdn(1) = 1.0;
    prdnd(1) = 0.0;
    vq_ztdn(2) = ptra(1);
    prdnd(2) = prefd(1);
    for (jk = 2; jk <= klev; ++jk) {
      ikp = jk + 1;
      zreflect = 1.0 / (1.0 - prefd(jk) * prdnd(jk));
      vq_ztdn(ikp) = ptdbt(jk) * ptra(jk) + (ptrad(jk) * ((vq_ztdn(jk) - ptdbt(jk)) + ptdbt(jk) * pref(jk) * prdnd(jk))) * zreflect;
      prdnd(ikp) = prefd(jk) + ptrad(jk) * ptrad(jk) * prdnd(jk) * zreflect;
    }
    for (jk = 1; jk <= klev + 1; ++jk) {
      zreflect = 1.0 / (1.0 - prdnd(jk) * prupd(jk));
      pfu(jk, kw) = (ptdbt(jk) * prup(jk) + (vq_ztdn(jk) - ptdbt(jk)) * prupd(jk)) * zreflect;
      pfd(jk, kw) = ptdbt(jk) + (vq_ztdn(jk) - ptdbt(jk) + ptdbt(jk) * prup(jk) * prdnd(jk)) * zreflect;
    }

  }

  // spcvmc_sw: two-stream fluxes of the McICA subcolumns (actual arguments of rrtmg_sw by name)
  KOKKOS_INLINE_FUNCTION void spcvmc_sw() {
    int klev;
    int ib1;
    int ib2;
    int ibm;
    int igt;
    int ikl;
    int ikp;
    int ikx;
    int iw;
    int jb;
    int jg;
    int jl;
    int jk;
    int itind;
    Real tblind;
    Real ze1;
    Real zclear;
    Real zcloud;
    Real zdbtmc;
    Real zdbtmo;
    Real zf;
    Real zgw;
    Real zreflect;
    Real zwf;
    Real tauorig;
    Real repclc;
    ib1 = istart;
    ib2 = iend;
    klev = nlayers;
    iw = 0;
    repclc = 1.0e-12;
    for (jk = 1; jk <= klev + 1; ++jk) {
      zbbcd(jk) = 0.0;
      zbbcu(jk) = 0.0;
      zbbfd(jk) = 0.0;
      zbbfu(jk) = 0.0;
      zbbcddir(jk) = 0.0;
      zbbfddir(jk) = 0.0;
      zuvcd(jk) = 0.0;
      zuvfd(jk) = 0.0;
      zuvcddir(jk) = 0.0;
      zuvfddir(jk) = 0.0;
      znicd(jk) = 0.0;
      znifd(jk) = 0.0;
      znicddir(jk) = 0.0;
      znifddir(jk) = 0.0;
    }
    taumol_sw();
    for (jb = ib1; jb <= ib2; ++jb) {
      ibm = jb - 15;
      igt = ngc(ibm);
      if (iout > 0 && ibm >= 2) iw = ngs(ibm - 1);
      for (jg = 1; jg <= igt; ++jg) {
        iw = iw + 1;
        zincflx(iw) = adjflux(jb) * zsflxzen(iw) * cossza;
        ztdbtc(1) = 1.0;
        ztdbtc_nodel(1) = 1.0;
        zdbtc(klev + 1) = 0.0;
        ztrac(klev + 1) = 0.0;
        ztradc(klev + 1) = 0.0;
        zrefc(klev + 1) = albdir(ibm);
        zrefdc(klev + 1) = albdif(ibm);
        zrupc(klev + 1) = albdir(ibm);
        zrupdc(klev + 1) = albdif(ibm);
        ztdbt(1) = 1.0;
        ztdbt_nodel(1) = 1.0;
        zdbt(klev + 1) = 0.0;
        ztra(klev + 1) = 0.0;
        ztrad(klev + 1) = 0.0;
        zref(klev + 1) = albdir(ibm);
        zrefd(klev + 1) = albdif(ibm);
        zrup(klev + 1) = albdir(ibm);
        zrupd(klev + 1) = albdif(ibm);
        for (jk = 1; jk <= klev; ++jk) {
          ikl = klev + 1 - jk;
          lrtchkclr(jk) = true;
          lrtchkcld(jk) = false;
          lrtchkcld(jk) = (zcldfmc(ikl, iw) > repclc);
          ztauc(jk) = ztaur(ikl, iw) + ztaug(ikl, iw) + ztaua(ikl, ibm);
          zomcc(jk) = ztaur(ikl, iw) * 1.0 + ztaua(ikl, ibm) * zomga(ikl, ibm);
          zgcc(jk) = zasya(ikl, ibm) * zomga(ikl, ibm) * ztaua(ikl, ibm) / zomcc(jk);
          zomcc(jk) = zomcc(jk) / ztauc(jk);
          zclear = 1.0 - zcldfmc(ikl, iw);
          zcloud = zcldfmc(ikl, iw);
          ze1 = ztauc(jk) / cossza;
          if (ze1 <= od_lo) {
            zdbtmc = 1.0 - ze1 + 0.5 * ze1 * ze1;
          } else {
            tblind = ze1 / (bpade + ze1);
            itind = int(tblint * tblind + 0.5);
            zdbtmc = exp_tbl(itind);
          }
          zdbtc_nodel(jk) = zdbtmc;
          ztdbtc_nodel(jk + 1) = zdbtc_nodel(jk) * ztdbtc_nodel(jk);
          tauorig = ztauc(jk) + ztaormc(ikl, iw);
          ze1 = tauorig / cossza;
          if (ze1 <= od_lo) {
            zdbtmo = 1.0 - ze1 + 0.5 * ze1 * ze1;
          } else {
            tblind = ze1 / (bpade + ze1);
            itind = int(tblint * tblind + 0.5);
            zdbtmo = exp_tbl(itind);
          }
          zdbt_nodel(jk) = zclear * zdbtmc + zcloud * zdbtmo;
          ztdbt_nodel(jk + 1) = zdbt_nodel(jk) * ztdbt_nodel(jk);
          zf = zgcc(jk) * zgcc(jk);
          zwf = zomcc(jk) * zf;
          ztauc(jk) = (1.0 - zwf) * ztauc(jk);
          zomcc(jk) = (zomcc(jk) - zwf) / (1.0 - zwf);
          zgcc(jk) = (zgcc(jk) - zf) / (1.0 - zf);
          if (icpr >= 1) {
            ztauo(jk) = ztauc(jk) + ztaucmc(ikl, iw);
            zomco(jk) = ztauc(jk) * zomcc(jk) + ztaucmc(ikl, iw) * zomgcmc(ikl, iw);
            zgco(jk) = (ztaucmc(ikl, iw) * zomgcmc(ikl, iw) * zasycmc(ikl, iw) + ztauc(jk) * zomcc(jk) * zgcc(jk)) / zomco(jk);
            zomco(jk) = zomco(jk) / ztauo(jk);
          } else if (icpr == 0) {
            ztauo(jk) = ztaur(ikl, iw) + ztaug(ikl, iw) + ztaua(ikl, ibm) + ztaucmc(ikl, iw);
            zomco(jk) = ztaua(ikl, ibm) * zomga(ikl, ibm) + ztaucmc(ikl, iw) * zomgcmc(ikl, iw) + ztaur(ikl, iw) * 1.0;
            zgco(jk) = (ztaucmc(ikl, iw) * zomgcmc(ikl, iw) * zasycmc(ikl, iw) + ztaua(ikl, ibm) * zomga(ikl, ibm) * zasya(ikl, ibm)) / zomco(jk);
            zomco(jk) = zomco(jk) / ztauo(jk);
            zf = zgco(jk) * zgco(jk);
            zwf = zomco(jk) * zf;
            ztauo(jk) = (1.0 - zwf) * ztauo(jk);
            zomco(jk) = (zomco(jk) - zwf) / (1.0 - zwf);
            zgco(jk) = (zgco(jk) - zf) / (1.0 - zf);
          }
        }
        reftra_sw(klev, lrtchkclr, zgcc, cossza, ztauc, zomcc, zrefc, zrefdc, ztrac, ztradc);
        reftra_sw(klev, lrtchkcld, zgco, cossza, ztauo, zomco, zrefo, zrefdo, ztrao, ztrado);
        for (jk = 1; jk <= klev; ++jk) {
          ikl = klev + 1 - jk;
          zclear = 1.0 - zcldfmc(ikl, iw);
          zcloud = zcldfmc(ikl, iw);
          zref(jk) = zclear * zrefc(jk) + zcloud * zrefo(jk);
          zrefd(jk) = zclear * zrefdc(jk) + zcloud * zrefdo(jk);
          ztra(jk) = zclear * ztrac(jk) + zcloud * ztrao(jk);
          ztrad(jk) = zclear * ztradc(jk) + zcloud * ztrado(jk);
          ze1 = ztauc(jk) / cossza;
          if (ze1 <= od_lo) {
            zdbtmc = 1.0 - ze1 + 0.5 * ze1 * ze1;
          } else {
            tblind = ze1 / (bpade + ze1);
            itind = int(tblint * tblind + 0.5);
            zdbtmc = exp_tbl(itind);
          }
          zdbtc(jk) = zdbtmc;
          ztdbtc(jk + 1) = zdbtc(jk) * ztdbtc(jk);
          ze1 = ztauo(jk) / cossza;
          if (ze1 <= od_lo) {
            zdbtmo = 1.0 - ze1 + 0.5 * ze1 * ze1;
          } else {
            tblind = ze1 / (bpade + ze1);
            itind = int(tblint * tblind + 0.5);
            zdbtmo = exp_tbl(itind);
          }
          zdbt(jk) = zclear * zdbtmc + zcloud * zdbtmo;
          ztdbt(jk + 1) = zdbt(jk) * ztdbt(jk);
        }
        vrtqdr_sw(klev, iw, zrefc, zrefdc, ztrac, ztradc, zdbtc, zrdndc, zrupc, zrupdc, ztdbtc, zcd, zcu);
        vrtqdr_sw(klev, iw, zref, zrefd, ztra, ztrad, zdbt, zrdnd, zrup, zrupd, ztdbt, zfd, zfu);
        for (jk = 1; jk <= klev + 1; ++jk) {
          ikl = klev + 2 - jk;
          zbbfu(ikl) = zbbfu(ikl) + zincflx(iw) * zfu(jk, iw);
          zbbfd(ikl) = zbbfd(ikl) + zincflx(iw) * zfd(jk, iw);
          zbbcu(ikl) = zbbcu(ikl) + zincflx(iw) * zcu(jk, iw);
          zbbcd(ikl) = zbbcd(ikl) + zincflx(iw) * zcd(jk, iw);
          zbbfddir(ikl) = zbbfddir(ikl) + zincflx(iw) * ztdbt_nodel(jk);
          zbbcddir(ikl) = zbbcddir(ikl) + zincflx(iw) * ztdbtc_nodel(jk);
          if (ibm >= 10 && ibm <= 13) {
            zuvcd(ikl) = zuvcd(ikl) + zincflx(iw) * zcd(jk, iw);
            zuvfd(ikl) = zuvfd(ikl) + zincflx(iw) * zfd(jk, iw);
            zuvcddir(ikl) = zuvcddir(ikl) + zincflx(iw) * ztdbtc_nodel(jk);
            zuvfddir(ikl) = zuvfddir(ikl) + zincflx(iw) * ztdbt_nodel(jk);
          } else if (ibm == 14 || ibm <= 9) {
            znicd(ikl) = znicd(ikl) + zincflx(iw) * zcd(jk, iw);
            znifd(ikl) = znifd(ikl) + zincflx(iw) * zfd(jk, iw);
            znicddir(ikl) = znicddir(ikl) + zincflx(iw) * ztdbtc_nodel(jk);
            znifddir(ikl) = znifddir(ikl) + zincflx(iw) * ztdbt_nodel(jk);
          }
        }
      }
    }

  }

  // inatm_sw (iplon = 1; dyofyr = 0, so adjflx = adjes)
  KOKKOS_INLINE_FUNCTION void inatm_sw() {
    constexpr Real amd = 28.9660;
    constexpr Real amw = 18.0160;
    int l, imol, ib, ig;
    Real amm, adjflx;
    nlayers = nlay;
    for (l = 1; l <= nlay + 1; ++l) {
      for (imol = 1; imol <= mxmol; ++imol) wkl(imol, l) = 0.0;
      for (ig = 1; ig <= ngptsw; ++ig) {
        cldfmc(ig, l) = 0.0;
        taucmc(ig, l) = 0.0;
        ssacmc(ig, l) = 1.0;
        asmcmc(ig, l) = 0.0;
        fsfcmc(ig, l) = 0.0;
        ciwpmc(ig, l) = 0.0;
        clwpmc(ig, l) = 0.0;
        cswpmc(ig, l) = 0.0;
      }
      reicmc(l) = 0.0;
      relqmc(l) = 0.0;
      resnmc(l) = 0.0;
      for (ib = 1; ib <= nbndsw; ++ib) {
        taua(l, ib) = 0.0;
        ssaa(l, ib) = 1.0;
        asma(l, ib) = 0.0;
      }
    }
    adjflx = adjes;
    // dyofyr = 0: no earth_sun
    for (ib = jpb1; ib <= jpb2; ++ib) {
      solvar(ib) = scon / rrsw_scon;
      adjflux(ib) = adjflx * solvar(ib);
    }
    tbound = tsfc;
    pz(0) = plev(iplon, 1);
    tz(0) = tlev(iplon, 1);
    for (l = 1; l <= nlayers; ++l) {
      pavel(l) = play(iplon, l);
      tavel(l) = tlay(iplon, l);
      pz(l) = plev(iplon, l + 1);
      tz(l) = tlev(iplon, l + 1);
      pdp(l) = pz(l - 1) - pz(l);
      wkl(1, l) = h2ovmr(iplon, l);
      wkl(2, l) = co2vmr(iplon, l);
      wkl(3, l) = o3vmr(iplon, l);
      wkl(4, l) = n2ovmr(iplon, l);
      wkl(6, l) = ch4vmr(iplon, l);
      wkl(7, l) = o2vmr(iplon, l);
      amm = (1.0 - wkl(1, l)) * amd + wkl(1, l) * amw;
      coldry(l) = (pz(l - 1) - pz(l)) * 1.0e3 * avogad / (1.0e2 * grav * amm * (1.0 + wkl(1, l)));
    }
    for (l = 1; l <= nlayers; ++l)
      for (imol = 1; imol <= nmol; ++imol) wkl(imol, l) = coldry(l) * wkl(imol, l);
    if (iaer >= 1) {
      for (l = 1; l <= nlayers; ++l)
        for (ib = 1; ib <= nbndsw; ++ib) {
          taua(l, ib) = tauaer(iplon, l, ib);
          ssaa(l, ib) = ssaaer(iplon, l, ib);
          asma(l, ib) = asmaer(iplon, l, ib);
        }
    }
    if (icld >= 1) {
      inflag = inflgsw;
      iceflag = iceflgsw;
      liqflag = liqflgsw;
      for (l = 1; l <= nlayers; ++l) {
        for (ig = 1; ig <= ngptsw; ++ig) {
          cldfmc(ig, l) = cldfmcl(ig, iplon, l);
          taucmc(ig, l) = taucmcl(ig, iplon, l);
          ssacmc(ig, l) = ssacmcl(ig, iplon, l);
          asmcmc(ig, l) = asmcmcl(ig, iplon, l);
          fsfcmc(ig, l) = fsfcmcl(ig, iplon, l);
          ciwpmc(ig, l) = ciwpmcl(ig, iplon, l);
          clwpmc(ig, l) = clwpmcl(ig, iplon, l);
          if (iceflag == 5) cswpmc(ig, l) = cswpmcl(ig, iplon, l);
        }
        reicmc(l) = reicmcl(iplon, l);
        relqmc(l) = relqmcl(iplon, l);
        if (iceflag == 5) resnmc(l) = resnmcl(iplon, l);
      }
    }
  }

  // mcica_subcol_sw + generate_stochastic_clouds_sw (irng = 0: KISS)
  KOKKOS_INLINE_FUNCTION void mcica_subcol_sw(int permuteseed) {
    if (icld == 0) return;
    const int nsubcol = ngptsw;
    for (int k = 1; k <= nlay; ++k) {
      reicmcl(1, k) = rei(1, k);
      relqmcl(1, k) = rel(1, k);
      resnmcl(1, k) = res(1, k);
      pmid(1, k) = play(1, k) * 1.e2;
    }
    const Real Zo = 2500.;
    const Real cldmin = 1.0e-20;
    const int overlap = icld;
    int ilev, isubcol;
    const int i = 1;
    for (ilev = 1; ilev <= nlay; ++ilev) {
      cldf(i, ilev) = cldfrac(i, ilev);
      if (cldf(i, ilev) < cldmin) cldf(i, ilev) = 0.;
    }
    if (pmid(i, 1) < pmid(i, 2)) SW_FAIL(SW_KISS_PMID);
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
      // overlap 5 (and anything else: the driver only allows 0..4)
      SW_FAIL(SW_OVERLAP5);
    }
    const int ngbm = ngb(1) - 1;
    for (ilev = 1; ilev <= nlay; ++ilev)
      for (isubcol = 1; isubcol <= nsubcol; ++isubcol) {
        const bool iscloudy = CDF(isubcol, i, ilev) >= 1. - cldf(i, ilev);
        if (iscloudy) {
          cldfmcl(isubcol, i, ilev) = 1.;
          clwpmcl(isubcol, i, ilev) = clwpth(i, ilev);
          ciwpmcl(isubcol, i, ilev) = ciwpth(i, ilev);
          cswpmcl(isubcol, i, ilev) = cswpth(i, ilev);
          const int n = ngb(isubcol) - ngbm;
          taucmcl(isubcol, i, ilev) = taucld(n, i, ilev);
          ssacmcl(isubcol, i, ilev) = ssacld(n, i, ilev);
          asmcmcl(isubcol, i, ilev) = asmcld(n, i, ilev);
          fsfcmcl(isubcol, i, ilev) = fsfcld(n, i, ilev);
        } else {
          cldfmcl(isubcol, i, ilev) = 0.;
          clwpmcl(isubcol, i, ilev) = 0.;
          ciwpmcl(isubcol, i, ilev) = 0.;
          cswpmcl(isubcol, i, ilev) = 0.;
          taucmcl(isubcol, i, ilev) = 0.;
          ssacmcl(isubcol, i, ilev) = 1.;
          asmcmcl(isubcol, i, ilev) = 0.;
          fsfcmcl(isubcol, i, ilev) = 0.;
        }
      }
  }

  // rrtmg_sw (ncol = 1; aer_opt = 0: iaer = 10; no WRF_CHEM)
  KOKKOS_INLINE_FUNCTION void rrtmg_sw() {
    const Real zepzen = 1.e-10;
    int i, ib, ig;
    iout = 0;
    istart = jpb1;
    iend = jpb2;
    icpr = 0;
    iaer = 10;
    inatm_sw();
    cldprmc_sw();
    if (ierr) return;
    icpr = 1;
    setcoef_sw();
    cossza = coszen;
    if (cossza <= zepzen) cossza = zepzen;
    for (ib = 1; ib <= 9; ++ib) {
      albdir(ib) = aldir;
      albdif(ib) = aldif;
    }
    albdir(nbndsw) = aldir;
    albdif(nbndsw) = aldif;
    for (ib = 10; ib <= 13; ++ib) {
      albdir(ib) = asdir;
      albdif(ib) = asdif;
    }
    if (icld == 0) {
      for (ig = 1; ig <= ngptsw; ++ig)
        for (i = 1; i <= nlay + 1; ++i) {
          zcldfmc(i, ig) = 0.0;
          ztaucmc(i, ig) = 0.0;
          ztaormc(i, ig) = 0.0;
          zasycmc(i, ig) = 0.0;
          zomgcmc(i, ig) = 1.0;
        }
    } else if (icld >= 1) {
      for (i = 1; i <= nlayers; ++i)
        for (ig = 1; ig <= ngptsw; ++ig) {
          zcldfmc(i, ig) = cldfmc(ig, i);
          ztaucmc(i, ig) = taucmc(ig, i);
          ztaormc(i, ig) = taormc(ig, i);
          zasycmc(i, ig) = asmcmc(ig, i);
          zomgcmc(i, ig) = ssacmc(ig, i);
        }
    }
    // iaer = 10
    for (i = 1; i <= nlayers; ++i)
      for (ib = 1; ib <= nbndsw; ++ib) {
        ztaua(i, ib) = taua(i, ib);
        ztauacln(i, ib) = 0.0;
        zasya(i, ib) = asma(i, ib);
        zomga(i, ib) = ssaa(i, ib);
      }
    for (i = 1; i <= nlayers + 1; ++i) {
      zbbcu(i) = 0.0;
      zbbcd(i) = 0.0;
      zbbfu(i) = 0.0;
      zbbfd(i) = 0.0;
      zbbcddir(i) = 0.0;
      zbbfddir(i) = 0.0;
      zuvcd(i) = 0.0;
      zuvfd(i) = 0.0;
      zuvcddir(i) = 0.0;
      zuvfddir(i) = 0.0;
      znicd(i) = 0.0;
      znifd(i) = 0.0;
      znicddir(i) = 0.0;
      znifddir(i) = 0.0;
    }
    spcvmc_sw();
    for (i = 1; i <= nlayers + 1; ++i) {
      swuflxc(iplon, i) = zbbcu(i);
      swdflxc(iplon, i) = zbbcd(i);
      swuflx(iplon, i) = zbbfu(i);
      swdflx(iplon, i) = zbbfd(i);
      uvdflx(i) = zuvfd(i);
      nidflx(i) = znifd(i);
      dirdflux(i) = zbbfddir(i);
      difdflux(i) = swdflx(iplon, i) - dirdflux(i);
      swdkdir(iplon, i) = dirdflux(i);
      swdkdif(iplon, i) = difdflux(i);
      swdkdirc(iplon, i) = zbbcddir(i);
      dirdnuv(i) = zuvfddir(i);
      difdnuv(i) = zuvfd(i) - dirdnuv(i);
      sibvisdir(iplon, i) = dirdnuv(i);
      sibvisdif(iplon, i) = difdnuv(i);
      dirdnir(i) = znifddir(i);
      difdnir(i) = znifd(i) - dirdnir(i);
      sibnirdir(iplon, i) = dirdnir(i);
      sibnirdif(iplon, i) = difdnir(i);
    }
    for (i = 1; i <= nlayers + 1; ++i) {
      swnflxc(i) = swdflxc(iplon, i) - swuflxc(iplon, i);
      swnflx(i) = swdflx(iplon, i) - swuflx(iplon, i);
    }
    for (i = 1; i <= nlayers; ++i) {
      const Real zdpgcp = heatfac / pdp(i);
      swhrc(iplon, i) = (swnflxc(i + 1) - swnflxc(i)) * zdpgcp;
      swhr(iplon, i) = (swnflx(i + 1) - swnflx(i)) * zdpgcp;
    }
    swhrc(iplon, nlayers) = 0.0;
    swhr(iplon, nlayers) = 0.0;
    for (i = 1; i <= nlayers + 1; ++i) {
      swuflxcln(iplon, i) = 0.0;
      swdflxcln(iplon, i) = 0.0;
    }
  }

  // RRTMG_SWRAD's i/j loop body for column (i,j)
  KOKKOS_INLINE_FUNCTION void column(const SwIn& a, int i, int j) {
    const Real ch4 = 1774.e-9, n2o = 319.e-9, o2 = 0.209488;
    const Real amdw = 1.607793, amdo = 0.603461;
    const Real co2 = a.co2;
    int k, nb;
    Real dz = 0, dzsum;

    for (k = kts; k <= kte; ++k) RHO1D(k) = a.rho3d(i, j, k);
    bool dorrsw = true;
    a.coszr(i, j, 1) = a.xcoszen(i, j, 1);
    const Real coszrs = a.xcoszen(i, j, 1);
    if (coszrs <= 0.0) dorrsw = false;
    if (!dorrsw) {
      for (k = kts; k <= kte; ++k) a.rthratensw(i, j, k) = 0.0;
      if (a.has_swupt) {
        a.swupt(i, j, 1) = 0.;
        a.swuptc(i, j, 1) = 0.;
        a.swdnt(i, j, 1) = 0.;
        a.swdntc(i, j, 1) = 0.;
        a.swupb(i, j, 1) = 0.;
        a.swupbc(i, j, 1) = 0.;
        a.swdnb(i, j, 1) = 0.;
        a.swdnbc(i, j, 1) = 0.;
      }
      a.swddir(i, j, 1) = 0.;
      a.swddni(i, j, 1) = 0.;
      a.swddif(i, j, 1) = 0.;
      a.swdownc(i, j, 1) = 0.0;
      a.swddnic(i, j, 1) = 0.0;
      a.swddirc(i, j, 1) = 0.0;
      a.swcf(i, j, 1) = 0.;
      // (gsw and swtenc keep their values, as in the Fortran)
      return;
    }

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
      P1D(k) = a.p3d(i, j, k) / 100.;
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
      // (F_ICE_PHY is not passed: no qs -> qi split)
    }
    for (k = kts; k <= kte; ++k) QV1D(k) = Kokkos::max(QV1D(k), 1.e-12);

    // nlay = kte - kts + 2; icld = cldovrlp (set by the driver)
    inflgsw = 2;
    iceflgsw = 3;
    liqflgsw = 1;
    if (a.icloud != 0) {
      // has_reqc = has_reqi = has_reqs = 0
      for (k = kts; k <= kte; ++k) recloud1d(ncol, k) = 5.0;
      for (k = kts; k <= kte; ++k) reice1d(ncol, k) = 10.;
      for (k = kts; k <= kte; ++k) resnow1d(ncol, k) = 10.0;
    }
    coszen = coszrs;
    scon = a.solcon;
    dyofyr = 0;
    adjes = 1.0;
    plev(ncol, 1) = Pw1D(1);
    tlev(ncol, 1) = Tw1D(1);
    tsfc = a.tsk(i, j, 1);
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
    }
    dzsum = 0.0;
    for (k = kts; k <= kte; ++k) {
      dz = DZ1D(k);
      hgt(ncol, k) = dzsum + 0.5 * dz;
      dzsum = dzsum + dz;
    }
    // extra layer from the model top to the top of the atmosphere
    play(ncol, kte + 1) = 0.5 * plev(ncol, kte + 1);
    tlay(ncol, kte + 1) = tlev(ncol, kte + 1) + 0.0;
    plev(ncol, kte + 2) = 1.0e-5;
    tlev(ncol, kte + 2) = tlev(ncol, kte + 1) + 0.0;
    h2ovmr(ncol, kte + 1) = h2ovmr(ncol, kte);
    co2vmr(ncol, kte + 1) = co2vmr(ncol, kte);
    o2vmr(ncol, kte + 1) = o2vmr(ncol, kte);
    ch4vmr(ncol, kte + 1) = ch4vmr(ncol, kte);
    n2ovmr(ncol, kte + 1) = n2ovmr(ncol, kte);
    hgt(ncol, kte + 1) = dzsum + 0.5 * dz;
    if (!inirad(o3mmr, plev, kts, kte, O3SUM, PPSUM, O3WIN, PPWIN)) SW_FAIL(SW_O3FRAC);
    for (k = kts; k <= kte + 1; ++k) {
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
    // sf_surface_physics /= 8
    asdir = a.albedo(i, j, 1);
    asdif = a.albedo(i, j, 1);
    aldir = a.albedo(i, j, 1);
    aldif = a.albedo(i, j, 1);

    // inflgsw = 2 (> 0), iceflgsw = 3, liqflgsw = 1, no progn
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
      relcalc(pver, tlay, landfrac, landm, icefrac, reliq, snowh);
      reicalc(pver, tlay, reice, retab);
      // iceflgsw = 3
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
        cswpth(ncol, k) = 0.0;
        res(ncol, k) = 10.0;
      }
      for (k = kts; k <= kte; ++k)
        for (nb = 1; nb <= nbndsw; ++nb) {
          taucld(nb, ncol, k) = 0.0;
          ssacld(nb, ncol, k) = 1.0;
          asmcld(nb, ncol, k) = 0.0;
          fsfcld(nb, ncol, k) = 0.0;
        }
    }
    clwpth(ncol, kte + 1) = 0.;
    ciwpth(ncol, kte + 1) = 0.;
    cswpth(ncol, kte + 1) = 0.;
    rel(ncol, kte + 1) = 10.;
    rei(ncol, kte + 1) = 10.;
    res(ncol, kte + 1) = 10.;
    cldfrac(ncol, kte + 1) = 0.;
    for (nb = 1; nb <= nbndsw; ++nb) {
      taucld(nb, ncol, kte + 1) = 0.;
      ssacld(nb, ncol, kte + 1) = 1.;
      asmcld(nb, ncol, kte + 1) = 0.;
      fsfcld(nb, ncol, kte + 1) = 0.;
    }
    mcica_subcol_sw(1);
    if (ierr) return;
    for (nb = 1; nb <= nbndsw; ++nb)
      for (k = kts; k <= kte + 1; ++k) {
        tauaer(ncol, k, nb) = 0.;
        ssaaer(ncol, k, nb) = 1.;
        asmaer(ncol, k, nb) = 0.;
      }
    // (tauaer3d_sw not associated; aer_opt = 0: ecaer = 0, unused with iaer = 10)

    rrtmg_sw();
    if (ierr) return;

    a.gsw(i, j, 1) = swdflx(1, 1) - swuflx(1, 1);
    a.swcf(i, j, 1) = (swdflx(1, kte + 2) - swuflx(1, kte + 2)) - (swdflxc(1, kte + 2) - swuflxc(1, kte + 2));
    if (a.has_swupt) {
      a.swupt(i, j, 1) = swuflx(1, kte + 2);
      a.swuptc(i, j, 1) = swuflxc(1, kte + 2);
      a.swdnt(i, j, 1) = swdflx(1, kte + 2);
      a.swdntc(i, j, 1) = swdflxc(1, kte + 2);
      a.swupb(i, j, 1) = swuflx(1, 1);
      a.swupbc(i, j, 1) = swuflxc(1, 1);
      a.swdnb(i, j, 1) = swdflx(1, 1);
      a.swdnbc(i, j, 1) = swdflxc(1, 1);
    }
    a.swddir(i, j, 1) = swdkdir(1, 1);
    a.swddni(i, j, 1) = a.swddir(i, j, 1) / coszrs;
    a.swddif(i, j, 1) = swdkdif(1, 1);
    a.swdownc(i, j, 1) = swdflxc(1, 1);
    a.swddirc(i, j, 1) = swdkdirc(1, 1);
    a.swddnic(i, j, 1) = a.swddirc(i, j, 1) / coszrs;
    for (k = kts; k <= kte; ++k) {
      TTEN1D(k) = swhr(ncol, k) / 86400.;
      a.rthratensw(i, j, k) = TTEN1D(k);
      a.swtenc(i, j, k) = swhrc(ncol, k) / 86400.0;
    }
  }
};

} // namespace

Real rrtmg_swrad(const SwTableData& tab, const SwRadArgs& a) {
  if (a.has_reqc != 0 || a.has_reqi != 0 || a.has_reqs != 0)
    throw std::runtime_error("cm1k rrtmg_swrad: has_reqc/has_reqi/has_reqs /= 0 is not ported");
  if (a.is_cammgmp_used) throw std::runtime_error("cm1k rrtmg_swrad: is_CAMMGMP_used is not ported");
  if (a.calc_clean_atm_diag > 0) throw std::runtime_error("cm1k rrtmg_swrad: calc_clean_atm_diag > 0 is not ported");
  if (a.mp_physics == 5 || a.mp_physics == 95)
    throw std::runtime_error("cm1k rrtmg_swrad: FER_MP_HIRES microphysics is not ported");
  if (a.cldovrlp < 0 || a.cldovrlp > 4) throw std::runtime_error("cm1k rrtmg_swrad: cldovrlp must be 0..4");
  if (a.aer_opt != 0) throw std::runtime_error("cm1k rrtmg_swrad: aer_opt /= 0 is not ported");
  if (a.sf_surface_physics == 8) throw std::runtime_error("cm1k rrtmg_swrad: sf_surface_physics = 8 is not ported");
  if (a.kts != 1) throw std::runtime_error("cm1k rrtmg_swrad: kts must be 1");

  // co2 = (280. + 90.*exp(0.02*(yr-2000)))*1.e-6
  const Real co2 = (280. + 90. * std::exp(0.02 * Real(a.yr - 2000))) * 1.e-6;

  SwIn in;
  in.rthratensw = a.rthratensw; in.swtenc = a.swtenc;
  in.swupt = a.swupt; in.swuptc = a.swuptc; in.swdnt = a.swdnt; in.swdntc = a.swdntc;
  in.swupb = a.swupb; in.swupbc = a.swupbc; in.swdnb = a.swdnb; in.swdnbc = a.swdnbc;
  in.swcf = a.swcf; in.gsw = a.gsw; in.coszr = a.coszr;
  in.swddir = a.swddir; in.swddni = a.swddni; in.swddif = a.swddif;
  in.swdownc = a.swdownc; in.swddnic = a.swddnic; in.swddirc = a.swddirc;
  in.xcoszen = a.xcoszen; in.albedo = a.albedo; in.tsk = a.tsk; in.xland = a.xland; in.xice = a.xice;
  in.snow = a.snow;
  in.t3d = a.t3d; in.t8w = a.t8w; in.p3d = a.p3d; in.p8w = a.p8w; in.rho3d = a.rho3d; in.dz8w = a.dz8w;
  in.cldfra3d = a.cldfra3d;
  in.qv3d = a.qv3d; in.qc3d = a.qc3d; in.qr3d = a.qr3d; in.qi3d = a.qi3d; in.qs3d = a.qs3d; in.qg3d = a.qg3d;
  in.o33d = a.o33d;
  in.g = a.g; in.co2 = co2; in.solcon = a.solcon;
  in.icloud = a.icloud; in.o3input = a.o3input;
  in.warm_rain = a.warm_rain;
  in.f_qc = a.f_qc; in.f_qr = a.f_qr; in.f_qi = a.f_qi; in.f_qs = a.f_qs; in.f_qg = a.f_qg;
  in.has_o33d = a.o33d.is_allocated();
  in.has_swupt = a.swupt.is_allocated();

  SwCol proto;
  static_cast<SwTables&>(proto) = tab.dev;
  proto.kte = a.kte;
  proto.nlay = (a.kte - a.kts + 1) + 1;
  proto.nlayers = proto.nlay;
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
  Kokkos::View<Real*, Device> rpool("rrtmg_sw_scratch_r", size_t(nr) * size_t(nc));
  Kokkos::View<int*, Device> ipool("rrtmg_sw_scratch_i", size_t(std::max<long>(ni, 1)) * size_t(nc));
  Kokkos::View<int*, Device> errs("rrtmg_sw_err", size_t(ncols));
  const int its = a.its, jts = a.jts;
  for (int c0 = 0; c0 < ncols; c0 += nc) {
    const int n = std::min(nc, ncols - c0);
    Real* rp = rpool.data();
    int* ip = ipool.data();
    Kokkos::parallel_for(
        "rrtmg_swrad", Kokkos::RangePolicy<Kokkos::IndexType<int>>(0, n), KOKKOS_LAMBDA(int c) {
          SwCol col = proto;
          Binder b(rp, ip, nc, c);
          col.layout(b);
          const int ic = c0 + c;
          const int i = its + ic % ni_, j = jts + ic / ni_;
          col.column(in, i, j);
          errs(ic) = col.ierr;
        });
  }
  auto he = Kokkos::create_mirror_view_and_copy(HostSpace(), errs);
  for (int ic = 0; ic < ncols; ++ic)
    if (he(ic) != 0)
      throw std::runtime_error(std::string("cm1k rrtmg_swrad: ") + sw_err_message(he(ic)) + " (column i=" +
                               std::to_string(a.its + ic % ni_) + ", j=" + std::to_string(a.jts + ic / ni_) + ")");
  return co2;
}

} // namespace cm1k::rrtmg
