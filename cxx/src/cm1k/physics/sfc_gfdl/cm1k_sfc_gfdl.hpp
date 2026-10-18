#ifndef CM1K_SFC_GFDL_HPP
#define CM1K_SFC_GFDL_HPP

// Port of the GFDL surface layer (sfcmodel=4, "GFDL surface layer" of HWRF):
//
//   src/module_sf_gfdl.F      SF_GFDL, MFLUX2 (Monin-Obukhov fluxes with the
//                             Hicks universal functions and Wegstein
//                             iterations, the GFDL slab T* prediction over land)
//   src/module_sf_exchcoef.F  znot_wind10m with icoef_sf = 6 (znot_m_v6,
//                             znot_t_v6: roughness lengths over water from
//                             the 10-m wind)
//   src/module_gfs_funcphys.F fpvs (the saturation vapor pressure table
//                             lookup) and the table it reads (gpvs/fpvsx)
//
// as src/turb.F (sfc_and_turb, lines 841-877) calls it. Bit-for-bit with the
// double-precision Fortran (tests/fortran_bridge/gfdl_tests.cpp, against the
// unmodified module through cm1f_bridge_gfdl.F90).
//
// What CM1's call fixes (turb.F 484-500), and so what is ported:
//   ntsflg = 1 (GFDL slab T* prediction over land, and the tsk patch at the
//   i = 1 and j = 1 rows), sfenth = 0, pert_cd = .false., icoef_sf = 6,
//   iwavecpl = 0, lcurr_sf = .false., isltyp = 14 (drysmc = 0, maxsmc = 1),
//   isfflx = 1, num_soil_layers = 1. Options also allows ntsflg = 0,
//   isfflx = 0, other sfenth and isltyp (1..30) for the component tests;
//   Options::check throws for icoef_sf /= 6 (the other znot_* fits are not
//   ported). iwavecpl = 1, lcurr_sf and the Cd perturbation (ran1) are not
//   reachable from CM1 and not ported. The dead PROGTM-era locals of SF_GFDL
//   (PRSLKI, THGB, THX) and of MFLUX2 (restar, rat, the 2-m-only logs) are
//   not computed; rat = 0 in MFLUX2, so zot = zo there.
//
// Land points: SF_GFDL reads the soil moisture as smois(i,1,j) of a dummy
// declared (ims:ime,jms:jme,1:num_soil_layers) (WRF's (i,layer,j) order),
// but CM1 passes its 2D smois(ib:ie,jb:je) with num_soil_layers = 1, so
// for j > 1 the Fortran reads past the end of smois (undefined). The kernel
// takes smois(i,j) = the value read at smois(i,1,j) (the component tests
// pass a (ib:ie,jb:je,1:nj) array with those values); the model only
// allows sfcmodel=4 over water (sfc::Options::check), where smois is not
// read.
//
// Storage: 2D arrays are FArray3 (ib:ie,jb:je,1:1) indexed a(i,j,1), as in
// physics/sfc. Interior points 1..ni x 1..nj; the tsk patch reads tsk at
// i = 2 and j = 2 (and the halo when ni or nj is 1).

#include "physics/sfclay/cm1k_wrf_common.hpp"

namespace cm1k::sfc_gfdl {

using wrfphys::WrfConst;

struct Options {
  int ntsflg = 1;       // turb.F: GFDL slab T* prediction over land
  Real sfenth = 0.0;    // turb.F: sea-spray enthalpy factor
  int icoef_sf = 6;     // turb.F: exchange-coefficient fits (only 6 ported)
  int isfflx = 1;       // turb.F: surface fluxes on
  int isltyp = 14;      // turb.F: soil type (water)
  void check() const;   // throws std::runtime_error for anything not ported
};

// MFLUX2's vapor-pressure tables (tab, table: 223 entries, DATA in the
// Fortran) and module_gfs_funcphys's tbpvs (gpvs: fpvsx on 180..330 K, 7501
// entries, built on the host with the scalar pow/exp the Fortran uses).
struct Tables {
  static constexpr int ntab = 223, nxpvs = 7501;
  Kokkos::View<Real*, Device> tab, table, tbpvs;
  Real c1xpvs = 0, c2xpvs = 0;
  static Tables make();
};

// Column inputs at k = 1 (turb.F: U3D = dum1, V3D = dum2, T3D = dum4,
// QV3D = divx, P3D = prs), and the 2D inputs.
struct Inputs {
  FArray3 u3d, v3d, t3d, qv3d, p3d;            // 3D, k = 1 read
  FArray3 psfc, xland, gsw, glw, zkmax, smois;  // 2D (zkmax = zh(:,:,1) in CM1; smois: land only)
};

// The 2D arrays SF_GFDL writes (cm1.F names; the dummy argument where it
// differs in brackets). inout: tsk, znt (z0t), mznt (z0), u10, v10 (the
// previous 10-m wind sets wind10, MFLUX2's input for z0 over water).
// cd_out [Cd_out] (read as cdm but never used), ust (read, unused).
struct Fields {
  FArray3 tsk, znt, mznt, ust, u10, v10;
  FArray3 chs, chs2, cqs2, cpm, psim, psih, hfx, qfx, lh, taux, tauy, flhc, flqc, qgh, qsfc, gz1oz0,
          wspd /*swspd*/, br, cd_out, ch_out, mol, zol, z0t, fm, fh;
  FArray3 tstrc;   // scratch: the new skin temperature before the patch
};

// SF_GFDL for all rows j = 1..nj (one parallel_for over the points; the
// tsk patch at the domain edges, ids = jds = 1, runs afterwards from the
// old tsk as the Fortran's i/j loop order makes it).
void sf_gfdl(const Options& o, const WrfConst& c, const Tables& t, Real dt, int ni, int nj, const Inputs& in,
             const Fields& f);

} // namespace cm1k::sfc_gfdl

#endif
