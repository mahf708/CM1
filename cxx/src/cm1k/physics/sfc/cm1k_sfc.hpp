#ifndef CM1K_SFC_HPP
#define CM1K_SFC_HPP

// Port of CM1's original surface layer (sfcmodel=1, "CM1 simple scheme
// (neutral stability)") as used with the LES TKE scheme and bbc=3
// (semi-slip lower boundary):
//
//   src/sfcphys.F       getcecd (Cd, z0, u*, Ch, Cq), sfcflux (surface
//                       fluxes of theta and qv), sfcdiags (hfx, qfx, qsfc,
//                       wspd, 2-m diagnostics)
//   src/turb.F          the surface block of sfc_and_turb (psfc, ground-
//                       relative winds, u1/v1/s1 at the lowest level) and
//                       sfcstress (bbc=3: t13/t23 at the surface)
//   src/init_surface.F  init_surface + getlanduse (initsfc=1)
//
// The pieces the surface feeds into the turbulence scheme (tkebc's
// kmh(k=1) = karman*z0*u*, the theta/qv flux at the surface in turbs, the
// surface dissipative heating in solve1) are in physics/turb.
//
// Supported (Options::check throws for anything else):
//   * sfcmodel = 0 (no surface layer: bbc=1 free slip), or sfcmodel = 1 with
//     bbc = 3 (param.F sets bbc=3 for sfcmodel>=1), isfcflx = 0 or 1,
//     oceanmodel = 1, tbc = 1
//   * sfcmodel = 2, 3 (the WRF/MM5 surface layers, physics/sfclay, with the
//     SLAB land model, physics/slab) with a PBL scheme (cm1setup = 2,
//     ipbl = 1 or 2): this file then provides the shared surface arrays,
//     init_surface and the bbc=3 prelude of sfc_and_turb (sfc_prelude);
//     param.F sets isfcflx = 1 and requires pertflx = 0; lu0 = 1..33,
//     tmn0 >= 100 K
//   * sfcmodel = 6 (the MYNN surface layer, physics/sfc_mynn, with SLAB over
//     land) with the MYNN PBL (cm1setup = 2, ipbl = 4, 5): lu0 = 1..33,
//     tmn0 >= 100 K (checked in pbl_mynn::Coupling)
//   * sfcmodel = 7 (the MYJ surface layer + SLAB, physics/myj) with the
//     MYJ PBL (cm1setup = 2, ipbl = 6); lu0 = 1..33, tmn0 >= 100 K
//   * cecd = 1, 2, 3; set_znt = 0/1, set_ust = 0/1, set_flx = 0/1;
//     pertflx = 0/1; cnstce, cnstcd, cnst_znt, cnst_ust, cnst_shflx, cnst_lhflx
//   * initsfc = 1 (tsk0, lu0 -> LANDUSE.TBL), season = 1, 2
//   * sfcmodel = 1: cm1setup = 1 with ipbl = 0 (LES), or cm1setup = 2 with
//     ipbl = 2 (the simple PBL scheme, physics/pbl), which adds gethpbl2
//     (bulk-Ri boundary-layer depth hpbl, before getcecd)
//   * testcase = 0 (so use_avg_sfc = .false.; without ipbl=2, hpbl stays at
//     its initial 100 m), imove = 0, no terrain, axisymm = 0
//   * imoist = 0, or imoist = 1 with qv = qa(:,:,:,1)
//
// Not ported (diagnostics that never feed back): the qv budget in sfcflux
// (qbudget(8)) and solve1 (qbudget(9)), stau, and the timers.
//
// Storage: every surface array is 2D in CM1, (ib:ie,jb:je); here they are
// FArray3 (ib:ie,jb:je,1:1), the layout src/cm1dump.F writes them in, so the
// model's field registry and dump I/O handle them like the 3D fields.
// Kernels index them as a(i,j,1).

#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

#include <string>
#include <utility>
#include <vector>

namespace cm1k::sfc {

// Options read from the `input` module (namelist param2/param12, param.F
// adjustments). Defaults are those of run/namelist.input.
struct Options {
  int sfcmodel = 0, isfcflx = 0, oceanmodel = 1, bbc = 1, tbc = 1;
  int cecd = 3, pertflx = 0, set_flx = 0, set_znt = 0, set_ust = 0;
  int initsfc = 1, lu0 = 16, season = 1;
  int testcase = 0, imove = 0, axisymm = 0;
  bool terrain_flag = false;
  int imoist = 0, ptype = 0;
  int cm1setup = 1, ipbl = 0;
  Real tsk0 = 299.28, tmn0 = 297.28, xland0 = 2.0, cnstce = 0.0011, cnstcd = 0.0011;
  Real cnst_shflx = 0.24, cnst_lhflx = 5.2e-5, cnst_znt = 0.16, cnst_ust = 0.25;
  bool dosfcflx = false;   // init_surface.F
  int output_sfcflx = 0;    // turb.F: thflux/qvflux from hfx/qfx also without dosfcflx

  // sfcmodel=1 (with bbc=3): the surface layer runs in sfc_and_turb
  bool active() const { return sfcmodel == 1; }
  // the WRF/MM5 surface layers (sfcmodel=2,3: physics/sfclay + slab)
  bool wrf() const { return sfcmodel == 2 || sfcmodel == 3; }
  // the MYNN surface layer (sfcmodel=6: physics/sfc_mynn, run by
  // pbl_mynn::surface_layer; the arrays and init_surface are these)
  bool mynn() const { return sfcmodel == 6; }
  // the MYJ surface layer (sfcmodel=7: physics/myj, run by
  // myj::surface_layer; the arrays and init_surface are these)
  bool myj() const { return sfcmodel == 7; }
  // any ported surface layer (the surface arrays exist and are dumped)
  bool has_layer() const { return active() || wrf() || mynn() || myj(); }
  static Options from_namelist(const Namelist& nl);
  void check() const;   // throws std::runtime_error for anything not ported
};

// The surface arrays (cm1.F names). The first group is what src/cm1dump.F
// writes (see dumped()); tsk, znt, xland, mavail are inputs that persist
// across steps, the rest are recomputed by every sfc_and_turb.
struct SfcState {
  FArray3 tsk, znt, xland, mavail, ust, u1, v1, s1, cd, ch, cq, thflux, qvflux, psfc,
          u10, v10, s10, hfx, qfx, qsfc, wspd;
  // not in the dumps: zntmp (znt with halos, for tkebc and the surface
  // dissipation), t1, hpbl, dsxy, and sfcdiags' other outputs
  FArray3 zntmp, t1, hpbl, dsxy, br, cda, gz1oz0, th2, t2, q2;
  void allocate(const Bounds& b);
  // (name, array) of every dumped array, in src/cm1dump_state.inc order
  std::vector<std::pair<std::string, FArray3>> dumped() const;
};

// The values cm1.F gives the surface arrays when it allocates them (zero,
// then znt = zntmp = ust = 1e-6, tsk = 300, psfc = 1e5, qsfc = 1e-5,
// hpbl = 100). The Model calls this once, at construction.
void set_defaults(const SfcState& s);

// init_surface.F for initsfc=1 (+ getlanduse, with the USGS table of
// run/LANDUSE.TBL built in): tsk, xland, znt, mavail, dsxy. Needs the grid
// (ruh, rvh). For sfcmodel=2,3 also see sfclay::init_wrf_surface (tmn,
// emiss, thc, tslb).
void init_surface(const Options& o, const Grid& g, const SfcState& s);

// The part of init_surface that a reference dump does not carry: dsxy
// (grid spacing for the Mahrt-Sun correction in sfcdiags).
void set_dsxy(const Options& o, const Grid& g, const SfcState& s);

// The bbc=3/sfcmodel=1 block of turb.F sfc_and_turb, on the new state:
// sfc_prelude, then getcecd, sfcflux (isfcflx=1), sfcdiags. Interior
// points (1..ni,1..nj) only; the halos of ust, u1, v1, s1, znt, wspd and
// zntmp are set by turb::sfc_and_turb (CM1's "bc/comms" block).
//   cgs1..3: param.F extrapolation coefficients to the surface
void sfc_layer(const Options& o, const Constants& c, const Grid& g, const BaseState& base,
               Real cgs1, Real cgs2, Real cgs3, const FArray3& ua, const FArray3& va,
               const FArray3& tha, const FArray4& qa, const FArray3& prs, const FArray3& rf,
               const SfcState& s);

// The part of turb.F sfc_and_turb's surface block every surface layer
// shares (lines 290-332 and 532-597): psfc (cgs extrapolation), u1/v1/s1/t1
// at k=1 (ground-relative winds, pertflx) and, for ipbl=2, gethpbl2.
// Interior points only.
void sfc_prelude(const Options& o, const Constants& c, const Grid& g, const BaseState& base,
                 Real cgs1, Real cgs2, Real cgs3, const FArray3& ua, const FArray3& va,
                 const FArray3& tha, const FArray4& qa, const FArray3& prs, const SfcState& s);

// sfcphys.F gethpbl2 (ipbl=2), as turb.F calls it: qsfc (rslf, or 0 when
// imoist=0) and hpbl from the bulk Richardson number (bric = 0.125) with the
// previous step's ust, thflux, qvflux and tsk; hpbl = -1 on the halos (and
// in columns without a crossing). ua/va minus u0/v0 with pertflx=1 (ugr/vgr).
void gethpbl2(const Options& o, const Constants& c, const Bounds& b, const FArray3& zh,
              const FArray3& th0, const FArray3& tha, const FArray4& qa, const FArray3& ua,
              const FArray3& va, const FArray3& u0, const FArray3& v0, const SfcState& s);

// sfcphys.F getcecd (use_avg_sfc = .false.). za: height of the lowest
// model level above the surface, here zh(i,j,1).
void getcecd(const Options& o, const Bounds& b, const FArray3& zh, const SfcState& s);

// sfcphys.F sfcflux (testcase = 0). qv: qa(:,:,:,nqv), or unused when imoist=0.
void sfcflux(const Options& o, const Constants& c, const Bounds& b, const FArray3& th0,
             const FArray3& tha, const FArray4& qa, const SfcState& s);

// sfcphys.F sfcdiags. qa(:,:,1,nqv) is CM1's divx (0 when imoist=0).
void sfcdiags(const Options& o, const Constants& c, const Bounds& b, const FArray3& zh,
              const FArray3& th0, const FArray3& tha, const FArray4& qa, const FArray3& rf,
              const SfcState& s);

// turb.F sfcstress, bbc=3 (Cartesian, use_avg_sfc = .false., testcase /= 7):
// t13, t23 at k=1 on 1..ni+1 x 1..nj+1 (needs the halos of ust, u1, v1, s1).
void sfcstress(const Bounds& b, const FArray3& rf, const SfcState& s,
               const FArray3& t13, const FArray3& t23);

// USGS land-use table (run/LANDUSE.TBL): soil moisture availability (slmo)
// and roughness length in cm (sfz0) for category 1..33 and season 1..2.
// Also the emissivity (sfem) and thermal inertia (therin, in
// cal/cm/K/s^0.5 times 100) that getlanduse reads for SLAB.
Real landuse_slmo(int cat, int season);
Real landuse_sfz0(int cat, int season);
Real landuse_sfem(int cat, int season);
Real landuse_therin(int cat, int season);

} // namespace cm1k::sfc

#endif
