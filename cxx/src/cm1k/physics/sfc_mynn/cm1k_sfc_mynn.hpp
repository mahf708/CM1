#ifndef CM1K_SFC_MYNN_HPP
#define CM1K_SFC_MYNN_HPP

// Port of the MYNN surface layer (sfcmodel=6), src/module_sf_mynn.F:
//
//   psi_init / mynn_sf_init_driver   the stability-function lookup tables
//                                    (psi_opt=0), built once by PsiTables::make
//   SFCLAY_mynn + SFCLAY1D_mynn      sfclay() (one (i,j) column per thread)
//   charnock_1955, davis_etal_2008, Taylor_Yelland_2001, zilitinkevich_1995,
//   garratt_1992, fairall_etal_2003, Andreas_2002, Li_etal_2010, zolrib,
//   psi{m,h}_{stable,unstable}{,_full}
//
// plus the CM1 glue around the call in src/turb.F (sfc_and_turb):
//   prepare_inputs()   the "variables for wrf physics" block (k2=2 levels)
//                      and the tmp2d packing of the mynn_sf_j_loop
//   finish()           the sfcmodel=6 post-processing: qsfc diagnosis
//                      (update_sfc=.false.), ch/cq/s10, thflux/qvflux
//
// Bit-for-bit with the double-precision Fortran (tests/fortran_bridge/
// mynn_sfc_tests.cpp, against the unmodified module through
// cm1f_bridge_mynn_sfc.F90).
//
// Reachable in CM1 and ported: COARE_OPT=3.0, psi_opt=0, spp_pbl=0 (CM1 fixes
// all three), isftcflx = 0,1,2,3 over water, iz0tlnd = 0,1,3 over land,
// snowh >= 0.1 (Andreas 2002) over land, isfflx = 0/1, both
// itimestep==1 (first-guess init) and later steps.
//
// Not ported (Options::check throws):
//   * isftcflx = 4: SFCLAY1D_mynn never sets z_t/z_q for it (undefined
//     values in the Fortran)
//   * iz0tlnd = 2 (Yang 2008): reads qstar, a local automatic array of
//     SFCLAY_mynn that is only initialized when itimestep==1
//   * any other isftcflx/iz0tlnd value (the Fortran leaves z_t undefined)
// Not reachable (constants in the module): edson_etal_2013,
// fairall_etal_2014, the GFS psi functions, the PSI_* alternatives, zolri.
//
// Storage: the 2D surface arrays are FArray3 (ib:ie,jb:je,1:1), indexed
// a(i,j,1), like physics/sfc. Only the interior 1..ni x 1..nj is touched.

#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"

namespace cm1k::sfc_mynn {

struct Options {
  int sfcmodel = 1;
  int isftcflx = 0;   // namelist param12 (default 0)
  int iz0tlnd = 0;    // namelist param12 (default 0)
  int imoist = 0;

  bool active() const { return sfcmodel == 6; }
  static Options from_namelist(const Namelist& nl);
  void check() const;   // throws std::runtime_error for anything not ported
};

// module_sf_mynn's psim_stab, psim_unstab, psih_stab, psih_unstab (0:1000),
// filled exactly as psi_init(psi_opt=0) does.
struct PsiTables {
  static constexpr int n = 1001;
  Kokkos::View<Real*, Device> psim_stab, psim_unstab, psih_stab, psih_unstab;
  static PsiTables make();
};

// The 2D arrays SFCLAY_mynn reads and writes (cm1.F names; the dummy
// argument it is passed as is in brackets where it differs).
//   inputs:    psfc [PSFCPA], hpbl [PBLH], mavail, xland, tsk, snowh
//   persistent (inout across steps): ust, mol, qsfc, znt, ustm, hfx, qfx,
//              zol, rmol, wspd
//   outputs:   chs, chs2, cqs2, cpmm [CPM], regime, psim, psih, lh, flhc,
//              flqc, qgh, u10, v10, th2, t2, q2, gz1oz0, br, ch, ck, cka,
//              cd, cda
// CM1 starts them all at 0 except znt (1e-6 unless init_surface sets it),
// ust (1e-6), tsk, psfc (1e5), qsfc (1e-5), hpbl (100).
struct Fields {
  FArray3 psfc, hpbl, mavail, xland, tsk, snowh;
  FArray3 ust, mol, qsfc, znt, ustm, hfx, qfx, zol, rmol, wspd;
  FArray3 chs, chs2, cqs2, cpmm, regime, psim, psih, lh, flhc, flqc, qgh,
          u10, v10, th2, t2, q2, gz1oz0, br, ch, ck, cka, cd, cda;
  void allocate(const Bounds& b);
};

// Column inputs at the lowest model levels, CM1-indexed (i,j,k):
//   u3d, v3d   ground-relative wind at scalar points, k=1,2    (dum1, dum2)
//   dz8w       layer depth dz*rmh, k=1,2                        (dum5)
//   t3d        temperature, k=1                                 (dum4)
//   qv3d       water-vapor mixing ratio, k=1 (0 if dry)         (divx)
//   p3d        pressure, k=1                                    (prs)
//   rho3d      density, k=1                                     (rho)
struct Inputs {
  FArray3 u3d, v3d, dz8w, t3d, qv3d, p3d, rho3d;
};

// turb.F's preparation for sfcmodel=6 (lines ~335-400 and the divx block):
// fills `in` (allocated by the caller on ib:ie,jb:je,kb:ke or wider) on
// 1..ni x 1..nj, k=1,2. ugr/vgr: ground-relative winds (ua/va when imove=0).
// qv: qa(:,:,:,nqv), ignored when imoist=0 (qv3d=0 then).
void prepare_inputs(const Options& o, const Grid& g, const FArray3& ugr, const FArray3& vgr,
                    const FArray3& th0, const FArray3& tha, const FArray3& pi0, const FArray3& ppi,
                    const FArray3& prs, const FArray3& rho, const FArray4& qa, int nqv, const Inputs& in);

// SFCLAY_mynn as turb.F calls it (one j row per call there; all rows at once
// here): itimestep = nstep, isfflx = 1, dx = CM1's dx.
//   cp, g, rovcp, R=rd, xlv, svp1..3, svpt0, karman come from Constants;
//   ep1 = rv/rd - 1, ep2 = rd/rv as turb.F computes them.
void sfclay(const Options& o, const Constants& c, const PsiTables& t, int ni, int nj, Real dx,
            int itimestep, int isfflx, const Inputs& in, const Fields& f);

// turb.F after the call (update_sfc = .false.): qsfc = qv(k=1) + qfx/flqc
// (qv(k=1) where flqc = 0), ch = chs2 then ck (f.ch is CM1's ch: SFCLAY_mynn
// is passed the same array), cq = cqs2 then ck, s10 = |(u10,v10)|; with
// dosfcflx (or output_sfcflx=1) thflux = hfx/(cpmm*rf(:,:,1)) and
// qvflux = qfx/rf(:,:,1). cq, s10, thflux, qvflux: CM1's (ib:ie,jb:je,1:1).
void finish(int ni, int nj, bool dosfcflx, const Inputs& in, const FArray3& rf, const Fields& f,
            const FArray3& cq, const FArray3& s10, const FArray3& thflux, const FArray3& qvflux);

} // namespace cm1k::sfc_mynn

#endif
