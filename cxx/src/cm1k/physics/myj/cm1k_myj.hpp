#ifndef CM1K_MYJ_HPP
#define CM1K_MYJ_HPP

// Port of the Mellor-Yamada-Janjic (MYJ) surface layer and boundary-layer
// scheme as CM1 runs them (sfcmodel=7, ipbl=6):
//
//   src/module_sf_myjsfc.F   MYJSFCINIT (the psi tables; Tables::make),
//                            MYJSFC + SFCDIF (myjsfc())
//   src/module_bl_myjpbl.F   MYJPBL + MIXLEN, PRODQ2, DIFCOF, VDIFQ,
//                            VDIFH, VDIFV (myjpbl())
//   src/module_ra_etc.F      module_model_constants (r_d, cp, g, pq0,
//                            epsq2, a2..a4, p608, xlv, xls)
//
// One (i,j) column per thread with the k loops serial and the Fortran's
// operation order (bit-for-bit with the double-precision Fortran, see
// tests/fortran_bridge/myj_tests.cpp). The scheme counts levels downward
// from the model top (K = KTE+1-k); the column work arrays keep that
// flipped indexing so the code reads side by side with the Fortran.
// Per-column scratch lives in a workspace view (Workspace).
//
// What CM1 fixes (and what is therefore ported):
//   * LOWLYR = 1 everywhere (MYJSFCINIT; never changed), so LMH = KTE and
//     PSFC = PINT(1)
//   * MYJPBL: STEPBL = 1, QCI present (flQI = .true., NSPEC = 4: theta_e,
//     q, qc, qi are mixed), QCS/QCR/QCG passed but unused (the BSF lines
//     are commented out), SNOW = SICE = 0 (cm1.F never sets them), so the
//     RLIVWV factor is unreachable; RHO, ZNT and the QCS/QCR/QCG
//     tendencies are not used (CM1 copies back what it passed in)
//   * MYJSFC: ISURBAN = 0, IVGTYP and IZ0TLND unused by the code;
//     isftcflx is PRESENT (any value; /= 0 turns on the high-wind
//     roughness), usemyjpbl = (ipbl == 6)
//   * the NMM_CORE branches are compiled out (ARW: NTSD == 1 is the first
//     step)
//
// Storage: 2D arrays are FArray3 (ib:ie,jb:je,1:1) indexed a(i,j,1), as in
// physics/sfc; 3D inputs/outputs are CM1-indexed (i,j,k), k = 1..nk (pint:
// k = 1..nk+1), only the interior 1..ni x 1..nj is touched.

#include "share/cm1k_farray.hpp"
#include "share/cm1k_types.hpp"

namespace cm1k::myj {

// MYJSFCINIT: module_sf_myjsfc's PSIM1/PSIH1 (water) and PSIM2/PSIH2 (land)
// tables (KZTM = 10001 points, built on the host with libm as the Fortran)
// and the scalars it sets (ZTMIN*/ZTMAX*/DZETA*/FH0*).
struct Tables {
  static constexpr int kztm = 10001;
  Kokkos::View<Real*, Device> psim1, psih1, psim2, psih2;   // index K-1 holds PSI(K)
  Real dzeta1 = 0, dzeta2 = 0, fh01 = 0, fh02 = 0, ztmax1 = 0, ztmax2 = 0, ztmin1 = 0, ztmin2 = 0;
  static Tables make();
};

// Column workspace (one row of scratch per (i,j) column).
struct Workspace {
  Kokkos::View<Real**, Kokkos::LayoutLeft, Device> w;
  int ni = 0, nj = 0, nk = 0;
  void allocate(int ni, int nj, int nk);
};

// MYJSFC inputs (turb.F's tmp2d packing, here for every j at once):
//   dz    layer depth dz*rmh                       (tmp2d 1)
//   pmid  pressure prs                             (tmp2d 2)
//   pint  pressure on w levels, k = 1..nk+1        (tmp2d 3; only k = 1 is used)
//   th    theta = th0+tha                          (tmp2d 4)
//   t     temperature theta*(pi0+ppi)              (tmp2d 5)
//   qv    water vapor mixing ratio, 0 if dry       (tmp2d 6)
//   qc    cloud water mixing ratio, 0 if absent    (tmp2d 7)
//   u, v  wind at scalar points (ground-relative)  (tmp2d 8, 9)
//   q2    tke_myj                                  (tmp2d 10)
//   ht    terrain height zs (2D)
//   rhosfc  rf(:,:,1) (2D view of rf: k = 1 is read)
struct SfcInputs {
  FArray3 dz, pmid, pint, th, t, qv, qc, u, v, q2, ht, rhosfc;
};

// MYJSFC 2D arguments (the CM1 array turb.F passes is in brackets where
// the name differs).
//   in:     tsk, xland, z0base, mavail
//   inout:  qsfc, thz0, qz0, uz0, vz0, ust [USTAR], znt [ZNT = zntmyj],
//           pblh [hpbl], akhs, akms
//   out:    rmol, rib [br], chs, chs2, cqs2, hfx, qfx, lh [FLX_LH], flhc,
//           flqc, qgh, cpm [cpmm], ct, u10, v10, t02 [t2], th02 [th2],
//           tshltr, th10, q02 [q2], qshltr, q10, pshltr, u10e, v10e, psim,
//           psih, fm, fhh [fh], z0out [znt]
struct SfcFields {
  FArray3 tsk, xland, z0base, mavail;
  FArray3 qsfc, thz0, qz0, uz0, vz0, ust, znt, pblh, akhs, akms;
  FArray3 rmol, rib, chs, chs2, cqs2, hfx, qfx, lh, flhc, flqc, qgh, cpm, ct, u10, v10, t02, th02, tshltr,
          th10, q02, qshltr, q10, pshltr, u10e, v10e, psim, psih, fm, fhh, z0out;
};

// MYJSFC (itimestep = CM1's nstep, P1000mb = 1e5) over 1..ni x 1..nj.
// isftcflx: the PRESENT optional argument (0 = original roughness over
// water).
void myjsfc(const Tables& t, const Workspace& ws, int ni, int nj, int nk, int itimestep, bool usemyjpbl,
            int isftcflx, const SfcInputs& in, const SfcFields& f);

// MYJPBL inputs (turb.F's domyj packing, from the get_vars arrays):
//   dz (dum5), pmid (prs), pint (dum6, k = 1..nk+1), th (dum3), t (dum4),
//   exner (dum7), qv (divx), qcw (ppten = qc), qci (dum8 = qi), u (dum1),
//   v (dum2); ht = zs (2D)
struct PblInputs {
  FArray3 dz, pmid, pint, th, t, exner, qv, qcw, qci, u, v, ht;
};

// MYJPBL arguments:
//   in:     tsk, xland, chklowq, elflx [lh]
//   inout:  qsfc, thz0, qz0, uz0, vz0, ust [USTAR], ct, akhs, akms,
//           tke_myj (k = 1..nk)
//   out:    pblh [hpbl], kpbl [kpbl2d, integer stored as Real], mixht,
//           el_myj (k = 1..nk; k = 1 is 0), exch_h, exch_m (k = 1..nk-1),
//           rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten (k = 1..nk)
struct PblFields {
  FArray3 tsk, xland, chklowq, elflx;
  FArray3 qsfc, thz0, qz0, uz0, vz0, ust, ct, akhs, akms, tke_myj;
  FArray3 pblh, kpbl, mixht, el_myj, exch_h, exch_m, rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten;
};

// MYJPBL (DT = dt, STEPBL = 1) over 1..ni x 1..nj.
void myjpbl(const Workspace& ws, int ni, int nj, int nk, Real dt, const PblInputs& in, const PblFields& f);

} // namespace cm1k::myj

#endif
