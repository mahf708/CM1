#ifndef CM1K_PBL_MYNN_HPP
#define CM1K_PBL_MYNN_HPP

// Port of the MYNN-EDMF boundary-layer scheme as CM1 runs it (ipbl=4: level
// 2.5, ipbl=5: level 3):
//
//   src/module_bl_mynn_wrapper.F   mynnedmf_wrapper_run (mixing ratio <->
//                                  specific humidity, tendency conversion)
//   src/module_bl_mynn.F           mynn_bl_driver and everything it reaches
//                                  with CM1's settings: mym_initialize,
//                                  GET_PBLH, SCALE_AWARE, mym_condensation
//                                  (bl_mynn_cloudpdf=2), DMP_mf +
//                                  condensation_edmf (mass flux),
//                                  mym_turbulence / mym_level2 / mym_length
//                                  (bl_mynn_mixlength=1) / boulac_length,
//                                  mym_predict, mynn_tendencies,
//                                  moisture_check, retrieve_exchange_coeffs,
//                                  tridiag2, qsat_blend, xl_blend, phim, phih
//   src/module_bl_mynn_common.F    constants
//   src/turb.F                     the MYNN block of sfc_and_turb (input
//                                  packing, initflag = (nstep==0), output
//                                  copies) and the get_vars preparation
//
// One (i,j) column per thread with the k loops serial and the Fortran order
// of operations; per-column scratch lives in a workspace view (columns are
// processed in chunks to bound its size). Bit-for-bit with the double-
// precision Fortran: tests/fortran_bridge/mynn_pbl_tests.cpp drives the
// unmodified mynnedmf_wrapper_run through cm1f_bridge_mynn_pbl.F90 exactly
// as turb.F does.
//
// Options. param.F fixes, whenever ipbl=4/5 (or sfcmodel=6):
//   bl_mynn_cloudpdf=2, icloud_bl=1, spp_pbl=0, grav_settling=0,
//   bl_mynn_tkebudget=1, bl_mynn_mixlength=1, bl_mynn_edmf=1,
//   bl_mynn_edmf_mom=1, bl_mynn_edmf_tke=0, bl_mynn_cloudmix=1,
//   bl_mynn_mixscalars=0, bl_mynn_mixqt=0, bl_mynn_output=1,
//   bl_mynn_tkeadvect=.true., closure = 2.5 (ipbl=4) / 3.0 (ipbl=5)
// and module_bl_mynn fixes bl_mynn_topdown=0, bl_mynn_edmf_dd=0,
// dheat_opt=1, env_subs=.false., bl_mynn_stfunc=1, scaleaware=1, CKmod=1.
// turb.F passes FLAG_QC=FLAG_QI=.true., FLAG_QNI=(ptype 3 or 5) (the qni it
// passes is 0), every other scalar flag .false., restart=cycling=.false.
// Those are the values ported; Options::check throws for anything else,
// except bl_mynn_edmf and bl_mynn_edmf_mom, which may also be 0 (mass flux
// off / no momentum transport; both paths are ported and tested).
//
// Not ported (unreachable from CM1): DDMF_JPL, topdown_cloudrad,
// mynn_mix_chem, mym_condensation cases 0/1, mym_length cases 0/2,
// boulac_length0, mixing of qnc/qni/aerosols (bl_mynn_mixscalars=0),
// bl_mynn_mixqt=1, tridiag3, the ozone mixing (it only produces dozone,
// which CM1 discards), SPP perturbations, the restart/cycling branches. The
// env_subs branch of DMP_mf (sub_*/det_* stay 0) and the plume quantities
// only it uses (envm_*, detrainment rates) are not computed. Not ported
// from turb.F: the cm1setup=4 qke zeroing in the LES subdomain (throws) and
// the td_diss/td_pbl budget diagnostics (they read only the last j row of
// tmp2d). MYNN is skipped on a restart; the caller does that.

#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"

#include <cstddef>

namespace cm1k::pbl_mynn {

struct Options {
  int ipbl = 0;
  int imoist = 0;
  int ptype = 0;
  int cm1setup = 1;
  // param.F's fixed MYNN settings (see above)
  int bl_mynn_cloudpdf = 2, icloud_bl = 1, spp_pbl = 0, bl_mynn_tkebudget = 1, bl_mynn_mixlength = 1,
      bl_mynn_edmf = 1, bl_mynn_edmf_mom = 1, bl_mynn_edmf_tke = 0, bl_mynn_cloudmix = 1,
      bl_mynn_mixscalars = 0, bl_mynn_mixqt = 0, bl_mynn_output = 1;
  bool bl_mynn_tkeadvect = true;

  bool active() const { return ipbl == 4 || ipbl == 5; }
  Real closure() const { return ipbl == 5 ? 3.0 : 2.5; }
  static Options from_namelist(const Namelist& nl);
  void check() const;   // throws std::runtime_error for anything not ported
};

// Column inputs, CM1-indexed (i,j,k), k=1..nk (turb.F's tmp2d packing):
//   dz     dz*rmh                              (dum5)
//   u, v   ground-relative wind at scalars     (dum1, dum2)
//   w      0.5*(wa(k)+wa(k+1))
//   th     th0+tha                             (dum3)
//   qv     water vapor mixing ratio, 0 if dry  (divx)
//   qc, qi cloud water / ice mixing ratio      (ppten, dum8)
//   p      pressure                            (prs)
//   exner  pi0+ppi                             (dum7)
//   rho    density                             (rho)
//   t3d    temperature th*exner                (dum4)
// (turb.F also packs rthraten, which only the top-down scheme reads.)
struct Inputs {
  FArray3 dz, u, v, w, th, qv, qc, qi, p, exner, rho, t3d;
};

// 2D arrays, (ib:ie,jb:je,1:1) like physics/sfc, indexed a(i,j,1).
//   in:     znt (unused by the scheme), xland, tsk [ts], qsfc (unused),
//           psfc [ps], ust, ch (unused), hfx, qfx, wspd
//   inout:  rmol (recomputed from the fluxes; the input only feeds
//           mym_initialize at nstep=0), hpbl [pblh]
//   out:    kpbl [kpbl2d], nupdraft, ktop_plume (integers stored as Real),
//           maxmf
struct Surface {
  FArray3 znt, xland, tsk, qsfc, psfc, ust, ch, hfx, qfx, wspd, rmol, hpbl, kpbl, nupdraft, maxmf,
          ktop_plume;
  void allocate(const Bounds& b);
};

// 3D MYNN arrays (cm1.F names), indexed (i,j,k), k=1..nk used.
//   qke_adv   in: qke after advection (turb.F passes qke3d); out: qke_adv
//   qke       out (= qke_adv; the input is never read)
//   sh3d, sm3d, tsq, qsq, cov, el_pbl, qc_bl, qi_bl, cldfra_bl: inout
//   edmf_*, sub_*/det_*, dqke, qwt, qshear, qbuoy, qdiss: out
struct State {
  FArray3 qke, qke_adv, sh3d, sm3d, tsq, qsq, cov, el_pbl, qc_bl, qi_bl, cldfra_bl, edmf_a, edmf_w,
          edmf_qt, edmf_thl, edmf_ent, edmf_qc, sub_thl3d, sub_sqv3d, det_thl3d, det_sqv3d, dqke, qwt,
          qshear, qbuoy, qdiss;
  void allocate(const Bounds& b);   // (ib:ie,jb:je,kb:ke), zero
};

// The tendencies and diffusivities turb.F copies out (k=1..nk):
//   upten, vpten, thpten, qvpten, qcpten, qipten (qv/qc/qi as mixing
//   ratios), xkzh = exch_h, xkzm = exch_m, dheat (dheat3d, the dissipative
//   heating rate included in thpten).
struct Tendencies {
  FArray3 upten, vpten, thpten, qvpten, qcpten, qipten, xkzh, xkzm, dheat;
  void allocate(const Bounds& b);   // (ib:ie,jb:je,kb:ke), zero
};

// turb.F's get_vars preparation for the MYNN call (1..ni x 1..nj x 1..nk).
// ugr/vgr: ground-relative winds (ua/va when imove=0). nqv/nqc/nqi: index
// of qv/qc/qi in qa, or 0 when absent (qv also 0 when imoist=0).
void prepare_inputs(const Options& o, const Grid& g, const FArray3& ugr, const FArray3& vgr,
                    const FArray3& wa, const FArray3& th0, const FArray3& tha, const FArray3& pi0,
                    const FArray3& ppi, const FArray3& prs, const FArray3& rho, const FArray4& qa, int nqv,
                    int nqc, int nqi, const Inputs& in);

// The domynn block of turb.F: mynnedmf_wrapper_run over 1..ni x 1..nj with
// delt = dt, dxc = dx, initflag = (nstep == 0), then the copies into the
// CM1 arrays. max_work_bytes bounds the per-column scratch (columns are
// processed in chunks).
void run(const Options& o, int ni, int nj, int nk, Real dt, Real dx, int nstep, const Inputs& in,
         const Surface& sfc, const State& st, const Tendencies& tend, std::size_t max_work_bytes = 256u << 20);

} // namespace cm1k::pbl_mynn

#endif
