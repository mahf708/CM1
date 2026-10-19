#ifndef CM1K_RADIATION_HPP
#define CM1K_RADIATION_HPP

// cm1.F's radiation for radopt = 2 (RRTMG): the radiation arrays and timing
// of cm1.F / param.F / init_physics.F, and src/radiation_driver.F's
// radopt = 2 body (radconst, calc_coszen, cal_cldfra1, RRTMG_LWRAD,
// RRTMG_SWRAD, the eqtset = 2 conversion and thraten).
//
// Timing (cm1.F): radtim starts at 0 (param.F); the startup pass (nstep = 0)
// always calls the radiation, and the pass that ends at mtime calls it when
// mtime >= radtim - 0.1*dt; after each call radtim advances by dtrad until
// it exceeds mtime. The call is at the end of a pass, after the dump and
// before sfc_and_turb: the Model makes it at the start of step(), before
// sfc_and_turb, on the state of that dump (see Model::step).
//
// Supported: radopt = 0 (nothing) or 2, with imoist = 1, ptype = 1 (Kessler:
// cal_cldfra1's qc-only branch, mp_physics = 1 as the cm1k fork patch of
// radiation_driver.F sets), sfcmodel = 2 or 3 (gsw, glw, emiss are the
// WRF surface layer's / SLAB's arrays; albd from LANDUSE.TBL), testcase = 0 or 8 (4, 5 throw),
// no MYNN (icloud_bl coupling) or MYJ (sfcmodel = 7), PBL ipbl = 0..3 (YSU and
// GFS-EDMF get rthraten = thraten), eqtset = 1 or 2. Anything else throws.
//
// The RRTMG tables are read from RRTMG_LW_DATA / RRTMG_SW_DATA located by
// rrtmg::find_data_file ($CM1K_RRTMG_DATA_DIR, the working directory, then
// the source tree's run/).

#include "share/cm1k_constants.hpp"
#include "share/cm1k_farray.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cm1k::rrtmg {

struct LwTableData;
struct SwTableData;

struct RadOptions {
  int radopt = 0;
  Real dtrad = 300.0, ctrlat = 36.68, ctrlon = -98.35;
  int year = 2009, month = 5, day = 15, hour = 21, minute = 38, second = 0;
  int eqtset = 2, imoist = 0, ptype = 0, testcase = 0, sfcmodel = 0, ipbl = 0, rterm = 0, psolver = 3;
  int lu0 = 16, season = 1;
  static RadOptions from_namelist(const Namelist& nl);
  bool active() const { return radopt != 0; }
  void check() const;   // throws for every option not ported
};

// radtrns3d.F julday: day of the year (1 = Jan 1).
int julday(int year, int month, int day);

// radiation_driver.F radconst and calc_coszen (one column; xlat, xlon in
// degrees). Exposed for tests.
void radconst(Real julian, Real degrad, Real dpd, Real& declin, Real& solcon);
void calc_coszen(Real julian, Real xtime, Real gmt, Real declin, Real degrad, Real xlon, Real xlat, Real& coszen,
                 Real& hrang);

class Radiation {
public:
  // Allocates the arrays (zero, as cm1.F) for bounds b. tsk, xland: the
  // surface arrays; gsw, glw, emiss: the SLAB arrays radiation writes
  // (gsw, glw) and reads (emiss). Fills albd (getlanduse: ALBD(lu0)/100).
  void setup(const RadOptions& o, const Constants& c, const Bounds& b, const FArray3& tsk, const FArray3& xland,
             const FArray3& gsw, const FArray3& glw, const FArray3& emiss);

  // init_physics.F (radopt >= 1): o30 = 1e-12 (and the start of the radiation
  // clock, radtim = 0). Call after a native init.
  void init_state();
  // The radiation clock after a restart from the dump of step nstep (the
  // passes 0 .. nstep-1 have run; see the header comment).
  void set_clock(int nstep, Real dt);
  // radtim before the radiation of pass nstep (dt: the time step)
  static double clock_at(int nstep, Real dt, Real dtrad);

  // The radiation of the pass that ended at mtime (step nstep), when due;
  // then radtim advances. The first call reads the tables and sets p_top
  // from prs0 (init_physics.F).
  void pass(double mtime, int nstep, Real dt, const Grid& g, const BaseState& base, const State& s);

  // radiation_driver: always runs (no timing).
  void driver(double mtime, int nstep, Real dt, const Grid& g, const BaseState& base, const State& s);

  double radtim() const { return m_radtim; }
  Real dtrad() const { return m_dtrad; }

  // The arrays src/cm1dump_state.inc writes for radopt = 2, in that order.
  std::vector<std::pair<std::string, FArray3>> dumped() const;

  // cm1.F arrays (ibr:ier = ib:ie, jbr:jer, kbr:ker = kb:ke)
  FArray3 swten, lwten, swtenc, lwtenc, cldfra, o30, thraten;
  FArray3 lwupt, lwuptc, lwdnt, lwdntc, lwupb, lwupbc, lwdnb, lwdnbc;   // 2-D (i,j,1)
  FArray3 swupt, swuptc, swdnt, swdntc, swupb, swupbc, swdnb, swdnbc;
  FArray3 lwcf, swcf, coszr, xice, xsnow, xlat, xlong, coszen, swddir, swddni, swddif, hrang;
  FArray3 albd, olr;     // albd (ibl:iel = ib:ie), olr (1:ni,1:nj,1:1)
  FArray3 tsk, xland, gsw, glw, emiss;   // shared with the surface arrays

private:
  void start(const Grid& g, const BaseState& base);

  RadOptions m_o;
  Constants m_c{};
  Bounds m_b;
  bool m_started = false;
  double m_radtim = 0.0;
  Real m_dtrad = 0.0;
  int m_jday = 0;
  int m_nlayers = 0;
  Real m_p_top = 0, m_cgs1 = 0, m_cgs2 = 0, m_cgs3 = 0, m_cgt1 = 0, m_cgt2 = 0, m_cgt3 = 0;
  std::shared_ptr<const LwTableData> m_lw;
  std::shared_ptr<const SwTableData> m_sw;
  // radiation_driver scratch (cm1.F dum1..dum9, sten, thten) and its locals
  FArray3 dum1, dum2, dum3, dum4, dum5, dum6, dum7, dum8, dum9, dum10, dum11;
  FArray3 swdownc, swddnic, swddirc;
};

} // namespace cm1k::rrtmg

#endif
