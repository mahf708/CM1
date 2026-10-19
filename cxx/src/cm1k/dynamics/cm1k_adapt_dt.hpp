#ifndef CM1K_ADAPT_DT_HPP
#define CM1K_ADAPT_DT_HPP

// Adaptive time step (adapt_dt = 1): the CFL checks of misclibs.F
// (calccflquick, calcksquick), getnewdt (new dt from cflmax/ksmax, limited
// to 2*dtl and ramped down ahead of output, restart and stat times) and
// cm1.F's output-time bookkeeping (taptim, stattim, rsttim and the
// dowriteout/dostat/dorestart flags of each step), which getnewdt reads.
// dtsmall (nsound for the new dt) is Dyncore::dtsmall.

#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"

#include <string>

namespace cm1k {

// misclibs.F calccflquick: max over the interior of the advective Courant
// number from (ua, va, wa) at dt (0 when nothing exceeds 0).
Real calccflquick(const Bounds& b, int axisymm, Real dt, Real rdx, Real rdy, Real rdz, const FArray1& uh,
                  const FArray1& vh, const FArray3& mh, const FArray3& ua, const FArray3& va, const FArray3& wa);

// misclibs.F calcksquick: max of the horizontal diffusion numbers of
// kmh, khh, kmv on k = 2..nk.
Real calcksquick(const Bounds& b, int axisymm, Real dt, Real rdx, Real rdy, const FArray1& uh, const FArray1& vh,
                 const FArray3& kmh, const FArray3& kmv, const FArray3& khh, const FArray3& khv);

// cm1.F / misclibs.F time-step state of adapt_dt = 1.
struct AdaptDt {
  bool active = false;     // adapt_dt == 1
  int cm1setup = 1, nx = 0, ny = 0;
  Real dtl = 0, tapfrq = 0, statfrq = 0, rstfrq = 0;
  double dbldt = 0;        // cm1.F dbldt (dt = dbldt)
  Real cflmax = 0, ksmax = 0;
  double taptim = 0, stattim = 0, rsttim = 0;   // param.F: 0, 0, rstfrq
  // flags of the last pass of the time loop (startup pass: write and stats)
  bool dowriteout = true, dostat = true, dorestart = false;

  // Reads the options; throws for the combinations whose timing is not
  // ported (radiation, domain diagnostics, azimuthal averages, high-
  // frequency output, parcels, rstfrq = 0) or whose ksmax reads arrays CM1
  // does not allocate (cm1setup >= 1 without eddy viscosities).
  static AdaptDt from_namelist(const Namelist& nl, const Bounds& b);

  // misclibs.F getnewdt up to `dt = dbldt` (dbldt from cflmax/ksmax at the
  // current dt; ramp-down ahead of taptim/rsttim/stattim unless doinit).
  // The caller then calls dtsmall and sets dt = dbldt.
  void getnewdt(double mtime, Real dt, bool doinit);
  // cm1.F after statpack / writeout / restart_write: advance the timers past
  // mtime for the flags of the pass just finished.
  void advance_timers(double mtime);
  // cm1.F at the top of a step: this step's flags.
  void set_flags(double mtime);
};

} // namespace cm1k

#endif
