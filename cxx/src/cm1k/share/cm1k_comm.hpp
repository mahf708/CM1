#ifndef CM1K_COMM_HPP
#define CM1K_COMM_HPP

// MPI domain decomposition and halo exchanges: port of CM1's MPI layout
// (cm1.F: nodex/nodey, ni/nj, myi/myj, myi1..myj2; param.F: neighbours via
// nabor, ibw/ibe/ibs/ibn) and of comm.F's message passing routines.
//
// Built with CM1K_ENABLE_MPI (CMake), which defines CM1K_HAVE_MPI. Without
// it (and for a Model built without a communicator) everything here is
// serial: Decomp::serial, Comm::mpi() == false, and the callers keep the
// serial (`#ifndef MPI`) code paths unchanged.
//
// Exchanges keep CM1's names and split into _start/_end exactly where the
// Fortran does: _start packs the interior values (on the device), copies
// them to host buffers and posts MPI_Irecv/MPI_Isend; _end waits and
// unpacks into the halos. What the Fortran passes as buffer arrays (uw31,
// ..., reqs_u) is a named slot here, e.g. comm_3u_start(u3d, "u") ...
// comm_3u_end(u3d, "u"). Each routine fills exactly the halo points the
// Fortran routine fills (e.g. comm_3s: i-halos for j=1..nj and j-halos for
// i=1..ni, k=1..nk; no corners), so the halo contents, and hence the
// results, match the MPI Fortran bit for bit.
//
// The corner patches inside comm.F's *_end routines (p2tch*) need ibw=1
// and ibs=1 at the same corner with one direction periodic, which param.F
// does not allow, so they are never active and are not ported.

#include "share/cm1k_farray.hpp"

#ifdef CM1K_HAVE_MPI
#include <mpi.h>
#endif

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cm1k {

// cm1.F / param.F decomposition. Serial: nodex = nodey = 1, mpi = false.
struct Decomp {
  bool mpi = false;   // CM1's `#ifdef MPI` code paths (set when built from a communicator)
  int numprocs = 1, myid = 0;
  int nx = 0, ny = 0;
  int nodex = 1, nodey = 1, myi = 1, myj = 1;
  int ni = 0, nj = 0;                            // local grid size
  int myi1 = 1, myi2 = 0, myj1 = 1, myj2 = 0;    // global index range of this rank
  int mywest = 0, myeast = 0, mysouth = 0, mynorth = 0;
  int mysw = 0, mynw = 0, myne = 0, myse = 0;
#ifdef CM1K_HAVE_MPI
  MPI_Comm comm = MPI_COMM_NULL;
#endif

  static Decomp serial(int nx, int ny);
#ifdef CM1K_HAVE_MPI
  // cm1.F (cm1r20.1): nodex, nodey from the number of ranks (target
  // ni = 2 nj), then the round-robin split of nx, ny (param.F ni, nj).
  static Decomp make(MPI_Comm comm, int nx, int ny);
#endif
  // comm.F: nabor
  static int nabor(int i, int j, int nx, int ny);
};

// Lateral boundary flags (param.F): 1 on a rank at a physical
// (non-periodic) boundary.
struct EdgeFlags {
  int ibw = 0, ibe = 0, ibs = 0, ibn = 0;
  static EdgeFlags make(const Decomp& d, int wbc, int ebc, int sbc, int nbc);
};

// An inclusive Fortran index box.
struct Box {
  int i1 = 1, i2 = 0, j1 = 1, j2 = 0, k1 = 1, k2 = 0;
  int size() const { return (i2 - i1 + 1) * (j2 - j1 + 1) * (k2 - k1 + 1); }
};

// Directions (also the slots of a HaloSpec)
enum HaloDir { HW = 0, HE = 1, HS = 2, HN = 3, HSW = 4, HSE = 5, HNW = 6, HNE = 7 };

// One exchange: for each used direction d, recv[d] is the halo box filled
// from neighbour d, and send[d] the interior box sent to neighbour d (which
// stores it in its recv[opposite(d)]; both boxes have the same shape).
struct HaloSpec {
  bool use[8] = {false, false, false, false, false, false, false, false};
  Box recv[8], send[8];
};

class Comm {
public:
  Comm() = default;   // serial
  Comm(const Decomp& d, const EdgeFlags& e, int ngxy, int nk);
  ~Comm();
  Comm(const Comm&) = delete;
  Comm& operator=(const Comm&) = delete;

  bool mpi() const { return m_d.mpi; }
  const Decomp& decomp() const { return m_d; }
  const EdgeFlags& edges() const { return m_e; }

  // ---- comm.F halo exchanges (3D arrays: k range as in the Fortran) ----
  void comm_3s_start(const FArray3& s, const std::string& slot);
  void comm_3s_end(const FArray3& s, const std::string& slot);
  void comm_3t_start(const FArray3& t, const std::string& slot);
  void comm_3t_end(const FArray3& t, const std::string& slot);
  void comm_3u_start(const FArray3& u, const std::string& slot);
  void comm_3u_end(const FArray3& u, const std::string& slot);
  void comm_3v_start(const FArray3& v, const std::string& slot);
  void comm_3v_end(const FArray3& v, const std::string& slot);
  void comm_3w_start(const FArray3& w, const std::string& slot);
  void comm_3w_end(const FArray3& w, const std::string& slot);
  void comm_1s_start(const FArray3& s, const std::string& slot);
  void comm_1s_end(const FArray3& s, const std::string& slot);
  void comm_1p_start(const FArray3& s, const std::string& slot);
  void comm_1p_end(const FArray3& s, const std::string& slot);   // comm_1p_end1 + comm_1p_end2
  void comm_1t_start(const FArray3& t, const std::string& slot);
  void comm_1t_end(const FArray3& t, const std::string& slot);
  void comm_1u_start(const FArray3& u, const std::string& slot);
  void comm_1u_end(const FArray3& u, const std::string& slot);
  void comm_1v_start(const FArray3& v, const std::string& slot);
  void comm_1v_end(const FArray3& v, const std::string& slot);
  void comm_1w_start(const FArray3& w, const std::string& slot);
  void comm_1w_end(const FArray3& w, const std::string& slot);
  void comm_2we_start(const FArray3& s, const std::string& slot);
  void comm_2we_end(const FArray3& s, const std::string& slot);
  void comm_2sn_start(const FArray3& s, const std::string& slot);
  void comm_2sn_end(const FArray3& s, const std::string& slot);
  // corner points (blocking, as in comm.F)
  void getcorner(const FArray3& s);    // k = 1..nk
  void getcornert(const FArray3& t);   // k = 1..nk+1
  void getcorneru(const FArray3& u);
  void getcornerv(const FArray3& v);
  void getcornerw(const FArray3& w);   // k = 1..nk (as comm.F)
  // ---- 2D fields, stored as level k of an FArray3 (CM1's 2D arrays) ----
  void comm_1s2d_start(const FArray3& s, int k, const std::string& slot);
  void comm_1s2d_end(const FArray3& s, int k, const std::string& slot);
  void comm_2d_start(const FArray3& s, int k, const std::string& slot);
  void comm_2dew_end(const FArray3& s, int k, const std::string& slot);
  void comm_2dns_end(const FArray3& s, int k, const std::string& slot);
  void comm_2d_corner(const FArray3& s, int k);

  // ---- reductions (same MPI calls as the Fortran) ----
  // MPI_ALLREDUCE(MPI_IN_PLACE, v, n, MPI_DOUBLE_PRECISION, MPI_SUM)
  void allreduce_sum(double* v, int n) const;
  // MPI_IALLREDUCE(MPI_IN_PLACE, v, n, ..., MPI_SUM); finish with iwait(h)
  int iallreduce_sum(double* v, int n);
  void iwait(int h);
  // MPI_ALLREDUCE with MPI_MAX / MPI_MIN on doubles, and MPI_SUM on ints
  void allreduce_max(double* v, int n) const;
  void allreduce_min(double* v, int n) const;
  void allreduce_sum(int* v, int n) const;

  // generic engine (public for tests and other modules)
  void start(const FArray3& a, const HaloSpec& sp, const std::string& slot);
  // dirmask: bit d set = finish direction d (default: all)
  void end(const FArray3& a, const std::string& slot, unsigned dirmask = 0xffu);

private:
  bool active(int d) const;
  int neighbour(int d) const;
  struct Pending;
  Decomp m_d;
  EdgeFlags m_e;
  int m_ngxy = 0, m_nk = 0;
  int m_tag = 0;
  std::map<std::string, std::shared_ptr<Pending>> m_pend;
#ifdef CM1K_HAVE_MPI
  std::vector<MPI_Request> m_ireq;
#endif
};

} // namespace cm1k

#endif
