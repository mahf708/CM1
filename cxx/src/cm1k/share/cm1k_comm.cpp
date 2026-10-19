#include "cm1k_comm.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace cm1k {

//===========================================================================
// Decomposition

int Decomp::nabor(int i, int j, int nx, int ny) {
  int newi = i, newj = j;
  if (newi < 1) newi = nx;
  if (newi > nx) newi = 1;
  if (newj < 1) newj = ny;
  if (newj > ny) newj = 1;
  return (newi - 1) + (newj - 1) * nx;
}

Decomp Decomp::serial(int nx, int ny) {
  Decomp d;
  d.nx = nx; d.ny = ny;
  d.ni = nx; d.nj = ny;
  d.myi1 = 1; d.myi2 = nx; d.myj1 = 1; d.myj2 = ny;
  return d;
}

#ifdef CM1K_HAVE_MPI
Decomp Decomp::make(MPI_Comm comm, int nx, int ny) {
  Decomp d;
  d.mpi = true;
  d.comm = comm;
  d.nx = nx; d.ny = ny;
  MPI_Comm_size(comm, &d.numprocs);
  MPI_Comm_rank(comm, &d.myid);
  const int numprocs = d.numprocs;
  // cm1.F (cm1r20.1): nodex, nodey (computed on rank 0 and broadcast there;
  // every rank gets the same answer here)
  int nodex = 1, nodey = 1;
  if (nx == 1 && ny == 1) {
    throw std::runtime_error("cm1k: cannot use MPI when nx = ny = 1");
  } else if (nx <= 3) {
    nodex = 1; nodey = numprocs;
  } else if (ny <= 3) {
    nodex = numprocs; nodey = 1;
  } else {
    const double smeps = 1.0e-60;
    double minval = 1.0e30;
    for (int n = numprocs; n >= 1; --n) {
      const double temy = double(n);
      const double temx = double(numprocs) / temy;
      const double temni = double(nx) / temx;
      const double temnj = double(ny) / temy;
      const double tem0 = std::fabs(temni / temnj - 2.0);
      if (std::fabs(double(int(temx)) - temx) < smeps) {
        if (tem0 < minval) {
          minval = tem0;
          nodex = int(temx);
          nodey = int(temy);
        }
      }
    }
  }
  nodex = std::max(1, nodex);
  nodey = std::max(1, nodey);
  d.nodex = nodex; d.nodey = nodey;
  d.myj = d.myid / nodex + 1;
  d.myi = d.myid - (d.myj - 1) * nodex + 1;
  // round-robin split of the points (cm1r20: ni may differ between ranks)
  std::vector<int> isum(nodex, 0), jsum(nodey, 0);
  d.ni = 0;
  for (int i = 1, id = 0; i <= nx; ++i) {
    id = id + 1;
    isum[id - 1] += 1;
    if (id == d.myi) d.ni += 1;
    if (id == nodex) id = 0;
  }
  d.nj = 0;
  for (int j = 1, id = 0; j <= ny; ++j) {
    id = id + 1;
    jsum[id - 1] += 1;
    if (id == d.myj) d.nj += 1;
    if (id == nodey) id = 0;
  }
  d.myi1 = 1;
  for (int i = 1; i <= d.myi - 1; ++i) d.myi1 += isum[i - 1];
  d.myi2 = d.myi1 + d.ni - 1;
  d.myj1 = 1;
  for (int j = 1; j <= d.myj - 1; ++j) d.myj1 += jsum[j - 1];
  d.myj2 = d.myj1 + d.nj - 1;
  if (d.ni < 3 || d.nj < 3)
    throw std::runtime_error("cm1k: both ni and nj must be >= 3 for MPI runs (ni=" + std::to_string(d.ni) +
                             ", nj=" + std::to_string(d.nj) + ")");
  // param.F: neighbours
  d.mynorth = nabor(d.myi, d.myj + 1, nodex, nodey);
  d.mysouth = nabor(d.myi, d.myj - 1, nodex, nodey);
  d.myeast  = nabor(d.myi + 1, d.myj, nodex, nodey);
  d.mywest  = nabor(d.myi - 1, d.myj, nodex, nodey);
  d.mysw = nabor(d.myi - 1, d.myj - 1, nodex, nodey);
  d.mynw = nabor(d.myi - 1, d.myj + 1, nodex, nodey);
  d.myne = nabor(d.myi + 1, d.myj + 1, nodex, nodey);
  d.myse = nabor(d.myi + 1, d.myj - 1, nodex, nodey);
  return d;
}
#endif

EdgeFlags EdgeFlags::make(const Decomp& d, int wbc, int ebc, int sbc, int nbc) {
  // param.F (cm1r20)
  EdgeFlags e;
  if (wbc != 1 && d.myi1 == 1) e.ibw = 1;
  if (ebc != 1 && d.myi2 == d.nx) e.ibe = 1;
  if (sbc != 1 && d.myj1 == 1) e.ibs = 1;
  if (nbc != 1 && d.myj2 == d.ny) e.ibn = 1;
  return e;
}

//===========================================================================
// Halo exchanges

namespace {

using DBuf = Kokkos::View<Real*, Device>;

void pack_box(const FArray3& a, const Box& b, const DBuf& buf) {
  const int i1 = b.i1, j1 = b.j1, k1 = b.k1;
  const int nib = b.i2 - b.i1 + 1, njb = b.j2 - b.j1 + 1;
  Kokkos::parallel_for("comm_pack", box3(b.i1, b.i2, b.j1, b.j2, b.k1, b.k2), KOKKOS_LAMBDA(int i, int j, int k) {
    buf((i - i1) + nib * ((j - j1) + njb * (k - k1))) = a(i, j, k);
  });
}

void unpack_box(const FArray3& a, const Box& b, const DBuf& buf) {
  const int i1 = b.i1, j1 = b.j1, k1 = b.k1;
  const int nib = b.i2 - b.i1 + 1, njb = b.j2 - b.j1 + 1;
  Kokkos::parallel_for("comm_unpack", box3(b.i1, b.i2, b.j1, b.j2, b.k1, b.k2), KOKKOS_LAMBDA(int i, int j, int k) {
    a(i, j, k) = buf((i - i1) + nib * ((j - j1) + njb * (k - k1)));
  });
}

constexpr int opposite(int d) {
  return d == HW ? HE : d == HE ? HW : d == HS ? HN : d == HN ? HS
       : d == HSW ? HNE : d == HNE ? HSW : d == HSE ? HNW : HSE;
}

Box mk(int i1, int i2, int j1, int j2, int k1, int k2) {
  Box b;
  b.i1 = i1; b.i2 = i2; b.j1 = j1; b.j2 = j2; b.k1 = k1; b.k2 = k2;
  return b;
}

// comm_3s/3t/3w/2d style: cmp-wide halos, i-halos for j=1..nj+ej, j-halos
// for i=1..ni+ei. su/sv: the staggered extra point of u (v) is local, so
// the west (south) neighbour gets i=2..cmp+1 and the east halo starts at
// ni+2 (nj+2).
HaloSpec wide(int ni, int nj, int cmp, int k1, int k2, bool su, bool sv) {
  HaloSpec sp;
  const int ei = su ? 1 : 0, ej = sv ? 1 : 0;
  for (int d = 0; d < 4; ++d) sp.use[d] = true;
  sp.recv[HE] = mk(ni + 1 + ei, ni + ei + cmp, 1, nj + ej, k1, k2);
  sp.recv[HW] = mk(1 - cmp, 0, 1, nj + ej, k1, k2);
  sp.recv[HS] = mk(1, ni + ei, 1 - cmp, 0, k1, k2);
  sp.recv[HN] = mk(1, ni + ei, nj + 1 + ej, nj + ej + cmp, k1, k2);
  sp.send[HW] = mk(1 + ei, cmp + ei, 1, nj + ej, k1, k2);
  sp.send[HE] = mk(ni - cmp + 1, ni, 1, nj + ej, k1, k2);
  sp.send[HN] = mk(1, ni + ei, nj - cmp + 1, nj, k1, k2);
  sp.send[HS] = mk(1, ni + ei, 1 + ej, cmp + ej, k1, k2);
  return sp;
}

// comm_1* style: one halo point
HaloSpec one(int ni, int nj, int k1, int k2, bool su, bool sv) {
  return wide(ni, nj, 1, k1, k2, su, sv);
}

// getcorner* style: the four diagonal corner points
HaloSpec corners(int ni, int nj, int k1, int k2, bool su, bool sv) {
  HaloSpec sp;
  const int ei = su ? 1 : 0, ej = sv ? 1 : 0;
  for (int d = 4; d < 8; ++d) sp.use[d] = true;
  sp.recv[HNW] = mk(0, 0, nj + 1 + ej, nj + 1 + ej, k1, k2);
  sp.recv[HSW] = mk(0, 0, 0, 0, k1, k2);
  sp.recv[HNE] = mk(ni + 1 + ei, ni + 1 + ei, nj + 1 + ej, nj + 1 + ej, k1, k2);
  sp.recv[HSE] = mk(ni + 1 + ei, ni + 1 + ei, 0, 0, k1, k2);
  sp.send[HSE] = mk(ni, ni, 1 + ej, 1 + ej, k1, k2);
  sp.send[HNE] = mk(ni, ni, nj, nj, k1, k2);
  sp.send[HSW] = mk(1 + ei, 1 + ei, 1 + ej, 1 + ej, k1, k2);
  sp.send[HNW] = mk(1 + ei, 1 + ei, nj, nj, k1, k2);
  return sp;
}

} // namespace

struct Comm::Pending {
  HaloSpec sp;
  bool posted[8] = {false, false, false, false, false, false, false, false};
  bool done[8] = {false, false, false, false, false, false, false, false};
  DBuf dsend[8], drecv[8];
  DBuf::host_mirror_type hsend[8], hrecv[8];
#ifdef CM1K_HAVE_MPI
  MPI_Request rreq[8], sreq[8];
#endif
};

Comm::Comm(const Decomp& d, const EdgeFlags& e, int ngxy, int nk) : m_d(d), m_e(e), m_ngxy(ngxy), m_nk(nk) {}

Comm::~Comm() = default;

bool Comm::active(int d) const {
  const bool w = m_e.ibw == 0, e = m_e.ibe == 0, s = m_e.ibs == 0, n = m_e.ibn == 0;
  switch (d) {
    case HW: return w;
    case HE: return e;
    case HS: return s;
    case HN: return n;
    case HSW: return w && s;
    case HSE: return e && s;
    case HNW: return w && n;
    default: return e && n;
  }
}

int Comm::neighbour(int d) const {
  switch (d) {
    case HW: return m_d.mywest;
    case HE: return m_d.myeast;
    case HS: return m_d.mysouth;
    case HN: return m_d.mynorth;
    case HSW: return m_d.mysw;
    case HSE: return m_d.myse;
    case HNW: return m_d.mynw;
    default: return m_d.myne;
  }
}

void Comm::start(const FArray3& a, const HaloSpec& sp, const std::string& slot) {
#ifdef CM1K_HAVE_MPI
  if (!m_d.mpi) throw std::runtime_error("cm1k comm: halo exchange '" + slot + "' on a serial model");
  if (m_pend.count(slot)) throw std::runtime_error("cm1k comm: exchange slot '" + slot + "' is already in flight");
  auto p = std::make_shared<Pending>();
  p->sp = sp;
  const int base = m_tag;
  m_tag = (m_tag + 8) % 32000;
  // receives first (as comm.F), then pack and send
  for (int d = 0; d < 8; ++d) {
    if (!sp.use[d] || !active(d)) continue;
    const int n = sp.recv[d].size();
    p->drecv[d] = DBuf("comm_recv", n);
    p->hrecv[d] = Kokkos::create_mirror_view(p->drecv[d]);
    MPI_Irecv(p->hrecv[d].data(), n, MPI_DOUBLE, neighbour(d), base + d, m_d.comm, &p->rreq[d]);
  }
  for (int d = 0; d < 8; ++d) {
    if (!sp.use[d] || !active(d)) continue;
    const int n = sp.send[d].size();
    p->dsend[d] = DBuf("comm_send", n);
    p->hsend[d] = Kokkos::create_mirror_view(p->dsend[d]);
    pack_box(a, sp.send[d], p->dsend[d]);
  }
  Kokkos::fence();
  for (int d = 0; d < 8; ++d) {
    if (!sp.use[d] || !active(d)) continue;
    Kokkos::deep_copy(p->hsend[d], p->dsend[d]);
    MPI_Isend(p->hsend[d].data(), sp.send[d].size(), MPI_DOUBLE, neighbour(d), base + opposite(d), m_d.comm,
              &p->sreq[d]);
    p->posted[d] = true;
  }
  m_pend[slot] = p;
#else
  (void)a; (void)sp;
  throw std::runtime_error("cm1k comm: halo exchange '" + slot + "' needs a build with CM1K_ENABLE_MPI");
#endif
}

void Comm::end(const FArray3& a, const std::string& slot, unsigned dirmask) {
#ifdef CM1K_HAVE_MPI
  auto it = m_pend.find(slot);
  if (it == m_pend.end()) throw std::runtime_error("cm1k comm: exchange slot '" + slot + "' was not started");
  auto& p = *it->second;
  for (int d = 0; d < 8; ++d) {
    if (!p.posted[d] || p.done[d] || !(dirmask & (1u << d))) continue;
    MPI_Wait(&p.rreq[d], MPI_STATUS_IGNORE);
    Kokkos::deep_copy(p.drecv[d], p.hrecv[d]);
    unpack_box(a, p.sp.recv[d], p.drecv[d]);
    p.done[d] = true;
  }
  bool all = true;
  for (int d = 0; d < 8; ++d) all = all && (!p.posted[d] || p.done[d]);
  if (all) {
    for (int d = 0; d < 8; ++d)
      if (p.posted[d]) MPI_Wait(&p.sreq[d], MPI_STATUS_IGNORE);
    Kokkos::fence();
    m_pend.erase(it);
  }
#else
  (void)a; (void)dirmask;
  throw std::runtime_error("cm1k comm: halo exchange '" + slot + "' needs a build with CM1K_ENABLE_MPI");
#endif
}

#define CM1K_COMM_PAIR(NAME, SPEC)                                                   \
  void Comm::NAME##_start(const FArray3& a, const std::string& slot) { start(a, SPEC, slot); } \
  void Comm::NAME##_end(const FArray3& a, const std::string& slot) { end(a, slot); }

CM1K_COMM_PAIR(comm_3s, wide(m_d.ni, m_d.nj, m_ngxy, 1, m_nk, false, false))
CM1K_COMM_PAIR(comm_3t, wide(m_d.ni, m_d.nj, m_ngxy, 1, m_nk + 1, false, false))
CM1K_COMM_PAIR(comm_3u, wide(m_d.ni, m_d.nj, m_ngxy, 1, m_nk, true, false))
CM1K_COMM_PAIR(comm_3v, wide(m_d.ni, m_d.nj, m_ngxy, 1, m_nk, false, true))
CM1K_COMM_PAIR(comm_3w, wide(m_d.ni, m_d.nj, m_ngxy, 2, m_nk, false, false))
CM1K_COMM_PAIR(comm_1s, one(m_d.ni, m_d.nj, 1, m_nk, false, false))
CM1K_COMM_PAIR(comm_1t, one(m_d.ni, m_d.nj, 1, m_nk + 1, false, false))
CM1K_COMM_PAIR(comm_1u, one(m_d.ni, m_d.nj, 1, m_nk, true, false))
CM1K_COMM_PAIR(comm_1v, one(m_d.ni, m_d.nj, 1, m_nk, false, true))
CM1K_COMM_PAIR(comm_1w, one(m_d.ni, m_d.nj, 2, m_nk, false, false))
#undef CM1K_COMM_PAIR

void Comm::comm_1p_start(const FArray3& s, const std::string& slot) {
  start(s, one(m_d.ni, m_d.nj, 1, m_nk, false, false), slot);
}
void Comm::comm_1p_end(const FArray3& s, const std::string& slot) { end(s, slot); }

void Comm::comm_2we_start(const FArray3& s, const std::string& slot) {
  HaloSpec sp = wide(m_d.ni, m_d.nj, 2, 1, m_nk, false, false);
  sp.use[HS] = sp.use[HN] = false;
  start(s, sp, slot);
}
void Comm::comm_2we_end(const FArray3& s, const std::string& slot) { end(s, slot); }
void Comm::comm_2sn_start(const FArray3& s, const std::string& slot) {
  HaloSpec sp = wide(m_d.ni, m_d.nj, 2, 1, m_nk, false, false);
  sp.use[HW] = sp.use[HE] = false;
  start(s, sp, slot);
}
void Comm::comm_2sn_end(const FArray3& s, const std::string& slot) { end(s, slot); }

void Comm::getcorner(const FArray3& s) {
  start(s, corners(m_d.ni, m_d.nj, 1, m_nk, false, false), "getcorner");
  end(s, "getcorner");
}
void Comm::getcornert(const FArray3& t) {
  start(t, corners(m_d.ni, m_d.nj, 1, m_nk + 1, false, false), "getcorner");
  end(t, "getcorner");
}
void Comm::getcorneru(const FArray3& u) {
  start(u, corners(m_d.ni, m_d.nj, 1, m_nk, true, false), "getcorner");
  end(u, "getcorner");
}
void Comm::getcornerv(const FArray3& v) {
  start(v, corners(m_d.ni, m_d.nj, 1, m_nk, false, true), "getcorner");
  end(v, "getcorner");
}
void Comm::getcornerw(const FArray3& w) {
  start(w, corners(m_d.ni, m_d.nj, 1, m_nk, false, false), "getcorner");
  end(w, "getcorner");
}

void Comm::comm_1s2d_start(const FArray3& s, int k, const std::string& slot) {
  start(s, one(m_d.ni, m_d.nj, k, k, false, false), slot);
}
void Comm::comm_1s2d_end(const FArray3& s, int, const std::string& slot) { end(s, slot); }
void Comm::comm_2d_start(const FArray3& s, int k, const std::string& slot) {
  start(s, wide(m_d.ni, m_d.nj, m_ngxy, k, k, false, false), slot);
}
void Comm::comm_2dew_end(const FArray3& s, int, const std::string& slot) {
  end(s, slot, (1u << HW) | (1u << HE));
}
void Comm::comm_2dns_end(const FArray3& s, int, const std::string& slot) {
  end(s, slot, (1u << HS) | (1u << HN));
}
void Comm::comm_2d_corner(const FArray3& s, int k) {
  start(s, corners(m_d.ni, m_d.nj, k, k, false, false), "comm_2d_corner");
  end(s, "comm_2d_corner");
}

//===========================================================================
// Reductions

void Comm::allreduce_sum(double* v, int n) const {
#ifdef CM1K_HAVE_MPI
  if (m_d.mpi) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, m_d.comm);
#else
  (void)v; (void)n;
#endif
}

double Comm::allreduce_sum_value(double v) const {
#ifdef CM1K_HAVE_MPI
  if (m_d.mpi) {
    double out = 0.0;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, m_d.comm);
    return out;
  }
#endif
  return v;
}

void Comm::allreduce_max(double* v, int n) const {
#ifdef CM1K_HAVE_MPI
  if (m_d.mpi) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_MAX, m_d.comm);
#else
  (void)v; (void)n;
#endif
}

void Comm::allreduce_min(double* v, int n) const {
#ifdef CM1K_HAVE_MPI
  if (m_d.mpi) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_MIN, m_d.comm);
#else
  (void)v; (void)n;
#endif
}

void Comm::allreduce_sum(int* v, int n) const {
#ifdef CM1K_HAVE_MPI
  if (m_d.mpi) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_INT, MPI_SUM, m_d.comm);
#else
  (void)v; (void)n;
#endif
}

int Comm::iallreduce_sum(double* v, int n) {
#ifdef CM1K_HAVE_MPI
  if (!m_d.mpi) return -1;
  m_ireq.emplace_back();
  MPI_Iallreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, m_d.comm, &m_ireq.back());
  return int(m_ireq.size()) - 1;
#else
  (void)v; (void)n;
  return -1;
#endif
}

void Comm::iwait(int h) {
#ifdef CM1K_HAVE_MPI
  if (h < 0) return;
  MPI_Wait(&m_ireq[h], MPI_STATUS_IGNORE);
  bool all = true;
  for (auto& r : m_ireq) all = all && (r == MPI_REQUEST_NULL);
  if (all) m_ireq.clear();
#else
  (void)h;
#endif
}

} // namespace cm1k
