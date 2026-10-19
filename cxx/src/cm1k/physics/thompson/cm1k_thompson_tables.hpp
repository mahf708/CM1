#ifndef CM1K_THOMPSON_TABLES_HPP
#define CM1K_THOMPSON_TABLES_HPP

// Thompson microphysics (src/thompson.F, module_mp_thompson): the constants
// and lookup tables that thompson_init builds.
//
// Everything here is computed on the host, in double precision, with the
// Fortran's operation order (see the .cpp), so the tables are bit-for-bit
// those of the double-precision Fortran (gfortran -fdefault-real-8, with the
// table loops kept scalar by `!GCC$ NOVECTOR`, i.e. glibc's scalar libm).
//
// thompson_init's three expensive tables (qr_acr_qg, qr_acr_qs, freezeH2O)
// take minutes; the Fortran can read them from .dat files and so can cm1k:
// Tables::get() keeps them in an on-disk cache (see cache_dir()), validated
// on load by recomputing a sample of entries.

#include "share/cm1k_types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cm1k::thompson {

// module_mp_thompson parameters (thompson.F lines 62-245)
namespace par {
constexpr double T_0 = 273.15;
constexpr double PI = 3.1415926536;
constexpr double rho_w = 1000.0, rho_s = 100.0, rho_g = 500.0, rho_i = 890.0;
constexpr double Nt_c_max = 1999.E6;
constexpr double naIN0 = 1.5E6, naIN1 = 0.5E6, naCCN0 = 300.0E6, naCCN1 = 50.0E6;
constexpr double mu_r = 0.0, mu_g = 0.0, mu_i = 0.0;
constexpr double mu_s = 0.6357, Kap0 = 490.6, Kap1 = 17.46, Lam0 = 20.78, Lam1 = 3.29;
constexpr double gonv_min = 1.E2, gonv_max = 1.E6;
constexpr double am_r = PI * rho_w / 6.0, bm_r = 3.0;
constexpr double am_s = 0.069, bm_s = 2.0;
constexpr double am_g = PI * rho_g / 6.0, bm_g = 3.0;
constexpr double am_i = PI * rho_i / 6.0, bm_i = 3.0;
constexpr double av_r = 4854.0, bv_r = 1.0, fv_r = 195.0;
constexpr double av_s = 40.0, bv_s = 0.55, fv_s = 100.0;
constexpr double av_g = 442.0, bv_g = 0.89;
constexpr double av_i = 1847.5, bv_i = 1.0;
constexpr double av_c = 0.316946E8, bv_c = 2.0;
constexpr double C_cube = 0.5, C_sqrd = 0.15;
constexpr double Ef_si = 0.05, Ef_rs = 0.95, Ef_rg = 0.75, Ef_ri = 0.95;
constexpr double R1 = 1.E-12, R2 = 1.E-6, eps = 1.E-15;
constexpr double TNO = 5.0, ATO = 0.304;
constexpr double rho_not = 101325.0 / (287.05 * 298.0);
constexpr double Sc = 0.632;
constexpr double HGFR = 235.16;
constexpr double Rv = 461.5, oRv = 1. / Rv, R = 287.04, Cp = 1004.0;
constexpr double cv = Cp - R, cvv = 1408.5, cpl = 4190.0, cpi = 2106.0;
constexpr double lsub = 2.834E6, lvap0 = 2.5E6, lfus = lsub - lvap0, olfus = 1. / lfus;
constexpr double xm0i = 1.E-12, D0c = 1.E-6, D0r = 50.E-6, D0s = 200.E-6, D0g = 250.E-6;
constexpr int nbins = 100, nbc = nbins, nbi = nbins, nbr = nbins, nbs = nbins, nbg = nbins;
constexpr int ntb_c = 37, ntb_i = 64, ntb_r = 37, ntb_s = 37, ntb_g = 37, ntb_g1 = 37, ntb_r1 = 37,
              ntb_i1 = 55, ntb_t = 9, ntb_IN = 55;
constexpr int IFDRY = 0;
} // namespace par

// Scalars and small arrays thompson_init derives (1-based like the Fortran;
// element 0 unused). Small enough to capture by value in kernels.
struct Consts {
  double Nt_c = 0, mu_c = 0, Sc3 = 0, D0i = 0, xm0s = 0, xm0g = 0;
  double cce[6][16] = {}, ccg[6][16] = {}, ocg1[16] = {}, ocg2[16] = {};
  double cie[8] = {}, cig[8] = {}, oig1 = 0, oig2 = 0, obmi = 0;
  double cre[14] = {}, crg[14] = {}, ore1 = 0, org1 = 0, org2 = 0, org3 = 0, obmr = 0;
  double cse[19] = {}, csg[19] = {}, oams = 0, obms = 0, ocms = 0;
  double cge[13] = {}, cgg[13] = {}, oge1 = 0, ogg1 = 0, ogg2 = 0, ogg3 = 0, oamg = 0, obmg = 0, ocmg = 0;
  double t1_qr_qc = 0, t1_qr_qi = 0, t2_qr_qi = 0, t1_qg_qc = 0, t1_qs_qc = 0, t1_qs_qi = 0;
  double t1_qr_ev = 0, t2_qr_ev = 0, t1_qs_sd = 0, t2_qs_sd = 0, t1_qg_sd = 0, t2_qg_sd = 0;
  double t1_qs_me = 0, t2_qs_me = 0, t1_qg_me = 0, t2_qg_me = 0;
  int nic1 = 0, nic2 = 0, nii2 = 0, nii3 = 0, nir2 = 0, nir3 = 0, nis2 = 0, nig2 = 0, nig3 = 0, niIN2 = 0;
  // bin edges the column code needs: Dr(1), Dr(nbr), Ds(1), Ds(nbs), t_Nc(1)
  double Dr1 = 0, Drn = 0, Ds1 = 0, Dsn = 0, t_Nc1 = 0;
};

// Host tables, Fortran order (first index fastest), 1-based index helpers.
struct HostTables {
  Consts c;
  // bins (index 1..nb; element 0 unused)
  std::vector<double> Dc, dtc, Di, dti, Dr, dtr, Ds, dts, Dg, dtg, t_Nc;

  // qr_acr_qg: (ntb_g1, ntb_g, ntb_r1, ntb_r)
  std::vector<double> tcg_racg, tmr_racg, tcr_gacr, tmg_gacr, tnr_racg, tnr_gacr;
  // qr_acr_qs: (ntb_s, ntb_t, ntb_r1, ntb_r)
  std::vector<double> tcs_racs1, tmr_racs1, tcs_racs2, tmr_racs2, tcr_sacr1, tms_sacr1,
      tcr_sacr2, tms_sacr2, tnr_racs1, tnr_racs2, tnr_sacr1, tnr_sacr2;
  // freezeH2O: (ntb_r, ntb_r1, 45, ntb_IN) and (ntb_c, nbc, 45, ntb_IN), but
  // only the ice-nuclei slices listed in `in_slices` are stored, each as
  // (ntb_r, ntb_r1, 45) resp. (ntb_c, nbc, 45).
  std::vector<int> in_slices;
  std::vector<double> tpi_qrfz, tni_qrfz, tpg_qrfz, tnr_qrfz, tpi_qcfz, tni_qcfz;
  // qi_aut_qs: (ntb_i, ntb_i1)
  std::vector<double> tps_iaus, tni_iaus, tpi_ide;
  // table_Efrw (nbr, nbc), table_Efsw (nbs, nbc)
  std::vector<double> t_Efrw, t_Efsw;
  // table_dropEvap: (nbc, ntb_c, nbc)
  std::vector<double> tpc_wev, tnc_wev;

  static size_t ig(int i, int j, int k, int m) {   // racg layout
    return size_t(i - 1) + par::ntb_g1 * (size_t(j - 1) + par::ntb_g * (size_t(k - 1) + par::ntb_r1 * size_t(m - 1)));
  }
  static size_t is(int i, int j, int k, int m) {   // racs layout
    return size_t(i - 1) + par::ntb_s * (size_t(j - 1) + par::ntb_t * (size_t(k - 1) + par::ntb_r1 * size_t(m - 1)));
  }
  // freezeH2O, slice number s (0-based position in in_slices), temperature index k
  static size_t irfz(int i, int j, int k, int s) {
    return size_t(i - 1) + par::ntb_r * (size_t(j - 1) + par::ntb_r1 * (size_t(k - 1) + 45 * size_t(s)));
  }
  static size_t icfz(int i, int j, int k, int s) {
    return size_t(i - 1) + par::ntb_c * (size_t(j - 1) + par::nbc * (size_t(k - 1) + 45 * size_t(s)));
  }
  static size_t i2(int i, int j, int n1) { return size_t(i - 1) + n1 * size_t(j - 1); }
  static size_t iwev(int i, int j, int k) {
    return size_t(i - 1) + par::nbc * (size_t(j - 1) + par::ntb_c * size_t(k - 1));
  }
};

// The ice-nuclei slice of freezeH2O's tables the scheme uses when it is not
// aerosol aware (CM1 always): xni = 1000 /m^3 -> idx_IN (mp_thompson).
int default_idx_IN(const Consts& c);

//---------------------------------------------------------------------------
// The pieces of thompson_init, as in the Fortran. km ranges are the
// Fortran's (zero-based) decomposition indices, so a test can build a
// subset: qr_acr_qg/qr_acr_qs km = (m-1)*ntb_r1 + (k-1) over 0..ntb_r*ntb_r1-1,
// freezeH2O km = (m-1)*45 + (k-1) over the IN slices m in `in_slices`.
//---------------------------------------------------------------------------
// constants, exponents/gamma values, bins (thompson_init before the tables);
// allocates and zeroes all tables
void init_constants(HostTables& t, double nt_c_in, std::vector<int> in_slices);
void table_Efrw(HostTables& t);
void table_Efsw(HostTables& t);
void table_dropEvap(HostTables& t);
void qr_acr_qg(HostTables& t, int km_s, int km_e);
void qr_acr_qs(HostTables& t, int km_s, int km_e);
void freezeH2O(HostTables& t, int k_s, int k_e);   // temperature index range 1..45, all in_slices
void qi_aut_qs(HostTables& t);

// Numerical Recipes helpers (double precision, as built by gfortran -r8)
double gammln(double xx);
double wgamma(double y);
double gammp(double a, double x);

// thompson_init in full (nt_c_in in 1/cm^3, the namelist nt_c).
std::shared_ptr<const HostTables> build(double nt_c_in);

// Cached build: in memory per process, and on disk in cache_dir() (unless
// disabled). A cache file is used only if a recomputed sample of its
// entries matches bit for bit; otherwise the tables are rebuilt and the
// file rewritten (atomically).
std::shared_ptr<const HostTables> get(double nt_c_in);

// $CM1K_THOMPSON_CACHE if set ("0", "off" or "" disable the disk cache),
// else $XDG_CACHE_HOME/cm1k or $HOME/.cache/cm1k.
std::string cache_dir();

//---------------------------------------------------------------------------
// Device copies of what the column code reads.
//---------------------------------------------------------------------------
struct DevTables {
  Consts c;
  int idx_IN0 = 0;   // the freezeH2O slice stored below
  LView<double****> tcg_racg, tmr_racg, tcr_gacr, tmg_gacr, tnr_racg, tnr_gacr;
  LView<double****> tcs_racs1, tmr_racs1, tcs_racs2, tmr_racs2, tcr_sacr1, tms_sacr1, tcr_sacr2,
      tms_sacr2, tnr_racs1, tnr_racs2, tnr_sacr1, tnr_sacr2;
  LView<double***> tpi_qrfz, tni_qrfz, tpg_qrfz, tnr_qrfz, tpi_qcfz, tni_qcfz;   // slice idx_IN0
  LView<double**> tps_iaus, tni_iaus, tpi_ide, t_Efrw, t_Efsw;
  LView<double***> tnc_wev;

  static DevTables upload(const HostTables& h);
};

} // namespace cm1k::thompson

#endif
