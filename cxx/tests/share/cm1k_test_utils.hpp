#ifndef CM1K_TEST_UTILS_HPP
#define CM1K_TEST_UTILS_HPP

#include "share/cm1k_dump_io.hpp"

#include <catch2/catch.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace cm1k::testing {

// Root of the Fortran reference dumps (one subdirectory per case), or "".
inline std::string ref_dir() {
  const char* e = std::getenv("CM1K_REF_DIR");
  if (e) return std::string(e);
#ifdef CM1K_DEFAULT_REF_DIR
  return CM1K_DEFAULT_REF_DIR;
#else
  return "";
#endif
}

// <ref>/<case>, or "" (and a Catch2 WARN) if references are not available.
inline std::string case_ref(const std::string& case_name) {
  const auto r = ref_dir();
  if (r.empty()) { WARN("CM1K_REF_DIR not set; skipping reference comparison"); return ""; }
  const auto p = std::filesystem::path(r) / case_name;
  if (!std::filesystem::exists(p)) { WARN("no reference dumps for case " << case_name << " in " << r); return ""; }
  return p.string();
}

inline std::string case_namelist(const std::string& case_name) {
  const char* e = std::getenv("CM1K_CASES_DIR");
#ifdef CM1K_DEFAULT_CASES_DIR
  const std::string dflt = CM1K_DEFAULT_CASES_DIR;
#else
  const std::string dflt = "cases";
#endif
  return (std::filesystem::path(e ? e : dflt) / case_name / "namelist.input").string();
}

} // namespace cm1k::testing

#endif
