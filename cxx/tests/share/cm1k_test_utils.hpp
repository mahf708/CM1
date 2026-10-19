#ifndef CM1K_TEST_UTILS_HPP
#define CM1K_TEST_UTILS_HPP

#include "share/cm1k_dump_io.hpp"

#include <catch2/catch.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

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

// True when CM1K_REQUIRE_REFS is set (CI): missing references fail instead of skipping.
inline bool refs_required() {
  const char* e = std::getenv("CM1K_REQUIRE_REFS");
  return e && *e && std::string(e) != "0";
}

// <ref>/<case>, or "" (and a Catch2 WARN) if references are not available.
inline std::string case_ref(const std::string& case_name) {
  const auto r = ref_dir();
  if (r.empty()) {
    if (refs_required()) FAIL("CM1K_REQUIRE_REFS is set but CM1K_REF_DIR is not");
    WARN("CM1K_REF_DIR not set; skipping reference comparison");
    return "";
  }
  const auto p = std::filesystem::path(r) / case_name;
  if (!std::filesystem::exists(p)) {
    if (refs_required()) FAIL("no reference dumps for case " << case_name << " in " << r);
    WARN("no reference dumps for case " << case_name << " in " << r);
    return "";
  }
  return p.string();
}

// Last dumped step a step test compares (CM1K_MAX_STEP; default: all). The
// slow bounds-checked debug build uses it to keep run times bounded; the
// first dump after the start is always compared.
inline int max_step() {
  const char* e = std::getenv("CM1K_MAX_STEP");
  return (e && *e) ? std::atoi(e) : 1 << 30;
}

inline void limit_steps(std::vector<int>& steps) {
  const int lim = max_step();
  std::vector<int> kept;
  for (size_t n = 0; n < steps.size(); ++n)
    if (n == 0 || steps[n] <= lim) kept.push_back(steps[n]);
  steps.swap(kept);
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

// Points the native initialization at the extra input files of a case (the
// input_sounding read for isnd = 7, 17: CM1K_INPUT_SOUNDING), as make_refs.sh
// copies them next to the namelist for the Fortran run.
inline void use_case_inputs(const std::string& case_name) {
  const auto snd = std::filesystem::path(case_namelist(case_name)).parent_path() / "input_sounding";
  if (std::filesystem::exists(snd)) setenv("CM1K_INPUT_SOUNDING", snd.string().c_str(), 1);
  else unsetenv("CM1K_INPUT_SOUNDING");
}

} // namespace cm1k::testing

#endif
