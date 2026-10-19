#ifndef CM1K_NAMELIST_HPP
#define CM1K_NAMELIST_HPP

#include "cm1k_types.hpp"

#include <map>
#include <string>
#include <vector>

namespace cm1k {

// Minimal reader for CM1's Fortran namelist.input files.
//
// Keys are lower-cased and unique across groups (true for CM1), so lookups
// take only the variable name. Values keep their Fortran spelling; the typed
// getters convert them (".true."/"T" -> true, "1.0e5"/"1.0d5" -> 1e5, ...).
// Missing keys fall back to the default passed by the caller, which should be
// the value param.F would use.
class Namelist {
public:
  Namelist() = default;
  static Namelist from_file(const std::string& path);
  static Namelist from_string(const std::string& text);

  bool has(const std::string& key) const;
  int geti(const std::string& key, int def) const;
  Real getr(const std::string& key, Real def) const;
  bool getb(const std::string& key, bool def) const;
  std::string gets(const std::string& key, const std::string& def) const;
  std::vector<std::string> getv(const std::string& key) const;

  // Override or add a value (used by tests and the Python bindings).
  void set(const std::string& key, const std::string& value);

  // group name for a key ("" if unknown)
  std::string group_of(const std::string& key) const;
  const std::map<std::string, std::vector<std::string>>& values() const { return m_values; }

  // Directory of the namelist file (from_file), "" otherwise. CM1 reads its
  // extra input files (input_grid_z, ...) from the run directory; cm1k looks
  // for them next to the namelist, or in the working directory.
  const std::string& dir() const { return m_dir; }
  void set_dir(const std::string& d) { m_dir = d; }
  // path of an extra input file: dir()/name, or name when dir() is empty
  std::string input_path(const std::string& name) const;

private:
  std::map<std::string, std::vector<std::string>> m_values;
  std::map<std::string, std::string> m_group;
  std::string m_dir;
};

} // namespace cm1k

#endif
