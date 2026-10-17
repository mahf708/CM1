#include "cm1k_namelist.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cm1k {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// Split on commas outside quotes.
std::vector<std::string> split_values(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  char q = 0;
  for (char c : s) {
    if (q) { cur += c; if (c == q) q = 0; continue; }
    if (c == '\'' || c == '"') { q = c; cur += c; continue; }
    if (c == ',') { out.push_back(trim(cur)); cur.clear(); continue; }
    cur += c;
  }
  if (!trim(cur).empty()) out.push_back(trim(cur));
  out.erase(std::remove_if(out.begin(), out.end(), [](const std::string& v) { return v.empty(); }), out.end());
  return out;
}

std::string unquote(const std::string& v) {
  if (v.size() >= 2 && (v.front() == '\'' || v.front() == '"') && v.back() == v.front())
    return v.substr(1, v.size() - 2);
  return v;
}

} // namespace

Namelist Namelist::from_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cm1k: cannot open namelist file '" + path + "'");
  std::stringstream ss;
  ss << f.rdbuf();
  return from_string(ss.str());
}

Namelist Namelist::from_string(const std::string& text) {
  Namelist nl;
  // Strip comments ('!' outside quotes), join everything into a token stream.
  std::string clean;
  {
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) {
      char q = 0;
      std::string l;
      for (char c : line) {
        if (q) { l += c; if (c == q) q = 0; continue; }
        if (c == '\'' || c == '"') { q = c; l += c; continue; }
        if (c == '!') break;
        l += c;
      }
      clean += l + "\n";
    }
  }

  std::string group;
  size_t pos = 0;
  while (pos < clean.size()) {
    const auto amp = clean.find('&', pos);
    if (amp == std::string::npos) break;
    auto eol = clean.find_first_of(" \t\r\n", amp);
    group = lower(clean.substr(amp + 1, eol - amp - 1));
    // group body ends at a '/' outside quotes
    size_t end = eol;
    char q = 0;
    for (; end < clean.size(); ++end) {
      const char c = clean[end];
      if (q) { if (c == q) q = 0; continue; }
      if (c == '\'' || c == '"') { q = c; continue; }
      if (c == '/') break;
    }
    const std::string body = clean.substr(eol, end - eol);
    pos = end + 1;

    // Parse "key = values ..." assignments; a key starts where an identifier
    // is followed by '=' (outside quotes).
    std::vector<std::pair<size_t, size_t>> eqs; // (key_begin, eq_pos)
    q = 0;
    for (size_t i = 0; i < body.size(); ++i) {
      const char c = body[i];
      if (q) { if (c == q) q = 0; continue; }
      if (c == '\'' || c == '"') { q = c; continue; }
      if (c == '=') {
        size_t k = i;
        while (k > 0 && std::isspace((unsigned char)body[k - 1])) --k;
        size_t kb = k;
        while (kb > 0 && (std::isalnum((unsigned char)body[kb - 1]) || body[kb - 1] == '_' ||
                          body[kb - 1] == '(' || body[kb - 1] == ')' || body[kb - 1] == '%'))
          --kb;
        eqs.emplace_back(kb, i);
      }
    }
    for (size_t n = 0; n < eqs.size(); ++n) {
      const auto [kb, eq] = eqs[n];
      const size_t vend = (n + 1 < eqs.size()) ? eqs[n + 1].first : body.size();
      const std::string key = lower(trim(body.substr(kb, eq - kb)));
      const std::string val = body.substr(eq + 1, vend - eq - 1);
      nl.m_values[key] = split_values(val);
      nl.m_group[key] = group;
    }
  }
  return nl;
}

bool Namelist::has(const std::string& key) const { return m_values.count(lower(key)) > 0; }

std::vector<std::string> Namelist::getv(const std::string& key) const {
  auto it = m_values.find(lower(key));
  if (it == m_values.end()) return {};
  return it->second;
}

std::string Namelist::gets(const std::string& key, const std::string& def) const {
  auto v = getv(key);
  return v.empty() ? def : unquote(v[0]);
}

int Namelist::geti(const std::string& key, int def) const {
  auto v = getv(key);
  if (v.empty()) return def;
  try {
    return std::stoi(v[0]);
  } catch (...) {
    throw std::runtime_error("cm1k: namelist value for '" + key + "' is not an integer: " + v[0]);
  }
}

Real Namelist::getr(const std::string& key, Real def) const {
  auto v = getv(key);
  if (v.empty()) return def;
  std::string s = lower(v[0]);
  std::replace(s.begin(), s.end(), 'd', 'e');
  try {
    return std::stod(s);
  } catch (...) {
    throw std::runtime_error("cm1k: namelist value for '" + key + "' is not a real: " + v[0]);
  }
}

bool Namelist::getb(const std::string& key, bool def) const {
  auto v = getv(key);
  if (v.empty()) return def;
  std::string s = lower(v[0]);
  if (!s.empty() && s[0] == '.') s = s.substr(1);
  if (!s.empty() && s[0] == 't') return true;
  if (!s.empty() && s[0] == 'f') return false;
  throw std::runtime_error("cm1k: namelist value for '" + key + "' is not a logical: " + v[0]);
}

void Namelist::set(const std::string& key, const std::string& value) {
  m_values[lower(key)] = split_values(value);
}

std::string Namelist::group_of(const std::string& key) const {
  auto it = m_group.find(lower(key));
  return it == m_group.end() ? "" : it->second;
}

} // namespace cm1k
