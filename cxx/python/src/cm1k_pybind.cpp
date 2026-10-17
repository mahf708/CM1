// pybind11 bindings for cm1k (module `cm1k._cm1k`; the public API is the
// `cm1k` package in ../cm1k, which wraps this module).
//
// Memory model
//   Field arrays are Kokkos views in Fortran (LayoutLeft) order with Fortran
//   lower bounds. When the model's memory space is host accessible (Serial,
//   OpenMP, CUDA UVM, ...), `get_field(name)` returns a zero-copy numpy view
//   whose base object keeps the Kokkos allocation alive. Otherwise (or with
//   copy=True) it returns a fresh numpy copy. `set_field` always copies into
//   the model's array after checking shape.
//
// Kokkos lifecycle
//   `initialize()` is idempotent; Model construction initializes Kokkos on
//   demand. `finalize()` (registered with atexit by the package) first
//   releases every live Model, then finalizes Kokkos. Kokkos cannot be
//   re-initialized after that, and released models raise on use. numpy views
//   that outlive finalize keep their memory (it is intentionally leaked
//   rather than freed after finalize, which Kokkos forbids).
#include "driver/cm1k_model.hpp"
#include "share/cm1k_dump_io.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Kokkos_Core.hpp>

#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace cm1k::python {

namespace {

constexpr bool kHostAccessible = Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible;

class PyModel;
std::set<PyModel*>& live_models() {
  static std::set<PyModel*> s;
  return s;
}

bool kokkos_finalized() { return Kokkos::is_finalized(); }

bool initialize(int num_threads, int device_id, const std::vector<std::string>& args) {
  if (Kokkos::is_initialized()) return false;
  if (kokkos_finalized())
    throw std::runtime_error("cm1k: Kokkos was already finalized in this process and cannot be re-initialized");
  std::vector<std::string> argv_s = {"cm1k"};
  argv_s.insert(argv_s.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (auto& s : argv_s) argv.push_back(s.data());
  argv.push_back(nullptr);
  int argc = int(argv_s.size());
  // Command-line style args first, then explicit settings override them.
  Kokkos::InitializationSettings settings;
  if (num_threads > 0) settings.set_num_threads(num_threads);
  if (device_id >= 0) settings.set_device_id(device_id);
  if (args.empty()) {
    Kokkos::initialize(settings);
  } else {
    if (num_threads > 0 || device_id >= 0)
      throw std::invalid_argument("cm1k.initialize: pass either args or num_threads/device_id, not both");
    Kokkos::initialize(argc, argv.data());
  }
  return true;
}

void ensure_initialized() {
  if (!Kokkos::is_initialized()) initialize(-1, -1, {});
}

class PyModel {
public:
  explicit PyModel(const Namelist& nl) {
    ensure_initialized();
    m_model = std::make_unique<Model>(nl);
    live_models().insert(this);
  }
  ~PyModel() {
    live_models().erase(this);
    release();
  }
  PyModel(const PyModel&) = delete;
  PyModel& operator=(const PyModel&) = delete;

  // Drop the C++ model. Its views must be freed while Kokkos is alive.
  void release() {
    if (!m_model) return;
    if (Kokkos::is_initialized()) m_model.reset();
    else (void)m_model.release();   // leak: freeing after finalize is an error
  }

  Model& get() const {
    if (!m_model)
      throw std::runtime_error("cm1k: this Model was released (Kokkos has been finalized)");
    return *m_model;
  }

private:
  std::unique_ptr<Model> m_model;
};

void finalize() {
  if (!Kokkos::is_initialized()) return;
  for (auto* m : live_models()) m->release();
  Kokkos::finalize();
}

void check_field(const Model& m, const std::string& name) {
  if (!m.has_field(name)) throw py::key_error("cm1k: unknown field '" + name + "'");
}

// numpy array viewing (or copying) a LayoutLeft Kokkos view.
template <typename V>
py::array to_numpy(const V& v, bool copy) {
  constexpr int R = V::rank;
  std::vector<py::ssize_t> shape(R), strides(R);
  for (int d = 0; d < R; ++d) {
    shape[d] = py::ssize_t(v.extent(d));
    strides[d] = py::ssize_t(v.stride(d) * sizeof(Real));
  }
  if constexpr (kHostAccessible) {
    if (!copy) {
      auto* keep = new V(v);
      py::capsule owner(keep, [](void* p) {
        auto* vp = static_cast<V*>(p);
        if (Kokkos::is_initialized()) delete vp;   // else leak (see header comment)
      });
      return py::array_t<Real>(shape, strides, v.data(), owner);
    }
  }
  // Copy into a fresh Fortran-ordered numpy array.
  py::array_t<Real, py::array::f_style> out(shape);
  using U = Kokkos::View<typename V::non_const_data_type, Kokkos::LayoutLeft, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  U h;
  if constexpr (R == 1) h = U(out.mutable_data(), v.extent(0));
  if constexpr (R == 3) h = U(out.mutable_data(), v.extent(0), v.extent(1), v.extent(2));
  Kokkos::deep_copy(h, v);
  Kokkos::fence();
  return std::move(out);
}

template <typename V>
void from_numpy(const std::string& name, const V& v,
                const py::array_t<Real, py::array::f_style | py::array::forcecast>& a) {
  constexpr int R = V::rank;
  auto shape_str = [&](auto get, int n) {
    std::ostringstream s;
    s << "(";
    for (int d = 0; d < n; ++d) s << get(d) << (d + 1 < n ? ", " : (n == 1 ? ",)" : ")"));
    return s.str();
  };
  bool ok = a.ndim() == R;
  for (int d = 0; ok && d < R; ++d) ok = size_t(a.shape(d)) == v.extent(d);
  if (!ok)
    throw py::value_error("cm1k: shape mismatch setting '" + name + "': expected " +
                          shape_str([&](int d) { return v.extent(d); }, R) + ", got " +
                          shape_str([&](int d) { return a.shape(d); }, int(a.ndim())));
  using U = Kokkos::View<typename V::const_data_type, Kokkos::LayoutLeft, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  U h;
  if constexpr (R == 1) h = U(a.data(), v.extent(0));
  if constexpr (R == 3) h = U(a.data(), v.extent(0), v.extent(1), v.extent(2));
  if (static_cast<const void*>(h.data()) == static_cast<const void*>(v.data())) return;   // own view
  Kokkos::deep_copy(v, h);
  Kokkos::fence();
}

py::dict bounds_dict(const Bounds& b) {
  py::dict d;
#define CM1K_B(x) d[#x] = b.x;
  CM1K_B(nx) CM1K_B(ny) CM1K_B(nz) CM1K_B(ni) CM1K_B(nj) CM1K_B(nk) CM1K_B(nkp1)
  CM1K_B(ngxy) CM1K_B(ngz)
  CM1K_B(ib) CM1K_B(ie) CM1K_B(jb) CM1K_B(je) CM1K_B(kb) CM1K_B(ke)
  CM1K_B(ibm) CM1K_B(iem) CM1K_B(jbm) CM1K_B(jem) CM1K_B(kbm) CM1K_B(kem)
  CM1K_B(ibt) CM1K_B(iet) CM1K_B(jbt) CM1K_B(jet) CM1K_B(kbt) CM1K_B(ket)
  CM1K_B(ibc) CM1K_B(iec) CM1K_B(jbc) CM1K_B(jec) CM1K_B(kbc) CM1K_B(kec)
  CM1K_B(numq) CM1K_B(nrain)
#undef CM1K_B
  return d;
}

std::string kokkos_configuration() {
  std::ostringstream s;
  if (Kokkos::is_initialized()) Kokkos::print_configuration(s, false);
  return s.str();
}

} // namespace
} // namespace cm1k::python

PYBIND11_MODULE(_cm1k, m) {
  using namespace cm1k;
  using namespace cm1k::python;
  m.doc() = "cm1k: C++/Kokkos port of CM1 (low-level bindings; use the `cm1k` package)";

  // ---------------------------------------------------------------- Kokkos
  m.def("initialize", &initialize, py::arg("num_threads") = -1, py::arg("device_id") = -1,
        py::arg("args") = std::vector<std::string>{},
        "Initialize Kokkos (no-op returning False if already initialized).");
  m.def("finalize", &finalize, "Release all models and finalize Kokkos (idempotent).");
  m.def("is_initialized", [] { return Kokkos::is_initialized(); });
  m.def("is_finalized", [] { return Kokkos::is_finalized(); });
  m.def("execution_space", [] { return std::string(ExeSpace::name()); },
        "Name of the Kokkos execution space cm1k kernels run in (e.g. 'Serial', 'OpenMP', 'Cuda').");
  m.def("memory_space", [] { return std::string(MemSpace::name()); });
  m.def("host_accessible", [] { return kHostAccessible; },
        "True when field arrays can be viewed from numpy without a copy.");
  m.def("concurrency", [] {
    if (!Kokkos::is_initialized()) throw std::runtime_error("cm1k: Kokkos is not initialized");
    return ExeSpace().concurrency();
  });
  m.def("kokkos_configuration", &kokkos_configuration);
  m.def("step_dir", &DumpReader::step_dir, py::arg("root"), py::arg("nstep"),
        "root/step_NNNNNN (dump directory naming).");

  // -------------------------------------------------------------- Namelist
  py::class_<Namelist>(m, "Namelist")
    .def(py::init<>())
    .def("copy", [](const Namelist& nl) { return Namelist(nl); })
    .def("__copy__", [](const Namelist& nl) { return Namelist(nl); })
    .def_static("from_file", &Namelist::from_file, py::arg("path"))
    .def_static("from_string", &Namelist::from_string, py::arg("text"))
    .def("has", &Namelist::has, py::arg("key"))
    .def("__contains__", &Namelist::has)
    .def("geti", &Namelist::geti, py::arg("key"), py::arg("default"))
    .def("getr", &Namelist::getr, py::arg("key"), py::arg("default"))
    .def("getb", &Namelist::getb, py::arg("key"), py::arg("default"))
    .def("gets", &Namelist::gets, py::arg("key"), py::arg("default"))
    .def("getv", &Namelist::getv, py::arg("key"), "Raw (Fortran-spelled) values of a key.")
    .def("set_raw", &Namelist::set, py::arg("key"), py::arg("value"),
         "Set a key from its Fortran spelling, e.g. set_raw('dtl', '3.0').")
    .def("group_of", &Namelist::group_of, py::arg("key"))
    .def("keys", [](const Namelist& nl) {
      std::vector<std::string> k;
      for (auto& [key, v] : nl.values()) k.push_back(key);
      return k;
    })
    .def("raw", [](const Namelist& nl) { return nl.values(); });

  // ----------------------------------------------------------------- Model
  py::class_<PyModel>(m, "Model")
    .def(py::init<const Namelist&>(), py::arg("namelist"))
    .def("init", [](PyModel& self) { self.get().init(); })
    .def("init_from_dump", [](PyModel& self, const std::string& dir) { self.get().init_from_dump(dir); },
         py::arg("step_dir"))
    .def("step", [](PyModel& self, int n) {
           if (n < 0) throw py::value_error("cm1k: step(n) needs n >= 0");
           auto& model = self.get();
           py::gil_scoped_release nogil;
           for (int s = 0; s < n; ++s) model.step();
         }, py::arg("n") = 1)
    .def("write_dump", [](const PyModel& self, const std::string& root) {
           self.get().write_dump(root);
           return DumpReader::step_dir(root, self.get().nstep());
         }, py::arg("root"), "Write all fields to root/step_NNNNNN; returns that directory.")
    .def_property_readonly("nstep", [](const PyModel& s) { return s.get().nstep(); })
    .def_property_readonly("mtime", [](const PyModel& s) { return s.get().mtime(); })
    .def_property_readonly("dt", [](const PyModel& s) { return s.get().dt(); })
    .def_property_readonly("namelist", [](const PyModel& s) { return s.get().namelist(); })
    .def_property_readonly("bounds", [](const PyModel& s) { return bounds_dict(s.get().bounds()); })
    .def_property_readonly("qname", [](const PyModel& s) { return s.get().state().qname; })
    .def_property_readonly("released", [](const PyModel& s) {
      try { s.get(); return false; } catch (...) { return true; }
    })
    .def("field_names", [](const PyModel& s) { return s.get().field_names(); })
    .def("has_field", [](const PyModel& s, const std::string& n) { return s.get().has_field(n); })
    .def("field_rank", [](const PyModel& s, const std::string& n) {
      check_field(s.get(), n);
      return s.get().field_rank(n);
    })
    .def("field_bounds", [](const PyModel& s, const std::string& n) {
           const auto& model = s.get();
           check_field(model, n);
           std::vector<int> lb, ub;
           if (model.field_rank(n) == 1) {
             auto a = model.field1(n);
             lb = {a.lb(0)}; ub = {a.ub(0)};
           } else {
             auto a = model.field3(n);
             for (int d = 0; d < 3; ++d) { lb.push_back(a.lb(d)); ub.push_back(a.ub(d)); }
           }
           return py::make_tuple(py::tuple(py::cast(lb)), py::tuple(py::cast(ub)));
         }, py::arg("name"), "(lbounds, ubounds) of a field, Fortran (inclusive) bounds.")
    .def("get_field", [](const PyModel& s, const std::string& n, bool copy) -> py::array {
           const auto& model = s.get();
           check_field(model, n);
           if (model.field_rank(n) == 1) return to_numpy(model.field1(n).view(), copy);
           return to_numpy(model.field3(n).view(), copy);
         }, py::arg("name"), py::arg("copy") = false,
         "Field as a Fortran-ordered numpy array (zero-copy view when host accessible and copy=False).")
    .def("set_field", [](const PyModel& s, const std::string& n,
                         const py::array_t<Real, py::array::f_style | py::array::forcecast>& a) {
           const auto& model = s.get();
           check_field(model, n);
           if (model.field_rank(n) == 1) from_numpy(n, model.field1(n).view(), a);
           else from_numpy(n, model.field3(n).view(), a);
         }, py::arg("name"), py::arg("array"));

  m.attr("HOST_ACCESSIBLE") = kHostAccessible;
}
