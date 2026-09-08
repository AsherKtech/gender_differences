// group_valid.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>

namespace py = pybind11;

py::array_t<bool> group_any_valid(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> missing_codes,
    int64_t n_groups
) {
    auto v = values.unchecked<2>();       // (n_rows, n_cols)
    auto g = group_ids.unchecked<1>();    // (n_rows,)
    auto m = missing_codes.unchecked<1>(); // missing-code list

    ssize_t n_rows = v.shape(0);
    ssize_t n_cols = v.shape(1);
    ssize_t n_missing = m.shape(0);

    // output: n_groups x n_cols, init false
    py::array_t<bool> result({n_groups, (int64_t)n_cols});
    auto r = result.mutable_unchecked<2>();
    for (int64_t i = 0; i < n_groups; i++)
        for (ssize_t j = 0; j < n_cols; j++)
            r(i, j) = false;

    for (ssize_t i = 0; i < n_rows; i++) {
        int64_t gid = g(i);
        for (ssize_t j = 0; j < n_cols; j++) {
            if (r(gid, j)) continue;  // already known valid, skip
            double val = v(i, j);
            if (std::isnan(val)) continue;
            bool is_missing = false;
            for (ssize_t k = 0; k < n_missing; k++) {
                if (val == m(k)) { is_missing = true; break; }
            }
            if (!is_missing) r(gid, j) = true;
        }
    }
    return result;
}

PYBIND11_MODULE(group_valid, m) {
    m.def("group_any_valid", &group_any_valid);
}