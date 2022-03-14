/*
 * Copyright (C) 2022 Euclid Science Ground Segment
 *
 * This library is free software; you can redistribute it and/or modify it under the terms of
 * the GNU Lesser General Public License as published by the Free Software Foundation;
 * either version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with this library;
 * if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301 USA
 */

#include "Nnpz/BruteForce.h"
#include "Nnpz/Scaling.h"
#include "Nnpz/Weights.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace Nnpz;

template <typename T>
class BufferContainer {
public:
  explicit BufferContainer(py::buffer_info const& info) : m_ptr(static_cast<T*>(info.ptr)), m_size(info.size) {}
  BufferContainer(const BufferContainer&) = default;
  BufferContainer(BufferContainer&&)      = default;

  T* data() const {
    return m_ptr;
  }

  std::size_t size() const {
    return m_size;
  }

  void resize(std::size_t) {
    throw std::runtime_error("Buffer wrapper can not be resized");
  }

private:
  T*          m_ptr;
  std::size_t m_size;
};

template <typename T>
const NdArray<T> bufferToNdArray(py::buffer_info const& buffer, const std::string& repr, unsigned axes) {
  if (buffer.format != py::format_descriptor<T>::format()) {
    throw std::runtime_error("Expecting " + py::format_descriptor<T>::format() + " for the " + repr + " buffer");
  }
  if (buffer.ndim != axes) {
    throw std::runtime_error("Expecting " + std::to_string(axes) + " axes for the " + repr + " buffer, got " +
                             std::to_string(buffer.ndim));
  }
  std::vector<size_t> shape(buffer.shape.begin(), buffer.shape.end());
  std::vector<size_t> strides(buffer.strides.begin(), buffer.strides.end());
  return {std::move(shape), std::move(strides), BufferContainer<T>(buffer)};
}

double scale_function_wrapper(ScaleFunction const& scale_func, py::buffer const& reference, py::buffer const& target) {
  py::buffer_info ref_info    = reference.request();
  py::buffer_info target_info = target.request();

  auto ref_ndarray    = bufferToNdArray<photo_t>(ref_info, "reference", 2);
  auto target_ndarray = bufferToNdArray<photo_t>(target_info, "target", 2);

  return scale_func(ref_ndarray, target_ndarray);
}

template <void Bruteforce(NdArray<photo_t> const&, NdArray<photo_t> const&, NdArray<scale_t>&, NdArray<index_t>&, int,
                          ScaleFunction*, int (*)(void))>
static void _bruteforce_wrapper(py::buffer const& reference, py::buffer const& all_targets, py::buffer& all_scales,
                                py::buffer& all_closest, int k, ScaleFunction* scaling) {
  py::buffer_info ref_info     = reference.request();
  py::buffer_info target_info  = all_targets.request();
  py::buffer_info scale_info   = all_scales.request(true);
  py::buffer_info closest_info = all_closest.request(true);

  auto ref_ndarray     = bufferToNdArray<photo_t>(ref_info, "reference", 3);
  auto target_ndarray  = bufferToNdArray<photo_t>(target_info, "target", 3);
  auto scales_ndarray  = bufferToNdArray<scale_t>(scale_info, "scales", 2);
  auto closest_ndarray = bufferToNdArray<index_t>(closest_info, "indexes", 2);

  Bruteforce(ref_ndarray, target_ndarray, scales_ndarray, closest_ndarray, k, scaling, &PyErr_CheckSignals);
}

static void weight_function_wrapper(WeightCalculator const& calculator, py::buffer const& neighbors,
                                    py::buffer const& target, py::buffer& out_weights, py::buffer& out_flags) {
  py::buffer_info neighbor_info = neighbors.request();
  py::buffer_info target_info   = target.request();
  py::buffer_info weight_info   = out_weights.request(true);
  py::buffer_info flag_info     = out_flags.request(true);

  auto neighbor_ndarray = bufferToNdArray<photo_t>(neighbor_info, "neighbors", 4);
  auto target_ndarray   = bufferToNdArray<photo_t>(target_info, "target", 3);
  auto weight_ndarray   = bufferToNdArray<weight_t>(weight_info, "weights", 2);
  auto flag_ndarray     = bufferToNdArray<flag_t>(flag_info, "flags", 1);

  calculator(neighbor_ndarray, target_ndarray, weight_ndarray, flag_ndarray);
}

PYBIND11_MODULE(_Nnpz, m) {
  m.doc() = "Nnpz helper functions";

  py::class_<ScaleFunction, std::shared_ptr<ScaleFunction>>(m, "ScaleFunction")
      .def("__call__", &scale_function_wrapper);

  py::class_<WeightCalculator>(m, "WeightCalculator")
      .def(py::init<std::string const&, std::string const&>())
      .def("__call__", &weight_function_wrapper);

  m.def("chi2_bruteforce", &_bruteforce_wrapper<chi2_bruteforce>);
  m.def("euclidean_bruteforce", &_bruteforce_wrapper<euclidean_bruteforce>);

  m.def("scaling_factory", &scaleFunctionFactory);

  py::class_<ScaleFunctionParams>(m, "ScaleFunctionParams")
      .def(py::init<std::size_t, double>())
      .def_readwrite("maxiter", &ScaleFunctionParams::maxiter)
      .def_readwrite("tolerance", &ScaleFunctionParams::tolerance);
}
