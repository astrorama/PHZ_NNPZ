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

#include "Weights.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace Nnpz;

PYBIND11_MODULE(_Nnpz, m) {
  m.doc() = "Nnpz helper functions";

  py::class_<WeightCalculator>(m, "WeightCalculator")
      .def(py::init<std::string const&, std::string const&>())
      .def("__call__", &WeightCalculator::operator());
}
