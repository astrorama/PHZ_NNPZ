/*
 * Copyright (C) 2022 Euclid Science Ground Segment
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 3.0 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "Nnpz/Weights.h"
#include "Nnpz/Distances.h"
#include <ElementsKernel/Exception.h>
#include <iostream>
#include <map>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace Nnpz {

constexpr float  kMinWeight              = std::numeric_limits<float>::min();
constexpr flag_t ALTERNATIVE_WEIGHT_FLAG = 1;

struct Likelihood {
  static weight_t weight(PhotoArray const& ref_obj, PhotoArray const& target_obj) {
    auto        u_ref    = ref_obj.unchecked<2>();
    auto        u_target = target_obj.unchecked<2>();
    py::ssize_t nbands   = target_obj.shape(0);

    double chi2 = 0.;
    for (py::ssize_t bi = 0; bi < nbands; ++bi) {
      double diff = u_ref(bi, 0) - u_target(bi, 0);
      double nom  = diff * diff;
      double den  = (u_ref(bi, 1) * u_ref(bi, 1)) + (u_target(bi, 1) * u_target(bi, 1));
      chi2 += nom / den;
    }
    return static_cast<weight_t>(std::exp(-0.5f * chi2));
  }
};

struct InverseChi2 {
  static weight_t weight(PhotoArray const& ref_obj, PhotoArray const& target_obj) {
    return 1.f / Chi2Distance::distance(1., ref_obj.data(0), target_obj.data(0), ref_obj.size());
  }
};

struct InverseEuclidean {
  static weight_t weight(PhotoArray const& ref_obj, PhotoArray const& target_obj) {
    return 1.f / EuclideanDistance::distance(1., ref_obj.data(0), target_obj.data(0), ref_obj.size());
  }
};

template <typename WeightFunctor>
static void computeWeights(PhotoArray const& ref_objs, PhotoArray const& target_obj, WeightArray& out_weight) {
  auto w = out_weight.mutable_unchecked<1>();
  for (py::ssize_t ni = 0; ni < ref_objs.shape(0); ++ni) {
    w(ni) = WeightFunctor::weight(PhotoArray(ref_objs[py::make_tuple(ni)]), target_obj);
  }
}

static std::map<std::string, WeightFunc> kWeightFunctions{{"Euclidean", computeWeights<InverseEuclidean>},
                                                          {"Chi2", computeWeights<InverseChi2>},
                                                          {"Likelihood", computeWeights<Likelihood>}};

WeightCalculator::WeightCalculator(std::string const& primary, std::string const& secondary) {
  auto i = kWeightFunctions.find(primary);
  if (i == kWeightFunctions.end()) {
    throw Elements::Exception("Primary weight function ") << primary << " unknown";
  }
  m_primary = i->second;
  i         = kWeightFunctions.find(secondary);
  if (i == kWeightFunctions.end()) {
    throw Elements::Exception("Secondary weight function") << secondary << " unknown";
  }
  m_secondary = i->second;
}

template <typename T1, typename T2, int F1, int F2>
static void checkShapeEqual(py::array_t<T1, F1> const& a1, py::array_t<T2, F2> const& a2, int axis1, int axis2,
                            const std::string& v1, const std::string& v2) {
  if (a1.shape(axis1) != a2.shape(axis2)) {
    throw Elements::Exception() << "Axis " << axis1 << " of " << v1 << " does not match the axis " << axis2 << " of "
                                << v2;
  }
}

void WeightCalculator::operator()(PhotoArray const& neighbors, PhotoArray const& target, WeightArray& out_weights,
                                  FlagArray& out_flags) const {
  checkShapeEqual(neighbors, target, 0, 0, "neighbors", "target");
  checkShapeEqual(neighbors, target, 2, 1, "neighbors", "target");
  checkShapeEqual(neighbors, out_weights, 0, 0, "neighbors", "weights");
  checkShapeEqual(neighbors, out_weights, 1, 1, "neighbors", "weights");
  checkShapeEqual(out_weights, out_flags, 0, 0, "weights", "flags");

  if (neighbors.shape(3) != 2) {
    throw Elements::Exception() << "Third axis for the neighbor photometry must be 2";
  }
  if (target.shape(2) != 2) {
    throw Elements::Exception() << "Second axis for the target photometry must be 2";
  }
  if (out_weights.strides(1) != sizeof(weight_t)) {
    throw Elements::Exception() << "The neighbor weights must be contiguous per target object";
  }

  auto        u_out_flags   = out_flags.mutable_unchecked<1>();
  auto        u_out_weights = out_weights.mutable_unchecked<2>();
  py::ssize_t k             = out_weights.shape(1);

  py::ssize_t out_size = out_weights.shape(0);
  for (py::ssize_t i = 0; i < out_size; ++i) {
    weight_t* weight_ptr = u_out_weights.mutable_data(i, 0);

    auto idx = py::make_tuple(i);

    PhotoArray  nn_row(neighbors[idx]);
    PhotoArray  target_row(target[idx]);
    WeightArray weight_row(out_weights[idx]);

    m_primary(nn_row, target_row, weight_row);

    if (std::all_of(weight_ptr, weight_ptr + k, [](weight_t w) { return w < kMinWeight; })) {
      m_secondary(nn_row, target_row, weight_row);
      u_out_flags(i) |= ALTERNATIVE_WEIGHT_FLAG;
    }
  }
}
}  // namespace Nnpz
