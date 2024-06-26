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
#include <ElementsKernel/Exception.h>
#include <MathUtils/distances/Weights.h>
#include <limits>
#include <map>

using namespace Euclid::MathUtils;

namespace Nnpz {

constexpr float kMinWeight = std::numeric_limits<float>::min();
namespace {
template <typename WeightFunctor>
void computeWeights(NdArray<photo_t> const& ref_objs, NdArray<photo_t> const& target_obj,
                    NdArray<weight_t>& out_weight) {
  size_t const k      = ref_objs.shape(0);
  size_t const nbands = ref_objs.shape(1);

  PhotoPtrIterator target_begin(&target_obj.at(0, 0));
  for (size_t ni = 0; ni < k; ++ni) {
    PhotoPtrIterator ref_begin(&ref_objs.at(ni, 0, 0));
    PhotoPtrIterator ref_end(ref_begin + nbands);
    out_weight.at(ni) = static_cast<weight_t>(WeightFunctor::weight(1.f, ref_begin, ref_end, target_begin));
  }
}

template <typename T1, typename T2>
void checkShapeEqual(NdArray<T1> const& a1, NdArray<T2> const& a2, int axis1, int axis2, const std::string& v1,
                     const std::string& v2) {
  if (a1.shape(axis1) != a2.shape(axis2)) {
    throw Elements::Exception() << "Axis " << axis1 << " of " << v1 << " does not match the axis " << axis2 << " of "
                                << v2;
  }
}

const std::map<std::string, WeightFunc, std::less<>> kWeightFunctions{{"Euclidean", computeWeights<InverseEuclidean>},
                                                                      {"Chi2", computeWeights<InverseChi2>},
                                                                      {"Likelihood", computeWeights<Likelihood>}};

}  // namespace

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

void WeightCalculator::operator()(NdArray<photo_t> const& neighbors, NdArray<photo_t> const& target,
                                  NdArray<weight_t>& out_weights, NdArray<flag_t>& out_flags) const {
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

  size_t out_size = out_weights.shape(0);

  for (size_t i = 0; i < out_size; ++i) {

    auto nn_row     = neighbors.slice(i);
    auto target_row = target.slice(i);
    auto weight_row = out_weights.slice(i);

    m_primary(nn_row, target_row, weight_row);

    if (std::all_of(weight_row.begin(), weight_row.end(), [](weight_t w) { return w < kMinWeight; })) {
      m_secondary(nn_row, target_row, weight_row);
      out_flags.at(i) |= ALTERNATIVE_WEIGHT_FLAG;
    }
  }
}
}  // namespace Nnpz
