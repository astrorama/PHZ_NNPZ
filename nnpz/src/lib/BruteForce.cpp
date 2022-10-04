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
#include <ElementsKernel/Exception.h>
#include <MathUtils/distances/Distances.h>

using namespace Euclid::MathUtils;

namespace Nnpz {

namespace {
struct NeighborTriplet {
  index_t index    = 0;
  float   distance = 0;
  scale_t scale    = 1;

  bool operator<(const NeighborTriplet& other) const {
    return distance < other.distance;
  }
};

void insert_if_best(size_t k, std::vector<NeighborTriplet>& heap, const NeighborTriplet& neighbor) {
  if (heap.size() < k) {
    heap.emplace_back(neighbor);
    std::push_heap(heap.begin(), heap.end());
  } else if (neighbor < heap.front()) {
    std::pop_heap(heap.begin(), heap.end());
    heap.back() = neighbor;
    std::push_heap(heap.begin(), heap.end());
  }
}

template <typename DistanceFunctor>
void _bruteforce(NdArray<photo_t> const& reference, NdArray<photo_t> const& all_targets, NdArray<scale_t>& all_scales,
                 NdArray<index_t>& all_closest, size_t k, ScaleFunction const* scaling, int (*cancel_callback)(void)) {
  size_t const ntargets = all_targets.shape(0);
  size_t const nrefs    = reference.shape(0);

  if (ntargets != all_scales.shape(0)) {
    throw Elements::Exception() << "The first axis for the scale array does not match the number of targets: "
                                << all_scales.shape(0) << " vs " << ntargets;
  }
  if (ntargets != all_closest.shape(0)) {
    throw Elements::Exception() << "The first axis for the index array does not match the number of targets: "
                                << all_scales.shape(0) << " vs " << ntargets;
  }

  if (reference.shape(1) != all_targets.shape(1)) {
    throw Elements::Exception("The number of bands for the reference and target sets do not match");
  }
  if (reference.shape(2) != 2) {
    throw Elements::Exception("The reference set is expected to have (value, error) on the third axis");
  }
  if (all_targets.shape(2) != 2) {
    throw Elements::Exception("The target set is expected to have (value, error) on the third axis");
  }

  if (reference.strides(2) != sizeof(photo_t) && reference.strides(1) != 2 * sizeof(photo_t)) {
    throw Elements::Exception() << "The last two axes of the reference photometry must be contiguous";
  }
  if (all_targets.strides(2) != sizeof(photo_t) && all_targets.strides(1) != 2 * sizeof(photo_t)) {
    throw Elements::Exception() << "The last two axes of the target photometry must be contiguous";
  }
  if (all_closest.strides(1) != sizeof(index_t)) {
    throw Elements::Exception() << "The last axis of the neighbor indexes must be contiguous";
  }
  if (all_scales.strides(1) != sizeof(scale_t)) {
    throw Elements::Exception() << "The last axis of the scales must be contiguous";
  }

  std::vector<NeighborTriplet> heap;
  size_t const                 nbands = reference.shape(1);
  heap.reserve(all_closest.shape(1));

  for (size_t ti = 0; ti < ntargets && !cancel_callback(); ++ti) {
    auto target_photo = all_targets.slice(ti);
    auto ref_photo    = reference.slice(0);

    heap.clear();

    for (index_t ri = 0; ri < nrefs; ++ri) {
      scale_t scale = 1.;
      if (scaling) {
        scale = static_cast<scale_t>((*scaling)(ref_photo, target_photo));
      }
      auto ref_begin    = PhotoPtrIterator(&ref_photo.front());
      auto ref_end      = ref_begin + nbands;
      auto target_begin = PhotoPtrIterator(&target_photo.front());
      auto dist         = static_cast<float>(DistanceFunctor::distance(scale, ref_begin, ref_end, target_begin));
      insert_if_best(k, heap, {ri, dist, scale});
      ref_photo.next_slice();
    }
    std::transform(heap.begin(), heap.end(), &all_closest.at(ti, 0), [](const NeighborTriplet& t) { return t.index; });
    std::transform(heap.begin(), heap.end(), &all_scales.at(ti, 0), [](const NeighborTriplet& t) { return t.scale; });
  }
}
}  // namespace

void chi2_bruteforce(NdArray<photo_t> const& reference, NdArray<photo_t> const& all_targets,
                     NdArray<scale_t>& all_scales, NdArray<index_t>& all_closest, size_t k,
                     ScaleFunction const* scaling, int (*cancel_callback)(void)) {
  _bruteforce<Chi2Distance>(reference, all_targets, all_scales, all_closest, k, scaling, cancel_callback);
}

void euclidean_bruteforce(NdArray<photo_t> const& reference, NdArray<photo_t> const& all_targets,
                          NdArray<scale_t>& all_scales, NdArray<index_t>& all_closest, size_t k,
                          ScaleFunction const* scaling, int (*cancel_callback)(void)) {
  _bruteforce<EuclideanDistance>(reference, all_targets, all_scales, all_closest, k, scaling, cancel_callback);
}

}  // namespace Nnpz
