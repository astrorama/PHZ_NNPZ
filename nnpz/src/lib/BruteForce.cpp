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
#include "Nnpz/Distances.h"
#include "Nnpz/MaxHeap.h"
#include <ElementsKernel/Exception.h>

namespace Nnpz {

template <typename DistanceFunctor>
static void _bruteforce(NdArray<photo_t> const& reference, NdArray<photo_t> const& all_targets,
                        NdArray<scale_t>& all_scales, NdArray<index_t>& all_closest, int k, ScaleFunction* scaling,
                        int (*cancel_callback)(void)) {
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

  if (all_closest.strides(1) != sizeof(index_t)) {
    throw Elements::Exception() << "The last axis of the neighbor indexes must be contiguous";
  }
  if (all_scales.strides(1) != sizeof(scale_t)) {
    throw Elements::Exception() << "The last axis of the scales must be contiguous";
  }

  std::vector<float> distances_buffer(all_closest.shape(1));

  for (size_t ti = 0; ti < ntargets && !cancel_callback(); ++ti) {
    MaxHeap heap(k, &distances_buffer[0], &all_closest.at(ti, 0), &all_scales.at(ti, 0));

    auto target_photo = all_targets.slice(ti);

    for (index_t ri = 0; ri < nrefs; ++ri) {
      auto ref_photo = reference.slice(ri);

      scale_t scale = 1.;
      if (scaling) {
        scale = static_cast<scale_t>((*scaling)(ref_photo, target_photo));
      }
      float dist = DistanceFunctor::distance(scale, ref_photo, target_photo);
      insert_if_best(heap, ri, dist, scale);
    }
  }
}

void chi2_bruteforce(NdArray<photo_t> const& reference, NdArray<photo_t> const& all_targets,
                     NdArray<scale_t>& all_scales, NdArray<index_t>& all_closest, int k, ScaleFunction* scaling,
                     int (*cancel_callback)(void)) {
  _bruteforce<Chi2Distance>(reference, all_targets, all_scales, all_closest, k, scaling, cancel_callback);
}

void euclidean_bruteforce(NdArray<photo_t> const& reference, NdArray<photo_t> const& all_targets,
                          NdArray<scale_t>& all_scales, NdArray<index_t>& all_closest, int k, ScaleFunction* scaling,
                          int (*cancel_callback)(void)) {
  _bruteforce<EuclideanDistance>(reference, all_targets, all_scales, all_closest, k, scaling, cancel_callback);
}

}  // namespace Nnpz