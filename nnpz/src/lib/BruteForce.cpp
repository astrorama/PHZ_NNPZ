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

namespace py = pybind11;

namespace Nnpz {

template <typename DistanceFunctor>
static void _bruteforce(PhotoArray const& reference, PhotoArray const& all_targets, ScaleArray& all_scales,
                        IndexArray& all_closest, int k, ScaleFunction* scaling) {
  if (reference.shape(1) != all_targets.shape(1)) {
    throw Elements::Exception("The number of bands for the reference and target sets do not match");
  }
  if (reference.shape(2) != 2) {
    throw Elements::Exception("The reference set is expected to have (value, error) on the third axis");
  }
  if (all_targets.shape(2) != 2) {
    throw Elements::Exception("The target set is expected to have (value, error) on the third axis");
  }

  auto u_reference = reference.unchecked<3>();
  auto u_targets   = all_targets.unchecked<3>();
  auto u_scales    = all_scales.mutable_unchecked<2>();
  auto u_closest   = all_closest.mutable_unchecked<2>();

  py::ssize_t const ntargets = all_targets.shape(0);
  py::ssize_t const nrefs    = reference.shape(0);
  py::ssize_t const nbands   = reference.shape(1);

  std::vector<float> distances_buffer(all_closest.shape(1));

  for (py::ssize_t ti = 0; ti < ntargets && PyErr_CheckSignals() == 0; ++ti) {
    MaxHeap heap(k, &distances_buffer[0], u_closest.mutable_data(ti, 0), u_scales.mutable_data(ti, 0));

    photo_t const* target_photo_ptr = u_targets.data(ti, 0, 0);

    for (index_t ri = 0; ri < nrefs; ++ri) {
      photo_t const* ref_photo_ptr = u_reference.data(ri, 0, 0);

      float scale = 1.;
      if (scaling) {
        scale = (*scaling)(ref_photo_ptr, target_photo_ptr, nbands);
      }
      float dist = DistanceFunctor::distance(scale, ref_photo_ptr, target_photo_ptr, nbands);
      insert_if_best(heap, ri, dist, scale);
    }
  }
}

void chi2_bruteforce(PhotoArray const& reference, PhotoArray const& all_targets, ScaleArray& all_scales,
                     IndexArray& all_closest, int k, ScaleFunction* scaling) {
  _bruteforce<Chi2Distance>(reference, all_targets, all_scales, all_closest, k, scaling);
}

void euclidean_bruteforce(PhotoArray const& reference, PhotoArray const& all_targets, ScaleArray& all_scales,
                          IndexArray& all_closest, int k, ScaleFunction* scaling) {
  _bruteforce<EuclideanDistance>(reference, all_targets, all_scales, all_closest, k, scaling);
}

}  // namespace Nnpz