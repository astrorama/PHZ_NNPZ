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
#include "Nnpz/MaxHeap.h"
#include <ElementsKernel/Exception.h>

namespace py = pybind11;

namespace Nnpz {

static float computeScale(photo_t const* reference, photo_t const* target, py::ssize_t nbands) {
  double nom = 0., den = 0.;

  for (py::ssize_t band_idx = 0; band_idx < nbands; ++band_idx) {
    photo_t ref_val = reference[band_idx * 2];
    photo_t tar_val = target[band_idx * 2];
    photo_t tar_err = target[band_idx * 2 + 1];
    photo_t err_sqr = (tar_err * tar_err);

    nom += (ref_val * tar_val) / err_sqr;
    den += (ref_val * ref_val) / err_sqr;
  }

  return static_cast<float>(std::min(1e5, std::max(1e-5, nom / den)));
}

struct Chi2Distance {
  static float distance(scale_t scale, photo_t const* reference, photo_t const* target, py::ssize_t nbands) {
    double acc = 0.;

    for (py::ssize_t band_idx = 0; band_idx < nbands; ++band_idx) {
      double ref_val = scale * reference[band_idx * 2];
      double ref_err = scale * reference[band_idx * 2 + 1];
      double tar_val = target[band_idx * 2];
      double tar_err = target[band_idx * 2 + 1];
      double nom     = (ref_val - tar_val) * (ref_val - tar_val);
      double den     = ref_err * ref_err + tar_err * tar_err;
      acc += nom / den;
    }

    return static_cast<float>(acc);
  }
};

struct EuclideanDistance {
  static float distance(scale_t scale, photo_t const* reference, photo_t const* target, py::ssize_t nbands) {
    double acc = 0.;

    for (py::ssize_t band_idx = 0; band_idx < nbands; ++band_idx) {
      double ref_val = scale * reference[band_idx * 2];
      double tar_val = target[band_idx * 2];
      double d       = ref_val - tar_val;
      acc += d * d;
    }

    return static_cast<float>(std::sqrt(acc));
  }
};

template <typename DistanceFunctor>
static void _bruteforce(PhotoArray const& reference, PhotoArray const& all_targets, ScaleArray& all_scales,
                        IndexArray& all_closest, int k, bool scaling) {
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

  for (py::ssize_t ti = 0; ti < ntargets; ++ti) {
    MaxHeap heap(k, &distances_buffer[0], u_closest.mutable_data(ti, 0), u_scales.mutable_data(ti, 0));

    photo_t const* target_photo_ptr = u_targets.data(ti, 0, 0);

    for (index_t ri = 0; ri < nrefs; ++ri) {
      photo_t const* ref_photo_ptr = u_reference.data(ri, 0, 0);

      float scale = 1.;
      if (scaling) {
        scale = computeScale(ref_photo_ptr, target_photo_ptr, nbands);
      }
      float dist = DistanceFunctor::distance(scale, ref_photo_ptr, target_photo_ptr, nbands);
      insert_if_best(heap, ri, dist, scale);
    }
  }
}

void chi2_bruteforce(PhotoArray const& reference, PhotoArray const& all_targets, ScaleArray& all_scales,
                     IndexArray& all_closest, int k, bool scaling) {
  _bruteforce<Chi2Distance>(reference, all_targets, all_scales, all_closest, k, scaling);
}

void euclidean_bruteforce(PhotoArray const& reference, PhotoArray const& all_targets, ScaleArray& all_scales,
                          IndexArray& all_closest, int k, bool scaling) {
  _bruteforce<EuclideanDistance>(reference, all_targets, all_scales, all_closest, k, scaling);
}

}  // namespace Nnpz