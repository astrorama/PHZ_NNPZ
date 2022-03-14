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

#ifndef PHZ_NNPZ_DISTANCES_H
#define PHZ_NNPZ_DISTANCES_H

#include "Types.h"
#include <cmath>

namespace Nnpz {

/**
 * Implement the distance, distance derivative and scale guessing for the χ^2 distance
 */
struct Chi2Distance {
  static double distance(scale_t scale, NdArray<photo_t> const& reference, NdArray<photo_t> const& target) {
    double acc    = 0.;
    size_t nbands = reference.shape(0);

    for (size_t band_idx = 0; band_idx < nbands; ++band_idx) {
      double ref_val = scale * reference.at(band_idx, 0);
      double ref_err = scale * reference.at(band_idx, 1);
      double tar_val = target.at(band_idx, 0);
      double tar_err = target.at(band_idx, 1);
      double nom     = (ref_val - tar_val) * (ref_val - tar_val);
      double den     = ref_err * ref_err + tar_err * tar_err;
      acc += nom / den;
    }

    return acc;
  }

  static double guessScale(NdArray<photo_t> const& reference, NdArray<photo_t> const& target) {
    double nom = 0., den = 0.;
    size_t nbands = reference.shape(0);

    for (size_t band_idx = 0; band_idx < nbands; ++band_idx) {
      photo_t ref_val = reference.at(band_idx, 0);
      photo_t tar_val = target.at(band_idx, 0);
      photo_t tar_err = target.at(band_idx, 1);
      photo_t err_sqr = (tar_err * tar_err);

      nom += (ref_val * tar_val) / err_sqr;
      den += (ref_val * ref_val) / err_sqr;
    }

    return nom / den;
  }

  /**
   * Implement the derivative of the chi2 distance
   *\f[
   *    \frac{\delta}{\delta a}\left[ \frac{(a f_{ref} - f_{target})^2}{(a e_{ref})^2 + e_{target}^2}\right] = \frac{2
   *(a f_{ref} - f_{target}) (e_{ref}^2 a f_{target} + f_{ref} e_{target}^2)}{(e_{ref}^2 a^2 + e_{target}^2)^2} \f]
   */
  static double daDistance(scale_t scale, NdArray<photo_t> const& reference, NdArray<photo_t> const& target) {
    double acc    = 0.;
    size_t nbands = reference.shape(0);

    for (size_t i = 0; i < nbands; ++i) {
      photo_t ref_val = reference.at(i, 0);
      photo_t ref_err = reference.at(i, 1);
      photo_t tar_val = target.at(i, 0);
      photo_t tar_err = target.at(i, 1);

      auto ref_err_sq = ref_err * ref_err;
      auto tar_err_sq = tar_err * tar_err;

      double nom = 2 * (scale * ref_val - tar_val) * (ref_err_sq * scale * tar_val + ref_val * tar_err_sq);
      double den = ref_err_sq * scale * scale * tar_err_sq;
      acc += nom / (den * den);
    }

    return acc;
  }
};

/**
 * Implement the distance, distance derivative and scale guessing for the Euclidean distance
 */
struct EuclideanDistance {
  static double distance(scale_t scale, NdArray<photo_t> const& reference, NdArray<photo_t> const& target) {
    double acc    = 0.;
    size_t nbands = reference.shape(0);

    for (size_t band_idx = 0; band_idx < nbands; ++band_idx) {
      double ref_val = scale * reference.at(band_idx, 0);
      double tar_val = target.at(band_idx, 0);
      double d       = ref_val - tar_val;
      acc += d * d;
    }

    return std::sqrt(acc);
  }

  static double guessScale(photo_t const* reference, photo_t const* target, ssize_t nbands) {
    double acc = 0.;

    for (ssize_t band_idx = 0; band_idx < nbands; ++band_idx) {
      photo_t ref_val = reference[band_idx * 2];
      photo_t tar_val = target[band_idx * 2];

      acc += ref_val / tar_val;
    }

    return acc / static_cast<double>(nbands);
  }

  /**
   *\f[
   * \frac{\delta}{\delta a}\left[ \sqrt{\sum_i^n (a f_{ref} - f_{target})^2} \right] = \frac{a \left(\sum_i^n
   *f_{ref}^2\right) - \left( \sum_i^n f_{ref} \times f_{target} \right)}{\sqrt{\sum_i^n (a f_{ref} - f_{target})^2}}
   *\f]
   */
  static double daDistance(scale_t scale, photo_t const* reference, photo_t const* target, ssize_t nbands) {
    double den = 0., nom_sum_sqr = 0., nom_sum_prod = 0.;

    for (ssize_t band_idx = 0; band_idx < nbands; ++band_idx) {
      double ref_val = scale * reference[band_idx * 2];
      double tar_val = target[band_idx * 2];
      nom_sum_sqr += ref_val * ref_val;
      nom_sum_prod -= ref_val * tar_val;
      den += (tar_val - scale * ref_val) * (tar_val - scale * ref_val);
    }

    return (scale * nom_sum_sqr + nom_sum_prod) / std::sqrt(den);
  }
};
}  // namespace Nnpz

#endif  // PHZ_NNPZ_DISTANCES_H
