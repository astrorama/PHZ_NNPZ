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

#ifndef PHZ_NNPZ_WEIGHTS_H
#define PHZ_NNPZ_WEIGHTS_H

#include "Nnpz/Types.h"
#include <string>

namespace Nnpz {

constexpr flag_t ALTERNATIVE_WEIGHT_FLAG = 1;

/**
 * Interface for the weight calculator
 */
class WeightCalculator {
public:
  /**
   * @param primary Primary weighting method
   * @param secondary Weighting method to be used if the primary weights are too small (under
   * std::numeric_limits<float>::min())
   */
  WeightCalculator(std::string const& primary, std::string const& secondary);

  void operator()(PhotoArray const& neighbors, PhotoArray const& target, WeightArray& out_weights,
                  FlagArray& out_flags) const;

private:
  WeightFunc m_primary, m_secondary;
};
}  // namespace Nnpz

#endif  // PHZ_NNPZ_WEIGHTS_H
