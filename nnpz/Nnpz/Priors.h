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

#ifndef PHZ_NNPZ_PRIORS_H
#define PHZ_NNPZ_PRIORS_H

#include "AlexandriaKernel/memory_tools.h"
#include <limits>

namespace Nnpz {

/**
 * Prior interface
 */
class Prior {
public:
  virtual ~Prior() = default;

  /**
   * @return Minimum and maximum value acceptable for the prior. Values outside this range
   *    should be clipped.
   */
  virtual std::pair<double, double> getValidRange() const = 0;

  /**
   * @param x
   * @return The prior value for x
   */
  virtual double operator()(double x) const = 0;

  /**
   * @param x
   * @return The derivative of the prior at x
   */
  virtual double dx(double x) const = 0;
};

/**
 * Built-in priors behave almost identically: they are 1. within a range
 * (even if the range is infinite, or infinitely small) and 0. outside.
 */
class GenericPrior : public Prior {
public:
  ~GenericPrior() override = default;

  std::pair<double, double> getValidRange() const final {
    return m_range;
  }

  double operator()(double x) const final {
    return x >= m_range.first && x <= m_range.second;
  };

  /**
   * Just at min and max the derivative is infinite, but the caller should be
   * clipping anyway.
   */
  double dx(double) const final {
    return 0;
  }

protected:
  explicit GenericPrior(double min = -std::numeric_limits<double>::infinity(),
                        double max = std::numeric_limits<double>::infinity())
      : m_range(min, max) {}

  std::pair<double, double> m_range;
};

/**
 * Uniform has an infinite range
 */
class Uniform final : public GenericPrior {
public:
  Uniform() : GenericPrior() {}
  ~Uniform() override = default;
};

/**
 * Tophat as a finite, configurable, range
 */
class Tophat final : public GenericPrior {
public:
  Tophat(double min, double max) : GenericPrior(min, max) {}
  ~Tophat() override = default;
};

/**
 * Delta has an infinitely small range
 */
class Delta final : public GenericPrior {
public:
  explicit Delta(double d) : GenericPrior(d, d) {}
  ~Delta() override = default;
};

}  // namespace Nnpz

#endif  // PHZ_NNPZ_PRIORS_H
