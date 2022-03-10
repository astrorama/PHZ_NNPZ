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

#ifndef PHZ_NNPZ_SCALING_H
#define PHZ_NNPZ_SCALING_H

#include "Types.h"
#include <string>

namespace Nnpz {

class ScaleFunction {
public:
  virtual ~ScaleFunction()                                          = default;
  virtual float operator()(photo_t const*, photo_t const*, ssize_t) = 0;
};

struct ScaleFunctionParams {
  std::size_t maxiter;
  double      tolerance;

  ScaleFunctionParams(std::size_t _maxiter, double _tolerance) : maxiter(_maxiter), tolerance(_tolerance) {}
};

std::shared_ptr<ScaleFunction> scaleFunctionFactory(std::string const& prior, ScaleFunctionParams const& params);

}  // namespace Nnpz

#endif  // PHZ_NNPZ_SCALING_H
