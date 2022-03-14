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

#ifndef PHZ_NNPZ_TYPES_H
#define PHZ_NNPZ_TYPES_H

#include "NdArray/NdArray.h"
#include <cstdint>
#include <functional>
#include <pybind11/numpy.h>

namespace Nnpz {

using index_t  = uint32_t;
using weight_t = float;
using photo_t  = double;
using scale_t  = float;
using flag_t   = uint32_t;
template <typename T>
using NdArray = Euclid::NdArray::NdArray<T>;

using WeightFunc = std::function<void(NdArray<photo_t> const&, NdArray<photo_t> const&, NdArray<weight_t>&)>;

}  // namespace Nnpz

#endif  // PHZ_NNPZ_TYPES_H
