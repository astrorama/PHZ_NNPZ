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

#include <cstdint>
#include <pybind11/numpy.h>

namespace Nnpz {

using index_t  = int32_t;
using weight_t = float;
using photo_t  = double;
using scale_t  = float;
using flag_t   = uint32_t;

using IndexArray  = pybind11::array_t<index_t>;
using PhotoArray  = pybind11::array_t<photo_t>;
using WeightArray = pybind11::array_t<weight_t>;
using FlagArray   = pybind11::array_t<flag_t>;
using ScaleArray  = pybind11::array_t<scale_t>;

using WeightFunc = std::function<void(PhotoArray const&, PhotoArray const&, WeightArray&)>;

}  // namespace Nnpz

#endif  // PHZ_NNPZ_TYPES_H
