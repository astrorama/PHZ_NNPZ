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

#ifndef PHZ_NNPZ_BRUTEFORCE_H
#define PHZ_NNPZ_BRUTEFORCE_H

#include "Nnpz/Scaling.h"
#include "Nnpz/Types.h"
#include <ElementsKernel/Export.h>

namespace Nnpz {

/**
 * Compute the chi2 distance between the target catalog and the reference sample.
 * Both reference and catalog photometries are expected to be 32 bits
 * floats in row major. The first axis corresponds to the different objects,
 * the second axis to the different bands, and the third axis has only two
 * positions: value / error
 *
 * all_scales and all_closest are expected to be a row-major matrix
 * of (ntarget x k) elements.
 */
ELEMENTS_API void chi2_bruteforce(PhotoArray const& reference, PhotoArray const& all_targets, ScaleArray& all_scales,
                                  IndexArray& all_closest, int k, ScaleFunction* scaling);

/**
 * Compute the euclidena distance between the target catalog and the reference sample.
 * @see chi2_bruteforce for a description of the memory layout
 */
ELEMENTS_API void euclidean_bruteforce(PhotoArray const& reference, PhotoArray const& all_targets,
                                       ScaleArray& all_scales, IndexArray& all_closest, int k,
                                       ScaleFunction* scaling);

}  // namespace Nnpz

#endif  // PHZ_NNPZ_BRUTEFORCE_H
