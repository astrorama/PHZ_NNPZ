#
#  Copyright (C) 2022 Euclid Science Ground Segment
#
#  This library is free software; you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation;
#  either version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with this library;
#  if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA
#
import numpy as np
from nnpz.flags import NnpzFlag

cimport numpy as np

np.import_array()

ctypedef np.double_t PHOTO_DTYPE_t
ctypedef np.float32_t SCALE_DTYPE_t
ctypedef np.float32_t WEIGHT_DTYPE_t
ctypedef np.uint32_t FLAG_DTYPE_t

WEIGHT_DTYPE = np.float32


def likelihood(const PHOTO_DTYPE_t[:,:,:] ref_objs, const PHOTO_DTYPE_t[:,:] target_obj):
    """
    Compute the weight as the likelihood of the chi2: $L = e^{-\\chi^2/2}$
    Note that the maximum weight can be 1 (when $\\chi^2 == 0$), and it gets
    asymptotically close to 0 as $\\chi^2$ grows.
    """
    val_1 = ref_objs[..., 0]
    err_1 = ref_objs[..., 1]
    val_2 = target_obj[..., 0, None].T
    err_2 = target_obj[..., 1, None].T

    nom = np.subtract(val_1, val_2, dtype=WEIGHT_DTYPE)
    nom = np.multiply(nom, nom, out=nom)

    den1 = np.multiply(err_1, err_1, dtype=WEIGHT_DTYPE)
    den2 = np.multiply(err_2, err_2, dtype=WEIGHT_DTYPE)
    den = np.add(den1, den2, out=den1)

    chi2 = np.sum(nom / den, axis=1, dtype=WEIGHT_DTYPE)
    return np.exp(-0.5 * chi2, out=chi2)


def inverse_chi2(const PHOTO_DTYPE_t[:,:,:] ref_objs, const PHOTO_DTYPE_t[:,:] target_obj):
    """
    Compute the weight as the inverse of the $\\chi^2$ distance.
    For two identical points, this will be infinity since their distance is 0.
    This distance should only be used as a fall-back for whenever the likelihood of all
    neighbors become too small.
    """
    val_1 = ref_objs[..., 0]
    err_1 = ref_objs[..., 1]
    val_2 = target_obj[..., 0, None].T
    err_2 = target_obj[..., 1, None].T

    nom = np.subtract(val_1, val_2, dtype=WEIGHT_DTYPE)
    nom = np.multiply(nom, nom, out=nom)

    den1 = np.multiply(err_1, err_1, dtype=WEIGHT_DTYPE)
    den2 = np.multiply(err_2, err_2, dtype=WEIGHT_DTYPE)
    den = np.add(den1, den2, out=den1)

    chi2 = np.sum(nom / den, axis=1, dtype=WEIGHT_DTYPE)
    return np.reciprocal(chi2, out=chi2)


def inverse_euclidean(const PHOTO_DTYPE_t[:,:,:] ref_objs, const PHOTO_DTYPE_t[:,:] target_obj):
    """
    Compute the weight as the inverse of the Euclidean distance.
    For two identical points, this will be infinity since their distance is 0.
    This distance should only be used as a fall-back for whenever the likelihood of all
    neighbors become too small.
    """
    val_1 = ref_objs[..., 0]
    val_2 = target_obj[..., 0, None].T

    diff = np.subtract(val_1, val_2, dtype=WEIGHT_DTYPE)
    prod = np.multiply(diff, diff, out=diff)
    sum = np.sum(prod, axis=1, dtype=WEIGHT_DTYPE)
    distance = np.sqrt(sum, out=sum)
    return np.reciprocal(distance, out=distance)


class WeightCalculator:
    """
    Wraps a primary and secondary weight calculator methods. Secondary is used if the
    primary returns a value too small to be represented by a float32.

    Args:
        primary: str
            Name of the primary method
        secondary: str
            Name of the secondary method

    Notes:
        The supported methods are the inverse of 'Euclidean' and 'Chi2', and 'Likelihood'
    """
    _weights = {
        'Euclidean': inverse_euclidean,
        'Chi2': inverse_chi2,
        'Likelihood': likelihood,
    }

    def __init__(self, primary:str, secondary:str):
        self.__primary_weight = WeightCalculator._weights[primary]
        self.__secondary_weight = WeightCalculator._weights[secondary]
        self.__min_weight = np.finfo(np.float32).tiny

    def __call__(self, const PHOTO_DTYPE_t[:,:,:,:] neighbor_photometry,
                       const PHOTO_DTYPE_t[:,:,:] target_photometry,
                       WEIGHT_DTYPE_t[:,:] output_weights,
                       FLAG_DTYPE_t[:] output_flags):
        assert neighbor_photometry.shape[0] == target_photometry.shape[0]
        assert neighbor_photometry.shape[0] == output_weights.shape[0]
        assert neighbor_photometry.shape[1] == output_weights.shape[1]
        assert neighbor_photometry.shape[2] == target_photometry.shape[1]
        assert neighbor_photometry.shape[3] == 2
        assert target_photometry.shape[2] == 2
        assert output_weights.shape[0] == output_flags.shape[0]

        cdef int output_size = len(target_photometry)
        cdef int i
        cdef WEIGHT_DTYPE_t[:] weight_array

        for i in range(output_size):
            neighbor_photo = neighbor_photometry[i]
            weights = self.__primary_weight(neighbor_photo, target_photometry[i])
            if not np.any(weights > self.__min_weight):
                weights = self.__secondary_weight(neighbor_photo, target_photometry[i])
                output_flags[i] |= NnpzFlag.ALTERNATIVE_WEIGHT_FLAG
            weight_array = weights
            output_weights[i] = weight_array
