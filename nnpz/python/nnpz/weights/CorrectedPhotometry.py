#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under the terms of
# the GNU Lesser General Public License as published by the Free Software Foundation;
# either version 3.0 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this library;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301 USA
#
from typing import List

import numpy as np
from ElementsKernel import Logging
from nnpz import NnpzFlag
from nnpz.photometry.SourceIndependantGalacticUnReddening import \
    SourceIndependantGalacticUnReddening
from nnpz.reference_sample import PhotometryProvider
from nnpz.weights import WeightPhotometryProvider

logger = Logging.getLogger(__name__)


class CorrectedPhotometry(WeightPhotometryProvider):
    """
    Similar to RecomputedPhotometry, but the correction is done by interpolation
    rather than re-computation, which is less CPU intensive.

    Args:
        ref_phot: A PhotometrtyProvider instance
        ref_filters: List of filters to use
        ebv_list: None, or a 1D array with the (E(B-V) corresponding to each entry in the
                target catalog
        filter_trans_mean_lists: A map with the filter_name as key, and a list/array with the
                filter mean corresponding to each entry in the target catalog
    """

    def __init__(self, ref_phot: PhotometryProvider, ref_filters: List[str],
                 ebv_list: np.ndarray = None,
                 filter_trans_mean_lists: dict = None):
        self.__filters = ref_filters
        self.__ref_photo = ref_phot.getData(*self.__filters)
        self.__ref_ebv_corr = ref_phot.getEBVCorrectionFactors(*self.__filters)
        self.__ref_shift_corr = ref_phot.getShiftCorrectionFactors(*self.__filters)
        self.__ebv_list = ebv_list

        # Requires the shift value, not the mean
        filter_trans_map = {f: ref_phot.getFilterTransmission(f) for f in ref_phot.getFilterList()}

        self.__filter_shifts = dict()
        if filter_trans_mean_lists:
            for filter_name, src_trans_mean in filter_trans_mean_lists.items():
                transmission = filter_trans_map[filter_name]
                trans_mean = np.average(transmission[:, 0], weights=transmission[:, 1])
                not_nan_mean = np.isfinite(src_trans_mean)
                shifts = np.zeros(src_trans_mean.shape)
                shifts[not_nan_mean] = src_trans_mean[not_nan_mean] - trans_mean
                self.__filter_shifts[filter_name] = shifts

    def __call__(self, ref_i: int, cat_i: int, flags: NnpzFlag):
        """
        Project the photometry of the reference sample as if it were seen through the same
        part of the detector as the target.

        Args:
            ref_i: The index of the reference sample for which re-compute the photometry
            cat_i: The index of the target to use for the re-computation
            flags: The flags objects to update

        Returns:
            A map with keys the filter names and values the photometry values
        """
        dtype = [(filter_name, np.float32) for filter_name in self.__filters]
        rest_photo = self.__ref_photo[ref_i]
        corrected_photo = np.zeros(2, dtype=dtype)
        ebv = self.__ebv_list[cat_i][np.newaxis] if self.__ebv_list is not None else 0

        # First, apply the filter correction
        for fi, fname in enumerate(self.__filters):
            shift_corr_a, shift_corr_b = self.__ref_shift_corr[ref_i, fi]
            ebv_corr = self.__ref_ebv_corr[ref_i, fi]
            # Copy value
            corrected_photo[fname] = rest_photo[fi]
            # Correct shift
            shift = self.__filter_shifts[fname][cat_i] if fname in self.__filter_shifts else 0
            shift_corr = shift_corr_a * shift * shift + shift_corr_b * shift + 1
            corrected_photo[fname] *= shift_corr
            # Then, the galactic reddening
            corrected_photo[fname] *= 10 ** (-0.4 * ebv_corr * ebv)

        return corrected_photo
