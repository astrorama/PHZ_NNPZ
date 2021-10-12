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
from typing import Tuple

import numpy as np
from nnpz.photometry.GalacticReddeningPrePostProcessor import GalacticReddeningPrePostProcessor
from nnpz.photometry.PhotometryCalculator import PhotometryCalculator
from nnpz.photometry.PhotometryPrePostProcessorInterface import PhotometryPrePostProcessorInterface


class PhotometryWithCorrectionsCalculator(object):

    def __init__(self, filter_map: dict, pre_post_processor: PhotometryPrePostProcessorInterface,
                 ebv_ref: float, shifts: np.ndarray):
        pre_post_ebv = GalacticReddeningPrePostProcessor(pre_post_processor, p_14_ebv=ebv_ref)
        self._calculator = PhotometryCalculator(filter_map, pre_post_processor, shifts=shifts)
        self._ebv_calculator = PhotometryCalculator(filter_map, pre_post_ebv)
        self._shifts = shifts
        self._ebv = ebv_ref

    def compute(self, sed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Compute the reference photometry
        photo, shifted = self._calculator.compute(sed)
        shift_corr = np.zeros(2, dtype=photo.dtype)

        # Apply Audrey Galametz's formula for the EBV correction
        reddened_photo = self._ebv_calculator.compute(sed)
        ebv_corr = np.zeros(1, dtype=reddened_photo.dtype)
        for filter_name in photo.dtype.names:
            rfx = reddened_photo[filter_name][0]
            fx = photo[filter_name][0]
            ebv_corr[filter_name] = -2.5 * np.log10(rfx / fx) / self._ebv

        # Apply Stephane Paltani's formula for the filter shift correction
        for filter_name in photo.dtype.names:
            if photo[filter_name][0] > 0:
                Ct = shifted[filter_name] / photo[filter_name][0]
                C_hat_t = (Ct - 1) / self._shifts
                # Obtain a and b
                shift_corr[filter_name] = np.polyfit(self._shifts, C_hat_t, deg=1)
            else:
                shift_corr[filter_name] = 0.

        # Return photometry and correction factors
        return photo, ebv_corr, shift_corr

    def __call__(self, sed: np.ndarray):
        return self.compute(sed)
