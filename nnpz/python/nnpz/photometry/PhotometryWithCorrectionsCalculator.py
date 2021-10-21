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
    """
    Computes photometry values and the correction factors for galactic reddening (EBV)
    and filter variations (average wavelength shifts).

    Given a SED index $\alpha$, and a filter transmission $T$:

    The EBV correction factor is computed with the formula
    \\f[
    a_{\\alpha,T} = -2.5 \\log_{10} \\left( \\frac{\\int
SED_{\\alpha}(\\lambda)*reddener(\\lambda)*Filter_T(\\lambda)}{\\int
SED_{\\alpha}(\\lambda)*Filter_T(\\lambda)}\\right) / {ebv\\_ref}
    \\f]

    Where the reddener functor is provided by GalacticReddeningPrePostProcessor

    For the filter variation, the correction factor for a given shift ($\\Delta\\lambda$)
    corresponds to

    \\f[
    C_{T,\\alpha}(\\Delta\\lambda) = \\frac{f_{T,\\alpha}(\\Delta\\lambda)}{f_{T,\\alpha}(0)}
    \\f]

    This correction factor is computed for a set of different deltas, and modeled by a
    second-degree polynomial. Since $C_{T,\alpha}(0) = 1$, the constant coefficient can be fixed
    to 1, and only two free parameters are to be found.

    This is converted to a linear regression by defining

    \\f[
    \\hat{C}_{T,\\alpha}(\\Delta\\lambda) = \\frac{C_{T,\\alpha}(\\Delta\\lambda) - 1}{\\Delta\\lambda}
    \\f]

    Which is defined for $\\Delta\\lambda \\ne 0$. In turn,
    $\\hat{C}$ can be approximated by a linear function:

    \\f[
    \\hat{C}_{T,\\alpha}(\\Delta\\lambda) \\approx a_{T,\\alpha}\\Delta\\lambda + b_{T,\\alpha}
    \\f]

    Which can be obtained by a least-square minimization. The correction factor can be
    finally computed by

    \\f[
    C_{T,\\alpha}(\\Delta\\lambda) \\approx a_{T,\\alpha}\\Delta\\lambda^2 + b_{T,\\alpha}\\Delta\\lambda + 1
    \\f]

    Args:
        filter_map: A dictionary with keys the filter names and values the
                filter transmissions as 2D numpy arrays, where the first axis
                represents the knots and the second axis has size 2 with the
                first element being the wavelength (expressed in Angstrom) and
                the second the filter transmission (in the range [0,1])
        pre_post_processor: An object which is used for controlling the
                type of the photometry produced, by performing unit conversions
                before and after integrating the SED. It must implement the
                PhotometryPrePostProcessorInterface.
        ebv_ref: The EBV value used to compute the correction factor. 0.3 works well.
        shifts: The shift values used to compute the correction factors to be fit by a linear
            model. 0 can *not* be one of the points.
        galactic_reddening_curve: The galactic reddening curve. See GalacticReddeningPrePostProcessor
    """

    def __init__(self, filter_map: dict, pre_post_processor: PhotometryPrePostProcessorInterface,
                 ebv_ref: float, shifts: np.ndarray, galactic_reddening_curve: str = None):
        if shifts is not None and 0 in shifts:
            raise ValueError('Ĉ(Δλ) is not defined for Δλ=0! Please, remove 0 from the shifts')

        pre_post_ebv = GalacticReddeningPrePostProcessor(pre_post_processor, p_14_ebv=ebv_ref,
                                                         galactic_reddening_curve=galactic_reddening_curve)
        self._calculator = PhotometryCalculator(filter_map, pre_post_processor, shifts=shifts)
        self._ebv_calculator = PhotometryCalculator(filter_map, pre_post_ebv)
        self._shifts = shifts
        self._ebv = ebv_ref

    def compute(self, sed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the photometry (for $\\Delta\\lambda  = 0$ and ${EBV} = 0$) and correction factors
        for the given set of SEDs
        Args:
            sed: The SED to compute the photometry for. It is a two
                dimensional numpy array of single precision floats. The first
                dimension has size same as the number of the knots and the
                second dimension has always size equal to two, with the first
                element representing the wavelength expressed in Angstrom and
                the second the energy value, expressed in erg/s/cm^2/Angstrom.

        Returns:
            A tuple (photometry, ebv_correction, shift_correction).
            - photometry is a structured array with the filter names as attributes, and
                one dimension with two positions: value and error.
            - ebv_correction is a structured array as photometry, with a single dimension
                with the unique EBV correction factor
            - shift_correction is a structured array as photometry, and one dimension with two
                position: shift correction factors a and b
        """
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
