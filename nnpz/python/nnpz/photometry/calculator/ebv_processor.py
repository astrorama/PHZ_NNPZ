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

"""
Created on: 19/02/2018
Author: Florian Dubath
"""

from __future__ import division, print_function

import numpy as np
from ElementsKernel.Auxiliary import getAuxiliaryPath

from nnpz.photometry.filter_provider import ListFileFilterProvider
from .photometry_processor_interface import PhotometryPrePostProcessorInterface


class EBVPrePostProcessor(PhotometryPrePostProcessorInterface):
    """
    Pre/Post processor for including the galactic absorption.

    This processor is a decorator around another PrePostProcessor, which apply
    galactic reddening to the sed beforehand.

    Args:
        pre_post_processor: The pre/post processor to be decorated.

        p_14_ebv: The P14 E(B-V) float value along the line of sight

        galactic_reddening_curve: The galactic reddening curve.
            The curve is a 2D numpy arrays, where the first axis
            represents the knots and the second axis has size 2 with the
            first element being the wavelength (expressed in Angstrom) and
            the second the rescaled galactic absorption value

    Note that the galactic_reddening_curve parameters
    is optional. If it is not given, the default behavior is to use the
    F99 extinction curve.
    """

    def __init__(self, pre_post_processor: PhotometryPrePostProcessorInterface, p_14_ebv: float,
                 galactic_reddening_curve: np.ndarray = None):
        self.__processor = pre_post_processor
        self.__p_14_ebv = p_14_ebv
        if galactic_reddening_curve is None:
            prov = ListFileFilterProvider(getAuxiliaryPath('GalacticExtinctionCurves.list'))
            self.__reddening_curve = prov.getFilterTransmission('extinction_curve')
        else:
            self.__reddening_curve = galactic_reddening_curve

    def _computeAbsorption(self, sed: np.ndarray) -> np.ndarray:
        # Compute A_lambda
        interp_absorption = np.interp(
            sed[:, 0], self.__reddening_curve[:, 0], self.__reddening_curve[:, 1], left=0, right=0
        )
        a_lambda = interp_absorption * self.__p_14_ebv
        absorption_factor = np.power(10, -0.4 * a_lambda)

        # Compute the reddened sed: build an array with the same shape as the SED
        # with 1s in the first column and the absorption factor in the second,
        # so the element-wise multiplication do not affect the lambda
        mult = np.ones((absorption_factor.shape[0], 2))
        mult[:, 1] = absorption_factor
        return np.multiply(sed, mult)

    def preProcess(self, sed: np.ndarray) -> np.ndarray:
        """
        Redden the SED according to the galactic absorption law and the
        E(B-V) parameter then redirects the call to the FnuPrePostProcessor
        """
        reddened_sed = self._computeAbsorption(sed)
        return self.__processor.preProcess(reddened_sed)

    def postProcess(self, intensity: float, filter_name: str) -> float:
        """
        Returns the flux as computed from the decorated pre/post processor.
        """
        return self.__processor.postProcess(intensity, filter_name)
