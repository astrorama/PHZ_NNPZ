"""
Created on: 18/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.photometry import PhotometryPrePostProcessorInterface


class FlambdaPrePostProcessor(PhotometryPrePostProcessorInterface):
    """Pre/Post processor for producing  photometry in erg/s/cm^2/A"""


    def __init__(self):
        self.__filter_norm = {}


    def preProcess(self, sed):
        """Returns the SED unmodified"""
        return sed


    def postProcess(self, intensity, filter_name, filter_trans):
        """Converts the intensity to flux density by normalizing for the filter.

        Args:
            intensity: The intensity
            filter_name: The filter name
            filter_trans: The filter transmission

        Returns:
            The flux density in erg/s/cm^2/A

        The given intensity is normalized using the total integral of the filter
        over the wavelength in Angstrom.

        The filter integration is performed using the trapezoidal rule.
        """

        # First get the filter normalization. We cache the factors to avoid
        # recomputing them when the processor is used for multiple SEDs.
        if not filter_name in self.__filter_norm:
            self.__filter_norm[filter_name] = np.trapz(filter_trans[:,1], x=filter_trans[:,0])
        norm = self.__filter_norm[filter_name]

        return intensity / norm