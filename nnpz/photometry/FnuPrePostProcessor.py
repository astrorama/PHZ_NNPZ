"""
Created on: 14/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.photometry import PhotometryPrePostProcessorInterface


class FnuPrePostProcessor(PhotometryPrePostProcessorInterface):
    """Pre/Post processor for producing  photometry in erg/s/cm^2/Hz"""


    def __init__(self):
        self.__filter_norm = {}


    def preProcess(self, sed):
        #switching to the photon equation
        #"""Returns the SED unmodified"""
        #return sed
        """Divides the SED with the wavelength"""
        res = sed.copy()
        res[:,1] = res[:,1] * res[:,0]
        return res


    def postProcess(self, intensity, filter_name, filter_trans):
        """Converts the intensity to flux density by normalizing for the filter.

        Args:
            intensity: The intensity
            filter_name: The filter name
            filter_trans: The filter transmission

        Returns:
            The flux density in erg/s/cm^2/Hz

        The given intensity is normalized for the filter as described in
        http://www.asvo.org.au/blog/2013/03/21/derivation-of-ab-magnitudes/ .
        According this recipe, the filter transmission is first multiplied with
        c/lambda^2 (to compensate for the fact that the integration of the
        filter is done in wavelength instead of frequency). Then the total
        integral of the result is used to normalize the intensity.

        The filter integration is performed using the trapezoidal rule.
        """

        # The speed of light in Angstrom/s
        c = 299792458E10

        # First get the filter normalization. We cache the factors to avoid
        # recomputing them when the processor is used for multiple SEDs.
        if not filter_name in self.__filter_norm:
            l = filter_trans[:,0]
            norm_f = filter_trans[:,1] / l # switching to the photon equation: remove the second / l
            self.__filter_norm[filter_name] = c * np.trapz(norm_f, x=l)
        norm = self.__filter_norm[filter_name]

        return intensity / norm
