"""
Created on: 13/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.photometry import PhotometryPrePostProcessorInterface


class PhotonPrePostProcessor(PhotometryPrePostProcessorInterface):
    """Pre/Post processor for producing photometry values in photon/cm^2/s"""

    def preProcess(self, sed):
        """Converts the given SED from ergs/cm^2/s/A to photon/cm^2/s/A.

        Args:
            sed: The SED to convert

        Returns:
            The SED expressed in photon/cm^2/s/A

        The conversion is taken from http://www.stsci.edu/~strolger/docs/UNITS.txt
        and is implemented as the equation:
        [Y photon/cm^2/s/A] = 5.03411250E+07 * [X1 erg/cm^2/s/A] * [X2 A]
        """
        result = np.ndarray(sed.shape, dtype=np.float32)
        result[:, 0] = sed[:, 0]
        result[:, 1] = 5.03411250E7 * sed[:, 1] * sed[:, 0]
        return result

    def postProcess(self, intensity, filter_name, filter_trans):
        """Returns the intensity unmodified. The filter info is ignored."""
        return intensity