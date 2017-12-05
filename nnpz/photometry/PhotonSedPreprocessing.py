"""
Created on: 05/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.photometry import SedPreprocessingInterface


class PhotonSedPreprocessing(SedPreprocessingInterface):
    """Converts the SED from ergs/cm^2/s/A to photon/cm^2/s/A.

    The conversion is taken from http://www.stsci.edu/~strolger/docs/UNITS.txt
    and is implemented as the equation:
    [Y photon/cm^2/s/A] = 5.03411250E+07 * [X1 erg/cm^2/s/A] * [X2 A]
    """


    def processSed(self, sed):
        result = np.ndarray(sed.shape, dtype=np.float32)
        result[:,0] = sed[:,0]
        result[:,1] = 5.03411250E7 * sed[:,1] * sed[:,0]
        return result


    def type(self):
        return "PHOTON_COUNT"