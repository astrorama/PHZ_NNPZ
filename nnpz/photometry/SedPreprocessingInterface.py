"""
Created on: 05/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc


class SedPreprocessingInterface(object):
    """Interface for pre-processing the SED templates before computing the photometry"""


    @abc.abstractmethod
    def processSed(self, sed):
        """Processes a SED.

        Args:
            sed: The SED to process. It is a two dimensional numpy array of single
                precision floats. The first dimension has size same as the
                number of the knots and the second dimension has always size
                equal to two, with the first element representing the wavelength
                and the second the energy value, expressed in erg/s/cm^2/Angstrom.

        Returns: The processed SED as a two dimensional numpy array with same
            format as the input.
        """
        return


    @abc.abstractmethod
    def type(self):
        """The type of the resulting SED as a string"""
        return