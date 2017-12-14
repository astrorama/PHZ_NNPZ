"""
Created on: 13/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc


class PhotometryPrePostProcessorInterface(object):
    """Interface defining the pre and post processing steps of the photometry calculation"""


    @abc.abstractmethod
    def preProcess(self, sed):
        """Pre-processes an SED.

        Args:
            sed: The SED to process. It is a two dimensional numpy array of
                single precision floats. The first dimension has size same as
                the number of the knots and the second dimension has always size
                equal to two, with the first element representing the wavelength
                expressed in Angstrom and the second the energy value, expressed
                in erg/s/cm^2/Angstrom.

        Returns: The pre-processed SED as a two dimensional numpy array with same
            format as the input.
        """
        return


    @abc.abstractmethod
    def postProcess(self, intensity, filter_name, filter_trans):
        """Post-processes a band intensity.

        Args:
            intesity: The intensity of the band, as computed by integrating the
                SED convolved with the filter
            filter_name: The name of the filter the intensity is for
            filter_trans: The filter transmission as a 2D numpy array of single
                precision floating point numbers. The first dimension represents
                the knots of the filter transmission and the second one has
                always size 2, representing the wavelength (expressed in
                Angstrom) and the transmission value (in the range [0,1]).

        Returns: The photometry value for the specific band
        """
        return
