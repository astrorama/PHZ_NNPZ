"""
Created on: 04/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc


class FilterProviderInterface(object):
    """Interface providing the filter transmissions"""


    @abc.abstractmethod
    def getFilterNames(self):
        """Provides a list with the names of the filters.

        Returns:
            A python list with the names of the filters as strings
        """
        return


    @abc.abstractmethod
    def getFilterTransmission(self, name):
        """Provides the transmission curve of the filter with the given name.

        Args:
            name: The name of the filter to get the transmission for

        Returns:
            A 2D numpy array of single precision floating point numbers. The
            first dimension represents the knots of the filter transmission
            and the second one has always size 2, representing the wavelength
            (expressed in Angstrom) and the transmission value (in the range
            [0,1]).

        Raises:
            UnknownNameException: If there is no filter with the given name
        """
        return
