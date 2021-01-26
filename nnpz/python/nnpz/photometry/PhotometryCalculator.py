#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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
Created on: 05/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np


class PhotometryCalculator(object):
    """
    Class for computing the photometry for a filter reference system
    """

    def __init__(self, filter_map, pre_post_processor):
        """Creates a new instance of the PhotometryCalculator.

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
        """
        self.__filter_trans_map = filter_map
        self.__pre_post_processor = pre_post_processor

        # Compute the ranges of the filters and the total range
        self.__filter_range_map = {}
        for f_name in self.__filter_trans_map:
            transmission = self.__filter_trans_map[f_name]
            self.__filter_range_map[f_name] = (transmission[0][0], transmission[-1][0])

        # Compute the total range of all filters
        ranges_arr = np.asarray(list(self.__filter_range_map.values()))
        self.__total_range = (ranges_arr[:, 0].min(), ranges_arr[:, 1].max())

    def compute(self, sed):
        """Computes the photometry for the given SED.

        Args:
            sed: The SED to compute the photometry for. It is a two
                dimensional numpy array of single precision floats. The first
                dimension has size same as the number of the knots and the
                second dimension has always size equal to two, with the first
                element representing the wavelength expressed in Angstrom and
                the second the energy value, expressed in erg/s/cm^2/Angstrom.

        Returns:
            A map with keys the filter names and values the photometry values

        The type of the photometry values computed depends on the type of the
        pre_post_processor passed to the constructor.

        The recipe used by this method is the following:

        - The SED is truncated to the full range covered by the filters
        - The SED is processed using the preProcess() method of the
            pre_post_processor given to the constructor. This step can perform
            any computations on the SED (like unit conversions, etc) that have
            to be performed before the SED integration.
        - Each filter transmission is interpolated to the knot values of the
            SED, using linear interpolation. The filter transmission outside the
            given filter range is assumed to be zero.
        - The SED template values are multiplied with each filter transmissions
        - The total area of the SED (intensity) is computed for each filter,
            using the trapezoidal rule
        - Each value computed is processed using the postProcess() method of the
            pre_post_processor given to the constructor. This step can perform
            any computations on the photometry values (like filter normalization
            etc) that have to be performed after the SED integration.
        """

        # Pre-process the SED
        sed = self.__pre_post_processor.preProcess(sed)

        # Iterate through the filters and compute the photometry values
        photometry_map = {}
        for filter_name in self.__filter_trans_map:
            filter_trans = self.__filter_trans_map[filter_name]

            # Interpolate the SED
            interp_sed = np.interp(filter_trans[:, 0], sed[:, 0], sed[:, 1], left=0, right=0)

            # Compute the SED through the filter
            filtered_sed = interp_sed * filter_trans[:, 1]

            # Compute the intensity of the filtered object
            intensity = np.trapz(filtered_sed, x=filter_trans[:, 0])

            # Post-process the intensity
            photometry = self.__pre_post_processor.postProcess(intensity, filter_name)

            # Add the computed photometry in the results
            photometry_map[filter_name] = photometry

        return photometry_map

    def __call__(self, sed):
        """
        Convenience method to make this class callable
        """
        return self.compute(sed)
