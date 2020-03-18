"""
Created on: 05/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np


class PhotometryCalculator(object):
    """Class for computing the photometry for a filter reference system
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
        for f in self.__filter_trans_map:
            t = self.__filter_trans_map[f]
            self.__filter_range_map[f] = (t[0][0], t[-1][0])

        # Compute the total range of all filters
        ranges_arr = np.asarray(list(self.__filter_range_map.values()))
        self.__total_range = (ranges_arr[:,0].min(), ranges_arr[:,1].max())


    def __truncateSed(self, sed, range):
        """Truncates the given SED at the given range"""

        min_i = np.searchsorted(sed[:, 0], range[0])
        if min_i > 0:
            min_i -= 1
        max_i = np.searchsorted(sed[:, 0], range[1])
        max_i += 1
        return sed[min_i:max_i+1, :]


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

        # First remove any part of the SED that is outside the total range
        sed = self.__truncateSed(sed, self.__total_range)

        # Pre-process the SED
        sed = self.__pre_post_processor.preProcess(sed)

        # Iterate through the filters and compute the photometry values
        photometry_map = {}
        for filter_name in self.__filter_trans_map:
            filter_trans = self.__filter_trans_map[filter_name]

            # Truncate the SED to the size of the filter
            trunc_sed = self.__truncateSed(sed, self.__filter_range_map[filter_name])

            # Interpolate to a superset of both filter and sed
            interp_grid = np.sort(np.concatenate([sed[:, 0], filter_trans[:, 0]]))

            # Interpolate the SED
            interp_sed = np.interp(interp_grid, trunc_sed[:, 0], trunc_sed[:, 1], left=0, right=0)

            # Interpolate the filter
            interp_filter = np.interp(interp_grid, filter_trans[:, 0], filter_trans[:, 1], left=0, right=0)

            # Compute the SED through the filter
            filtered_sed = interp_sed * interp_filter

            # Compute the intensity of the filtered object
            intensity = np.trapz(filtered_sed, x=interp_grid)

            # Post-process the intensity
            photometry = self.__pre_post_processor.postProcess(intensity, filter_name, filter_trans)

            # Add the computed photometry in the results
            photometry_map[filter_name] = photometry

        return photometry_map


    def __call__(self, sed):
        """
        Convenience method to make this class callable
        """
        return self.compute(sed)
