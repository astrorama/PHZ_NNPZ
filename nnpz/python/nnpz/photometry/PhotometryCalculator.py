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
Created on: 05/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from typing import Callable

import numpy as np
from scipy import signal


def correction_func(a, b):
    return a ** 2 + b + 1


class PhotometryCalculator(object):
    """
    Class for computing the photometry for a filter reference system
    """

    def __init__(self, filter_map, pre_post_processor, shifts: np.ndarray = None):
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
            shifts: Compute the photometry with these shifts in order to compute
                the correction factors.
        """
        self.__filter_trans_map = filter_map
        self.__pre_post_processor = pre_post_processor
        self.__shifts = shifts

        # Compute the ranges of the filters and the total range
        self.__filter_range_map = {}
        for f_name in self.__filter_trans_map:
            transmission = self.__filter_trans_map[f_name]
            self.__filter_range_map[f_name] = (transmission[0][0], transmission[-1][0])

        # Compute the total range of all filters
        ranges_arr = np.asarray(list(self.__filter_range_map.values()))
        self.__total_range = (ranges_arr[:, 0].min(), ranges_arr[:, 1].max())

    def __compute_interp_grid(self, trans, sed, shifts):
        # Mask areas where there is no transmission
        tmask = trans[:, 1] > 0
        # Unmask the knots just before and just after the first/last transmission
        tidx = np.argwhere(tmask)[0]
        first, last = tidx[0] - 1, np.flip(tidx)[0] + 1
        tmask[max(0, first):min(last, len(tmask) - 1)] = True
        # Pick the interval between the first and last >=0 transmission, and widen the area
        # according to the shifts
        start = trans[tmask, 0].min() + np.min(shifts)
        end = trans[tmask, 0].max() + np.max(shifts)
        # SED mask
        smask = (sed[:, 0] >= start) & (sed[:, 0] <= end)
        # Equally-spaced grid
        step = np.diff(trans[tmask, 0]).min(initial=min(1, np.diff(sed[smask, 0]).mean()))
        grid = np.arange(start, end + step, step=step)
        # Find emission lines, if any, plus the preceding and following knots
        peaks, _ = signal.find_peaks(sed[smask, 1], threshold=np.mean(sed[smask, 1]))
        pidx = np.concatenate([peaks, peaks - 1, peaks + 1])
        # Add the delta knots to the grid
        grid = np.unique(np.concatenate([grid, sed[pidx, 0]]))
        return grid

    def __compute_value(self, filter_name: str, trans: np.ndarray, sed: np.ndarray,
                        shifts: np.ndarray, grid_method):
        if grid_method is None:
            grid_method = self.__compute_interp_grid
        # Compute a reasonable interpolation grid
        interp_grid = grid_method(trans, sed, shifts)
        # Interpolate SED
        interp_sed = np.interp(interp_grid, sed[:, 0], sed[:, 1], left=0, right=0)
        # Interpolate shifted transmissions
        interp_trans = np.zeros(shape=(len(shifts), len(interp_grid)), dtype=np.float32)
        for i, shift in enumerate(shifts):
            interp_trans[i, :] = np.interp(interp_grid, trans[:, 0] + shift, trans[:, 1], left=0,
                                           right=0)
        # Broadcast the SED
        interp_sed = np.broadcast_to(interp_sed, shape=(len(shifts), len(interp_sed)))
        # Compute the SED through the filter
        filtered_sed = interp_sed * interp_trans
        # Compute the intensity of the filtered object
        intensity = np.trapz(filtered_sed, x=interp_grid)
        # Post-process the intensity
        return self.__pre_post_processor.postProcess(intensity, filter_name)

    def compute(self, sed, interp_grid_callback: Callable = None):
        """Computes the photometry for the given SED.

        Args:
            sed: The SED to compute the photometry for. It is a two
                dimensional numpy array of single precision floats. The first
                dimension has size same as the number of the knots and the
                second dimension has always size equal to two, with the first
                element representing the wavelength expressed in Angstrom and
                the second the energy value, expressed in erg/s/cm^2/Angstrom.

        Returns:
            A structured array with the filter names as attributes, and one dimension with two
            positions: value and error. Optionally, the photometry for the shifted filters.

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
        dtype = [(filter_name, np.float32) for filter_name in self.__filter_trans_map]

        # Pre-process the SED
        sed = self.__pre_post_processor.preProcess(sed)
        if self.__shifts is not None:
            photometry_shifted = np.zeros(len(self.__shifts), dtype=dtype)
        else:
            photometry_shifted = None

        # Iterate through the filters and compute the photometry values
        photometry_map = np.zeros(2, dtype=dtype)
        for filter_name in self.__filter_trans_map:
            filter_trans = self.__filter_trans_map[filter_name]
            # Add the computed photometry in the results
            photometry_map[filter_name][0] = self.__compute_value(filter_name, filter_trans, sed,
                                                                  shifts=np.asarray([0]),
                                                                  grid_method=interp_grid_callback)
            if self.__shifts is not None:
                photometry_shifted[filter_name][:] = self.__compute_value(filter_name, filter_trans,
                                                                          sed, shifts=self.__shifts,
                                                                          grid_method=interp_grid_callback)
        if self.__shifts is not None:
            return photometry_map, photometry_shifted
        return photometry_map

    def __call__(self, sed: np.ndarray):
        """
        Convenience method to make this class callable
        """
        return self.compute(sed)
