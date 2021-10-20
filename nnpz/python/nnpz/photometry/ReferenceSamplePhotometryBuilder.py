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
Created on: 04/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from nnpz.exceptions import WrongTypeException
from nnpz.photometry.FilterProviderInterface import FilterProviderInterface
from nnpz.photometry.PhotometryPrePostProcessorInterface import PhotometryPrePostProcessorInterface
from nnpz.photometry.PhotometryWithCorrectionsCalculator import PhotometryWithCorrectionsCalculator


class ReferenceSamplePhotometryBuilder(object):
    """
    Class for creating photometry from a reference sample
    """

    def __init__(self, filter_provider, pre_post_processor, ebv: float, shifts: np.array):
        """Creates a new instance of ReferenceSamplePhotometryBuilder

        Args:
            filter_provider: An instance of FilterProviderInterface from where
                the filter data are being retrieved
            pre_post_processor: An instance of PhotometryPrePostProcessorInterface
                which defines the type of photometry being produced
            ebv: To be used for the computation of the EBV correction factor
            shifts: Compute the photometry with these shifts in order to compute
                the correction factors.
        Raises:
            WrongTypeException: If the filter_provider is not an implementation
                of FilterProviderInterface
            WrongTypeException: If the pre_post_processor is not an
                implementation of PhotometryPrePostProcessorInterface
        """

        # Perform a check that the inputs implement the correct interfaces
        if not isinstance(filter_provider, FilterProviderInterface):
            raise WrongTypeException('filter_provider must implement FilterProviderInterface')
        if not isinstance(pre_post_processor, PhotometryPrePostProcessorInterface):
            raise WrongTypeException(
                'pre_post_processor must implement PhotometryPrePostProcessorInterface')

        self._filter_provider = filter_provider
        self._pre_post_processor = pre_post_processor
        self._ebv = ebv
        self._shifts = shifts
        self._filter_map = {}

        # By default we produce photometry for every available filter
        self.setFilters(filter_provider.getFilterNames())

    def setFilters(self, filter_list):
        """Sets the filters for which the photometry is produced.

        Args:
            filter_list: A list with the filter names

        Raises:
            UnknownNameException: If there is any filter which is not known by
                the filter_provider passed to the constructor

        Note that if this method is not called at all, the default behavior is
        to build the photometry for all filters available.
        """
        self._filter_map = {}
        for f in filter_list:
            self._filter_map[f] = self._filter_provider.getFilterTransmission(f)

    def buildPhotometry(self, sample_iter, progress_listener=None):
        """Computes the photometry of the SEDs the given iterator traverses.

        Args:
            sample_iter: An iterator to reference sample objects (or any type
                which provides the sed property)
            progress_listener: A function which is called at each iteration with
                parameter the current iteration number. It is ignored if None is
                passed (default).

        Returns:
            A dictionary where the keys are the filter names and the values are
            numpy arrays of single precision floats containing the photometry
            values of the filter for the iterated SEDs

        Note that if the sample_iter reach an object for which the SED is set
        to None it will stop the iteration and return the already computed
        photometry values.

        For more details in the computation recipe see the documentation of the
        PhotometryCalculator class.
        """

        # Create the calculator which will be used for the photometry computation
        calculator = PhotometryWithCorrectionsCalculator(self._filter_map,
                                                         self._pre_post_processor, self._ebv,
                                                         self._shifts)

        # Create the result map with empty list assigned to each filter
        photo_list = {}
        ebv_corr_list = {}
        shift_corr_list = {}
        for f in self._filter_map:
            photo_list[f] = []
            ebv_corr_list[f] = []
            shift_corr_list[f] = []

        # Iterate through all the elements the iterator points to
        for progress, element in enumerate(sample_iter):

            # Report the progress
            if progress_listener is not None:
                progress_listener(progress)

            # If we have reached a missing SED stop the iteration
            if element.sed is None:
                break

            # Compute the photometry and update the photo_list_map
            photo, ebv_corr, shift_corr = calculator.compute(element.sed)
            for f in photo.dtype.names:
                photo_list[f].append(photo[f][0])
                ebv_corr_list[f].append(ebv_corr[f])
                shift_corr_list[f].append(shift_corr[f])

        # Convert the photometry lists to numpy arrays
        photo_map = {}
        ebv_corr_map = {}
        shift_corr_map = {}
        for f in photo_list:
            photo_map[f] = np.asarray(photo_list[f], dtype=np.float32)
            ebv_corr_map[f] = np.asarray(ebv_corr_list[f], dtype=np.float32)
            shift_corr_map[f] = np.asarray(shift_corr_list[f], dtype=np.float32)

        return photo_map, ebv_corr_map, shift_corr_map
