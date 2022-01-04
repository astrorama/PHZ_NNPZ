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
Created on: 30/11/19
Author: Alejandro Alvarez Ayllon
"""
import multiprocessing
import os
from typing import List

import numpy as np
from ElementsKernel import Logging

from nnpz.photometry.calculator import PhotometryPrePostProcessorInterface, \
    PhotometryWithCorrectionsCalculator
from nnpz.photometry.filter_provider import FilterProviderInterface

logger = Logging.getLogger(__name__)


class PhotometryBuilder:
    """
    Class for creating photometry from a reference sample.

    Args:
        filter_provider: An instance of FilterProviderInterface from where
            the filter data are being retrieved
        pre_post_processor: An instance of PhotometryPrePostProcessorInterface
            which defines the type of photometry being produced
        ebv: To be used for the computation of the EBV correction factor
        shifts: To be used for the computation of the filter variation correction factors
        ncores:
            Number of cores to use

    Raises:
        WrongTypeException: If the filter_provider is not an implementation
            of FilterProviderInterface
        WrongTypeException: If the pre_post_processor is not an
            implementation of PhotometryPrePostProcessorInterface
    """

    class SedIter(object):
        """
        A generator for only the sed attribute of a reference object.
        """

        def __init__(self, objects):
            self.__objects = objects

        def __iter__(self):
            for o in self.__objects:
                yield o.sed

    def setFilters(self, filter_list: List[str]):
        """
        Sets the filters for which the photometry is produced.

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

    def __init__(self, filter_provider: FilterProviderInterface,
                 pre_post_processor: PhotometryPrePostProcessorInterface,
                 ebv: float, shifts: np.ndarray, ncores: int = None):
        if not isinstance(filter_provider, FilterProviderInterface):
            raise TypeError('filter_provider must implement FilterProviderInterface')
        if not isinstance(pre_post_processor, PhotometryPrePostProcessorInterface):
            raise TypeError('pre_post_processor must implement PhotometryPrePostProcessorInterface')

        self._filter_provider = filter_provider
        self._pre_post_processor = pre_post_processor
        self._ebv = ebv
        self._shifts = shifts
        self._filter_map = {}

        # By default we produce photometry for every available filter
        self.setFilters(filter_provider.getFilterNames())
        self.__ncores = os.cpu_count() if not ncores else ncores

    def buildPhotometry(self, sample_iter, progress_listener=None):
        """
        Computes the photometry of the SEDs the given iterator traverses.

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

        logger.info('Computing photometries using %d processes', self.__ncores)
        with multiprocessing.Pool(self.__ncores) as pool:
            elements = pool.imap(calculator, self.SedIter(sample_iter), chunksize=100)
            for progress, (photo, ebv_corr, shift_corr) in enumerate(elements):

                # Report the progress
                if progress_listener is not None:
                    progress_listener(progress)

                # Update the photo_list_map
                for f in photo.dtype.names:
                    photo_list[f].append(photo[f][0])
                    ebv_corr_list[f].append(ebv_corr[f])
                    shift_corr_list[f].append(shift_corr[f])

        # Convert the photometry lists to numpy arrays
        result_map = {}
        ebv_corr_map = {}
        shift_corr_map = {}
        for f in photo_list:
            result_map[f] = np.asarray(photo_list[f], dtype=np.float32)
            ebv_corr_map[f] = np.asarray(ebv_corr_list[f], dtype=np.float32)
            shift_corr_map[f] = np.asarray(shift_corr_list[f], dtype=np.float32)

        return result_map, ebv_corr_map, shift_corr_map
