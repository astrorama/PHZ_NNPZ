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
import itertools
import multiprocessing
import os

import numpy as np
from ElementsKernel import Logging
from nnpz.photometry import ReferenceSamplePhotometryBuilder, PhotometryCalculator

logger = Logging.getLogger('BuildPhotometry')


class ReferenceSamplePhotometryParallelBuilder(ReferenceSamplePhotometryBuilder):
    """
    Class for creating photometry from a reference sample using multiple cores
    """

    class SedIter(object):
        """
        A generator for only the sed attribute of a reference object.
        """

        def __init__(self, objects, batch_size):
            self.__iter = iter(objects)
            self.__batch_size = batch_size

        def __iter__(self):
            chunk = list(map(lambda o: o.sed, itertools.islice(self.__iter, self.__batch_size)))
            while chunk:
                yield np.stack(chunk, axis=0)
                chunk = list(map(lambda o: o.sed, itertools.islice(self.__iter, self.__batch_size)))

    def __init__(self, filter_provider, pre_post_processor, shifts, ncores=None,
                 batch_size: int = 100):
        """Creates a new instance of ReferenceSamplePhotometryBuilder

        Args:
            filter_provider: An instance of FilterProviderInterface from where
                the filter data are being retrieved
            pre_post_processor: An instance of PhotometryPrePostProcessorInterface
                which defines the type of photometry being produced
            shifts: To be used for the computation of the correction factors
            ncores:
                Number of cores to use
            batch_size:
                Processing chunk size

        Raises:
            WrongTypeException: If the filter_provider is not an implementation
                of FilterProviderInterface
            WrongTypeException: If the pre_post_processor is not an
                implementation of PhotometryPrePostProcessorInterface
        """
        super(ReferenceSamplePhotometryParallelBuilder, self).__init__(
            filter_provider, pre_post_processor, shifts, batch_size=batch_size)
        self.__ncores = os.cpu_count() if not ncores else ncores

    def buildPhotometry(self, n_items: int, sample_iter, progress_listener=None):
        """Computes the photometry of the SEDs the given iterator traverses.

        Args:
            n_items: Number of items to be processed, used to pre-allocate arrays
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
        dtype = [(filter_name, np.float32) for filter_name in self._filter_map]

        # Create the calculator which will be used for the photometry computation
        calculator = PhotometryCalculator(self._filter_map, self._pre_post_processor, self._shifts)

        # Create the result map with empty list assigned to each filter
        photo_list_map = np.zeros(n_items, dtype=dtype)
        photo_corr_map = np.zeros((n_items, 2), dtype=dtype)

        logger.info('Computing photometries using %d processes', self.__ncores)
        with multiprocessing.Pool(self.__ncores) as pool:
            elements = pool.imap(calculator,
                                 ReferenceSamplePhotometryParallelBuilder.SedIter(sample_iter,
                                                                                  self._batch_size),
                                 chunksize=1)
            for progress, (photo, corr) in enumerate(elements):

                # Report the progress
                if progress_listener is not None:
                    progress_listener(progress * self._batch_size)

                # Update the photo_list_map
                batch_slice = slice(progress * self._batch_size, (progress + 1) * self._batch_size)
                photo_list_map[batch_slice] = photo
                photo_corr_map[batch_slice] = corr

        return photo_list_map, photo_corr_map
