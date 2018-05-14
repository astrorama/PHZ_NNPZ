"""
Created on: 04/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.exceptions import *
from nnpz.photometry import FilterProviderInterface, PhotometryPrePostProcessorInterface, PhotometryCalculator


class ReferenceSamplePhotometryBuilder(object):
    """Class for creating photometry from a reference sample"""


    def __init__(self, filter_provider, pre_post_processor):
        """Creates a new instance of ReferenceSamplePhotometryBuilder

        Args:
            filter_provider: An instance of FilterProviderInterface from where
                the filter data are being retrieved
            pre_post_processor: An instance of PhotometryPrePostProcessorInterface
                which defines the type of photometry being produced

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
            raise WrongTypeException('pre_post_processor must implement PhotometryPrePostProcessorInterface')

        self.__filter_provider = filter_provider
        self.__pre_post_processor = pre_post_processor

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

        self.__filter_map = {}
        for f in filter_list:
            self.__filter_map[f] = self.__filter_provider.getFilterTransmission(f)


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
        calculator = PhotometryCalculator(self.__filter_map, self.__pre_post_processor)

        # Create the result map with empty list assigned to each filter
        photo_list_map = {}
        for f in self.__filter_map:
            photo_list_map[f] = []

        # Iterate through all the elements the iterator points to
        for progress, element in enumerate(sample_iter):

            # Report the progress
            if not progress_listener is None:
                progress_listener(progress)

            # If we have reached a missing SED stop the iteration
            if element.sed is None:
                break

            # Compute the photometry and update the photo_list_map
            photo = calculator.compute(element.sed)
            for f in photo:
                photo_list_map[f].append(photo[f])

        # Convert the photometry lists to numpy arrays
        result_map = {}
        for f in photo_list_map:
            result_map[f] = np.asarray(photo_list_map[f], dtype=np.float32)

        return result_map