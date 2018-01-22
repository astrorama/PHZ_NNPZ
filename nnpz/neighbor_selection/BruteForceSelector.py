"""
Created on: 18/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc
import numpy as np

from nnpz.exceptions import *
from nnpz.neighbor_selection import NeighborSelectorInterface


class BruteForceSelector(NeighborSelectorInterface):
    """Brute force implementation of the NeighborSelectorInterface.

    The distance computation and the neighbor selection methods are delegated
    to encapsulated interfaces.
    """


    class DistanceMethodInterface(object):
        """Interface for computing the reference distances for a coordinate"""
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def __call__(self, ref_data_values, ref_data_errors, coord_values, coord_errors):
            """Must be implemented to return the reference distances.

            Args:
                ref_data_values: A two dimensional numpy array containing the
                    reference coordinate values. The first axis has the size of
                    the number of the reference objects and the second the size
                    of the dimensionality of the parameter space.
                ref_data_errors:  A two dimensional numpy array containing the
                    reference coordinate errors. It has the same axes sizes as
                    the ref_data_values.
                coord_values: A 1D numpy array containing the coordinate values
                    to compute the distances for. It has the same size as the
                    second axis of the ref_data_values.
                coord_errors: A 1D numpy array containing the coordinate errors.
                    It has the same size as the coord_values.

            Returns:
                A 1D numpy array containing the distances of the given coordinate
                to all the reference data. It must have size same as the first
                axis of the ref_data_values and the order must match.
            """
            return


    class SelectionMethodInterface(object):
        """Interface for selecting the neighbors based on the distances"""
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def __call__(self, distances):
            """Must be implemented to return the neighbors.

            Args:
                distances: A 1D numpy array containing the distances.

            Returns:
                A 1D numpy array containing the indices of the neighbors selected
                from the distances array.
            """
            return


    def __init__(self, distance_method, selection_method):
        """Creates a new instance of BruteForceSelector

        Args:
            distance_method: The method used for computing the distances between
                a requested coordinate and all the reference data. It must
                implement the DinstanceMethodInterface.
            selection_method: The method used to select the neighbors based on
                the computed distances. It must implement the SelectionMethodInterface.

        Raises:
            WrongTypeException: If any of the given methods does not implement
                the interfaces
        """

        if not isinstance(distance_method, BruteForceSelector.DistanceMethodInterface):
            raise WrongTypeException('The distance_method must implement the DistanceMethodInterface')
        if not isinstance(selection_method, BruteForceSelector.SelectionMethodInterface):
            raise WrongTypeException('The selection_method must implement the SelectionMethodInterface')

        self.__distance = distance_method
        self.__selection = selection_method


    def _initializeImpl(self, ref_data):
        """Initializes the selector with the given data.

        For argument description see the interface documentation.
        """

        self.__ref_data_values = ref_data[:, :, 0]
        self.__ref_data_errors = ref_data[:, :, 1]


    def _findNeighborsImpl(self, coordinate):
        """Returns the neighbors of the given coordinate in the reference sample.

        For argument and return description see the interface documentation.

        Note that the returned neighbors are not required to be ordered in any
        way. Internally this method delegates the distance computation and the
        neighbor selection to the methods given to the constructor.
        """

        obj_values = coordinate[:, 0]
        obj_errors = coordinate[:, 1]

        distances = self.__distance(self.__ref_data_values, self.__ref_data_errors, obj_values, obj_errors)
        neighbor_ids = self.__selection(distances)
        neighbor_distances = distances[neighbor_ids]

        return neighbor_ids, neighbor_distances
