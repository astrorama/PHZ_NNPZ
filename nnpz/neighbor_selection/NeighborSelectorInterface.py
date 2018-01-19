"""
Created on: 18/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc

from nnpz.exceptions import *


class NeighborSelectorInterface(object):
    """Interface for selecting neighbors from a reference sample.
    """
    __metaclass__ = abc.ABCMeta


    def __init__(self):
        self.__dimensionality = None


    @abc.abstractmethod
    def _initializeImpl(self, ref_data):
        """Must be implemented to initialize the object with the reference data.

        Args:
            ref_data: The data of the reference sample to use when searching for
                neighbors. It is a three dimensional numpy array of single
                precision floating point numbers with the following dimensions:
                - First axis: Same size as the number of objects in the sample
                - Second axis: Size of dimensionality of the parameter space
                - Third axis: Always size two, where first element represents
                    the coordinate value and the second the uncertainty

        Note that the dimensions of the ref_data are guaranteed to be correct.
        The implementations do not need to perform any extra checks.
        """
        return


    def initialize(self,ref_data):
        """Initializes the selector with the given reference data.

        Args:
            ref_data: The data of the reference sample to use when searching for
                neighbors. It is a three dimensional numpy array of single
                precision floating point numbers with the following dimensions:
                - First axis: Same size as the number of objects in the sample
                - Second axis: Size of dimensionality of the parameter space
                - Third axis: Always size two, where first element represents
                    the coordinate value and the second the uncertainty

        Returns:
            The self object, to allow for chaining method calls.

        Raises:
            InvalidDimensionsException: If the ref_data parameter has wrong dimensions

        Internally this method delegates the initialization to the children
        _initializeImpl() method.
        """

        if ref_data.ndim != 3:
            raise InvalidDimensionsException('ref_data must have three dimensions')
        if ref_data.shape[2] != 2:
            raise InvalidDimensionsException('ref_data third axis must have size 2')

        self._initializeImpl(ref_data)
        self.__dimensionality = ref_data.shape[1]

        return self


    @abc.abstractmethod
    def _findNeighborsImpl(self, coordinate):
        """Must be implemented to find the neighbors in the reference sample.

        Args:
            coordinate: The coordinates to search the neighbors for. It is a
                two dimensional numpy array of single precision floats with the
                following dimensions:
                - First axis: Size of dimensionality of the parameter space
                - Second axis: Always size two, where first element represents
                    the coordinate value and the second the uncertainty

        Returns:
            The indices of the neighbors in the first axis of the ref_data given
            to the initialize() method, as a numpy array of long integers.

        Implementations are allowed to return any number of neighbors. Note that
        this method is guaranteed to be executed after the _initializeImpl() has
        been called. Also the coordinate parameter is guaranteed to have the
        correct dimensions and no extra validations need to be performed.
        """
        return


    def findNeighbors(self, coordinate):
        """Returns the neighbors of the given coordinate in the reference sample.

        Args:
            coordinate: The coordinates to search the neighbors for. It is a
                two dimensional numpy array of single precision floats with the
                following dimensions:
                - First axis: Size of dimensionality of the parameter space
                - Second axis: Always size two, where first element represents
                    the coordinate value and the second the uncertainty

        Returns:
            The indices of the neighbors in the first axis of the ref_data given
            to the initialize() method, as a numpy array of long integers.

        Raises:
            UninitializedException: If the initialize() method is not called yet
            InvalidDimensionsException: If the given coordinate has wrong dimensions

        Note that this method can return any number of neighbors. The returned
        neighbors are not ordered in any way.
        """

        if self.__dimensionality is None:
            raise UninitializedException('Uninitialized selector')

        if coordinate.ndim != 2:
            raise InvalidDimensionsException('coordinate must have two dimensions')
        if coordinate.shape[0] != self.__dimensionality:
            raise InvalidDimensionsException('Invalid parameter space size (' +
                coordinate.shape[0] + ' instead of ' + self.__dimensionality + ')')
        if coordinate.shape[1] != 2:
            raise InvalidDimensionsException('coordinate second axis must have size 2')

        return self._findNeighborsImpl(coordinate)