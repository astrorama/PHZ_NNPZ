"""
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.neighbor_selection import NeighborSelectorInterface


class AffectedSourcesFinder(object):
    """Class that maps the reference sample objects to the input sources they affect.

    This class delegates the searching of the neighbors to an instance of the
    NeighborSelectorInterface. Its responsibility is to retrieve the neighbors
    for the full input list of objects and to organize the result based on the
    reference sample order.
    """

    def __init__(self, neighbor_selector):
        """Creates a new AffectedSourcesFinder.

        Args:
            neighbor_selector: The NeighborSelectorInterface implementation to
                be used for finding the neighbors in the reference sample for
                each input source.
        """
        assert isinstance(neighbor_selector, NeighborSelectorInterface)
        self.__selector = neighbor_selector

    def findAffected(self, input_coord_iter, progress_listener=None):
        """Finds the input affected by each of the reference sample objects.

        Args:
            input_coord_iter: An iterable object which returns the coordinates
                of the input sources
            progress_listener: A callable object, which will be called with the
                index of the currently processed input object, to report the
                progress of the search.

        Returns:
            A map where the keys are the indices of the reference sample objects
            and values are lists of the input catalog indices that are affected
            by this reference object.
        """
        result = {}
        for i, in_data in enumerate(input_coord_iter):
            if progress_listener:
                progress_listener(i+1)
            neighbor_indices, _ = self.__selector.findNeighbors(in_data)
            for ref_i in neighbor_indices:
                if not ref_i in result:
                    result[ref_i] = []
                result[ref_i].append(i)
        return result
