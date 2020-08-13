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
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.neighbor_selection import NeighborSelectorInterface
from .NeighborSet import NeighborSet


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
        if not isinstance(neighbor_selector, NeighborSelectorInterface):
            raise TypeError(
                'Expected a NeighborSelectorInterface, got {}'.format(type(neighbor_selector))
            )
        self.__selector = neighbor_selector

    def findAffected(self, input_coord_iter, flags_iter, progress_listener=None):
        """Finds the input affected by each of the reference sample objects.

        Args:
            input_coord_iter: An iterable object which returns the coordinates
                of the input sources
            flags_iter: An iterable object which returns the NnpzFlag instances
                to update with the neighbor selection related flags for each
                input source
            progress_listener: A callable object, which will be called with the
                index of the currently processed input object, to report the
                progress of the search.

        Returns:
            A map where the keys are the indices of the reference sample objects
            and values are lists of the input catalog indices that are affected
            by this reference object.
        """
        result = {}
        for i, (in_data, flags) in enumerate(zip(input_coord_iter, flags_iter)):
            if progress_listener:
                progress_listener(i + 1)
            neighbor_indices, distances, scales = self.__selector.findNeighbors(in_data, flags)
            for ref_i, distance, scale in zip(neighbor_indices, distances, scales):
                if ref_i not in result:
                    result[ref_i] = NeighborSet()
                result[ref_i].append(i, distance=distance, scale=scale)
        return result
