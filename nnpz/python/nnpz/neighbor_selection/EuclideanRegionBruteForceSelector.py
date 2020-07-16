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
Created on: 22/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from nnpz.exceptions import InvalidDimensionsException
from nnpz.neighbor_selection import NeighborSelectorInterface, KDTreeSelector, BruteForceSelector
from nnpz.neighbor_selection.brute_force_methods import Chi2Distance, SmallestSelector

class EuclideanRegionBruteForceSelector(NeighborSelectorInterface):
    """Implementation of NeighborSelectorInterface which combines KDTree and brute force.

    This method initially uses a KDTree to efficiently find a big batch of
    closest neighbors in Euclidean distance. Then it uses brute force to return
    the closest neighbors in the batch, using chi2 distance. When the size of
    the batch is the same with the size of the neighbors, the behavior is
    identical with the KDTree search, meaning that the returned neighbors are
    the closest ones in Euclidean distance. When the size of the batch is the
    same as the reference sample, the behavior is identical with a brute force
    search using chi2, meaning that the returned neighbors are the closest ones
    in chi2 distance. Any size of the batch between these two values will return
    more accurately the neighbors using chi2 distance than the KDTree, but it
    will have a penalty at performance. Note that even for batches of big sizes
    (like 10.000 objects) this method will perform fairly well compared with the
    KDTree (around 10 times slower, but still with O(logn) complexity). The
    complexity of the algorithm is O(log n_tot) + O(n_batch), where n_tot is the
    total size of the reference sample and n_batch is the size of the Euclidean
    batch.
    """

    def __init__(self, neighbors_no, brute_force_batch_size=10000, balanced_tree=True,
                 distance_method=Chi2Distance()):
        """Create a new instance of EuclideanRegionBruteForceSelector.

        Args:
            neighbors_no: The number of closest neighbors to search for
            brute_force_batch_size: The number of Euclidean neighbors to use as
                the batch to search in
            balanced: If true, the median will be used to split the data, generating
                a more compact tree
        """
        super(EuclideanRegionBruteForceSelector, self).__init__()
        self.__neighbors_no = neighbors_no
        self.__brute_force_batch_size = brute_force_batch_size
        self.__balanced_tree = balanced_tree
        self.__distance = distance_method
        self.__kdtree = None
        self.__ref_data = None

    def _initializeImpl(self, ref_data):
        """Initializes the selector with the given data.

        For argument description see the interface documentation.
        """
        if self.__brute_force_batch_size > len(ref_data):
            raise InvalidDimensionsException(
                'The batch size is bigger than the number of reference objects: {} > {}'.format(
                    self.__brute_force_batch_size, len(ref_data)
                )
            )
        self.__ref_data = ref_data
        self.__kdtree = KDTreeSelector(self.__brute_force_batch_size,
                                       self.__balanced_tree).initialize(ref_data)

    def _findNeighborsImpl(self, coordinate, flags):
        """Returns te n closest neighbors to the given coordinate.

        For argument and return description see the interface documentation.

        For an explanation of the neighbors returned see the class documentation.
        """

        # If we do not have NaN values in the coordinate we use the KDTree to
        # reduce the number of euclidean neighbors we run brute force for. If
        # there are NaN values, we cannot use the KDTree, so we run brute force
        # for the full reference sample
        if np.isnan(coordinate).any():
            batch = np.arange(self.__ref_data.shape[0])
        else:
            batch, _, _ = self.__kdtree.findNeighbors(coordinate, flags)

        # Get the slice of the reference data which covers the batch
        batch_data = self.__ref_data[batch, :, :]

        # Now use a BruteForceSelector to find the chi2 neighbors in the batch
        brute_force = BruteForceSelector(
            self.__distance, SmallestSelector(self.__neighbors_no)
        ).initialize(batch_data)
        bf_ids, chi2, scales = brute_force.findNeighbors(coordinate, flags)

        # Retrieve the IDs of the objects in the full reference sample
        ids = batch[bf_ids]

        return ids, chi2, scales
