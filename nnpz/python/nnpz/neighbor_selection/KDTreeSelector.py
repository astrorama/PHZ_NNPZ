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

import threading

import numpy as np
import scipy.spatial as spatial
from ElementsKernel import Logging
from nnpz.neighbor_selection import NeighborSelectorInterface

logger = Logging.getLogger('KDTreeSelector')


class KDTreeSelector(NeighborSelectorInterface):
    """Implementation of the NeigborSelectorInterface based on KDTree.

    This implementation organizes the reference data in a KDTree and it uses
    this structure to speed up the process of searching for the n neighbors
    with the closest Euclidean distance.

    WARNING: This implementation ignores the uncertainties for both the
    reference sample and requested objects. The neighbors returned are the
    closest neighbors using Euclidean distance.
    """

    def __init__(self, neighbors_no, balanced_tree=True):
        """Create a new instance of KDTreeSelector.

        Args:
            neighbors_no: The number of closest neighbors to search for
            balanced_tree: If true, the median will be used to split the data, generating
                a more compact tree
        """
        super(KDTreeSelector, self).__init__()
        self.__neighbors_no = neighbors_no
        self.__balanced_tree = balanced_tree

    def _initializeImpl(self, ref_data):
        """Initializes the selector with the given data.

        For argument description see the interface documentation.
        """

        def _warn_long_execution():
            logger.warning('Building the KD-tree seems to be taking too long')
            logger.warning(
                'Some particular cases can trigger a worse-case performance when building the tree')
            logger.warning('You can try disabling the creation of a balanced tree')

        # We ignore the errors
        values = ref_data[:, :, 0]
        timer = threading.Timer(120, _warn_long_execution)
        timer.start()
        # False positive of pylint
        # pylint: disable=no-member
        self.__kdtree = spatial.cKDTree(values, balanced_tree=self.__balanced_tree)
        timer.cancel()

    def _findNeighborsImpl(self, coordinate, flags):
        """Returns the n closest neighbors to the given coordinate.

        For argument and return description see the interface documentation.

        WARNING: This method will return the closest neighbors in Euclidean
        distance and it will ignore the uncertainties both of the reference data
        and the given coordinate.
        """

        # We ignore the errors
        values = coordinate[:, 0]
        result = self.__kdtree.query(values, k = self.__neighbors_no)

        neighbor_ids = np.asarray(result[1], dtype=np.int64)
        neighbor_distances = np.asarray(result[0], dtype=np.float32)
        return neighbor_ids, neighbor_distances, np.ones(result[0].shape)