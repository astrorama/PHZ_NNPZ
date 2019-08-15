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

log = Logging.getLogger(__name__)


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
        self.__neighbors_no = neighbors_no
        self.__balanced_tree = balanced_tree


    def _initializeImpl(self, ref_data):
        """Initializes the selector with the given data.

        For argument description see the interface documentation.
        """
        def _warn_long_execution():
            log.warning('Building the KD-tree seems to be taking too long')
            log.warning('Some particular cases can trigger a worse-case performance when building the tree')
            log.warning('You can try disabling the creation of a balanced tree')

        # We ignore the errors
        values = ref_data[:,:,0]
        timer = threading.Timer(120, _warn_long_execution)
        timer.start()
        self.__kdtree = spatial.cKDTree(values, balanced_tree=self.__balanced_tree)
        timer.cancel()


    def _findNeighborsImpl(self, coordinate, flags):
        """Returns te n closest neighbors to the given coordinate.

        For argument and return description see the interface documentation.

        WARNING: This method will return the closest neighbors in Euclidean
        distance and it will ignore the uncertainties both of the reference data
        and the given coordinate.
        """

        # We ignore the errors
        values = coordinate[:,0]
        result = self.__kdtree.query(values, k = self.__neighbors_no)

        return np.asarray(result[1], dtype=np.int64), np.asarray(result[0], dtype=np.float32)
