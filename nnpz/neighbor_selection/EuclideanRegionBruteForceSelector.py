"""
Created on: 22/01/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.neighbor_selection import NeighborSelectorInterface, KDTreeSelector, BruteForceSelector

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


    def __init__(self, neighbors_no, brute_force_batch_size=10000):
        """Create a new instance of EuclideanRegionBruteForceSelector.

        Args:
            neighbors_no: The number of closest neighbors to search for
            brute_force_batch_size: The number of Euclidean neighbors to use as
                the batch to search in
        """
        self.__neighbors_no = neighbors_no
        self.__brute_force_batch_size = brute_force_batch_size


    def _initializeImpl(self, ref_data):
        """Initializes the selector with the given data.

        For argument description see the interface documentation.
        """
        self.__ref_data = ref_data
        self.__kdtree = KDTreeSelector(self.__brute_force_batch_size).initialize(ref_data)


    def _findNeighborsImpl(self, coordinate):
        """Returns te n closest neighbors to the given coordinate.

        For argument and return description see the interface documentation.

        For an explanation of the neighbors returned see the class documentation.
        """

        # First we find the euclidean neighbors to run brute force for
        batch, _ = self.__kdtree.findNeighbors(coordinate)

        # Get the slice of the reference data which covers the batch
        batch_data = self.__ref_data[batch,:,:]

        # Now use a BruteForceSelector to find the chi2 neighbors in the batch
        import nnpz.neighbor_selection.brute_force_methods as bfm
        brute_force = BruteForceSelector(bfm.Chi2Distance(), bfm.SmallestSelector(self.__neighbors_no)).initialize(batch_data)
        bf_ids, chi2 = brute_force.findNeighbors(coordinate)

        # Retrieve the IDs of the objects in the full reference sample
        ids = batch[bf_ids]

        return ids, chi2
