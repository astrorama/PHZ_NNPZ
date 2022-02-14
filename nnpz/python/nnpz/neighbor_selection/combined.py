#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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
import threading
from typing import Callable, Tuple

import numpy as np
from nnpz.exceptions import InvalidDimensionsException, UninitializedException
from nnpz.neighbor_selection import SelectorInterface
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry
from scipy import spatial

from .kdtree import _warn_long_execution
from ..utils.distances import chi2


class CombinedSelector(SelectorInterface):
    """
    This method first finds a batch of neighbors in Euclidean
    distance using a KDTree, and then it finds the closest neighbors inside
    the batch, by using chi2 distance.
    Args:
        k: int
            Number of neighbors
        batch: int
            Number of reference objects to look for in Euclidean distance for reducing the
            bruteforce search space 
        balanced: bool
            Train a balanced KDTree. See scipy cKDTree implementation.
        bruteforce: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
            A function that computes the distances from all reference objects (first parameter),
            to a target object (second parameter). A third parameter, out, must be accepted
            to avoid allocating a new array. The method must return out, or a newly allocated
            array if out was None
    """

    def __init__(self, k: int, batch: int, balanced: bool = True,
                 bruteforce: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = chi2):
        self.__k = k
        self.__batch = batch
        self.__balanced = balanced
        self.__reference_photo = None
        self.__kdtree = None
        self.__system = None
        self.__bruteforce_method = bruteforce

    def fit(self, train: Photometry, system: PhotometricSystem):
        """
        See Also:
            SelectorInterface.fit
        """
        timer = threading.Timer(120, _warn_long_execution)
        timer.start()
        # False positive of pylint
        # pylint: disable=no-member
        self.__reference_photo = train
        self.__kdtree = spatial.cKDTree(train.get_fluxes(system.bands),
                                        balanced_tree=self.__balanced)
        timer.cancel()
        self.__system = system

    def query(self, target: Photometry) -> Tuple[np.ndarray, np.ndarray]:
        """
        See Also:
            SelectorInterface.query
        """
        if self.__kdtree is None:
            raise UninitializedException()
        if target.system != self.__system:
            raise InvalidDimensionsException()
        assert target.unit == self.__reference_photo.unit

        _, candidates = self.__kdtree.query(target.values[:, :, 0], k=self.__batch)
        neighbors = np.zeros((len(target), self.__k), dtype=np.int32)
        distances = np.zeros(self.__batch, dtype=np.float32) * target.unit
        for i, t in enumerate(target):
            candidate_photo = self.__reference_photo[candidates[i]]
            self.__bruteforce_method(
                candidate_photo.get_fluxes(self.__system.bands, return_error=True), t,
                out=distances)
            final_neighbors = np.argsort(distances)[:self.__k]
            neighbors[i, :] = candidates[i, final_neighbors]
        scales = np.ones_like(neighbors, dtype=np.float32)
        return neighbors, scales
