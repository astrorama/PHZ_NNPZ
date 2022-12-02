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
from typing import Tuple

import numpy as np
from ElementsKernel import Logging
from nnpz.exceptions import InvalidDimensionsException, UninitializedException
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry
from sklearn.neighbors import KDTree

logger = Logging.getLogger('KDTreeSelector')


class KDTreeSelector:
    """
    Fastest method, finding the neighbors using Euclidean distance.
    Args:
        k: int
            Number of neighbors
        leafsize: int
            Number of points at which to switch to brute-force
    Warnings:
           All errors are ignored when this method is used.
    """

    def __init__(self, k: int, leafsize: int = 16):
        self.__k = k
        self.__leafsize = leafsize
        self.__kdtree = None
        self.__system = None

    def fit(self, train: Photometry, system: PhotometricSystem):
        """
        See Also:
            SelectorInterface.fit
        """
        self.__kdtree = KDTree(train.get_fluxes(system.bands), leaf_size=self.__leafsize)
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
        _, neighbors = self.__kdtree.query(target.values[:, :, 0], k=self.__k)
        scales = np.ones_like(neighbors, dtype=np.float32)
        return neighbors, scales
