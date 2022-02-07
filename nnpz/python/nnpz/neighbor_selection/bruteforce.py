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
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry
from nnpz.utils.distances import chi2


class BruteForceSelector:
    def __init__(self, k: int):
        self.__k = k
        self.__reference = None

    def fit(self, train: Photometry, system: PhotometricSystem):
        self.__reference = train.subsystem(system.bands)

    def query(self, target: Photometry) -> Tuple[np.ndarray, np.ndarray]:
        assert target.unit == self.__reference.unit
        assert target.system == self.__reference.system

        neighbors = np.zeros((len(target), self.__k), dtype=int)
        scales = np.ones_like(neighbors, dtype=np.float32)

        distances = np.zeros(len(self.__reference), dtype=np.float32) * target.unit
        for i, t in enumerate(target):
            chi2(self.__reference.values, t, out=distances)
            neighbors[i, :] = np.argpartition(distances, kth=self.__k)[:self.__k]
        return neighbors, scales
