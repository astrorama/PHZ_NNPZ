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
from typing import Callable, Tuple

import numpy as np
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry
from nnpz.utils.distances import chi2


class ScaledBruteForceSelector:
    def __init__(self, k: int, scaler: Callable[[Photometry, Photometry], np.ndarray]):
        self.__k = k
        self.__prior = scaler
        self.__reference = None
        self.__scaler = scaler

    def fit(self, train: Photometry, system: PhotometricSystem):
        self.__reference = train.subsystem(system.bands)

    def query(self, target: Photometry) -> Tuple[np.ndarray, np.ndarray]:
        assert target.unit == self.__reference.unit
        assert target.system == self.__reference.system

        neighbors = np.zeros((len(target), self.__k), dtype=int)
        scales = np.ones_like(neighbors, dtype=np.float32)

        distances = np.zeros(len(self.__reference), dtype=np.float32) * target.unit
        for i, t in enumerate(target):
            all_scales = self.__scaler(self.__reference.values, t)
            scaled_ref = self.__reference.values * all_scales[..., np.newaxis, np.newaxis]
            chi2(scaled_ref, t, out=distances)
            neighbors[i] = np.argpartition(distances, kth=self.__k)[:self.__k]
            scales[i] = all_scales[neighbors[i]]
        return neighbors, scales
