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
from nnpz.neighbor_selection.bruteforce import BruteForceSelector
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry


class ScaledBruteForceSelector:
    def __init__(self, k: int, scaler: Callable[[Photometry, Photometry], np.ndarray]):
        self.__k = k
        self.__prior = scaler
        self.__reference = None
        self.__reference_photo = None
        self.__scaler = scaler

    def fit(self, train: Photometry, system: PhotometricSystem):
        self.__reference = train.subsystem(system.bands)
        self.__reference_photo = train

    def query(self, target: Photometry) -> Tuple[np.ndarray, np.ndarray]:
        assert target.unit == self.__reference.unit
        assert target.system == self.__reference.system

        scales = self.__scaler(self.__reference, target)
        bruteforce_selector = BruteForceSelector(self.__k)
        bruteforce_selector.fit(self.__reference * scales[:, np.newaxis], self.__reference.system)
        neighbors, _ = bruteforce_selector.query(target)

        return neighbors, scales
