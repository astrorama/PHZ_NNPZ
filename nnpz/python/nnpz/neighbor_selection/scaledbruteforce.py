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
from nnpz.scaling.Chi2Scaling import Chi2Scaling


class ScaledBruteForceSelector:
    def __init__(self, k: int, scale_prior: Callable[[float], float],
                 batch_size: int, max_iter: int, rtol: float):
        self.__k = k
        self.__prior = scale_prior
        self.__batch_size = batch_size
        self.__max_iter = max_iter
        self.__rtol = rtol
        self.__reference = None
        self.__reference_photo = None
        self.__scaling = Chi2Scaling(scale_prior, batch_size=batch_size, max_iter=max_iter,
                                     rtol=rtol)

    def fit(self, train: Photometry, system: PhotometricSystem):
        self.__reference = train.subsystem(system.bands)
        self.__reference_photo = train

    def query(self, target: Photometry) -> Tuple[np.ndarray, np.ndarray]:
        assert target.unit == self.__reference.unit
        assert target.system == self.__reference.system

        scales = self.__scaling(self.__reference, target)
        bruteforce_selector = BruteForceSelector(self.__k)
        bruteforce_selector.fit(self.__reference * scales[:, np.newaxis], self.__reference.system)
        neighbors, _ = bruteforce_selector.query(target)

        return neighbors, scales
