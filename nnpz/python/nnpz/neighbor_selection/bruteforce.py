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
from _Nnpz import chi2_bruteforce, euclidean_bruteforce, scaling_factory, ScaleFunctionParams
from nnpz.exceptions import InvalidDimensionsException, UninitializedException
from nnpz.neighbor_selection import SelectorInterface
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry


class BruteForceSelector(SelectorInterface):
    """
    Look for neighbors using a bruteforce method (i.e. compare with every single reference
    object)

    Args:
        k: int
            Number of neighbors
        method: string
            Distance kernel to use
    """

    def __init__(self, k: int, method: str = 'Chi2', scale_prior: str = None):
        self.__k = k
        self.__reference = None
        if method.lower() == 'chi2':
            self.__method = chi2_bruteforce
        elif method.lower() == 'euclidean':
            self.__method = euclidean_bruteforce
        else:
            raise ValueError(f'Unknown distance method {method}')

        params = ScaleFunctionParams(20, 1e-8)
        self.__scaling = scaling_factory(scale_prior, params) if scale_prior else None

    def fit(self, train: Photometry, system: PhotometricSystem):
        """
        See Also:
            SelectorInterface.fit
        """
        self.__reference = train.subsystem(system.bands)

    def query(self, target: Photometry) -> Tuple[np.ndarray, np.ndarray]:
        """
        See Also:
            SelectorInterface.query
        """
        if self.__reference is None:
            raise UninitializedException()
        if target.system != self.__reference.system:
            raise InvalidDimensionsException()
        assert target.unit == self.__reference.unit

        neighbors = np.zeros((len(target), self.__k), dtype=np.uint32)
        scales = np.ones_like(neighbors, dtype=np.float32)

        self.__method(self.__reference.values, target.values.value, scales, neighbors, self.__k,
                      self.__scaling)
        return neighbors, scales
