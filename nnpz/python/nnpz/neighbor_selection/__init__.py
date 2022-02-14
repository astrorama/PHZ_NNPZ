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
import abc
from typing import Tuple

import numpy as np
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry


class SelectorInterface(abc.ABC):
    """
    Interface to be implemented by the neighbor selecting algorithms
    """

    @abc.abstractmethod
    def fit(self, train: Photometry, system: PhotometricSystem):
        """
        Train the selector on the given reference sample
        Args:
            train: Photometry
                Reference sample photometry
            system: PhotometricSystem
                Subset from the bands from the reference photometry to use for the matching
        Warnings:
            system *must* match the target photometric system
        """
        pass

    @abc.abstractmethod
    def query(self, target: Photometry) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the selector for a given *set* of targets
        Args:
            target: Photometry
                Target photometry. It *must* match the system passed to fit
        Returns:
            (neighbor_indexes, scaling) : (np.ndarray, np.ndarray)
                The *index* (not the ID!!) of the matched neighbors, and the applied scaling
                factor (which must be 1 for those selectors that do not apply scaling)
        """
        pass
