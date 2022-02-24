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

from collections import namedtuple

import numpy as np

SedIter = namedtuple('Iter', ['sed'])


class SedGenerator:
    """
    Generates lazily a list of SEDs at different redshift positions
    """

    def __init__(self):
        self.__seds = []
        self.__zs = []

    def add(self, sed: np.ndarray, z: np.ndarray):
        """
        Add a new pair of SED / set of redshift to the generator
        Args:
            sed:
                A 2D numpy array, where the first axis corresponds to the number of knots,
                and the second always to 2: wavelength and flux
            z:
                A 1D iterable with different redshifts samples
        """
        self.__seds.append(sed)
        self.__zs.append(z)

    def __iter__(self):
        """
        Generator

        Returns:
            It yields a SedIter that mimics the ReferenceSample iterator, so it can be
            used as a drop-in replacement for the photometry computation
        """
        for sed, zs in zip(self.__seds, self.__zs):
            for z in zs:
                yield SedIter(sed=np.stack([sed[:, 0] * (z + 1), sed[:, 1] / (1 + z) ** 2], axis=1))

    def __len__(self):
        return len(self.__seds) * len(self.__zs[0])
