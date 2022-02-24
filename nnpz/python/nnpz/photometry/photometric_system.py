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
from collections import OrderedDict
from typing import Iterable, List, Union

import numpy as np


class PhotometricSystem:
    """
    Model the photometric system used to measure the photometry of a set of sources. i.e.:
    bands and their associated transmissions if known.

    Args:
        bands:
            Either a list of band names, or an *ordered* dictionary of band name -> transmission

    Notes:
        The dictionary *must* be ordered so the band position matches the columns from
        the photometry array.

    See Also:
        nnpz.photometry.Photometry
    """

    def __init__(self, bands: Union[List[str], OrderedDict[str, np.ndarray]]):
        if isinstance(bands, list):
            self.__bands = {b: None for b in bands}
        elif isinstance(bands, OrderedDict):
            self.__bands = bands
        else:
            raise ValueError('bands must be a list or an ordered dictionary')
        self.__band_names = list(self.__bands.keys())

    @property
    def bands(self) -> List[str]:
        """
        Returns:
            The list of band names.
        """
        return self.__band_names

    def get_transmission(self, band: str) -> np.ndarray:
        """
        Args:
            band: str
                Band name.
        Returns:
            Transmission for the given band.
        """
        return self.__bands[band]

    def get_band_indexes(self, bands: List[str]) -> np.ndarray:
        """
        Args:
            bands: List[str]
                List of bands.
        Returns:
            The column indexes corresponding to the given bands.
        """
        return np.array([self.__band_names.index(b) for b in bands])

    def __len__(self) -> int:
        return len(self.__bands)

    def __getitem__(self, bands: Iterable[str]) -> 'PhotometricSystem':
        """
        Args:
            bands: Iterable[str]
                Subset of the bands contained within this system.
        Returns:
            Another PhotometricSystem containing only the bands in bands.
        """
        subset = OrderedDict()
        for i in bands:
            if i not in self.__band_names:
                raise KeyError('Band {} not found'.format(i))
            subset[i] = self.__bands[i]
        return PhotometricSystem(subset)

    def __eq__(self, other) -> bool:
        return isinstance(other, PhotometricSystem) and other.__band_names == self.__band_names

    def __repr__(self) -> str:
        return '{' + ' '.join(self.__band_names) + '}'
