#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
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
from typing import List, Union

import numpy as np


class PhotometricSystem:
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
        return self.__band_names

    def get_transmission(self, band: str):
        return self.__bands[band]

    def get_band_indexes(self, bands: List[str]) -> np.ndarray:
        return np.array([self.__band_names.index(b) for b in bands])

    def __len__(self):
        return len(self.__bands)

    def __getitem__(self, item):
        subset = OrderedDict()
        for i in item:
            if i not in self.__band_names:
                raise KeyError('Band {} not found'.format(i))
            subset[i] = self.__bands[i]
        return PhotometricSystem(subset)

    def __eq__(self, other):
        return isinstance(other, PhotometricSystem) and other.__band_names == self.__band_names

    def __repr__(self):
        return '{' + ' '.join(self.__band_names) + '}'
