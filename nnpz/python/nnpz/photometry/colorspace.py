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
import numpy as np


class ColorSpace:
    def __init__(self, **factors):
        self.__factors = factors
        lens = list(
            map(len, filter(lambda x: isinstance(x, (list, np.ndarray)), self.__factors.values()))
        )
        self.__len = lens[0] if lens else 1
        for l in lens:
            assert l == self.__len

    def copy(self):
        return ColorSpace(**{k: v.copy() for k, v in self.__factors.items()})

    def __len__(self):
        return self.__len

    def get(self, item):
        return self.__factors[item]

    def __getitem__(self, item):
        return ColorSpace(**{k: v[item] for k, v in self.__factors.items()})

    def __str__(self) -> str:
        if not self.__factors:
            return 'RestFrame ColorSpace'
        return 'ColorSpace with {} factors:\n\t{}'.format(
            len(self.__factors),
            '\n\t'.join(self.__factors.keys())
        )

    def __eq__(self, other: 'ColorSpace') -> bool:
        if self.__factors.keys() != other.__factors.keys() or len(self) != len(other):
            return False
        equal = np.full(self.__len, True)
        for k, v in self.__factors.items():
            equal &= other.__factors[k] == v
        return equal


RestFrame = ColorSpace()
