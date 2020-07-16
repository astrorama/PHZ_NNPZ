#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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

"""
Created on: 19/04/2018
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function
from six import with_metaclass

import numpy as np

_flag_names = [
    'AlternativeWeightFlag',
]

if len(_flag_names) != len(set(_flag_names)):
    raise Exception('{}._flag_names contains duplicate entries'.format('__name__'))

# The flag info keys are tuples containing the following:
# - The index of the flag byte in the array
# - A byte with the flag bit set
_flag_info_map = {name: (i // 8, 1 << (i - (i // 8) * 8)) for i, name in enumerate(_flag_names)}

_array_size = max([x for x, _ in _flag_info_map.values()]) + 1


class _NnpzFlagType(type):
    """The metaclass of the NnpzFlag class. It adds to the class one attribute
    per available flag"""

    def __getattribute__(cls, name):
        """Overriden to return a new instance of NnpzFlag when the name is a valid
        flag name. We need to create a new instance because the flags are mutable."""
        if name in _flag_info_map:
            return NnpzFlag(name)
        return super(_NnpzFlagType, cls).__getattribute__(name)

    def __dir__(cls):
        """Implemented to support autocomplete"""
        return ['getFlagNames()'] + _flag_names


class NnpzFlag(with_metaclass(_NnpzFlagType, object)):
    """
    Wraps all flags that can be set for a target object
    """

    @staticmethod
    def getFlagNames():
        """
        Returns: list
            List of known flag names
        """
        return _flag_names

    @staticmethod
    def getArraySize():
        """
        Returns: int
            Required array size so all flags can be stored
        """
        return _array_size

    def __init__(self, *flag_list):
        self.__array = np.zeros(_array_size, dtype=np.uint8)
        for flag in flag_list:
            if isinstance(flag, NnpzFlag):
                self.__array |= flag._NnpzFlag__array
            else:
                arr_i, flag_bits = _flag_info_map[flag]
                self.__array[arr_i] |= flag_bits

    def __str__(self):
        res = '['
        res += '.'.join(['{:08b}'.format(x) for x in self.__array])
        for flag in _flag_names:
            if not (self & NnpzFlag(flag)).isClear():
                res += ', ' + flag
        res += ']'
        return res

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return np.all(self.__array == other._NnpzFlag__array)

    def __invert__(self):
        self.__array == ~self.__array

    def __and__(self, other):
        res = NnpzFlag()
        res._NnpzFlag__array = self.__array & other._NnpzFlag__array
        return res

    def __or__(self, other):
        res = NnpzFlag()
        res._NnpzFlag__array = self.__array | other._NnpzFlag__array
        return res

    def __xor__(self, other):
        res = NnpzFlag()
        res._NnpzFlag__array = self.__array ^ other._NnpzFlag__array
        return res

    def __iand__(self, other):
        self.__array &= other._NnpzFlag__array
        return self

    def __ior__(self, other):
        self.__array |= other._NnpzFlag__array
        return self

    def __ixor__(self, other):
        self.__array ^= other._NnpzFlag__array
        return self

    def isClear(self):
        """
        Returns: bool
            True if no flag is set
        """
        return bool(np.all(self.__array == 0))

    def __bool__(self):
        return not self.isClear()

    def __nonzero__(self):
        return self.__bool__()

    def isSet(self, flag):
        """
        Args:
            flag: NnpzFlag
        Returns: bool
            True if the flags inside `flag` are set within self
        """
        return not (self & flag).isClear()

    def asArray(self):
        """
        Returns: numpy.array
            Array with the flag values
        """
        return self.__array.copy()
