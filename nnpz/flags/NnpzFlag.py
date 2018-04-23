"""
Created on: 19/04/2018
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

_flag_names = [
    'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
]

if len(_flag_names) != len(set(_flag_names)):
    raise Exception('{}._flag_names contains duplicate entries'.format('__name__'))

# The flag info keys are tuples containing the following:
# - The index of the flag byte in the array
# - A byte with the flag bit set
_flag_info_map = {name : (i//8, 1 << (i-(i//8)*8)) for i, name in enumerate(_flag_names)}

_array_size = max([x for x,_ in _flag_info_map.values()]) + 1


class _NnpzFlagType(type):
    """The metaclass of the NnpzFlag class. It adds to the class one attribute
    per available flag"""

    def __getattribute__(self, name):
        """Overriden to return a new instance of NnpzFlag when the name is a valid
        flag name. We need to create a new instance because the flags are mutable."""
        if name in _flag_info_map:
            return NnpzFlag(name)
        else:
            return super(_NnpzFlagType, self).__getattribute__(name)

    def __dir__(self):
        """Implemented to support autocomplete"""
        return ['getFlagNames()'] + _flag_names


class NnpzFlag():

    __metaclass__ = _NnpzFlagType

    @staticmethod
    def getFlagNames():
        return _flag_names

    @staticmethod
    def getArraySize():
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
        for f in _flag_names:
            if not (self & NnpzFlag(f)).isClear():
                res += ', ' + f
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
        return bool(np.all(self.__array == 0))

    def __bool__(self):
        return not self.isClear()

    def __nonzero__(self):
        return self.__bool__()

    def isSet(self, flag):
        return not (self & flag).isClear()

    def asArray(self):
        return self.__array.copy()
