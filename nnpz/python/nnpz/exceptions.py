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
NNPZ specific exception types
"""

from __future__ import division, print_function


class NnpzException(Exception):
    """
    Base class for NNPZ specific exceptions
    """
    pass


class DuplicateIdException(NnpzException):
    """
    The reference sample already has an object with the given ID
    """
    pass


class IdMismatchException(NnpzException):
    """
    Tried to set an attribute for an object ID that has not been registered
    """
    pass


class InvalidDimensionsException(NnpzException):
    """
    The shape and/or dimensionality of a numpy array does not match the expectations
    """
    pass


class InvalidAxisException(NnpzException):
    """
    The axis values of an array do not match the expectations
    """
    pass


class AlreadySetException(NnpzException):
    """
    Tried to re-define an attribute on an existing reference object
    """
    pass


class UninitializedException(NnpzException):
    """
    The initialization sequence has not been done or done incorrectly
    """
    pass


class FileNotFoundException(NnpzException):
    """
    File does not exist
    """
    pass


class UnknownNameException(NnpzException):
    """
    An input table misses some required attributes
    """
    pass


class AmbiguityException(NnpzException):
    """
    TODO: Unused
    """
    pass


class CorruptedFileException(NnpzException):
    """
    Either the file has been corrupted, or received a file with the wrong format
    """
    pass


class InvalidPositionException(CorruptedFileException):
    """
    The file format is not corrupted, but an index value is invalid
    """
    pass


class InvalidPathException(NnpzException):
    """
    Equivalent to NotADirectoryError
    """
    pass


class WrongTypeException(NnpzException):
    """
    A class does not implement an expected method
    """
    pass


class WrongFormatException(NnpzException):
    """
    A file does not follow the expected format
    """
    pass


class MissingDataException(NnpzException):
    """
    Required data is not present
    """
    pass
