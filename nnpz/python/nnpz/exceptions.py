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

"""
NNPZ specific exception types
"""



class NnpzException(Exception):
    """
    Base class for NNPZ specific exceptions
    """


class DuplicateIdException(NnpzException):
    """
    The reference sample already has an object with the given ID
    """


class IdMismatchException(NnpzException):
    """
    Tried to set an attribute for an object ID that has not been registered
    """


class InvalidDimensionsException(NnpzException):
    """
    The shape and/or dimensionality of a numpy array does not match the expectations
    """


class InvalidAxisException(NnpzException):
    """
    The axis values of an array do not match the expectations
    """


class AlreadySetException(NnpzException):
    """
    Tried to re-define an attribute on an existing reference object
    """


class UninitializedException(NnpzException):
    """
    The initialization sequence has not been done or done incorrectly
    """


class FileNotFoundException(NnpzException):
    """
    File does not exist
    """


class UnknownNameException(NnpzException):
    """
    An input table misses some required attributes
    """


class CorruptedFileException(NnpzException):
    """
    Either the file has been corrupted, or received a file with the wrong format
    """


class InvalidPositionException(CorruptedFileException):
    """
    The file format is not corrupted, but an index value is invalid
    """


class InvalidPathException(NnpzException):
    """
    Equivalent to NotADirectoryError
    """


class WrongTypeException(NnpzException):
    """
    A class does not implement an expected method
    """


class WrongFormatException(NnpzException):
    """
    A file does not follow the expected format
    """


class MissingDataException(NnpzException):
    """
    Required data is not present
    """
