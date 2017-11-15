"""
Created on: 09/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function


class NnpzException(Exception):
    pass


class DuplicateIdException(NnpzException):
    pass


class IdMismatchException(NnpzException):
    pass


class InvalidDimensionsException(NnpzException):
    pass


class InvalidAxisException(NnpzException):
    pass


class AlreadySetException(NnpzException):
    pass


class UninitializedException(NnpzException):
    pass


class FileNotFoundException(NnpzException):
    pass


class UnknownNameException(NnpzException):
    pass


class AmbiguityException(NnpzException):
    pass