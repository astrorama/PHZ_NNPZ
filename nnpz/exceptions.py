"""
Created on: 09/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function


class DuplicateIdException(Exception):
    pass


class IdMismatchException(Exception):
    pass


class InvalidDimensionsException(Exception):
    pass


class InvalidAxisException(Exception):
    pass


class AlreadySetException(Exception):
    pass


class UninitializedException(Exception):
    pass