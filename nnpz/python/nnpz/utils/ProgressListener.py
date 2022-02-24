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
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""
import logging

from ElementsKernel import Logging


class ProgressListener:
    """
    Listen for progress changes, and logs into the terminal at least every 1%

    Args:
        total: int
            Expected number of items to process
        prefix:
            Message to prefix when logging progress changes
        logger:
            Logger to use for writing the output
    """

    def __init__(self, total: int, prefix: str = '', logger: logging.Logger = Logging.getLogger()):
        self.__current = -1
        self.__total = total
        self.__prefix = prefix
        self.__logger = logger

    def __call__(self, i: int):
        if (100 * i) // self.__total > self.__current:
            self.__current = (100 * i) // self.__total
            self.__logger.info(self.__prefix + str(self.__current) + '%')
        if i == self.__total and hasattr(self.__logger, 'finalize'):
            self.__logger.finalize()
