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

"""
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""


import sys


class ProgressListener(object):
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

    # noinspection PyMethodMayBeStatic
    class DefaultLogger(object):
        """
        Wrapper for writing directly into stdout
        """

        # pylint: disable=no-self-use
        def info(self, msg):
            """
            Log a message into the standard output
            """
            print('\r{}'.format(' ' * (len(msg) + 3)), end='')
            print('\r{}'.format(msg), end='')
            sys.stdout.flush()

        # pylint: disable=no-self-use
        def finalize(self):
            """
            Print a new line
            """
            print('')

    def __init__(self, total, prefix='', logger=None):
        self.__current = -1
        self.__total = total
        self.__prefix = prefix
        if logger is not None:
            self.__logger = logger
        else:
            self.__logger = ProgressListener.DefaultLogger()

    def __call__(self, i):
        if int(100 * i / self.__total) > self.__current:
            self.__current = int(100 * i / self.__total)
            self.__logger.info(self.__prefix + str(self.__current) + '%')
        if i == self.__total and hasattr(self.__logger, 'finalize'):
            self.__logger.finalize()
