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
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import sys


class ProgressListener(object):

    class DefaultLogger(object):

        def info(self, m):
            print('\r{}'.format(' ' * (len(m) + 3)), end='')
            print('\r{}'.format(m), end='')
            sys.stdout.flush()

        def finalize(self):
            print('')


    def __init__(self, total, prefix='', logger=None):
        self.__current = -1
        self.__total = total
        self.__prefix = prefix
        if not logger is None:
            self.__logger = logger
        else:
            self.__logger = ProgressListener.DefaultLogger()

    def __call__(self, i):
        if int(100 * i / self.__total) > self.__current:
            self.__current = int(100 * i / self.__total)
            self.__logger.info(self.__prefix + str(self.__current) + '%')
        if i == self.__total and hasattr(self.__logger, 'finalize'):
            self.__logger.finalize()
