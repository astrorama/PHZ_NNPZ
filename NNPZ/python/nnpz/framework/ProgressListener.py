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
