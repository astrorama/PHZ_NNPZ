"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc

from nnpz.utils import Logging


logger = Logging.getLogger('Configuration')


_handler_list = []

class ConfigManager(object):


    class ConfigHandler(object):
        __metaclass__ = abc.ABCMeta

        def _checkParameterExists(self, param, args):
            if param not in args:
                logger.error('Missing parameter: {}'.format(param))
                exit(-1)

        @abc.abstractmethod
        def parseArgs(self, args):
            pass


    @staticmethod
    def addHandler(handler):
        assert isinstance(handler, ConfigManager.ConfigHandler)
        _handler_list.append(handler)


    def __init__(self, args):
        self.__objects = {}
        for handler in _handler_list:
            self.__objects.update(handler.parseArgs(args))


    def getAvailableObjectList(self):
        return self.__objects.keys()


    def getObject(self, name):
        return self.__objects[name]