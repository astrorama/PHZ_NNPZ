"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc

from nnpz.utils import Logging


logger = Logging.getLogger('Configuration')


_handler_map = {}

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
    def addHandler(handler_type):
        assert issubclass(handler_type, ConfigManager.ConfigHandler)
        _handler_map[handler_type] = handler_type()


    @staticmethod
    def getHandler(handler_type):
        return _handler_map.get(handler_type)


    def __init__(self, args):
        self.__objects = {}
        for handler in _handler_map.values():
            self.__objects.update(handler.parseArgs(args))


    def getAvailableObjectList(self):
        return self.__objects.keys()


    def getObject(self, name):
        return self.__objects[name]