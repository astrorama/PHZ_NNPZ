"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc

from ElementsKernel import Configuration, Logging

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

    def __init__(self, config_file, extra_arguments):
        # Populate values from the configuration file
        if not config_file:
            config_file = Configuration.getConfigurationPath('nnpz.conf', True)
        args = {}
        exec(open(config_file).read(), args)

        self.parseExtraArgs(args, extra_arguments)

        self.__objects = {}
        for handler in _handler_map.values():
            self.__objects.update(handler.parseArgs(args))

    def parseExtraArgs(self, args, extra_arguments):
        """
        Overload/add new key/values to args from additional command-line arguments
        Args:
            args:
                The dictionary to update
            extra_arguments:
                List of additional arguments
        Raises:
            If any of the options is invalid
        """
        i = 0
        while i < len(extra_arguments):
            key = extra_arguments[i]
            if not key.startswith('--'):
                raise ValueError('Invalid option: {}'.format(key))
            key = key[2:]
            if '=' in key:
                key, value = key.split('=', 1)
            else:
                i += 1
                value = extra_arguments[i]

            try:
                args[key] = eval(value)
            except Exception:
                # If the evaluation failed use the argument as a string
                args[key] = value

            i += 1

    def getAvailableObjectList(self):
        return self.__objects.keys()

    def getObject(self, name):
        return self.__objects[name]
