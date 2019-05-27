"""
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import abc
import os
import sys
from argparse import ArgumentParser

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

    def __init__(self, argv=sys.argv[1:], description=None):
        parser = ArgumentParser(description=description)
        parser.add_argument(
            '--config-file', type=str, metavar='FILE', help='Configuration file'
        )
        arguments, extra_arguments = parser.parse_known_args(argv)
        self.__config_file = arguments.config_file

        # If not specified, try to find the configuration on a default location
        if self.__config_file is None:
            conf_path = os.path.abspath(os.path.dirname(__file__))  # Remove the filename
            conf_path = os.path.dirname(conf_path)  # Remove the bin directory
            self.__config_file = os.path.join(conf_path, 'config', 'nnpz.conf')

        # Populate values from the configuration file
        args = {}
        exec(open(self.__config_file).read(), args)

        try:
            self.parseExtraArgs(args, extra_arguments)
        except Exception as e:
            parser.error(str(e))

        if 'log_level' in args:
            Logging.enableStdErrLogging(args['log_level'])
        else:
            Logging.enableStdErrLogging()

        logger.debug('Configuration file: {}'.format(self.__config_file))
        logger.debug('Running {} with the following options:'.format(parser.prog))
        for key in sorted(args):
            if key.startswith('_'):
                continue
            value = args[key]
            logger.debug('    {} : {}'.format(key, value))
        logger.debug('')

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
