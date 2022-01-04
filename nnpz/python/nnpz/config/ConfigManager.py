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
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""


import abc

from ElementsKernel import Configuration, Logging

logger = Logging.getLogger('Configuration')

_handler_map = {}


class ConfigManager(object):
    """
    The ConfigManager handles the lifetime and interdependency of the different classes that model
    the configuration of NNPZ.
    Each individual component (input, output, search, scaling, weighting) should have
    its own associated configuration class.
    These classes must inherit from ConfigManager.ConfigHandler, and, in order to register
    them, call `ConfigManager.addHandler(ConfigClass)` at the end of the module
    where they are defined.

    Args:
        config_file: str
            Path to the configuration file. If None, it will default to 'nnpz.conf' under
            any of the known configuration paths (See Elements documentation to know how this is
            resolved)
        extra_arguments: list of str
            List of additional arguments, as strings. They will be evaluated as Python
            expressions, so you could use 'range(10)', for instance.
    """

    class ConfigHandler(object):
        """
        Configuration classes must implement this interface
        """
        __metaclass__ = abc.ABCMeta

        @staticmethod
        def _checkParameterExists(param, args):
            if param not in args:
                logger.error('Missing parameter: %s', param)
                exit(-1)

        @abc.abstractmethod
        def parseArgs(self, args):
            """
            Parse the arguments the class knows about. Ignore any others.
            Args:
                args: dict
                    A dictionary with the configuration key/values
            """

    @staticmethod
    def addHandler(handler_type):
        """
        Register a configuration handler
        Args:
            handler_type: A *type* that inherits from ConfigHandler
        """
        if not issubclass(handler_type, ConfigManager.ConfigHandler):
            raise TypeError('Expected a ConfigHandler, got {}'.format(type(handler_type)))
        _handler_map[handler_type] = handler_type()

    @staticmethod
    def getHandler(handler_type):
        """
        Get the instance (singleton) of a given configuration handler
        Args:
            handler_type: A *type* that inherits from ConfigHandler
        Returns: handler_type
            The registered instance of handler_type
        """
        return _handler_map.get(handler_type)

    def __init__(self, config_file, extra_arguments):
        # Populate values from the configuration file
        if not config_file:
            config_file = Configuration.getConfigurationPath('nnpz.conf', True)
        args = {}
        # pylint: disable=exec-used
        exec(open(config_file).read(), args)

        self._parseExtraArgs(args, extra_arguments)

        self.__objects = {}
        for handler in _handler_map.values():
            self.__objects.update(handler.parseArgs(args))

    @staticmethod
    def _parseExtraArgs(args, extra_arguments):
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
                # pylint: disable=eval-used
                args[key] = eval(value)
            except (SyntaxError, NameError):
                # If the evaluation failed use the argument as a string
                args[key] = value

            i += 1

    def getAvailableObjectList(self):
        """
        Returns: list of str
            Known configuration keys
        """
        return list(self.__objects.keys())

    def getObject(self, name):
        """
        Args:
            name: str
                Configuration key
        Returns: object
            Configuration value
        """
        return self.__objects[name]
