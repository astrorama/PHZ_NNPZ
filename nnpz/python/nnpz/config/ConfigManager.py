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
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

import abc
from ast import literal_eval
from typing import Any, Dict, List, Type

from ElementsKernel import Configuration, Logging

logger = Logging.getLogger('Configuration')

_handler_map = {}


class ConfigManager:
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

    class ConfigHandler:
        """
        Configuration classes must implement this interface
        """
        __metaclass__ = abc.ABCMeta

        @staticmethod
        def _exists_parameter(param: str, args: Dict[str, Any]):
            if param not in args:
                logger.error('Missing parameter: %s', param)
                exit(-1)

        @abc.abstractmethod
        def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
            """
            Parse the arguments the class knows about. Ignore any others.
            Args:
                args: dict
                    A dictionary with the configuration key/values
            """

    @staticmethod
    def add_handler(handler_type: Type[ConfigHandler]):
        """
        Register a configuration handler
        Args:
            handler_type: A *type* that inherits from ConfigHandler
        """
        if not issubclass(handler_type, ConfigManager.ConfigHandler):
            raise TypeError('Expected a ConfigHandler, got {}'.format(type(handler_type)))
        _handler_map[handler_type] = handler_type()

    @staticmethod
    def get_handler(handler_type: Type[ConfigHandler]) -> ConfigHandler:
        """
        Get the instance (singleton) of a given configuration handler
        Args:
            handler_type: A *type* that inherits from ConfigHandler
        Returns: handler_type
            The registered instance of handler_type
        """
        return _handler_map.get(handler_type)

    def __init__(self, config_file: str, extra_arguments: List[str]):
        # Populate values from the configuration file
        if not config_file:
            config_file = Configuration.getConfigurationPath('nnpz.conf', True)
        args = {}
        # pylint: disable=exec-used
        exec(open(config_file).read(), args)  # nosec

        self._parse_extra_args(args, extra_arguments)

        self.__objects = {}
        for handler in _handler_map.values():
            self.__objects.update(handler.parse_args(args))

    @staticmethod
    def _parse_extra_args(args: Dict[str, Any], extra_arguments: List[str]):
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
                args[key] = literal_eval(value)
            except (SyntaxError, NameError, ValueError):
                # If the evaluation failed use the argument as a string
                args[key] = value

            i += 1

    def get_available_object_list(self) -> List[str]:
        """
        Returns: list of str
            Known configuration keys
        """
        return list(self.__objects.keys())

    def get(self, name: str) -> Any:
        """
        Args:
            name: str
                Configuration key
        Returns: object
            Configuration value
        """
        return self.__objects[name]
