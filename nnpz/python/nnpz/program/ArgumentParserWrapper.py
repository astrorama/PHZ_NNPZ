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
Created on: 15/08/19
Author: Alejandro Alvarez Ayllon
"""

import argparse


class ArgumentParserWrapper(object):
    """
    Wraps an ArgumentParser, "tricking" Elements, so the configuration file can be handled
    by us directly (it is Python code).
    It does so via via the attribute _actions, parse_args and parse_known_args
    """

    def __init__(self, *args, **kwargs):
        self.__parser = argparse.ArgumentParser(*args, **kwargs)

    def add_argument_group(self, *args, **kwargs):
        return self.__parser.add_argument_group(*args, **kwargs)

    def parse_known_args(self, *args, **kwargs):
        """
        One of the places where we trick Elements: we intercept the --config-file, and let Element
        parse /dev/null.
        We can not pass the actual config file, because otherwise Elements is going to convert every line to
        an argument
        """
        args, extra = self.__parser.parse_known_args(*args, **kwargs)
        args.config_file = '/dev/null'
        return args, extra

    @property
    def _actions(self):
        """
        Second place for the trick: we return a dummy action that will say it knows about any argument.
        Additional arguments are evaluated as Python code as well (suppressing the --), so we can not use the
        regular argument parser to do this.
        """
        class DummyAction:
            @property
            def option_strings(self):
                return self

            @property
            def dest(self):
                return None

            def __contains__(self, item):
                return True

        return [DummyAction()]

    def parse_args(self, *args, **kwargs):
        """
        Instead of using parse_args, we use parse_known_args, so the extra arguments can be handled
        by the ConfigManager
        """
        arguments, extra_arguments = self.__parser.parse_known_args(*args, **kwargs)
        setattr(arguments, 'extra_arguments', extra_arguments)
        return arguments
