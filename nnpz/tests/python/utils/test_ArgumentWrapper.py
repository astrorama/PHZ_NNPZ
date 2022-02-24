#
#  Copyright (C) 2022 Euclid Science Ground Segment
#
#  This library is free software; you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation;
#  either version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with this library;
#  if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA
#
import pytest
from nnpz.utils.ArgumentParserWrapper import ArgumentParserWrapper


@pytest.fixture
def elements_parser():
    """
    See ElementsKernel.Program._parseParameters
    """
    parser = ArgumentParserWrapper(description='Mock')
    group = parser.add_argument_group('Generic Options')
    group.add_argument('--config-file', help='Name of a configuration file')
    group.add_argument('--log-file', help='Log file')
    group.add_argument('--log-level', help='Log level')
    return parser


###############################################################################

def test_config_file_hidden(elements_parser: ArgumentParserWrapper):
    args = ['--config-file=Test.py']
    config_file = elements_parser.parse_known_args(args)[0].config_file
    assert config_file == '/dev/null'
    arguments = elements_parser.parse_args(args)
    assert arguments.config_file == 'Test.py'


###############################################################################

def test_parameters(elements_parser):
    args = ['--config-file=Test.py', '--something', '1', '--foo=bar']
    arguments = elements_parser.parse_args(args)
    assert arguments.extra_arguments == ['--something', '1', '--foo=bar']

###############################################################################
