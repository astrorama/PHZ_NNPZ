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
from typing import Any, Dict, List

import pytest
from ElementsKernel.Temporary import TempFile
from nnpz.config.ConfigManager import ConfigManager, _handler_map


###############################################################################

@pytest.fixture
def config_file():
    cfg = TempFile()
    with open(cfg.path(), 'wt') as fd:
        print("""
# Comment
value_int = 1234
value_str = "string"

import numpy as np

parameters = np.arange(10)
""", file=fd)
    return cfg


###############################################################################

@pytest.fixture
def extra_args():
    return ['--foo', '1', '--bar=something', '--otherwise="else"', '--life', 'itis42']


###############################################################################

class MockConfigHandler(ConfigManager.ConfigHandler):

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._exists_parameter('foo', args)

        for k, v in args.items():
            setattr(self, k, v)
        return {'set-by-the': 'handler'}


###############################################################################

def test_ConfigManager(config_file: TempFile, extra_args: List):
    _handler_map.clear()
    ConfigManager.add_handler(MockConfigHandler)

    config = ConfigManager(config_file.path(), extra_args)

    handler = config.get_handler(MockConfigHandler)
    assert handler is not None
    assert handler.foo == 1
    assert handler.bar == 'something'
    assert handler.otherwise == 'else'
    assert handler.life == 'itis42'

    assert config.get_available_object_list() == ['set-by-the']
    assert config.get('set-by-the') == 'handler'

###############################################################################
