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
File: tests/python/RunPyTest_test.py

Created on: 07/11/18
Author: nikoapos
"""

import os

import pytest
import sys


def _runTests():
    return len([s for s in sys.argv if s.endswith('RunPyTest_test.py')]) == 0


@pytest.mark.skipif(_runTests(), reason='File is not called explicitly')
class TestRunPyTestInDir:
    """
    @class TestTempClass
    @brief Unit Test class
    !!! Test class example for python             !!!
    !!! Please remove it and add your tests there !!!
    """

    def testRunPyTestInDir(self):
        assert pytest.cmdline.main(['--ignore=' + __file__, os.path.dirname(__file__)]) == 0
