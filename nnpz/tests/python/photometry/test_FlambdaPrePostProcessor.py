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
Created on: 18/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from platform import processor

import pytest
import numpy as np

from nnpz.photometry import FlambdaPrePostProcessor


###############################################################################

def test_preProcess():
    """Test the preProcess() method"""

    # Given
    sed = np.asarray([(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.2)], dtype=np.float32)

    # When
    processor = FlambdaPrePostProcessor({})
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result, sed)

###############################################################################

def test_postProcess():
    """Test the postProcess() method"""

    # Given
    intensity = 1
    filter_name = 'name'
    filter_trans = np.asarray([(1,0), (2,4), (3,9), (4,0)], dtype=np.float32)
    expected = 1. / 13.

    # When
    processor = FlambdaPrePostProcessor({filter_name: filter_trans})
    result = processor.postProcess(intensity, filter_name)

    # Then
    assert result == pytest.approx(expected)


###############################################################################
