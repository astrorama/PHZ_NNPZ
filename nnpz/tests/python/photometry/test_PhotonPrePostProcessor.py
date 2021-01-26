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
Created on: 14/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from platform import processor

import pytest
import numpy as np

from nnpz.photometry import PhotonPrePostProcessor


###############################################################################

def test_preProcess():
    """Test the preProcess() method"""

    # Given
    sed = np.asarray([(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.2)], dtype=np.float32)
    expected = np.ndarray(len(sed), dtype=np.float32)
    expected = sed[:, 1] * 5.03411250E+07 * sed[:, 0]

    # When
    processor = PhotonPrePostProcessor({})
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result[:, 0], sed[:, 0])
    assert result[0][1] == pytest.approx(expected[0])
    assert result[1][1] == pytest.approx(expected[1])
    assert result[2][1] == pytest.approx(expected[2])
    assert result[3][1] == pytest.approx(expected[3])


###############################################################################

def test_postProcess():
    """Test the postProcess() method"""

    # Given
    intensity = 1.5

    # When
    processor = PhotonPrePostProcessor({})
    result = processor.postProcess(intensity, None)

    # Then
    assert result == intensity

###############################################################################
