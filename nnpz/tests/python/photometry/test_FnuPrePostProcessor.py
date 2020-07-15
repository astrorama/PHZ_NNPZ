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

from nnpz.photometry import FnuPrePostProcessor


###############################################################################

def test_preProcess():
    """Test the preProcess() method"""

    # Given
    sed = np.asarray([(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.2)], dtype=np.float32)
    # Photon Equation
    expected = np.asarray([(1, 0.1), (2, 0.2), (3, 0.6), (4, 0.8)], dtype=np.float32)
    # Energy Equation
    # expected = np.asarray([(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.2)], dtype=np.float32)

    # When
    processor = FnuPrePostProcessor()
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result, expected)

###############################################################################

def test_postProcess():
    """Test the postProcess() method"""

    # Given
    c = 299792458E10
    intensity = c
    filter_name = 'name'
    # Photon Equation
    filter_trans = np.asarray([(1,0), (2,2), (3,3), (4,0)], dtype=np.float32)
    # Energy Equation
    # filter_trans = np.asarray([(1,0), (2,4), (3,9), (4,0)], dtype=np.float32)

    expected = 0.5

    # When
    processor = FnuPrePostProcessor()
    result = processor.postProcess(intensity, filter_name, filter_trans)

    # Then
    assert result == pytest.approx(expected)


###############################################################################

def test_lambdaZero():
    """
    A lambda of 0 does not really make sense, but there is no reason why a filter could
    not be defined starting at an arbitrary low lambda (i.e. 10), and, after a filter shift,
    end being 0 or negative.
    As long as the transmission is 0 at those regions, we should be fine just dropping those points
    """

    # Given
    c = 299792458E10
    intensity = c
    filter_name = 'name'
    # Photon Equation
    filter_trans = np.asarray([(-1, 0), (0, 0), (1, 0), (2, 2), (3, 3), (4, 0)], dtype=np.float32)
    expected = 0.5

    # When
    processor = FnuPrePostProcessor()
    result = processor.postProcess(intensity, filter_name, filter_trans)

    # Then
    assert result == pytest.approx(expected)

###############################################################################

def test_lambdaZeroWithValue():
    """
    We can not drop if there is a tranmission at <=0, which doesn't make sense at all.
    """

    # Given
    c = 299792458E10
    intensity = c
    filter_name = 'name'
    # Photon Equation
    filter_trans = np.asarray([(-1, 0), (0, 0.1), (1, 0), (2, 2), (3, 3), (4, 0)], dtype=np.float32)
    expected = 0.5

    # When
    processor = FnuPrePostProcessor()

    # Then
    with pytest.raises(ValueError):
        processor.postProcess(intensity, filter_name, filter_trans)

###############################################################################
