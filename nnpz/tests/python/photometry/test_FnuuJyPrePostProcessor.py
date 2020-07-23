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

import pytest
import numpy as np

from nnpz.photometry import FnuPrePostProcessor, FnuuJyPrePostProcessor


###############################################################################

def test_preProcess():
    """Test the preProcess() method"""

    # Given
    fnu = FnuPrePostProcessor()
    sed = np.asarray([(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.2)], dtype=np.float32)
    expected = fnu.preProcess(sed)

    # When
    processor = FnuuJyPrePostProcessor()
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result, expected)

###############################################################################

def test_postProcess():
    """Test the postProcess() method"""

    # Given
    fnu = FnuPrePostProcessor()
    intensity = 1.
    filter_name = 'name'
    filter_trans = np.asarray([(1,0), (2,4), (3,9), (4,0)], dtype=np.float32)
    expected = fnu.postProcess(intensity, filter_name, filter_trans) * 1E23 * 1E6

    # When
    processor = FnuuJyPrePostProcessor()
    result = processor.postProcess(intensity, filter_name, filter_trans)

    # Then
    assert result == expected


###############################################################################
