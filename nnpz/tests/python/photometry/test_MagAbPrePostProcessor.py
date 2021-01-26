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

import math

import numpy as np
from nnpz.photometry import FnuPrePostProcessor, MagAbPrePostProcessor


###############################################################################

def test_preProcess():
    """Test the preProcess() method"""

    # Given
    fnu = FnuPrePostProcessor({})
    sed = np.asarray([(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.2)], dtype=np.float32)
    expected = fnu.preProcess(sed)

    # When
    processor = MagAbPrePostProcessor({})
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result, expected)


###############################################################################

def test_postProcess():
    """Test the postProcess() method"""

    filter_name = 'name'
    filter_trans = np.asarray([(1, 0), (2, 4), (3, 9), (4, 0)], dtype=np.float32)

    # Given
    fnu = FnuPrePostProcessor({filter_name: filter_trans})
    intensity = 1.
    expected = -2.5 * math.log10(fnu.postProcess(intensity, filter_name)) - 48.6

    # When
    processor = MagAbPrePostProcessor({filter_name: filter_trans})
    result = processor.postProcess(intensity, filter_name)

    # Then
    assert result == expected

###############################################################################
