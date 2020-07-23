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


@pytest.fixture
def reference_values():
    # Reference values are situated on a cube around the center
    ref = np.zeros((27, 3, 2))
    i = 0
    for x in np.arange(-1., 1.1):
        for y in np.arange(-1., 1.1):
            for z in np.arange(-1, 1.1):
                ref[i, 0, 0] = x
                ref[i, 1, 0] = y
                ref[i, 2, 0] = z
                i += 1
    return ref


@pytest.fixture
def target_values():
    """
    Return the coordinates for a point centered on the origin.
    Makes it easier to reason about the distances
    """
    return np.array([(0., 0.), (0., 0.), (0., 0.)])
