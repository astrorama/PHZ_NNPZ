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

import astropy.units as u
import numpy as np
import pytest
from nnpz.photometry.colorspace import RestFrame
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry


@pytest.fixture
def system() -> PhotometricSystem:
    return PhotometricSystem(['x', 'y', 'z'])


@pytest.fixture
def reference_values(system: PhotometricSystem) -> Photometry:
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
    return Photometry(np.arange(len(ref)), values=ref * u.uJy, system=system, colorspace=RestFrame)


@pytest.fixture
def target_values(system: PhotometricSystem) -> Photometry:
    """
    Return the coordinates for a point centered on the origin.
    Makes it easier to reason about the distances
    """
    photo = np.zeros((1, 3, 2))
    photo[:, :, 1] = 0.00001  # Set the error to something so chi2 works
    return Photometry(np.array([0]), values=photo * u.uJy,
                      system=system, colorspace=RestFrame)
