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
Created: 11/04/2018
Author: Alejandro Alvarez Ayllon
"""


from nnpz.flags import NnpzFlag
from nnpz.weights import CopiedPhotometry

# noinspection PyUnresolvedReferences
from .fixtures import *


###############################################################################

def test_copiedPhotometryAll(reference_photo_fixture):
    # When
    copied = CopiedPhotometry(reference_photo_fixture)

    # Then
    ref_photo = reference_photo_fixture.getData('Y', 'g', 'vis')
    for ref_i in range(len(reference_photo_fixture.getIds())):
        phot = copied(ref_i, 1, NnpzFlag())
        assert phot['Y'].shape == (2,)
        original = ref_photo[ref_i]
        assert np.array_equal(phot['Y'], original[0])
        assert np.array_equal(phot['g'], original[1])
        assert np.array_equal(phot['vis'], original[2])
