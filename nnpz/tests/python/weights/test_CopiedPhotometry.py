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
Created: 11/04/2018
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

from nnpz.flags import NnpzFlag
from nnpz.reference_sample import PhotometryProvider
from nnpz.weights import CopiedPhotometry
from ..reference_sample.fixtures import *


###############################################################################

def test_copiedPhotometryAll(photometry_file_fixture):

    # Given
    provider = PhotometryProvider(photometry_file_fixture)

    # When
    copied = CopiedPhotometry(provider.getData())

    # Then
    for ref_i in range(len(provider.getIds())):
        phot = copied(ref_i, 1, NnpzFlag())
        assert np.array_equal(phot, provider.getData()[ref_i])
