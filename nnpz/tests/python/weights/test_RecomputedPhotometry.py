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

from nnpz.weights import RecomputedPhotometry
from .fixtures import *

###############################################################################

def test_recomputedPhotometry(reference_sample_fixture, filters_fixture, target_fixture):

    # Given
    filter_map = dict(filters_fixture)
    phot_type = 'F_nu_uJy'
    ebv = target_fixture['ebv']
    filter_means = target_fixture['filter_means']

    # When
    recomputed = RecomputedPhotometry(
        reference_sample_fixture, sorted(filter_map.keys()), filter_map, phot_type, ebv, filter_means
    )

    # Then

    # The first one is not shifted at all, so the peak of the SED falls only on Y
    phot = recomputed(0, 0, None)
    assert phot.shape == (len(filters_fixture), 2)
    assert phot[0, 0] > 1.
    assert phot[1, 0] <= np.finfo(np.float).eps
    assert phot[2, 0] <= np.finfo(np.float).eps

    # The second one shifts VIS +1500, so it touches the peak of the SED
    phot = recomputed(0, 1, None)
    assert phot[0, 0] > 1.
    assert phot[1, 0] <= np.finfo(np.float).eps
    assert phot[2, 0] > 1.

    # The third one shift VIS +1500 (touches first peak) and g +1000 (touches second peak)
    phot = recomputed(0, 2, None)
    assert phot[0, 0] > 1.
    assert phot[1, 0] > 1.
    assert phot[2, 0] > 1.

    # The fourth one shift VIS +1500, g +1000 and Y -999, so Y stops touching the peak
    phot = recomputed(0, 3, None)
    assert phot[0, 0] <= np.finfo(np.float).eps
    assert phot[1, 0] > 1.
    assert phot[2, 0] > 1.

    # The last one has nan, it should behave as the first
    phot = recomputed(0, 4, None)
    assert phot.shape == (len(filters_fixture), 2)
    assert phot[0, 0] > 1.
    assert phot[1, 0] <= np.finfo(np.float).eps
    assert phot[2, 0] <= np.finfo(np.float).eps

###############################################################################

def test_recomputedPhotometryNanMeans(reference_sample_fixture, filters_fixture, target_fixture):

    # Given
    filter_map = dict(filters_fixture)
    phot_type = 'F_nu_uJy'
    ebv = target_fixture['ebv']
    filter_means = target_fixture['filter_means']
    for k in filter_means:
        filter_means[k][:] = np.nan

    # When
    recomputed = RecomputedPhotometry(
        reference_sample_fixture, sorted(filter_map.keys()), filter_map, phot_type, ebv, filter_means
    )

    # Then

    # It should, effectively, behave like the shift is 0
    for i in range(len(filter_means['vis'])):
        phot = recomputed(0, i, None)
        assert phot.shape == (len(filters_fixture), 2)
        assert phot[0, 0] > 1.
        assert phot[1, 0] <= np.finfo(np.float).eps
        assert phot[2, 0] <= np.finfo(np.float).eps
