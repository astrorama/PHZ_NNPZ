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

from __future__ import division, print_function

from nnpz.weights import CorrectedPhotometry

from .fixtures import *


###############################################################################

def test_correctedPhotometry(reference_photo_fixture, target_fixture):
    # Given
    ebv = target_fixture['ebv']
    filter_means = target_fixture['filter_means']
    ref_photo = reference_photo_fixture.getData()
    filter_list = reference_photo_fixture.getFilterList()

    # When
    corrected = CorrectedPhotometry(
        reference_photo_fixture, filter_means.keys(), ebv, filter_means
    )

    # Then

    # The first one is not shifted at all
    phot = corrected(1, 0, None)
    assert len(phot.dtype) == len(filter_list)
    assert phot['Y'].shape == (2,)
    assert np.isclose(phot['Y'][0], ref_photo[1, filter_list.index('Y'), 0])
    assert np.isclose(phot['g'][0], ref_photo[1, filter_list.index('g'), 0])
    assert np.isclose(phot['vis'][0], ref_photo[1, filter_list.index('vis'), 0])

    # The second one shifts VIS +1500
    phot = corrected(1, 1, None)
    assert np.isclose(phot['Y'][0], ref_photo[1, filter_list.index('Y'), 0])
    assert np.isclose(phot['g'][0], ref_photo[1, filter_list.index('g'), 0])
    assert np.isclose(phot['vis'][0], 2250001 * ref_photo[1, filter_list.index('vis'), 0])

    # The third one shift VIS +1500 and g +1000
    phot = corrected(1, 2, None)
    assert np.isclose(phot['Y'][0], ref_photo[1, filter_list.index('Y'), 0])
    assert np.isclose(phot['g'][0], 1001 * ref_photo[1, filter_list.index('g'), 0])
    assert np.isclose(phot['vis'][0], 2250001 * ref_photo[1, filter_list.index('vis'), 0])

    # The fourth one shift VIS +1500, g +1000 and Y -999 (but Y is constant!)
    phot = corrected(1, 3, None)
    assert np.isclose(phot['Y'][0], ref_photo[1, filter_list.index('Y'), 0])
    assert np.isclose(phot['g'][0], 1001 * ref_photo[1, filter_list.index('g'), 0])
    assert np.isclose(phot['vis'][0], 2250001 * ref_photo[1, filter_list.index('vis'), 0])

    # The last one has nan, it should behave as the first
    phot = corrected(1, 4, None)
    assert len(phot.dtype) == len(reference_photo_fixture.getFilterList())
    assert phot['Y'].shape == (2,)
    assert np.isclose(phot['Y'][0], ref_photo[1, filter_list.index('Y'), 0])
    assert np.isclose(phot['g'][0], ref_photo[1, filter_list.index('g'), 0])
    assert np.isclose(phot['vis'][0], ref_photo[1, filter_list.index('vis'), 0])


###############################################################################

def test_correctedPhotometryNanMeans(reference_photo_fixture, target_fixture):
    # Given
    ebv = target_fixture['ebv']
    ref_photo = reference_photo_fixture.getData()
    filter_list = reference_photo_fixture.getFilterList()
    filter_means = target_fixture['filter_means']
    for k in filter_means:
        filter_means[k][:] = np.nan

    # When
    recomputed = CorrectedPhotometry(
        reference_photo_fixture, filter_means.keys(), ebv, filter_means
    )

    # Then
    # It should, effectively, behave like the shift is 0
    for i in range(len(filter_means['vis'])):
        phot = recomputed(1, i, None)
        print(phot)
        assert len(phot.dtype) == len(reference_photo_fixture.getFilterList())
        assert np.isclose(phot['Y'][0], ref_photo[1, filter_list.index('Y'), 0])
        assert np.isclose(phot['g'][0], ref_photo[1, filter_list.index('g'), 0])
        assert np.isclose(phot['vis'][0], ref_photo[1, filter_list.index('vis'), 0])
