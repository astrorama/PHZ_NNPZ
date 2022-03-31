#
#  Copyright (C) 2022 Euclid Science Ground Segment
#
#  This library is free software; you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation;
#  either version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with this library;
#  if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA
#
from nnpz.neighbor_selection.bruteforce import BruteForceSelector
from nnpz.pipeline.neighbor_finder import NeighborFinder

from ..photometry.fixtures import *


###############################################################################

def test_findNeighborsNoEBV(reference_photometry: Photometry, target_photometry: Photometry):
    finder = NeighborFinder(
        dict(
            reference_photometry=reference_photometry,
            source_independent_ebv=None,
            neighbor_selector=BruteForceSelector(2, method='euclidean'),
            neighbor_no=2,
            target_system=target_photometry.system
        ))
    out = np.zeros(len(target_photometry),
                   dtype=[('NEIGHBOR_INDEX', int, 2), ('NEIGHBOR_SCALING', float, 2),
                          ('NEIGHBOR_PHOTOMETRY', float, (2, len(target_photometry.system), 2))])

    finder(target_photometry, out=out)

    nn_idx = list(map(set, out['NEIGHBOR_INDEX']))

    np.testing.assert_array_equal(out['NEIGHBOR_SCALING'], 1.)

    assert nn_idx[1] in [{0, 1}, {1, 2}]
    assert nn_idx[2] in [{1, 2}, {2, 3}]
    assert nn_idx[3] in [{2, 3}, {3, 4}]
    assert nn_idx[4] == {3, 4}

    assert nn_idx[0] == {0, 1}
    np.testing.assert_array_equal(out['NEIGHBOR_PHOTOMETRY'][0, out['NEIGHBOR_INDEX'][0, 0]],
                                  reference_photometry.values[0].value)
    np.testing.assert_array_equal(out['NEIGHBOR_PHOTOMETRY'][0, out['NEIGHBOR_INDEX'][0, 1]],
                                  reference_photometry.values[1].value)


###############################################################################

def test_findNeighborsEBV(reference_photometry: Photometry, target_photometry: Photometry):
    class MockEBV:
        @u.quantity_input
        def deredden(self, photo: u.uJy, ebv: np.ndarray):
            """
            Flip the order
            """
            assert len(photo) == len(target_photometry)
            assert len(photo) == len(ebv)
            return np.flip(photo, axis=0).copy()

    print(target_photometry.values.strides)
    finder = NeighborFinder(
        dict(
            reference_photometry=reference_photometry,
            source_independent_ebv=MockEBV(),
            neighbor_selector=BruteForceSelector(2, method='euclidean'),
            neighbor_no=2,
            target_system=target_photometry.system
        ))

    out = np.zeros(len(target_photometry),
                   dtype=[('NEIGHBOR_INDEX', int, 2), ('NEIGHBOR_SCALING', float, 2),
                          ('NEIGHBOR_PHOTOMETRY', float, (2, len(target_photometry.system), 2))])

    finder(target_photometry, out=out)

    nn_idx = list(map(set, out['NEIGHBOR_INDEX']))

    np.testing.assert_array_equal(out['NEIGHBOR_SCALING'], 1.)
    assert nn_idx[0] == {3, 4}
    assert nn_idx[1] in [{2, 3}, {3, 4}]
    assert nn_idx[2] in [{1, 2}, {2, 3}]
    assert nn_idx[3] in [{0, 1}, {1, 2}]
    assert nn_idx[4] == {0, 1}

###############################################################################
