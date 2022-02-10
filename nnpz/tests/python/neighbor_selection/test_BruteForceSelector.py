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

from nnpz.exceptions import InvalidDimensionsException, UninitializedException
from nnpz.neighbor_selection.bruteforce import BruteForceSelector
from nnpz.utils.distances import euclidean

from .fixtures import *


###############################################################################

def test_BruteForceNotInitialized(target_values: Photometry):
    """
    Querying before initializing must throw
    """
    bf_selector = BruteForceSelector(k=4, method=euclidean)
    with pytest.raises(UninitializedException):
        bf_selector.query(target_values)


###############################################################################

def test_BruteForceInvalidDimensions(reference_values: Photometry, target_values: Photometry):
    """
    Querying with an invalid dimensionality must throw
    """
    bf_selector = BruteForceSelector(k=4, method=euclidean)
    bf_selector.fit(reference_values, reference_values.system)
    with pytest.raises(InvalidDimensionsException):
        bf_selector.query(target_values.subsystem(['x', 'y']))


###############################################################################

def test_BruteForceSmallest(reference_values: Photometry, target_values: Photometry):
    """
    Query only for the closest neighbor, which is the center
    """
    bf_selector = BruteForceSelector(k=1, method=euclidean)
    bf_selector.fit(reference_values, reference_values.system)

    idx, scales = bf_selector.query(target_values)
    # Only the center is a hit
    distances = euclidean(reference_values.values[idx[0]], target_values.values[0])
    assert (len(idx) == len(scales))
    assert (len(idx) == 1)
    assert (np.all(distances.value <= 0.1))
    assert (np.all(scales == 1.))
