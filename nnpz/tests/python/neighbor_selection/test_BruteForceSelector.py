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

from nnpz.exceptions import UninitializedException, InvalidDimensionsException
from nnpz.neighbor_selection import BruteForceSelector
from nnpz.neighbor_selection.brute_force_methods import EuclideanDistance, LessThanSelector, SmallestSelector

from .fixtures import *


###############################################################################

def test_BruteForceNotInitialized(target_values):
    """
    Querying before initializing must throw
    """
    bf_selector = BruteForceSelector(EuclideanDistance(), LessThanSelector(1.))
    with pytest.raises(UninitializedException):
        bf_selector.findNeighbors(target_values, None)


###############################################################################

def test_BruteForceInvalidDimensions(reference_values, target_values):
    """
    Querying with an invalid dimensionality must throw
    """
    bf_selector = BruteForceSelector(EuclideanDistance(), LessThanSelector(1.))
    bf_selector.initialize(reference_values)
    with pytest.raises(InvalidDimensionsException):
        bf_selector.findNeighbors(target_values[:, 1], None)


###############################################################################

def test_BruteForceLessThan(reference_values, target_values):
    """
    Query neighbors closer than 1.01 (because is strict less than!)
    Points centered on the faces and the center fall there

    See Also: 3d_neighbors.png
    """
    bf_selector = BruteForceSelector(EuclideanDistance(), LessThanSelector(1.01))
    bf_selector.initialize(reference_values)

    idx, distances, scales = bf_selector.findNeighbors(target_values, None)
    # Only the center point and those that are within a sphere of radius 1.01
    assert (len(idx) == len(distances))
    assert (len(idx) == 7)
    assert (np.all(distances <= 1.))
    assert (np.all(scales == 1.))


###############################################################################

def test_BruteForceSmallest(reference_values, target_values):
    """
    Query only for the closest neighbor, which is the center
    """
    bf_selector = BruteForceSelector(EuclideanDistance(), SmallestSelector(1))
    bf_selector.initialize(reference_values)

    idx, distances, scales = bf_selector.findNeighbors(target_values, None)
    # Only the center is a hit
    assert (len(idx) == len(distances))
    assert (len(idx) == 1)
    assert (np.all(distances <= 0.1))
    assert (np.all(scales == 1.))


###############################################################################

def test_BruteForceLessThanNan(reference_values, target_values):
    """
    Query for neighbors closer than 1.01, but this time the target has a NaN, so
    effectively we are flattening the cube along that axis, into a square.
    Some points that were away due to the Z dimension, become a hit.

    See Also: 2d_neighbors.png
        Note that in each visible point there are 3 with the same X,Y coordinate (3 different Z)
    """
    bf_selector = BruteForceSelector(EuclideanDistance(), LessThanSelector(1.01))
    bf_selector.initialize(reference_values)

    target_values[2, :] = np.nan
    idx, distances, scales = bf_selector.findNeighbors(target_values, None)
    # Only the center point and those that are within a circle of radius 1.01
    assert (len(idx) == len(distances))
    assert (len(idx) == 15)
    assert (np.all(distances <= 1.))
    assert (np.all(scales == 1.))


###############################################################################

def test_BruteForceSmallest(reference_values, target_values):
    """
    Query for neighbors closer than 1.01, but this time the target has a NaN, so
    effectively we are flattening the cube along that axis, into a square.
    Some points that were away due to the Z dimension, become a hit.

    See Also: 2d_neighbors.png
        Note that in each visible point there are 3 with the same X,Y coordinate (3 different Z)
    """
    bf_selector = BruteForceSelector(EuclideanDistance(), SmallestSelector(1))
    bf_selector.initialize(reference_values)

    target_values[2, :] = np.nan
    idx, distances, scales = bf_selector.findNeighbors(target_values, None)
    # Only the center is a hit
    assert (len(idx) == len(distances))
    assert (len(idx) == 1)
    assert (np.all(distances <= 0.1))
    assert (np.all(scales == 1.))
