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

from nnpz.exceptions import UninitializedException, InvalidDimensionsException
from nnpz.neighbor_selection import EuclideanRegionBruteForceSelector
from nnpz.neighbor_selection.brute_force_methods import EuclideanDistance, LessThanSelector, SmallestSelector

from .fixtures import *


###############################################################################

def test_EuclidenRegionBruteForceNotInitialized(target_values):
    """
    Querying before initializing must throw
    """
    erbf_selector = EuclideanRegionBruteForceSelector(7, 3)
    with pytest.raises(UninitializedException):
        erbf_selector.findNeighbors(target_values, None)


###############################################################################

def test_EuclidenRegionBruteForceInvalidDimensions(reference_values, target_values):
    """
    Querying with an invalid dimensionality must throw
    """
    erbf_selector = EuclideanRegionBruteForceSelector(3, 9)
    erbf_selector.initialize(reference_values)
    with pytest.raises(InvalidDimensionsException):
        erbf_selector.findNeighbors(target_values[:, 1], None)


###############################################################################

def test_EuclidenRegionBruteForce(reference_values, target_values):
    """
    Query for the 7 nearest neighbors, which are those falling on the center of each face, plus the
    middle one.
    """
    erbf_selector = EuclideanRegionBruteForceSelector(7, 18, distance_method=EuclideanDistance())
    erbf_selector.initialize(reference_values)

    idx, distances, scales = erbf_selector.findNeighbors(target_values, None)
    assert (len(idx) == len(distances))
    assert (len(idx) == 7)
    assert (np.all(distances <= 1.01))
    assert (np.all(scales == 1.))


###############################################################################

def test_EuclidenRegionBruteForce2(reference_values, target_values):
    """
    Query for the 10 nearest neighbors, which are those falling on the center of each face, plus the
    middle one, and some of the vertex. In this case, we check that not all distances are the same.
    """
    erbf_selector = EuclideanRegionBruteForceSelector(10, 18, distance_method=EuclideanDistance())
    erbf_selector.initialize(reference_values)

    idx, distances, scales = erbf_selector.findNeighbors(target_values, None)
    assert (len(idx) == len(distances))
    assert (len(idx) == 10)
    assert ((distances <= 1.01).sum() == 7)
    assert ((distances > 1.01).sum() == 3)
    assert (np.all(scales == 1.))


###############################################################################

def test_EuclidenRegionBruteForceNan(reference_values, target_values):
    """
    This time the target has a NaN, so effectively we are flattening the cube along that axis, into a square.
    Some points that were away due to the Z dimension, become a hit.

    We query for 10 nearest, all should be at less than 1.01 as the Z axis is discarded

    See Also: 2d_neighbors.png
    """
    erbf_selector = EuclideanRegionBruteForceSelector(10, 18, distance_method=EuclideanDistance())
    erbf_selector.initialize(reference_values)

    target_values[2, :] = np.nan
    idx, distances, scales = erbf_selector.findNeighbors(target_values, None)
    # Only the center is a hit
    assert (len(idx) == len(distances))
    assert (len(idx) == 10)
    assert (np.all(distances <= 1.01))
    assert (np.all(scales == 1.))
