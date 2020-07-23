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
from nnpz.neighbor_selection import AdaptiveSelector, KDTreeSelector
from nnpz.neighbor_selection.brute_force_methods import EuclideanDistance, LessThanSelector, SmallestSelector

from .fixtures import *


###############################################################################

def test_AdaptiveNotInitialized(target_values):
    """
    Querying before initializing must throw
    """
    adaptive_selector = AdaptiveSelector(KDTreeSelector(neighbors_no=5), np.zeros((10, 3, 2)))
    with pytest.raises(UninitializedException):
        adaptive_selector.findNeighbors(target_values, None)


###############################################################################

def test_AdaptiveInvalidDimensions(reference_values, target_values):
    """
    Querying with an invalid dimensionality must throw
    """
    adaptive_selector = AdaptiveSelector(KDTreeSelector(neighbors_no=5), np.zeros((10, 3, 2)))
    adaptive_selector.initialize(reference_values)
    with pytest.raises(InvalidDimensionsException):
        adaptive_selector.findNeighbors(target_values[:, 1], None)


###############################################################################

def test_Adaptive(reference_values, target_values):
    """
    Query for the 7 nearest neighbors, which are those falling on the center of each face, plus the
    middle one.

    See Also: 3d_neighbors.png
    """
    adaptive_selector = AdaptiveSelector(KDTreeSelector(neighbors_no=7), target_values[np.newaxis, :, :])
    adaptive_selector.initialize(reference_values)

    idx, distances, scales = adaptive_selector.findNeighbors(target_values, None)
    assert (len(idx) == len(distances))
    assert (len(idx) == 7)
    assert (np.all(distances <= 1.01))
    assert (np.all(scales == 1.))


###############################################################################

def test_KDTree2(reference_values, target_values):
    """
    Query for the 10 nearest neighbors, which are those falling on the center of each face, plus the
    middle one, and some of the vertex. In this case, we check that not all distances are the same.
    """
    adaptive_selector = AdaptiveSelector(KDTreeSelector(neighbors_no=10), target_values[np.newaxis, :, :])
    adaptive_selector.initialize(reference_values)

    idx, distances, scales = adaptive_selector.findNeighbors(target_values, None)
    assert (len(idx) == len(distances))
    assert (len(idx) == 10)
    assert ((distances <= 1.01).sum() == 7)
    assert ((distances > 1.01).sum() == 3)
    assert (np.all(scales == 1.))


###############################################################################

def test_KDTreeNan(reference_values, target_values):
    """
    This time the target has a NaN, so effectively we are flattening the cube along that axis, into a square.
    Some points that were away due to the Z dimension, become a hit.

    We query for 10 nearest, all should be at less than 1.01 as the Z axis is discarded

    See Also: 2d_neighbors.png

    Notes: This is where KDTree breaks, so Adaptive should make it work!
    """
    target_values[2, :] = np.nan
    adaptive_selector = AdaptiveSelector(KDTreeSelector(neighbors_no=10), target_values[np.newaxis, :, :])
    adaptive_selector.initialize(reference_values)

    idx, distances, scales = adaptive_selector.findNeighbors(target_values, None)
    assert (len(idx) == len(distances))
    assert (len(idx) == 10)
    assert (np.all(distances <= 1.01))
    assert (np.all(scales == 1.))
