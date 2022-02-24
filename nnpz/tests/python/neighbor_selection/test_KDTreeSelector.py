#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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
from nnpz.neighbor_selection.kdtree import KDTreeSelector
from nnpz.utils.distances import euclidean

from .fixtures import *


###############################################################################

def test_KDTreeNotInitialized(target_values: Photometry):
    """
    Querying before initializing must throw
    """
    kd_selector = KDTreeSelector(k=2)
    with pytest.raises(UninitializedException):
        kd_selector.query(target_values)


###############################################################################

def test_BruteForceInvalidDimensions(reference_values: Photometry, target_values: Photometry):
    """
    Querying with an invalid dimensionality must throw
    """
    kd_selector = KDTreeSelector(k=5)
    kd_selector.fit(reference_values, reference_values.system)
    with pytest.raises(InvalidDimensionsException):
        kd_selector.query(target_values.subsystem(['x', 'y']))


###############################################################################

def test_KDTree(reference_values: Photometry, target_values: Photometry):
    """
    Query for the 7 nearest neighbors, which are those falling in the center of each face, plus the
    middle one.

    See Also: 3d_neighbors.png
    """
    kd_selector = KDTreeSelector(k=7)
    kd_selector.fit(reference_values, reference_values.system)

    idx, scales = kd_selector.query(target_values)
    distances = euclidean(reference_values.values[idx[0]], target_values.values[0]).value
    assert (len(idx) == len(scales))
    assert (len(idx[0]) == 7)
    np.testing.assert_array_less(distances, 1.01)
    np.testing.assert_allclose(scales, 1.)


###############################################################################

def test_KDTree2(reference_values, target_values):
    """
    Query for the 10 nearest neighbors, which are those falling in the center of each face, plus the
    middle one, and some vertex. In this case, we check that not all distances are the same.
    """
    kd_selector = KDTreeSelector(k=10)
    kd_selector.fit(reference_values, reference_values.system)

    idx, scales = kd_selector.query(target_values)
    distances = euclidean(reference_values.values[idx[0]], target_values.values[0]).value
    assert (len(idx) == len(scales))
    assert (len(idx[0]) == 10)
    assert ((distances <= 1.01).sum() == 7)
    assert ((distances > 1.01).sum() == 3)
    np.testing.assert_allclose(scales, 1.)

###############################################################################
