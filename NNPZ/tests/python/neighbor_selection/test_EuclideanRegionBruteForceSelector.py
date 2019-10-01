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

    idx, distances = erbf_selector.findNeighbors(target_values, None)
    assert (len(idx) == len(distances))
    assert (len(idx) == 7)
    assert (np.all(distances <= 1.01))


###############################################################################

def test_EuclidenRegionBruteForce2(reference_values, target_values):
    """
    Query for the 10 nearest neighbors, which are those falling on the center of each face, plus the
    middle one, and some of the vertex. In this case, we check that not all distances are the same.
    """
    erbf_selector = EuclideanRegionBruteForceSelector(10, 18, distance_method=EuclideanDistance())
    erbf_selector.initialize(reference_values)

    idx, distances = erbf_selector.findNeighbors(target_values, None)
    assert (len(idx) == len(distances))
    assert (len(idx) == 10)
    assert ((distances <= 1.01).sum() == 7)
    assert ((distances > 1.01).sum() == 3)


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
    idx, distances = erbf_selector.findNeighbors(target_values, None)
    # Only the center is a hit
    assert (len(idx) == len(distances))
    assert (len(idx) == 10)
    assert (np.all(distances <= 1.01))
