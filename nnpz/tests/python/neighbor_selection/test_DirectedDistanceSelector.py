import itertools

import numpy as np
import pytest
from nnpz.neighbor_selection.brute_force_methods import DirectedDistance

###############################################################################

@pytest.fixture
def directed_distance():
    return DirectedDistance()

###############################################################################

def test_DirectedDistanceSelf(directed_distance):
    a = np.array([[2, 2]])
    assert directed_distance(a, None, a[0], None)[0] == 0

###############################################################################

def test_DirectedDistance(directed_distance):
    axis = np.arange(0, 41, 20)
    reference = np.array(list(itertools.product(axis, axis)))
    point = np.array([20, 20])

    distances = directed_distance(reference, None, point, None)

    assert len(distances) == len(reference)

    expected = np.array([
        np.nan,  2.0e+02, 4.0e+02,
        2.0e+02, 0.0e+00, 4.00e+01,
        4.0e+02, 4.0e+01, 1.78e-13
    ])

    # Skip nan
    assert np.isclose(expected[1:], distances[1:]).all()
