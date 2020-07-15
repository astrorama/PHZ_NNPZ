import pytest
import numpy as np


@pytest.fixture
def reference_values():
    # Reference values are situated on a cube around the center
    ref = np.zeros((27, 3, 2))
    i = 0
    for x in np.arange(-1., 1.1):
        for y in np.arange(-1., 1.1):
            for z in np.arange(-1, 1.1):
                ref[i, 0, 0] = x
                ref[i, 1, 0] = y
                ref[i, 2, 0] = z
                i += 1
    return ref


@pytest.fixture
def target_values():
    """
    Return the coordinates for a point centered on the origin.
    Makes it easier to reason about the distances
    """
    return np.array([(0., 0.), (0., 0.), (0., 0.)])
