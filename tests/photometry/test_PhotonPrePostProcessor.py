"""
Created on: 14/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from platform import processor

import pytest
import numpy as np

from nnpz.photometry import PhotonPrePostProcessor, PhotonPrePostProcessor


###############################################################################

def test_preProcess():
    """Test the preProcess() method"""

    # Given
    sed = np.asarray([(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.2)], dtype=np.float32)
    expected = np.ndarray(len(sed), dtype=np.float32)
    expected = sed[:,1] * 5.03411250E+07 * sed[:,0]


    # When
    processor = PhotonPrePostProcessor()
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result[:,0], sed[:,0])
    assert result[0][1] == pytest.approx(expected[0])
    assert result[1][1] == pytest.approx(expected[1])
    assert result[2][1] == pytest.approx(expected[2])
    assert result[3][1] == pytest.approx(expected[3])

###############################################################################

def test_postProcess():
    """Test the postProcess() method"""

    # Given
    intensity = 1.5

    # When
    processor = PhotonPrePostProcessor()
    result = processor.postProcess(intensity, None, None)

    # Then
    assert result == intensity

###############################################################################