"""
Created on: 18/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from platform import processor

import pytest
import numpy as np

from nnpz.photometry import FlambdaPrePostProcessor


###############################################################################

def test_preProcess():
    """Test the preProcess() method"""

    # Given
    sed = np.asarray([(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.2)], dtype=np.float32)

    # When
    processor = FlambdaPrePostProcessor()
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result, sed)

###############################################################################

def test_postProcess():
    """Test the postProcess() method"""

    # Given
    intensity = 1
    filter_name = 'name'
    filter_trans = np.asarray([(1,0), (2,4), (3,9), (4,0)], dtype=np.float32)
    expected = 1. / 13.

    # When
    processor = FlambdaPrePostProcessor()
    result = processor.postProcess(intensity, filter_name, filter_trans)

    # Then
    assert result == pytest.approx(expected)


###############################################################################
