"""
Created on: 14/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from platform import processor

import pytest
import numpy as np

from nnpz.photometry import FnuPrePostProcessor


###############################################################################

def test_preProcess():
    """Test the preProcess() method"""

    # Given
    sed = np.asarray([(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.2)], dtype=np.float32)

    # When
    processor = FnuPrePostProcessor()
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result, sed)

###############################################################################

def test_postProcess():
    """Test the postProcess() method"""

    # Given
    c = 299792458E10
    intensity = c
    filter_name = 'name'
    filter_trans = np.asarray([(1,0), (2,4), (3,9), (4,0)], dtype=np.float32)
    expected = 0.5

    # When
    processor = FnuPrePostProcessor()
    result = processor.postProcess(intensity, filter_name, filter_trans)

    # Then
    assert result == pytest.approx(expected)


###############################################################################
