"""
Created on: 05/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import pytest
import numpy as np

from nnpz.photometry import DefaultSedPreprocessing

###############################################################################

def test_processSed():
    """Test the processSed() returns its input unmodified"""

    # Given
    expected = np.asarray([(1, 0.1), (2, 0.2), (3, 0.3)], dtype=np.float32)

    # When
    processor = DefaultSedPreprocessing()
    sed = processor.processSed(expected)

    # Then
    assert np.array_equal(sed, expected)

###############################################################################

def test_type():
    """Test the type() returns correctly"""

    # Given
    expected = "FLUX_ENERGY"

    # When
    processor = DefaultSedPreprocessing()
    type = processor.type()

    # Then
    assert type == expected

###############################################################################