import pytest
import numpy as np

from nnpz.photometry import FnuPrePostProcessor, FnuuJyPrePostProcessor


###############################################################################

def test_preProcess():
    """Test the preProcess() method"""

    # Given
    fnu = FnuPrePostProcessor()
    sed = np.asarray([(1, 0.1), (2, 0.1), (3, 0.2), (4, 0.2)], dtype=np.float32)
    expected = fnu.preProcess(sed)

    # When
    processor = FnuuJyPrePostProcessor()
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result, expected)

###############################################################################

def test_postProcess():
    """Test the postProcess() method"""

    # Given
    fnu = FnuPrePostProcessor()
    intensity = 1.
    filter_name = 'name'
    filter_trans = np.asarray([(1,0), (2,4), (3,9), (4,0)], dtype=np.float32)
    expected = fnu.postProcess(intensity, filter_name, filter_trans) * 1E23 * 1E6

    # When
    processor = FnuuJyPrePostProcessor()
    result = processor.postProcess(intensity, filter_name, filter_trans)

    # Then
    assert result == expected


###############################################################################