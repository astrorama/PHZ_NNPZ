"""
Created on: 05/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import pytest
import numpy as np

from nnpz.photometry import PhotonSedPreprocessing

###############################################################################

def test_type():
    """Test the type() returns correctly"""

    # Given
    expected = "PHOTON_COUNT"

    # When
    processor = PhotonSedPreprocessing()
    type = processor.type()

    # Then
    assert type == expected

###############################################################################

def test_processSed():
    """Test the processSed() converts the input to photon counts"""

    # Given
    energy_sed = np.asarray([(1, 0.1), (2, 0.2), (3, 0.3)], dtype=np.float32)

    # When
    processor = PhotonSedPreprocessing()
    photon_sed = processor.processSed(energy_sed)

    # Then
    for energy, photon in zip(energy_sed, photon_sed):
        assert photon[0] == energy[0]
        assert pytest.approx(photon[1], photon[1] * 1E-10) == float(5.03411250E7 * energy[1] * energy[0])

###############################################################################

