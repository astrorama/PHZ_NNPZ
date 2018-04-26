"""
Created: 11/04/2018
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

from nnpz.flags import NnpzFlag
from nnpz.photometry import PhotometryProvider
from nnpz.weights import CopiedPhotometry
from tests.photometry.fixtures import *


###############################################################################

def test_copiedPhotometryAll(photometry_file_fixture):

    # Given
    provider = PhotometryProvider(photometry_file_fixture)

    # When
    copied = CopiedPhotometry(provider.getData())

    # Then
    for ref_i in range(len(provider.getIds())):
        phot = copied(ref_i, 1, NnpzFlag())
        assert np.array_equal(phot, provider.getData()[ref_i])
