"""
Created: 11/04/2018
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import pytest
import numpy as np

from nnpz.reference_sample.ReferenceSample import ReferenceSample
from nnpz.weights import RecomputedPhotometry
from .fixtures import *

###############################################################################

def test_recomputedPhotometry(reference_sample_fixture, filters_fixture, target_fixture):

    # Given
    filter_map = dict(filters_fixture)
    phot_type = 'F_nu_uJy'
    ebv = target_fixture['ebv']
    filter_means = target_fixture['filter_means']

    # When
    recomputed = RecomputedPhotometry(
        reference_sample_fixture, filter_map.keys(), filter_map, phot_type, ebv, filter_means
    )

    # Then
    for ref_i, ref in enumerate(reference_sample_fixture.iterate()):
        for cat_i in range(5):
            phot = recomputed(ref_i, cat_i)
            print(phot)
            assert phot.shape == (len(filters_fixture), 2)
