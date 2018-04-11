"""
Created: 11/04/2018
Author: Alejandro Alvarez Ayllon
"""

from __future__ import division, print_function

import numpy as np
import os
import pytest

from astropy.table import Table
from nnpz.reference_sample.ReferenceSample import ReferenceSample

from tests.util_fixtures import temp_dir_fixture

@pytest.fixture()
def reference_sample_fixture(temp_dir_fixture):
    """
    Generates a reference sample to be used by the weight tests
    """
    NReferences = 5

    # Use this SED as a sample
    test_sed_file = os.path.join(os.path.dirname(__file__), 'test.sed')
    test_sed_table = Table.read(test_sed_file, format='ascii')
    test_sed = np.asarray((test_sed_table['col1'], test_sed_table['col2'])).transpose()

    os.rmdir(temp_dir_fixture)
    ref_sample = ReferenceSample.createNew(temp_dir_fixture)
    for id in range(1, NReferences+1):
        ref_sample.createObject(id)

        # Generate a SED shifted by a random Z
        z = np.random.random() * 4
        new_sed = np.ndarray(test_sed.shape)
        new_sed[:, 0] = test_sed[:, 0] * z
        new_sed[:, 1] = test_sed[:, 1] / z
        ref_sample.addSedData(id, new_sed)

        # Can skip the PDZ for these tests

    return ref_sample


@pytest.fixture()
def filters_fixture():
    """Returns the filters to be used for testing"""
    filters = {}
    filters['vis'] = np.asarray([
        (3340.0, 1.57817697001e-05),
        (3950.0, 1.10903981897e-05),
        (5300.0, 0.0120291676715),
        (6200.0, 0.747178052878),
        (6270.0, 0.74811143401),
        (6900.0, 0.733246736066),
        (7190.0, 0.713413655568),
        (7510.0, 0.690543251343),
        (8890.0, 0.428761967225),
        (10790.0, 2.50137842435e-06)
    ], dtype=np.float32)
    filters['g'] = np.asarray([
        (3280, 0.0),
        (3460, 0.00023),
        (4200, 0.19075),
        (6830, 2e-05),
        (7100, 3e-05),
        (7360, 6e-05),
        (7860, 0.00055),
        (8720, 0.00065),
        (8730, 0.00065),
        (9220, 0.0006)
    ], dtype=np.float32)
    filters['Y'] = np.asarray([
        (5690.0, 1.5850049282e-12),
        (6310.0, 4.50368644306e-12),
        (7170.0, 8.85156294771e-13),
        (7390.0, 8.73947334478e-13),
        (8280.0, 3.71894675455e-12),
        (15520.0, 0.00013080550664),
        (15990.0, 6.24355648417e-05),
        (17820.0, 0.000114044428357),
        (18140.0, 0.000160034738868),
        (19470.0, 0.000110527769945)
    ], dtype=np.float32)

    return filters


@pytest.fixture()
def target_fixture(filters_fixture):
    """
    Generates a target catalog with E(B-V) and filter shifts per target.
    The filter shifts are generated from the filters_fixture
    """
    NTargets = 10
    filter_means = {}
    for filter_name, transmissions in filters_fixture.iteritems():
        trans_avg = np.mean(transmissions[:, 0])
        filter_means[filter_name] = trans_avg + np.random.randint(-100, 100, size=NTargets)

    return {
        'ID': np.asarray(range(1, NTargets+1), dtype=np.float32),
        'ebv': np.ones(NTargets),
        'filter_means': filter_means,
    }
