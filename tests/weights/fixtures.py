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

from ..util_fixtures import temp_dir_fixture

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
        (1000, 0.0),
        (1500, 0.5),
        (2000, 0.2),
        (2500, 0.0),
        (3000, 0.0),
        (3500, 0.0),
        (4000, 0.0),
        (4500, 0.0),
        (5000, 0.0)
    ], dtype=np.float32)
    filters['g'] = np.asarray([
        (1000, 0.0),
        (1500, 0.0),
        (2000, 0.0),
        (2500, 0.0),
        (3000, 0.0),
        (3500, 0.0),
        (4000, 0.3),
        (4500, 0.5),
        (5000, 0.1)
    ], dtype=np.float32)
    filters['Y'] = np.asarray([
        (1000, 0.0),
        (1500, 0.0),
        (2000, 0.2),
        (2500, 0.5),
        (3000, 0.5),
        (3500, 0.0),
        (4000, 0.0),
        (4500, 0.0),
        (5000, 0.0)
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
    for filter_name, transmissions in filters_fixture.items():
        trans_avg = np.mean(transmissions[:, 0])
        filter_means[filter_name] = trans_avg + np.random.randint(-50, 50, size=NTargets)

    return {
        'ID': np.asarray(range(1, NTargets+1), dtype=np.float32),
        'ebv': np.zeros(NTargets),
        'filter_means': filter_means,
    }
