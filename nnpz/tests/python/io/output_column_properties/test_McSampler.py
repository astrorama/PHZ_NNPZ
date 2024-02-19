#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under the terms of
# the GNU Lesser General Public License as published by the Free Software Foundation;
# either version 3.0 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this library;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301 USA
#

# noinspection PyUnresolvedReferences
import numpy as np

from .fixtures import *


###############################################################################


def test_take_samples(contributions, mock_provider):
    sampler = McSampler(take_n=200, mc_provider=mock_provider)
    sampler.generate_output(np.arange(len(contributions)), contributions, None)

    # First object can not have any sample from 2, and the weight is higher for 1
    samples = sampler.get_samples()[0]
    assert samples.shape == (200,)
    assert samples.dtype.names == ('P1', 'P2', 'I1', 'M1')
    vals, counts = np.unique(samples['P1'], return_counts=True)
    assert 2 not in vals
    assert 0 in vals
    assert 1 in vals
    assert counts[0] < counts[1]
    assert counts[0] + counts[1] == 200

    # Second object must have the most samples from 2, and more from 0 than from 1
    samples = sampler.get_samples()[1]
    assert samples.shape == (200,)
    assert samples.dtype.names == ('P1', 'P2', 'I1', 'M1')
    vals, counts = np.unique(samples['P1'], return_counts=True)
    assert counts[0] > counts[1]
    assert counts[2] > counts[1]
    assert counts.sum() == 200


###############################################################################

def test_take_weight_0(reference_ids, contributions, mock_provider):
    sampler = McSampler(take_n=200, mc_provider=mock_provider)
    contributions['NEIGHBOR_WEIGHTS'][0] = 0
    sampler.generate_output(np.arange(len(contributions)), contributions, None)

    # All samples are 0 for the first source
    samples = sampler.get_samples()[0]
    assert samples.shape == (200,)
    assert samples.dtype.names == ('P1', 'P2', 'I1', 'M1')
    assert (samples['P1'] == 0).all()
    assert (samples['P2'] == 0).all()
    assert (samples['I1'] == 0).all()

    # Second object must be the same as before
    samples = sampler.get_samples()[1]
    assert samples.shape == (200,)
    assert samples.dtype.names == ('P1', 'P2', 'I1', 'M1')
    vals, counts = np.unique(samples['P1'], return_counts=True)
    assert counts[0] > counts[1]
    assert counts[2] > counts[1]
    assert counts.sum() == 200


###############################################################################

def test_take_weight_nans(reference_ids, contributions, mock_provider):
    sampler = McSampler(take_n=200, mc_provider=mock_provider)
    contributions['NEIGHBOR_WEIGHTS'][0] = np.nan
    sampler.generate_output(np.arange(len(contributions)), contributions, None)

    # All samples are 0 for the first source (should behave as weight 0)
    samples = sampler.get_samples()[0]
    assert samples.shape == (200,)
    assert samples.dtype.names == ('P1', 'P2', 'I1', 'M1')
    assert (samples['P1'] == 0).all()
    assert (samples['P2'] == 0).all()
    assert (samples['I1'] == 0).all()

    # Second object must be the same as before
    samples = sampler.get_samples()[1]
    assert samples.shape == (200,)
    assert samples.dtype.names == ('P1', 'P2', 'I1', 'M1')
    vals, counts = np.unique(samples['P1'], return_counts=True)
    assert counts[0] > counts[1]
    assert counts[2] > counts[1]
    assert counts.sum() == 200

###############################################################################
