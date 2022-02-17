#
#  Copyright (C) 2022 Euclid Science Ground Segment
#
#  This library is free software; you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation;
#  either version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with this library;
#  if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA
#
from nnpz.io.output_column_providers.McSamples import McSamples

from .fixtures import *


def test_McSamples(sampler: McSampler, mock_output_handler: MockOutputHandler,
                   contributions: np.ndarray):
    samples = McSamples(sampler, parameters=['P1', 'P2', 'I1'])
    mock_output_handler.add_column_provider(sampler)
    mock_output_handler.add_column_provider(samples)

    mock_output_handler.initialize(nrows=len(contributions))
    mock_output_handler.write_output_for(np.arange(len(contributions)), contributions)

    columns = mock_output_handler.get_data_for_provider(samples)

    assert len(columns.dtype.fields) == 3
    assert columns['MC_SAMPLES_P1'].shape == (2, 200)
    assert columns['MC_SAMPLES_P2'].shape == (2, 200)
    assert columns['MC_SAMPLES_I1'].shape == (2, 200)

    assert columns['MC_SAMPLES_P1'].dtype == np.float32
    assert columns['MC_SAMPLES_P2'].dtype == np.float32
    assert columns['MC_SAMPLES_I1'].dtype == np.int32

    # First object can not have any sample from 2, and the weight is higher for 1
    hist, _ = np.histogram(columns['MC_SAMPLES_P1'][0], bins=np.arange(0, 4))
    assert hist[2] == 0.
    assert hist[1] > hist[0]

    # Second object must have the most samples from 2, and more from 0 than from 1
    hist, _ = np.histogram(columns['MC_SAMPLES_P1'][1], bins=np.arange(0, 4))
    assert np.argmax(hist) == 2
    assert hist[0] > hist[1]
