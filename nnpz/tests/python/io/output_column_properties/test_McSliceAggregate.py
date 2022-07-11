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

from nnpz.io.output_column_providers.McSliceAggregate import McSliceAggregate

from .fixtures import *


def test_slice(sampler: McSampler, mock_output_handler: MockOutputHandler,
               contributions: np.ndarray, mock_provider: MockProvider):
    slicer = McSliceAggregate(
        sampler, target_param='P1', slice_param='I1', suffix='AVG',
        aggregator=np.mean, binning=np.arange(0, 6, dtype=np.float32) - 0.5,
        unit=u.fortnight
    )
    mock_output_handler.add_column_provider(sampler)
    mock_output_handler.add_column_provider(slicer)
    mock_output_handler.initialize(nrows=len(contributions))
    mock_output_handler.write_output_for(np.arange(len(contributions)), contributions)
    columns = mock_output_handler.get_data_for_provider(slicer)

    ref0 = mock_provider.get_data(0)
    ref1 = mock_provider.get_data(1)
    ref2 = mock_provider.get_data(2)

    assert len(columns.dtype.fields) == 1
    assert 'MC_SLICE_AGGREGATE_P1_I1_AVG' in columns.dtype.fields
    column = columns['MC_SLICE_AGGREGATE_P1_I1_AVG']
    assert column.shape == (2, 5)
    assert u.Unit(column.unit) == u.fortnight

    # First object can not have any sample from 2
    np.testing.assert_array_equal(column[0, 2], -99.)
    np.testing.assert_almost_equal(column[0, 0].mean(), ref0['P1'].mean())
    np.testing.assert_almost_equal(column[0, 1].mean(), ref1['P1'].mean())
    # Second object
    np.testing.assert_almost_equal(column[1, 0].mean(), ref0['P1'].mean())
    np.testing.assert_almost_equal(column[1, 1].mean(), ref1['P1'].mean())
    np.testing.assert_almost_equal(column[1, 2].mean(), ref2['P1'].mean())
