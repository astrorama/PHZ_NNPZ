#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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

from astropy.table import Column

# noinspection PyUnresolvedReferences
from nnpz.io.output_column_providers.McSliceAggregate import McSliceAggregate

from .fixtures import *


def test_slice(sampler, mock_provider):
    slicer = McSliceAggregate(
        sampler, target_param='P1', slice_param='I1', suffix='AVG',
        aggregator=np.mean, binning=np.arange(0, 6, dtype=np.float) - 0.5
    )
    columns = slicer.getColumns()
    ref0 = mock_provider.getData(0)
    ref1 = mock_provider.getData(1)
    ref2 = mock_provider.getData(2)

    assert len(columns) == 1
    column = columns[0]
    assert isinstance(column, Column)
    assert column.name == 'MC_SLICE_AGGREGATE_P1_I1_AVG'
    assert column.shape == (2, 5)

    # First object can not have any sample from 2
    assert np.equal(column[0, 2], -99.)
    assert np.isclose(column[0, 0].mean(), ref0['P1'].mean())
    assert np.isclose(column[0, 1].mean(), ref1['P1'].mean())
    # Second object
    assert np.isclose(column[1, 0].mean(), ref0['P1'].mean())
    assert np.isclose(column[1, 1].mean(), ref1['P1'].mean())
    assert np.isclose(column[1, 2].mean(), ref2['P1'].mean())
