#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
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
from nnpz.io.output_column_providers.McCounter import McCounter

# noinspection PyUnresolvedReferences
from .fixtures import *


def test_count_samples(sampler):
    counter = McCounter(sampler, param_name='I1', binning=np.arange(0, 10))
    columns = counter.getColumns()

    assert len(columns) == 1
    column = columns[0]
    assert isinstance(column, Column)
    assert column.name == 'MC_COUNT_I1'
    assert column.shape == (2, 10)
    # First object can not have any sample from 2, and the weight is higher for 1
    assert column[0, 2] == 0
    assert column[0, 0] > 0
    assert column[0, 1] > column[0, 0]
    assert column[0].sum() == 200
    # Second object must have the most samples from 2, and more from 0 than from 1
    assert column[1, 1] > 0
    assert column[1, 0] > column[1, 1]
    assert column[1, 2] > column[1, 0]
    assert column[1].sum() == 200
