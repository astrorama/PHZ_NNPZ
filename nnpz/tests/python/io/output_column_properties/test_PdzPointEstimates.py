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
from nnpz.io.output_column_providers.CoaddedPdz import CoaddedPdz
from nnpz.io.output_column_providers.PdzPointEstimates import PdzPointEstimates

from .fixtures import *


def test_PdzPointEstimates(reference_sample: MockReferenceSample,
                           mock_output_handler: MockOutputHandler,
                           neighbor_info: np.ndarray):
    coadded_pdf = CoaddedPdz(reference_sample)
    pdz_point = PdzPointEstimates(reference_sample, ['median', 'mean', 'mode'])
    mock_output_handler.add_column_provider(coadded_pdf)
    mock_output_handler.add_column_provider(pdz_point)

    mock_output_handler.initialize(len(neighbor_info))
    mock_output_handler.write_output_for(np.arange(len(neighbor_info)), neighbor_info)

    columns = mock_output_handler.get_data_for_provider(pdz_point)
    assert len(columns.dtype.fields) == 3

    assert np.array_equal(columns['REDSHIFT_MODE'], np.arange(len(columns)))
    assert np.allclose(columns['REDSHIFT_MEDIAN'], np.arange(len(columns)), atol=1)
    assert np.all(columns['REDSHIFT_MEAN'] > np.arange(len(columns)))
