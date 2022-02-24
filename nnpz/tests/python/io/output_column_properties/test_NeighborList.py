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
from nnpz.io.output_column_providers.NeighborList import NeighborList

from .fixtures import *


def test_NeighborList(mock_output_handler: MockOutputHandler):
    nlist = NeighborList(ref_ids=np.flip(np.arange(100)), n_neighbors=2)
    mock_output_handler.add_column_provider(nlist)

    neighbor_info = np.zeros(1, dtype=[('NEIGHBOR_INDEX', int, 2),
                                       ('NEIGHBOR_WEIGHTS', np.float32, 2),
                                       ('NEIGHBOR_SCALING', np.float32, 2)])
    neighbor_info['NEIGHBOR_INDEX'][0] = [0, 1]
    neighbor_info['NEIGHBOR_WEIGHTS'] = 1.

    mock_output_handler.initialize(len(neighbor_info))
    mock_output_handler.write_output_for([0], neighbor_info)

    columns = mock_output_handler.get_data_for_provider(nlist)
    assert len(columns.dtype.fields) == 3

    np.testing.assert_array_equal(columns['NEIGHBOR_IDS'][0], [99, 98])
    np.testing.assert_array_equal(columns['NEIGHBOR_WEIGHTS'][0], [1., 1.])
    np.testing.assert_array_equal(columns['NEIGHBOR_SCALING'][0], [0., 0.])
