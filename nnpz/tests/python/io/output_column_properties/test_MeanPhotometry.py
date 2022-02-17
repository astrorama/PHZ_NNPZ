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
from nnpz.io.output_column_providers.MeanPhotometry import MeanPhotometry

from .fixtures import *


def test_MeanPhotometry(reference_photometry: Photometry, target_photometry: Photometry,
                        reference_matched_photometry: np.ndarray,
                        mock_output_handler: MockOutputHandler):
    mean_photo = MeanPhotometry(['A', 'B', 'C'], [0, 1, 2], reference_photometry.unit, None, None)
    mock_output_handler.add_column_provider(mean_photo)
    mock_output_handler.initialize(len(target_photometry))

    neighbor_info = np.zeros(1, dtype=[('NEIGHBOR_INDEX', int, 2),
                                       ('NEIGHBOR_WEIGHTS', np.float32, 2),
                                       ('NEIGHBOR_PHOTOMETRY', np.float32, (2, 3, 2))])

    neighbor_info['NEIGHBOR_INDEX'][0] = [0, 1]
    neighbor_info['NEIGHBOR_WEIGHTS'] = 1.
    neighbor_info['NEIGHBOR_PHOTOMETRY'][0] = reference_matched_photometry

    mock_output_handler.write_output_for([0], neighbor_info)

    columns = mock_output_handler.get_data_for_provider(mean_photo)
    assert len(columns.dtype.fields) == 6
    assert set(columns.dtype.fields) == {'A_MEAN', 'A_MEAN_ERR', 'B_MEAN', 'B_MEAN_ERR', 'C_MEAN',
                                         'C_MEAN_ERR'}

    assert np.isclose(columns['A_MEAN'][0], np.mean(reference_matched_photometry[0, :, 0, 0]))
    assert np.isclose(columns['B_MEAN'][0], np.mean(reference_matched_photometry[0, :, 1, 0]))
    assert np.isclose(columns['C_MEAN'][0], np.mean(reference_matched_photometry[0, :, 2, 0]))
