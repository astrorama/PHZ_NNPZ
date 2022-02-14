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
import astropy.units as u
import numpy as np
import pytest
from nnpz.io.output_column_providers.UniformPhotometry import UniformPhotometry
from nnpz.photometry.colorspace import RestFrame
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry

# noinspection PyUnresolvedReferences
from .fixtures import MockOutputHandler, mock_output_handler


class DummyPhotometry:
    def __init__(self):
        self._data = np.zeros((2, 2),
                              dtype=[('A', np.float32), ('B', np.float32), ('C', np.float32)])
        self._data['A'][:, 0] = [0.5626045, 0.94242]
        self._data['B'][:, 0] = [1.7679665, 1.8930919]
        self._data['C'][:, 0] = [3.5187345, 2.606762]

    def getData(self, *filter_list):
        data = np.zeros((len(self._data), len(filter_list), 2), dtype=np.float32)
        for i, f in enumerate(filter_list):
            data[:, i, 0] = self._data[f][:, 0]
        return data


@pytest.fixture
def reference_photometry():
    return Photometry(
        ids=np.arange(2), values=np.array(
            [((0.5626045, 0.), (1.7679665, 0.), (3.5187345, 0.)),
             ((0.94242, 0.), (1.8930919, 0.), (2.606762, 0.))]) * u.uJy,
        system=PhotometricSystem(['A', 'B', 'C']), colorspace=RestFrame)


@pytest.fixture()
def reference_matched_photometry(reference_photometry):
    matched = np.zeros((1, 2, 3, 2), dtype=np.float32)
    matched[0, :, 0, 0] = reference_photometry.values[:, 0, 0] * np.array([1., 2.])
    matched[0, :, 1, 0] = reference_photometry.values[:, 1, 0] * np.array([1.5, 1.7])
    matched[0, :, 2, 0] = reference_photometry.values[:, 2, 0] * np.array([1.2, 1.5])
    return matched


@pytest.fixture
def target_photometry():
    return Photometry(
        ids=np.array([0]), values=np.array(
            [((10., 0.), (10., 0.), (10., 0.))]) * u.uJy,
        system=PhotometricSystem(['A', 'B', 'C']), colorspace=RestFrame)


def test_uniform_photometry(reference_photometry, target_photometry, reference_matched_photometry,
                            mock_output_handler: MockOutputHandler):
    uniform = UniformPhotometry(
        target_photometry, reference_photometry, {
            ('MY_A_A', 'MY_A_A_ERR'): ('A', 'A', 'FLUX_A', 'FLUX_A_ERR'),
            ('MY_B_B', 'MY_B_B_ERR'): ('B', 'B', 'FLUX_B', 'FLUX_B_ERR'),
            ('MY_C_C', 'MY_C_C_ERR'): ('C', 'C', 'FLUX_C', 'FLUX_C_ERR'),
            ('MY_C_A', 'MY_C_A_ERR'): ('C', 'A', 'FLUX_A', 'FLUX_A_ERR')
        }
    )
    mock_output_handler.add_column_provider(uniform)
    mock_output_handler.initialize(len(target_photometry))

    neighbor_info = np.zeros(1, dtype=[('NEIGHBOR_INDEX', int, 2),
                                       ('NEIGHBOR_WEIGHTS', np.float32, 2),
                                       ('NEIGHBOR_PHOTOMETRY', np.float32, (2, 3, 2))])
    assert neighbor_info['NEIGHBOR_PHOTOMETRY'].shape == reference_matched_photometry.shape

    neighbor_info['NEIGHBOR_INDEX'][0] = [0, 1]
    neighbor_info['NEIGHBOR_WEIGHTS'] = 1.
    neighbor_info['NEIGHBOR_PHOTOMETRY'][0] = reference_matched_photometry
    mock_output_handler.write_output_for([0], neighbor_info)

    columns = mock_output_handler.get_data_for_provider(uniform)
    assert len(columns.dtype.fields) == 8

    assert 'MY_A_A' in columns.dtype.fields
    assert 'MY_A_A_ERR' in columns.dtype.fields
    assert 'MY_B_B' in columns.dtype.fields
    assert 'MY_B_B_ERR' in columns.dtype.fields
    assert 'MY_C_C' in columns.dtype.fields
    assert 'MY_C_C_ERR' in columns.dtype.fields
    assert 'MY_C_A' in columns.dtype.fields
    assert 'MY_C_A_ERR' in columns.dtype.fields

    target_a = target_photometry.get_fluxes('A')[0].value
    target_b = target_photometry.get_fluxes('B')[0].value
    target_c = target_photometry.get_fluxes('C')[0].value

    assert np.isclose(columns['MY_A_A'][0], target_a * 0.75)
    assert np.isclose(columns['MY_B_B'][0], target_b * 0.62745)
    assert np.isclose(columns['MY_C_C'][0], target_c * 0.75)
    # In this case the output uniform photometry is C*, so the ratio is computed
    # as C* / A
    #   First  reference: 3.5187345 / (1*0.5626045) = 6.254366077768664
    #   Second reference: 2.6067620 / (2*0.9424200) = 1.3830150039260625
    #   So the mean ratio is 3.818690540847363
    assert np.isclose(columns['MY_C_A'][0], target_a * 3.818690540847363)
