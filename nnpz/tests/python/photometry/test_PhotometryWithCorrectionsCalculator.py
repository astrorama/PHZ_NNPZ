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
from nnpz.photometry.calculator import PhotometryPrePostProcessorInterface
from nnpz.photometry.calculator.photometry_with_corrections_calculator import \
    PhotometryWithCorrectionsCalculator

from .fixtures import *


###############################################################################

class MockPrePostProcessor(PhotometryPrePostProcessorInterface):

    def pre_process(self, sed):
        return sed

    def post_process(self, intensity, filter_name):
        return intensity


###############################################################################

@pytest.fixture
def filter_trans_map():
    return {'First': np.asarray([(1, 0.1), (2, 0.2), (3, 0.4)], dtype=np.float32),
            'Second': np.asarray([(4, 0.4), (5, 0.5), (6, 0.6)], dtype=np.float32),
            'Third': np.asarray([(7, 0.7), (8, 0.8), (9, 0.9)], dtype=np.float32),
            'Fourth': np.asarray([(1, 0.11), (2, 0.22), (3, 0.44)], dtype=np.float32)}


###############################################################################

@pytest.fixture
def sed_fixture():
    return np.array(
        [[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
         [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]],
        dtype=np.float32).T


###############################################################################

@pytest.fixture
def reddening_fixture():
    return np.array(
        [[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
         [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.]],
        dtype=np.float32).T


###############################################################################

def test_photometryWithCorrectionsCalculator(filter_trans_map: Dict[str, np.ndarray],
                                             sed_fixture: np.ndarray,
                                             reddening_fixture: np.ndarray):
    shifts = np.array([-1, 1, 2, 3, 4, 5], dtype=np.float32)
    calculator = PhotometryWithCorrectionsCalculator(filter_trans_map,
                                                     MockPrePostProcessor(),
                                                     ebv_ref=0.3, shifts=shifts,
                                                     galactic_reddening_curve=reddening_fixture)
    photo_map, ebv_corr_map, shift_corr_map = calculator.compute(sed_fixture)

    # Types and shape
    for f in filter_trans_map.keys():
        assert f in photo_map.dtype.names
        assert photo_map[f].dtype == np.float32
    assert photo_map.dtype == ebv_corr_map.dtype
    assert photo_map.dtype == shift_corr_map.dtype
    assert photo_map.shape == (2,)
    assert ebv_corr_map.shape == (1,)
    assert shift_corr_map.shape == (2,)

    # Photometry values should match EBV 0.0 and no shift
    assert np.isclose(photo_map['First'][0], 0.6)
    assert np.isclose(photo_map['Second'][0], 4.1)
    assert np.isclose(photo_map['Third'][0], 11.3)

    # EBV extinction is greater the highest the lambda
    assert ebv_corr_map['First'] > ebv_corr_map['Second']
    assert ebv_corr_map['Second'] > ebv_corr_map['Third']
    assert ebv_corr_map['Fourth'] > ebv_corr_map['Second']  # Fourth band overlap with the first!

    # Shift corrections
    # The SED is just y=x-1, so the correction factors should be able to interpolate well
    # for those filters within the SED boundaries
    grid = np.linspace(sed_fixture[:, 0].min() - 5, sed_fixture[:, 0].max() + 5, 50)
    sinterp = np.interp(grid, sed_fixture[:, 0], sed_fixture[:, 1], left=0, right=0)
    for fname in ['First', 'Second', 'Fourth']:
        trans = filter_trans_map[fname]
        a, b = shift_corr_map[fname]
        for dx in shifts:
            finterp = np.interp(grid, trans[:, 0] + dx, trans[:, 1], left=0, right=0)
            expected = np.trapz(finterp * sinterp, x=grid)
            corr = a * dx * dx + b * dx + 1
            corrected = photo_map[fname][0] * corr
            assert (np.isclose(expected, corrected, rtol=0.3))


###############################################################################


def test_shift0(filter_trans_map):
    shifts = np.linspace(-5, 5, 11)
    with pytest.raises(ValueError):
        _ = PhotometryWithCorrectionsCalculator(filter_trans_map,
                                                MockPrePostProcessor(),
                                                ebv_ref=0.3, shifts=shifts)

###############################################################################
