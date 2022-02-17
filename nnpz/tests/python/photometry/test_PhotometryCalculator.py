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

"""
Created on: 13/12/17
Author: Nikolaos Apostolakos
"""

try:
    from mock import Mock
except ModuleNotFoundError:
    from unittest.mock import Mock
import numpy as np

from nnpz.photometry.calculator import PhotometryPrePostProcessorInterface
from nnpz.photometry.calculator.photometry_calculator import PhotometryCalculator


###############################################################################

def test_compute():
    # Given
    sed = np.asarray([
        (0.0, 1), (0.1, 2), (0.2, 3), (0.3, 4), (0.4, 5), (0.5, 5), (0.6, 4), (0.7, 3), (0.8, 2),
        (0.9, 1),
        (1.0, 1), (1.1, 2), (1.2, 3), (1.3, 4), (1.4, 5), (1.5, 5), (1.6, 4), (1.7, 3), (1.8, 2),
        (1.9, 1),
        (2.0, 1), (2.1, 2), (2.2, 3), (2.3, 4), (2.4, 5), (2.5, 5), (2.6, 4), (2.7, 3), (2.8, 2),
        (2.9, 1),
        (3.0, 1), (3.1, 2), (3.2, 3), (3.3, 4), (3.4, 5), (3.5, 5), (3.6, 4), (3.7, 3), (3.8, 2),
        (3.9, 1),
        (4.0, 1), (4.1, 2), (4.2, 3), (4.3, 4), (4.4, 5), (4.5, 5), (4.6, 4), (4.7, 3), (4.8, 2),
        (4.9, 1),
        (5.0, 1), (5.1, 2), (5.2, 3), (5.3, 4), (5.4, 5), (5.5, 5), (5.6, 4), (5.7, 3), (5.8, 2),
        (5.9, 1),
        (6.0, 1), (6.1, 2), (6.2, 3), (6.3, 4), (6.4, 5), (6.5, 5), (6.6, 4), (6.7, 3), (6.8, 2),
        (6.9, 1),
        (7.0, 1), (7.1, 2), (7.2, 3), (7.3, 4), (7.4, 5), (7.5, 5), (7.6, 4), (7.7, 3), (7.8, 2),
        (7.9, 1),
        (8.0, 1), (8.1, 2), (8.2, 3), (8.3, 4), (8.4, 5), (8.5, 5), (8.6, 4), (8.7, 3), (8.8, 2),
        (8.9, 1),
        (9.0, 1), (9.1, 2), (9.2, 3), (9.3, 4), (9.4, 5), (9.5, 5), (9.6, 4), (9.7, 3), (9.8, 2),
        (9.9, 1),
        (10.0, 1)], dtype=np.float32)

    filter_map = {
        'first': np.asarray([(0.35, 0), (0.51, 1), (0.72, 0)], dtype=np.float32),
        'second': np.asarray([(2.11, 0), (3.21, 1), (4.19, 0)], dtype=np.float32)
    }

    pre_post_processor = Mock(spec_set=PhotometryPrePostProcessorInterface)
    pre_post_processor.pre_process.side_effect = lambda x: x
    pre_post_processor.post_process.side_effect = lambda x, y: x

    # When
    calculator = PhotometryCalculator(filter_map, pre_post_processor)
    photometry = calculator.compute(sed)

    # Then

    # Compute the expected photometry values. Note that the grid here is not the same
    # used in PhotometryCalculator, so we need to give a considerable tolerance
    grid = np.linspace(sed[:, 0].min(), sed[:, 0].max(), 10000)
    expected = {}
    filter_1 = filter_map['first']
    filter_1 = np.interp(grid, filter_1[:, 0], filter_1[:, 1])
    sed_interp = np.interp(grid, sed[:, 0], sed[:, 1])
    filtered_sed_1 = sed_interp * filter_1
    expected['first'] = np.trapz(filtered_sed_1, x=grid)

    filter_2 = filter_map['second']
    filter_2 = np.interp(grid, filter_2[:, 0], filter_2[:, 1])
    filtered_sed_2 = sed_interp * filter_2
    expected['second'] = np.trapz(filtered_sed_2, x=grid)

    # Check the postProcess() has been called once per filter with the correct
    # parameters
    for args in pre_post_processor.post_process.call_args_list:
        intensity, filter_name = args[0]
        assert filter_name in expected
        np.testing.assert_approx_equal(intensity, expected[filter_name], significant=1)

    # Check that the result has the correct values
    np.testing.assert_approx_equal(photometry['first'][0], expected['first'], significant=1)
    np.testing.assert_approx_equal(photometry['second'][0], expected['second'], significant=1)

###############################################################################
