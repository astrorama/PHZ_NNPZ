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

from .fixtures import *


###############################################################################

def test_CoaddedPdz(reference_sample: MockReferenceSample, neighbor_info: np.ndarray):
    coadded = CoaddedPdz(reference_sample, kernel=None, bandwidth=0)
    col_defs = coadded.get_column_definition()
    assert len(col_defs) == 1
    assert col_defs[0][0] == 'REDSHIFT_PDF'
    assert col_defs[0][3] == (1500,)

    output = np.zeros(len(neighbor_info), dtype=[(col_defs[0][0], col_defs[0][1], col_defs[0][3])])
    coadded.generate_output(np.arange(len(neighbor_info)), neighbor_info, output)

    # Always the first one if the most weighted, and the first one increments
    np.testing.assert_array_equal(output['REDSHIFT_PDF'].argmax(axis=1), np.arange(50))


###############################################################################

def avg_and_std(pdz_bins, weights):
    avg = np.average(pdz_bins, weights=weights)
    std = np.sqrt(np.sum(weights * (pdz_bins - avg) ** 2) / np.sum(weights))
    return avg, std


def test_CoaddedPdzSmooth(reference_sample: MockReferenceSample):
    neighbor_info = np.zeros(3,
                             dtype=[('NEIGHBOR_INDEX', int, 30), ('NEIGHBOR_WEIGHTS', float, 30)])
    neighbor_info['NEIGHBOR_WEIGHTS'] = 0.
    neighbor_info['NEIGHBOR_INDEX'] = (1 + np.arange(30)[np.newaxis])
    neighbor_info['NEIGHBOR_WEIGHTS'][0][6] = 1.
    neighbor_info['NEIGHBOR_WEIGHTS'][1][20] = 1.
    neighbor_info['NEIGHBOR_WEIGHTS'][2][10] = 1.

    coadded = CoaddedPdz(reference_sample, kernel='gaussian', bandwidth=1.5)
    col_defs = coadded.get_column_definition()
    output = np.zeros(len(neighbor_info), dtype=[(col_defs[0][0], col_defs[0][1], col_defs[0][3])])
    coadded.generate_output(np.arange(len(neighbor_info)), neighbor_info, output)

    pdz_bins = reference_sample.get_provider('pdz').get_redshift_bins()

    avg0, std0 = avg_and_std(pdz_bins, output['REDSHIFT_PDF'][0])
    np.testing.assert_allclose([avg0, std0], [6., 1.5], rtol=1e-4)

    avg1, std1 = avg_and_std(pdz_bins, output['REDSHIFT_PDF'][1])
    np.testing.assert_allclose([avg1, std1], [20., 1.5], rtol=1e-4)

    avg2, std2 = avg_and_std(pdz_bins, output['REDSHIFT_PDF'][2])
    np.testing.assert_allclose([avg2, std2], [10., 1.5], rtol=1e-4)
