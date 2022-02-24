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
    coadded = CoaddedPdz(reference_sample)
    col_defs = coadded.get_column_definition()
    assert len(col_defs) == 1
    assert col_defs[0][0] == 'REDSHIFT_PDF'
    assert col_defs[0][3] == (1500,)

    output = np.zeros(len(neighbor_info), dtype=[(col_defs[0][0], col_defs[0][1], col_defs[0][3])])
    coadded.generate_output(np.arange(len(neighbor_info)), neighbor_info, output)

    # Always the first one if the most weighted, and the first one increments
    np.testing.assert_array_equal(output['REDSHIFT_PDF'].argmax(axis=1), np.arange(50))

###############################################################################
