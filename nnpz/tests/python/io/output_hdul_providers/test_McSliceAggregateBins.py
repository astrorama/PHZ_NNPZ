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
import fitsio.hdu
from nnpz.io.output_hdul_providers.McSliceAggregateBins import McSliceAggregateBins

from .fixtures import *


def test_McSliceAggregateBins(fits_fixture: FITS):
    bins = np.exp(np.arange(1, 10))
    agg_bin = McSliceAggregateBins(target_param='dependent', slice_param='independent',
                                   suffix='foo', slice_binning=bins)
    agg_bin.add_extensions(fits_fixture)
    assert 'BINS_MC_SLICE_AGGREGATE_DEPENDENT_INDEPENDENT_FOO' in fits_fixture
    hdu = fits_fixture['BINS_MC_SLICE_AGGREGATE_DEPENDENT_INDEPENDENT_FOO']
    assert isinstance(hdu, fitsio.hdu.TableHDU)
    assert np.allclose(hdu['INDEPENDENT'][:], bins)
