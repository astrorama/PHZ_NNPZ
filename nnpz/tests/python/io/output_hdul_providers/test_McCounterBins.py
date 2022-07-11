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
from nnpz.io.output_hdul_providers.McCounterBins import McCounterBins

# noinspection PyUnresolvedReferences
from .fixtures import *


def test_McCounterBins(fits_fixture: FITS):
    bins = np.arange(0, 10, dtype=int)
    counter = McCounterBins(param_name='param', binning=bins, unit=u.watt)
    counter.add_extensions(fits_fixture)
    assert 'BINS_MC_COUNT_PARAM' in fits_fixture
    hdu = fits_fixture['BINS_MC_COUNT_PARAM']
    assert isinstance(hdu, fitsio.hdu.TableHDU)
    np.testing.assert_array_equal(hdu['BINS'][:], bins)
    assert u.Unit(hdu.read_header().get('TUNIT1')) == u.watt
