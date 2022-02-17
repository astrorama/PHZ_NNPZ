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
import fitsio
from nnpz.io.output_hdul_providers.McPdf1DBins import McPdf1DBins

from .fixtures import *


def test_McPdf1DBins(fits_fixture: FITS):
    bins = 10 ** np.linspace(-1, 10, 11)
    pdf1bins = McPdf1DBins('param', bins)
    pdf1bins.add_extensions(fits_fixture)
    assert 'BINS_MC_PDF_1D_PARAM' in fits_fixture
    hdu = fits_fixture['BINS_MC_PDF_1D_PARAM']
    assert isinstance(hdu, fitsio.hdu.TableHDU)
    # Note that the output is the bin mid-point, so the shape correspond to the histogram values
    np.testing.assert_almost_equal(hdu['BINS_PDF'][:], (bins[1:] + bins[:-1]) / 2)
