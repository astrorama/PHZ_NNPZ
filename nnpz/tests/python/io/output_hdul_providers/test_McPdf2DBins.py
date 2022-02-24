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
from nnpz.io.output_hdul_providers.McPdf2DBins import McPdf2DBins

from .fixtures import *


def test_McPdf2DBins(fits_fixture: FITS):
    bins_x = np.linspace(0, 1, 10)
    bins_y = np.linspace(0, 100, 10)
    pdfbins = McPdf2DBins(param_names=('param_x', 'param_y'), binning=(bins_x, bins_y))
    pdfbins.add_extensions(fits_fixture)
    assert 'BINS_MC_PDF_2D_PARAM_X_PARAM_Y' in fits_fixture
    hdu = fits_fixture['BINS_MC_PDF_2D_PARAM_X_PARAM_Y']
    assert isinstance(hdu, fitsio.hdu.TableHDU)
    # Note that the output is the bin mid-point, so the shape correspond to the histogram values
    expected = np.meshgrid((bins_x[1:] + bins_x[:-1]) / 2, (bins_y[1:] + bins_y[:-1]) / 2)
    np.testing.assert_almost_equal(hdu['PARAM_X'][:], expected[0].T.ravel())
    np.testing.assert_almost_equal(hdu['PARAM_Y'][:], expected[1].T.ravel())
