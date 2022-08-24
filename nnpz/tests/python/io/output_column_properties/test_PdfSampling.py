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
from nnpz.io.output_column_providers.PdfSampling import PdfSampling

from .fixtures import *


def test_PdfSampling(reference_sample: MockReferenceSample, mock_output_handler: MockOutputHandler,
                     neighbor_info: np.ndarray):
    coadded_pdf = CoaddedPdz(reference_sample, kernel=None, bandwidth=0)
    pdf_sampling = PdfSampling(reference_sample, quantiles=[0.25, 0.75], mc_samples=1000)
    mock_output_handler.add_column_provider(coadded_pdf)
    mock_output_handler.add_column_provider(pdf_sampling)

    mock_output_handler.initialize(len(neighbor_info))
    mock_output_handler.write_output_for(np.arange(len(neighbor_info)), neighbor_info)

    columns = mock_output_handler.get_data_for_provider(pdf_sampling)
    assert len(columns.dtype.fields) == 2

    np.testing.assert_array_less(columns['REDSHIFT_PDF_QUANTILES'][:, 0],
                                 columns['REDSHIFT_PDF_QUANTILES'][:, 1])
    np.testing.assert_array_less(columns['REDSHIFT_PDF_QUANTILES'][:-1, 0],
                                 columns['REDSHIFT_PDF_QUANTILES'][1:, 0])

    bins = reference_sample.get_provider('pdz').get_redshift_bins()
    for i, c in enumerate(columns['REDSHIFT_PDF_MC']):
        count, _ = np.histogram(c, bins=bins)
        # Roughly, the most samples must come from around the peak
        assert i - 2 <= count.argmax() <= i + 2
