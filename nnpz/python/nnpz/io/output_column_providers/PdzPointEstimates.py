#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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
Created on: 15/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from astropy.table import Column

from nnpz.io import OutputHandler
from nnpz.io.output_column_providers import PdfSampling
import numpy as np


class PdzPointEstimates(OutputHandler.OutputColumnProviderInterface):

    def __init__(self, pdf_provider, estimates):
        self.__estimate_impl = {}
        self.__pdf_provider = pdf_provider
        self.__estimates = estimates
        for e in self.__estimates:
            if not hasattr(self, 'getEstimate' + e.capitalize()):
                raise Exception('Unknown redshift PDF estimate {}'.format(e))

    def addContribution(self, reference_sample_i, neighbor, flags):
        pass

    def getEstimateMedian(self, *_):
        # We use the PdfSampling provider for getting the 50% quantile
        sampl_prov = PdfSampling(self.__pdf_provider, [0.5])
        quant_col = [x for x in sampl_prov.getColumns() if x.name == 'REDSHIFT_PDF_QUANTILES'][0]
        median_data = quant_col.data
        return [Column(median_data, "REDSHIFT_MEDIAN")]

    def getEstimateMean(self, bins, pdfs):
        avg = np.average(np.tile(bins, (len(pdfs), 1)), weights=pdfs, axis=1)
        return [Column(avg, 'REDSHIFT_MEAN')]

    def getEstimateMode(self, bins, pdfs):
        modes = bins[np.argmax(pdfs, axis=1)]
        return [Column(modes, 'REDSHIFT_MODE')]

    def getColumns(self):
        pdfs = self.__pdf_provider.getColumns()[0]
        bins = self.__pdf_provider.getPdzBins()
        columns = []
        for e in self.__estimates:
            get_impl = getattr(self, 'getEstimate' + e.capitalize())
            columns.extend(get_impl(bins, pdfs))
        return columns
