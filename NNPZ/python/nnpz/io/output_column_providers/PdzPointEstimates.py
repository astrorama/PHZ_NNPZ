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

    def addContribution(self, reference_sample_i, catalog_i, weight, flags):
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
