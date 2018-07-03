"""
Created on: 15/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from astropy.table import Column

from nnpz.io import OutputHandler
from nnpz.io.output_column_providers import PdfSampling

class MedianRedshiftFromQuantile(OutputHandler.OutputColumnProviderInterface):


    def __init__(self, pdf_provider):
        self.__pdf_provider = pdf_provider

    def addContribution(self, reference_sample_i, catalog_i, weight, flags):
        pass

    def getColumns(self):
        # We use the PdfSampling provider for getting the 50% quantile
        sampl_prov = PdfSampling(self.__pdf_provider, [0.5])
        quant_col = [x for x in sampl_prov.getColumns() if x.name == 'REDSHIFT_PDF_QUANTILES'][0]
        median_data = quant_col.data
        return [Column(median_data, "REDSHIFT_MEDIAN")]

