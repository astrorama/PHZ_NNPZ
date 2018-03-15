"""
Created on: 15/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from astropy.table import Column

from nnpz.io import OutputHandler

class PdfHalfQuantile(OutputHandler.OutputColumnProviderInterface):


    def __computeMedian(self, pdf, bins):
        bin_size = bins[1] - bins[0]
        cum_prob = (pdf * bin_size).cumsum()
        for i, prob in enumerate(cum_prob):
            if prob > 0.5:
                break
        return bins[i] - (bins[i] - bins[i-1]) * (cum_prob[i] - 0.5) / (cum_prob[i] - cum_prob[i-1])


    def __init__(self, pdf_provider, col_name):
        self.__pdf_provider = pdf_provider
        self.__col_name = col_name


    def addContribution(self, reference_sample_i, catalog_i, weight):
        pass


    def getColumns(self):
        bins = self.__pdf_provider.getPdzBins()
        pdfs = self.__pdf_provider.getColumns()[0].data
        medians = np.asarray([self.__computeMedian(pdf, bins) for pdf in pdfs], dtype=np.float32)
        col = Column(medians, self.__col_name)
        return [col]

