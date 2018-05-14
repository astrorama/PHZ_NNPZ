"""
Created on: 15/02/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np
from scipy import interpolate
from astropy.table import Column

from nnpz.io import OutputHandler

class PdfSampling(OutputHandler.OutputColumnProviderInterface):


    def __sample(self, pdf, bins, quantiles):
        cum_prob = np.zeros(len(bins))
        cum_prob[1:] = np.cumsum(np.diff(bins) * ((pdf[:-1] + pdf[1:]) / 2.))
        inv_cum = interpolate.interp1d(cum_prob/max(cum_prob), bins, kind='linear')
        return inv_cum(quantiles)


    def __init__(self, pdf_provider, col_name, quantiles=[], mc_samples=0):
        self.__pdf_provider = pdf_provider
        self.__col_name = col_name
        self.__qs = quantiles
        self.__mc_no = mc_samples


    def addContribution(self, reference_sample_i, catalog_i, weight, flags):
        pass


    def getColumns(self):
        bins = self.__pdf_provider.getPdzBins()
        pdfs = self.__pdf_provider.getColumns()[0].data

        cols = []

        fixed_probs = np.asarray([self.__sample(pdf, bins, self.__qs) for pdf in pdfs], dtype=np.float32)
        for i, q, in enumerate(self.__qs):
            cols.append(Column(fixed_probs[:,i], "{}_{}".format(self.__col_name, int(q * 100))))

        if self.__mc_no > 0:
            mc_vals = np.asarray([self.__sample(pdf, bins, np.random.rand(self.__mc_no)) for pdf in pdfs], dtype=np.float32)
            cols.append(Column(mc_vals, "{}_mc".format(self.__col_name)))

        return cols

