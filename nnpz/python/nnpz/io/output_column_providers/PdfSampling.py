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

import numpy as np
from scipy import interpolate
from astropy.table import Column

from nnpz.io import OutputHandler

class PdfSampling(OutputHandler.OutputColumnProviderInterface,
                  OutputHandler.HeaderProviderInterface):


    def __sample(self, pdf, bins, quantiles):
        cum_prob = np.zeros(len(bins))
        cum_prob[1:] = np.cumsum(np.diff(bins) * ((pdf[:-1] + pdf[1:]) / 2.))
        inv_cum = interpolate.interp1d(cum_prob/max(cum_prob), bins, kind='linear')
        return inv_cum(quantiles)


    def __init__(self, pdf_provider, quantiles=[], mc_samples=0):
        self.__pdf_provider = pdf_provider
        self.__qs = quantiles
        self.__mc_no = mc_samples


    def addContribution(self, reference_sample_i, neighbor, flags):
        pass


    def getColumns(self):
        bins = self.__pdf_provider.getPdzBins()
        pdfs = self.__pdf_provider.getColumns()[0].data

        cols = []

        if self.__qs:
            fixed_probs = np.asarray([self.__sample(pdf, bins, self.__qs) for pdf in pdfs], dtype=np.float32)
            cols.append(Column(fixed_probs, "REDSHIFT_PDF_QUANTILES"))

        if self.__mc_no > 0:
            mc_vals = np.asarray([self.__sample(pdf, bins, np.random.rand(self.__mc_no)) for pdf in pdfs], dtype=np.float32)
            cols.append(Column(mc_vals, "REDSHIFT_PDF_MC"))

        return cols


    def getHeaderKeywords(self):
        keys = {}
        if self.__qs:
            keys["PDFQUAN"] =' '.join([str(q) for q in self.__qs])
        return keys

