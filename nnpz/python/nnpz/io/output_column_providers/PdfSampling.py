#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
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
from nnpz.io import OutputHandler
from scipy import interpolate


class PdfSampling(OutputHandler.OutputColumnProviderInterface,
                  OutputHandler.HeaderProviderInterface):
    """
    Generate a set of samples from a PDF

    Args:
        pdf_provider: OutputColumnProviderInterface
            Must implement the methods getPdzBins and getPdz
            (i.e. CoaddedPdz or TrueRedshiftPdz)
        quantiles: list of float
            Quantiles to compute (between 0 and 1)
        mc_samples:
            How many Montecarlo samples to generate
    """

    def __sample(self, pdf, bins, quantiles):
        cum_prob = np.zeros(len(bins))
        cum_prob[1:] = np.cumsum(np.diff(bins) * ((pdf[:-1] + pdf[1:]) / 2.))
        inv_cum = interpolate.interp1d(cum_prob / max(cum_prob), bins, kind='linear')
        return inv_cum(quantiles)

    def __init__(self, pdf_provider, quantiles=None, mc_samples=0):
        self.__pdf_provider = pdf_provider
        self.__qs = quantiles if quantiles else []
        self.__mc_no = mc_samples
        self.__quantiles = None
        self.__mc = None

    def getColumnDefinition(self):
        def_cols = []
        if self.__qs:
            def_cols.append(('REDSHIFT_PDF_QUANTILES', np.float32, len(self.__qs)))
        if self.__mc_no > 0:
            def_cols.append(('REDSHIFT_PDF_MC', np.float32, self.__mc_no))
        return def_cols

    def setWriteableArea(self, output_area):
        if self.__qs:
            self.__quantiles = output_area['REDSHIFT_PDF_QUANTILES']
        if self.__mc_no > 0:
            self.__mc = output_area['REDSHIFT_PDF_MC']

    def addContribution(self, reference_sample_i, neighbor, flags):
        pass

    def fillColumns(self):
        bins = self.__pdf_provider.getPdzBins()
        pdfs = self.__pdf_provider.getPdz()

        if self.__qs:
            for i, pdf in enumerate(pdfs):
                self.__quantiles[:] = self.__sample(pdf, bins, self.__qs)

        if self.__mc_no > 0:
            for i, pdf in enumerate(pdfs):
                samples = np.random.rand(self.__mc_no)
                self.__mc[:] = self.__sample(pdf, bins, samples)
