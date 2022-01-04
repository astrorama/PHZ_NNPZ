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


import numpy as np
from nnpz.io import OutputHandler
from scipy import interpolate


class PdzPointEstimates(OutputHandler.OutputColumnProviderInterface):
    """
    Compute point estimates from the PDZ

    Args:
        pdf_provider: OutputColumnProviderInterface
            Must implement the methods getPdzBins and getPdz
            (i.e. CoaddedPdz or TrueRedshiftPdz)
        estimates: list of str
            Point estimates to compute: from median, mean and mode
    """

    def __init__(self, pdf_provider, estimates):
        self.__estimate_impl = {}
        self.__pdf_provider = pdf_provider
        self.__estimates = estimates
        for e in self.__estimates:
            if not hasattr(self, 'getEstimate' + e.capitalize()):
                raise Exception('Unknown redshift PDF estimate {}'.format(e))
        self.__output_area = None

    def getColumnDefinition(self):
        return [
            ('REDSHIFT_{}'.format(estimate.upper()), np.float32)
            for estimate in self.__estimates
        ]

    def setWriteableArea(self, output_area):
        self.__output_area = output_area

    def addContribution(self, reference_sample_i, neighbor, flags):
        pass

    def getEstimateMedian(self, bins, pdfs, out):
        cum_prob = np.zeros(len(bins))
        dbins = np.diff(bins)

        for i, pdf in enumerate(pdfs):
            np.cumsum(dbins * ((pdf[:-1] + pdf[1:]) / 2.), out=cum_prob[1:])
            if max(cum_prob):
                inv_cum = interpolate.interp1d(cum_prob / max(cum_prob), bins, kind='linear')
                out[i] = inv_cum(0.5)
            else:
                out[i] = np.nan

    def getEstimateMean(self, bins, pdfs, out):
        zero_mask = np.sum(pdfs, axis=1) > 0
        out[zero_mask] = np.average(
            np.tile(bins, (len(pdfs[zero_mask]), 1)),
            weights=pdfs[zero_mask], axis=1
        )
        out[~zero_mask] = np.nan

    def getEstimateMode(self, bins, pdfs, out):
        out[:] = bins[np.argmax(pdfs, axis=1)]

    def fillColumns(self):
        pdfs = self.__pdf_provider.getPdz()
        bins = self.__pdf_provider.getPdzBins()
        for estimate in self.__estimates:
            estimate_name = 'REDSHIFT_{}'.format(estimate.upper())
            get_impl = getattr(self, 'getEstimate' + estimate.capitalize())
            get_impl(bins, pdfs, out=self.__output_area[estimate_name])
