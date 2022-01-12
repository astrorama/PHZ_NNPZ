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
from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np
from nnpz.io import OutputHandler
from nnpz.reference_sample.ReferenceSample import ReferenceSample
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

    def __init__(self, ref_sample: ReferenceSample, estimates):
        self.__estimate_impl = {}
        self.__pdz_bins = ref_sample.getProvider('pdz').getRedshiftBins()
        self.__estimates = estimates
        for e in self.__estimates:
            if not hasattr(self, 'get_estimate_' + e.lower()):
                raise Exception('Unknown redshift PDF estimate {}'.format(e))

    def get_column_definition(self) \
            -> List[Tuple[str, np.dtype, u.Unit, Optional[Tuple[int, ...]]]]:
        return [
            ('REDSHIFT_{}'.format(estimate.upper()), np.float32, u.dimensionless_unscaled)
            for estimate in self.__estimates
        ]

    def get_estimate_median(self, pdfs: np.dtype, out: np.ndarray):
        cum_prob = np.zeros(len(self.__pdz_bins))
        dbins = np.diff(self.__pdz_bins)
        for i, pdf in enumerate(pdfs):
            np.cumsum(dbins * ((pdf[:-1] + pdf[1:]) / 2.), out=cum_prob[1:])
            if max(cum_prob):
                inv_cum = interpolate.interp1d(cum_prob / max(cum_prob), self.__pdz_bins,
                                               kind='linear')
                out[i] = inv_cum(0.5)
            else:
                out[i] = np.nan

    def get_estimate_mean(self, pdfs: np.dtype, out: np.ndarray):
        zero_mask = np.sum(pdfs, axis=1) > 0
        out[zero_mask] = np.average(
            np.tile(self.__pdz_bins, (len(pdfs[zero_mask]), 1)),
            weights=pdfs[zero_mask], axis=1
        )
        out[~zero_mask] = np.nan

    def get_estimate_mode(self, pdfs: np.dtype, out: np.ndarray):
        out[:] = self.__pdz_bins[np.argmax(pdfs, axis=1)]

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray,
                        output: np.ndarray):
        # From CoaddedPdz
        pdfs = output['REDSHIFT_PDF']
        for estimate in self.__estimates:
            estimate_name = 'REDSHIFT_{}'.format(estimate.upper())
            get_impl = getattr(self, 'get_estimate_' + estimate.lower())
            get_impl(pdfs, out=output[estimate_name])
