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
from typing import Any, Dict, List, Optional, Tuple

import astropy.units as u
import numpy as np
from nnpz.io import OutputHandler
from nnpz.reference_sample import ReferenceSample


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

    def __sample(self, pdfs: np.ndarray, bins: np.ndarray, quantiles: np.ndarray,
                 output: np.ndarray):
        if len(quantiles.shape) == 1:
            quantiles = np.tile(quantiles, (len(output), 1))

        cum_prob = np.zeros((len(pdfs), len(bins)))
        bin_diff = np.diff(bins)[np.newaxis]
        pdfs_med = (pdfs[:, :-1] + pdfs[:, 1:]) / 2.
        cum_prob[:, 1:] = np.cumsum(bin_diff * pdfs_med, axis=-1)
        cum_prob[:] /= cum_prob[:, -1, np.newaxis]
        for i in range(len(output)):
            output[i] = np.interp(quantiles[i], cum_prob[i], bins)
        return output

    def __init__(self, ref_sample: ReferenceSample, quantiles: List[float] = None,
                 mc_samples: int = 0):
        self.__pdz_bins = ref_sample.getProvider('pdz').getRedshiftBins()
        self.__qs = np.asarray(quantiles) if quantiles else []
        self.__mc_no = mc_samples

    def get_column_definition(self) \
            -> List[Tuple[str, np.dtype, u.Unit, Optional[Tuple[int, ...]]]]:
        def_cols = []
        if self.__qs is not None:
            def_cols.append(
                ('REDSHIFT_PDF_QUANTILES', np.float32, u.dimensionless_unscaled, len(self.__qs))
            )
        if self.__mc_no > 0:
            def_cols.append(('REDSHIFT_PDF_MC', np.float32, u.dimensionless_unscaled, self.__mc_no))
        return def_cols

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray,
                        output: np.ndarray):
        # From CoaddedPdz
        pdfs = output['REDSHIFT_PDF']

        if self.__qs is not None:
            self.__sample(pdfs, self.__pdz_bins, self.__qs, output=output['REDSHIFT_PDF_QUANTILES'])

        if self.__mc_no > 0:
            samples = np.random.rand(len(output), self.__mc_no)
            self.__sample(pdfs, self.__pdz_bins, samples, output=output['REDSHIFT_PDF_MC'])

    def get_headers(self) -> Dict[str, Any]:
        keys = {}
        if self.__qs is not None:
            keys["PDFQUAN"] = ' '.join([str(q) for q in self.__qs])
        return keys
