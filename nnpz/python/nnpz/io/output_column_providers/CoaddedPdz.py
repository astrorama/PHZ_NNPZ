#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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
Created on: 28/03/2018
Author: Alejandro Alvarez Ayllon
"""
from typing import List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
from nnpz.io import OutputHandler
from nnpz.reference_sample.ReferenceSample import ReferenceSample
# pylint: disable=no-member
from scipy.interpolate import interp1d
from scipy.stats import norm

smallest_f32 = np.finfo(np.float32).tiny
biggest_f32 = np.finfo(np.float32).max


class CoaddedPdz(OutputHandler.OutputColumnProviderInterface):
    """
    The CoaddedPdz output provider generates a PDZ out of the weighted sum
    of the neighbors PDZ, and re-normalized (its integral is 1)
    It generates two columns:
    - CoaddedPdz, with the PDZ values
    - CoaddedPdzBins, with the PDZ bins
    It assumes all reference samples have the same PDZ bins.

    Args:
        reference_sample: ReferenceSample
            ReferenceSample instance with the PDZ provider setup
        kernel: Union[None, str]
        bandwidth: Union[str, float]
            Kernel bandwidth. It can be a value, 'auto' or a callable
    """

    def __iqr(self, pdz: np.ndarray) -> np.ndarray:
        """
        Inter-quantile range (Q3 - Q1)
        """
        iqr = np.ones(len(pdz))
        cum_prob = np.zeros_like(pdz)
        np.cumsum(self.__pdz_dbins * ((pdz[:, :-1] + pdz[:, 1:]) / 2.), out=cum_prob[:, 1:],
                  axis=-1)
        for i in range(len(pdz)):
            max_prob = np.max(cum_prob[i])
            if max_prob:
                inv_cum = interp1d(cum_prob[i] / max_prob, self.__pdz_bins, kind='linear')
                iqr[i] = inv_cum(0.75) - inv_cum(0.25)
        return iqr

    def __auto_bandwidth(self, pdz: np.ndarray) -> np.ndarray:
        """
        Compute the bandwidth of the gaussian kernel (σ) applying Silverman's rule of thumb.
        Since the PDZ is a histogram, we have a weighted-sample. We estimate the number of
        datapoints (neff) as scipy's `gaussian_kde` module does.

        See Also:
            https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
        """
        iqr = self.__iqr(pdz)
        tiled_pdz = np.tile(self.__pdz_bins, (len(pdz), 1))
        avg = np.average(tiled_pdz, weights=pdz, axis=1)
        var = np.sum(pdz * (tiled_pdz - avg[:, np.newaxis]) ** 2, axis=-1) / np.sum(pdz, axis=-1)
        neff = np.sum(pdz, axis=1) ** 2 / np.sum(pdz ** 2, axis=1)
        return 0.9 * np.minimum(np.sqrt(var), iqr / 1.34) * (neff ** -0.2)

    def __init__(self, reference_sample: ReferenceSample,
                 kernel: Union[None, str], bandwidth: Union[str, float],
                 store_kernel: bool = False):
        super(CoaddedPdz, self).__init__()
        self.__pdz_provider = reference_sample.get_provider('pdz')
        self.__pdz_bins = self.__pdz_provider.get_redshift_bins()
        self.__pdz_dbins = np.diff(self.__pdz_bins)[np.newaxis, :]
        self.__kernel = kernel
        if bandwidth == 'auto':
            self.__bandwidth = self.__auto_bandwidth
        else:
            self.__bandwidth = lambda pdz: np.full(len(pdz), fill_value=bandwidth)
        self.__store_kernel = store_kernel

        # Make sure the kernel has an odd number of bins
        self.__pdz_odd = self.__pdz_bins if len(self.__pdz_bins) % 2 == 1 else self.__pdz_bins[:-1]
        self.__mz = self.__pdz_odd[len(self.__pdz_odd) // 2]

    def get_column_definition(self) -> List[Tuple[str, np.dtype, u.Unit, Optional[Tuple[int, ...]]]]:
        columns = [
            ('REDSHIFT_PDF', np.float32, u.dimensionless_unscaled, (len(self.__pdz_bins),))
        ]
        if self.__store_kernel:
            columns.append((
                'REDSHIFT_PDF_KERNEL', np.float32, u.dimensionless_unscaled, (len(self.__pdz_bins))
            ))
        return columns

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray, output: np.ndarray):
        neighbor_idx = neighbor_info['NEIGHBOR_INDEX']
        neighbor_weights = neighbor_info['NEIGHBOR_WEIGHTS']
        output_pdz = output['REDSHIFT_PDF']

        neighbor_pdz = self.__pdz_provider.get_pdz_for_index(neighbor_idx.ravel())
        neighbor_pdz = neighbor_pdz.reshape(*neighbor_idx.shape, -1)

        # Coadd
        neighbor_pdz *= neighbor_weights[..., np.newaxis]
        total_weight = np.sum(neighbor_weights, axis=-1)
        np.sum(neighbor_pdz, axis=1, out=output_pdz)
        np.divide(output_pdz, total_weight[..., np.newaxis], out=output_pdz)
        
        # Clip negatives
        np.clip(output_pdz, a_min=0, a_max=None, out=output_pdz)

        # Normalize
        integral = np.trapz(output_pdz, self.__pdz_bins, axis=-1)
        np.reciprocal(integral, out=integral)
        np.multiply(output_pdz, integral[..., np.newaxis], out=output_pdz)

        # Smooth
        if not self.__kernel:
            return

        bandwidths = self.__bandwidth(output_pdz)
        for i, pdz in enumerate(output_pdz):
            sigma = bandwidths[i]
            if sigma > 0:
                kernel = norm.pdf(self.__pdz_odd, loc=self.__mz, scale=sigma)
                norm_scale = np.sum(kernel)
                if norm_scale > 0:
                    pdz[:] = np.convolve(pdz, kernel, mode='same') / norm_scale
                if self.__store_kernel:
                    output['REDSHIFT_PDF_KERNEL'][i] = kernel
