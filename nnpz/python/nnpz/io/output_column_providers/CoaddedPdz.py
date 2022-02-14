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
from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np
from nnpz.io import OutputHandler
from nnpz.reference_sample.ReferenceSample import ReferenceSample

# pylint: disable=no-member
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
    """

    def __init__(self, reference_sample: ReferenceSample):
        super(CoaddedPdz, self).__init__()
        self.__pdz_provider = reference_sample.get_provider('pdz')
        self.__pdz_bins = self.__pdz_provider.get_redshift_bins()

    def get_column_definition(self) \
            -> List[Tuple[str, np.dtype, u.Unit, Optional[Tuple[int, ...]]]]:
        return [
            ('REDSHIFT_PDF', np.float32, u.dimensionless_unscaled, (len(self.__pdz_bins),))
        ]

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

        # Normalize
        integral = np.trapz(output_pdz, self.__pdz_bins, axis=-1)
        np.reciprocal(integral, out=integral)
        np.multiply(output_pdz, integral[..., np.newaxis], out=output_pdz)
