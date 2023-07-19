#
#  Copyright (C) 2022 Euclid Science Ground Segment
#
#  This library is free software; you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation;
#  either version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with this library;
#  if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA
#
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
from datetime import datetime
from typing import Dict, Union

import astropy.units as u
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.neighbors.WeightConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.target
import numpy as np
from nnpz.config import ConfigManager
from nnpz.photometry.photometry import Photometry
from nnpz.reference_sample import IndexProvider


class ComputeWeights:
    """
    Compute the weights for the selected neighbors for a set of target objects.
    (May or may not follow the photometry correction).
    Weights based on distance are corrected with the absolute weight coming from the reference
    sample.
    """

    def __init__(self, conf_manager: Union[ConfigManager, Dict]):
        target_system = conf_manager.get('target_system')
        ref_system = conf_manager.get('reference_system')
        self.__index: IndexProvider = conf_manager.get('reference_sample').index
        self.__ref_filter_indexes = ref_system.get_band_indexes(target_system.bands)
        self.__weight_calculator = conf_manager.get('weight_calculator')

    @u.quantity_input
    def __call__(self, target: Photometry, neighbor_index: np.ndarray, neighbor_photo: u.uJy,
                 neighbor_scales: np.ndarray, out_weights: np.ndarray, out_flags: np.ndarray):
        # This makes a copy
        nn_photo = neighbor_photo.take(self.__ref_filter_indexes, axis=2)
        # The Scaling of the photometry is done by the neighbor finder
        # nn_photo *= neighbor_scales[..., np.newaxis, np.newaxis]

        # FITS stores in big-endian, and we can't pass that directly to cython
        if nn_photo.dtype.byteorder == '>':
            nn_photo = nn_photo.newbyteorder().byteswap(inplace=True)

        self.__weight_calculator(nn_photo.value, target.values.value, out_weights, out_flags)

        # Correct the weights with the reference absolute weights
        out_weights *= self.__index.get_weight_for_index(neighbor_index)
