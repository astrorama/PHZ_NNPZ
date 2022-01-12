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


class ComputeWeights:
    def __init__(self, conf_manager: ConfigManager):
        target_system = conf_manager.getObject('target_system')
        ref_system = conf_manager.getObject('reference_system')
        self.__ref_filter_indexes = ref_system.get_band_indexes(target_system.bands)
        self.__weight_calculator = conf_manager.getObject('weight_calculator')

    @u.quantity_input
    def __call__(self, target: Photometry, neighbor_photo: u.uJy,
                 neighbor_scales: np.ndarray,
                 out_weights: np.ndarray, out_flags: np.ndarray):
        # This makes a copy
        nn_photo = neighbor_photo[:, :, self.__ref_filter_indexes, :]
        nn_photo *= neighbor_scales[..., np.newaxis, np.newaxis]

        # FITS stores in big-endian, and we can't not pass that directly to cython
        if nn_photo.dtype.byteorder == '>':
            nn_photo = nn_photo.newbyteorder().byteswap(inplace=True)

        self.__weight_calculator(nn_photo.value, target.values.value,
                                 output_weights=out_weights, output_flags=out_flags)
