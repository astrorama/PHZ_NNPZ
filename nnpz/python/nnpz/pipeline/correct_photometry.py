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

import astropy.units as u
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.reference
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.target
import numpy as np
from ElementsKernel import Logging
from nnpz.config.ConfigManager import ConfigManager
from nnpz.photometry.photometry import Photometry
from nnpz.photometry.projection.ebv import correct_ebv
from nnpz.photometry.projection.filter_variation import correct_filter_variation

logger = Logging.getLogger(__name__)


class CorrectPhotometry:
    def __init__(self, conf_manager: ConfigManager):
        self.__ref_system = conf_manager.getObject('reference_system')
        self.__ebv_corr_coefs = conf_manager.getObject('reference_ebv_correction')
        self.__filter_corr_coefs = conf_manager.getObject('reference_filter_variation_correction')

    @u.quantity_input
    def __call__(self, target: Photometry, neighbor_idx: np.ndarray, neighbor_photo: u.uJy,
                 out: u.uJy = None):
        if out is None:
            out = neighbor_photo

        assert out.shape == neighbor_photo.shape
        assert out.shape[0] == len(target)
        assert out.shape[2] == len(self.__ref_system.bands)
        assert out.shape[3] == 2

        if 'ebv' in target.colorspace:
            chunk_ebv_corr_coefs = self.__ebv_corr_coefs[neighbor_idx]
            for filter_idx, filter_name in enumerate(self.__ref_system.bands):
                logger.info('Correcting %s for EBV', filter_name)
                nn_filter_photo = neighbor_photo[:, :, filter_idx, :]
                nn_filter_out = out[:, :, filter_idx, :]
                correct_ebv(nn_filter_photo,
                            corr_coef=chunk_ebv_corr_coefs[:, :, filter_idx],
                            ebv=target.colorspace.ebv,
                            out=nn_filter_out)

        if 'shifts' in target.colorspace:
            chunk_filter_corr_coefs = self.__filter_corr_coefs[neighbor_idx]
            for filter_idx, filter_name in enumerate(self.__ref_system.bands):
                if filter_name not in target.colorspace.shifts.dtype.names:
                    continue
                logger.info('Correcting for %s filter variation', filter_name)
                nn_filter_out = out[:, :, filter_idx, :]
                correct_filter_variation(nn_filter_out,
                                         corr_coef=chunk_filter_corr_coefs[:, :, filter_idx],
                                         shift=target.colorspace.shifts[filter_name],
                                         out=nn_filter_out)
