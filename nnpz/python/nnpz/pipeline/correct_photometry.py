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

from typing import Dict, Union

import astropy.units as u
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.reference
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
from typing import OrderedDict
import nnpz.config.target
import numpy as np
from ElementsKernel import Logging
from nnpz.config.ConfigManager import ConfigManager
from nnpz.photometry.photometry import Photometry
from nnpz.photometry.photometry import PhotometricSystem
from nnpz.photometry.projection import source_independent_ebv
from nnpz.photometry.projection.ebv import correct_ebv
from nnpz.photometry.projection.filter_variation import correct_filter_variation

logger = Logging.getLogger(__name__)


class CorrectPhotometry:
    """
    Correct the photometry for the selected neighbors for a set of target objects.

    See Also:
        correct_ebv, correct_filter_variation
    """
    def __init__(self, conf_manager: Union[ConfigManager, Dict]):
        self.__ref_system = conf_manager.get('reference_system')
        self.__ebv_corr_coefs = conf_manager.get('reference_ebv_correction')
        self.__filter_corr_coefs = conf_manager.get('reference_filter_variation_correction')

    @u.quantity_input
    def __call__(self, target: Photometry, neighbor_idx: np.ndarray, neighbor_photo: u.uJy,
                 out: u.uJy = None):
        if out is None:
            out = neighbor_photo.copy()

        assert out.shape == neighbor_photo.shape
        assert out.shape[0] == len(target)
        assert out.shape[2] == len(self.__ref_system.bands)
        assert out.shape[3] == 2

        if 'ebv' in target.colorspace:
            chunk_ebv_corr_coefs = self.__ebv_corr_coefs[neighbor_idx]
            # Check if the ebv_corr_coeff are different of 0, if not apply grey correction
            for filter_idx, filter_name in enumerate(self.__ref_system.bands):
                shape = chunk_ebv_corr_coefs[:, :, filter_idx].shape
                total = shape[0]*shape[1]

                nn_filter_photo = neighbor_photo[:, :, filter_idx, :]
                nn_filter_out = out[:, :, filter_idx, :]
                if np.sum(chunk_ebv_corr_coefs[:, :, filter_idx]==0) != total:
                    logger.info('Correcting %s for EBV using neighbours photometry _EBV_CORR coefficients', filter_name)
                    correct_ebv(nn_filter_photo,
                                    corr_coef=chunk_ebv_corr_coefs[:, :, filter_idx],
                                    ebv=target.colorspace.ebv,
                                    out=nn_filter_out)
                else:
                    logger.warning('Correcting %s for EBV using grey correction as the reference sample photometry file is missing _EBV_CORR coefficient or they are all zero.', filter_name)
                    transmission = self.__ref_system.get_transmission(filter_name)
                    sub_dict =  OrderedDict()
                    sub_dict[filter_name]= transmission
                    sub_system = PhotometricSystem(sub_dict)
                    reddener = source_independent_ebv.SourceIndependentGalacticEBV(sub_system)
                    # We use the SourceIndependentGalacticEBV which expoect data with another form factor: therefore we need to loop over the neighbours
                    for index in range(nn_filter_out.shape[1]):
                        data = nn_filter_photo[:,index,:]
                        data = data.reshape([nn_filter_out.shape[0],1,2])
                        reddener.redden(data, target.colorspace.ebv, out=data)
                        data = data.reshape([nn_filter_out.shape[0],2])
                        nn_filter_photo[:,index,:] = data
                            
                            

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
        return out
