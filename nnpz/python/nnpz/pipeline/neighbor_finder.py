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
import copy
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
from typing import Dict, Union

# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.neighbors.GalacticUnreddenerConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.neighbors.NeighborSelectorConfig
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

logger = Logging.getLogger(__name__)


class NeighborFinder:
    """
    Find the neighbors for a set of target objects
    """

    def __init__(self, conf_manager: Union[ConfigManager, Dict]):
        self.__ref_data: Photometry = conf_manager.get('reference_photometry')
        self.__source_independent_ebv = conf_manager.get('source_independent_ebv')
        self.__selector = conf_manager.get('neighbor_selector')
        self.__knn = conf_manager.get('neighbor_no')
        # Train the selector
        self.__selector.fit(self.__ref_data, conf_manager.get('target_system'))

    def __call__(self, target: Photometry, out: np.ndarray):
        assert 'NEIGHBOR_INDEX' in out.dtype.names
        assert 'NEIGHBOR_SCALING' in out.dtype.names
        assert 'NEIGHBOR_PHOTOMETRY' in out.dtype.names
        assert len(target) == len(out)

        if self.__source_independent_ebv is not None:
            logger.info('De-redden')
            target = copy.copy(target)
            target.values = self.__source_independent_ebv.deredden(target.values,
                                                                   ebv=target.colorspace.ebv)

        logger.info('Looking for neighbors')
        all_idx, all_scales = self.__selector.query(target)

        out['NEIGHBOR_INDEX'] = all_idx
        out['NEIGHBOR_SCALING'] = all_scales
        out['NEIGHBOR_PHOTOMETRY'] = self.__ref_data.values[all_idx]
        out['NEIGHBOR_PHOTOMETRY'] *= all_scales[..., np.newaxis, np.newaxis]
