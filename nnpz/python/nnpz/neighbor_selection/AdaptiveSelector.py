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

import numpy as np
from ElementsKernel import Logging

from .NeighborSelectorInterface import NeighborSelectorInterface

logger = Logging.getLogger('AdaptiveSelector')


class AdaptiveSelector(NeighborSelectorInterface):
    """
    Wraps another NeighborSelectorInterface, and remove from consideration any dimension
    for which all target sources have a NaN value.
    """

    def __init__(self, selector, target_phot, filter_names=None):
        """
        Constructor
        Args:
            selector: Decorated implementation of NeighborSelector
            target_phot: Photometries from the target catalog
            filter_names: For logging purposes
        """
        if not isinstance(selector, NeighborSelectorInterface):
            raise TypeError(
                'Expecting a NeighborSelectorInterface for selector, got {}'.format(type(selector))
            )
        super(AdaptiveSelector, self).__init__()
        self.__selector = selector
        n_bands = target_phot.shape[1]
        self.__valid_bands = [
            i for i in range(n_bands) if not np.all(np.isnan(target_phot[:, i, 0]))
        ]
        # If by chance all bands are empty, that's likely an empty catalog, so do not remove any!
        if not self.__valid_bands:
            self.__valid_bands = range(n_bands)
        self.__invalid_bands = [i for i in range(n_bands) if i not in self.__valid_bands]
        if len(self.__valid_bands) < n_bands:
            if filter_names:
                logger.warning(
                    'Discarding columns %s from the neighbor search',
                    ', '.join([filter_names[i][0] for i in self.__invalid_bands])
                )
            else:
                logger.warning(
                    'Discarding %d bands from the neighbor search',
                    n_bands - len(self.__valid_bands)
                )

    def _findNeighborsImpl(self, coordinate, flags):
        return self.__selector.findNeighbors(coordinate[self.__valid_bands, :], flags)

    def _initializeImpl(self, ref_data):
        return self.__selector.initialize(ref_data[:, self.__valid_bands, :])
