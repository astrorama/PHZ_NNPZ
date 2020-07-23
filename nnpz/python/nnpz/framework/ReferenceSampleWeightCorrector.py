#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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

from nnpz.framework import ReferenceSampleWeightCalculator


class ReferenceSampleWeightCorrector(object):
    """
    This class wraps the ReferenceSampleWeightCalculator if
    the user requires a correction of the relative weights (based on distance)
    by an absolute weight (individual per entry on the reference sample).

    The resulting weight would be absolute * relative
    """

    def __init__(self, wrapped, absolute_weights):
        """
        Constructor
        Args:
            wrapped: An instance of ReferenceSampleWeightCalculator
            absolute_weights: A list, or numpy array, with the weights corresponding to *each* entry
                in the reference catalog
        """
        assert isinstance(wrapped, ReferenceSampleWeightCalculator)
        self.__wrapped = wrapped
        self.__absolute_weights = absolute_weights

    def computeWeights(self, affected, target_data, result_flags, progress_listener):
        """
        Args:
            affected: The output of AffectedSourcesFinder.findAffected
            target_data: Target catalog data
            result_flags: A list of NnpzFlag, one per entry on the target catalog
            progress_listener: An object implementing ProgressListener

        Returns:
            A map where the keys are the indices of the reference sample objects
            and values are lists of the computed weight of the reference sample object per each
            object in the target catalog
        """
        weights = self.__wrapped.computeWeights(
            affected, target_data, result_flags, progress_listener
        )
        for weight in weights.keys():
            for i in range(len(weights[weight])):
                weights[weight][i] *= self.__absolute_weights[weight]
        return weights
