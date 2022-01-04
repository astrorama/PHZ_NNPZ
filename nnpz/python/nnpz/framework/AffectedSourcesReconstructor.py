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
Created on: 27/05/19
Author: Alejandro Alvarez Ayllon
"""


from nnpz.framework.NeighborSet import NeighborSet


class AffectedSourcesReconstructor(object):
    """
    This class reconstructs the affected map, as returned by AffectedSourcesFinder,
    from a NNPZ output catalof
    """

    def __init__(self):
        pass

    @staticmethod
    def reconstruct(ref_ids, src_indexes, neighbors, weights, scales, progress_listener):
        """

        Args:
            ref_ids:
                Reference sample IDS.
            src_indexes:
                The indexes of the rows from the input catalog. Note that this is *not* the
                object ID, but the row position within the table.
            neighbors:
                An iterable object where each position holds the neighbors of each source in
                the catalog.
            weights:
                An iterable object where each position holds the weights for the neighbors of each
                source in the catalog.
            scales:
                An iterable object where each position holds the scaling for the neighbors of each
                source in the catalog.
            progress_listener:
                A callable that will be notified by the progress
        Returns:
            A map where the keys are the indices of the reference sample objects
            and values are lists of the input catalog indices that are affected
            by this reference object, and another map with the weights instead.
        """
        ref_id_to_idx = {}
        for ref_idx, ref_id in enumerate(ref_ids):
            ref_id_to_idx[ref_id] = ref_idx
        affected = {}
        for i, src_idx in enumerate(src_indexes):
            for nn_id, nn_weight, nn_scale in zip(neighbors[i], weights[i], scales[i]):
                nn_idx = ref_id_to_idx[nn_id]
                if nn_idx not in affected:
                    affected[nn_idx] = NeighborSet()
                affected[nn_idx].append(src_idx, weight=nn_weight, scale=nn_scale)
            progress_listener(i)
        return affected
