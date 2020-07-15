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

class NeighborSet(object):
    """
    NeighborSet wraps a set of neighbors, storing their attributes column-wise, so
    you can access or set all neighbors attribute at once as neighbor_set.attribute_name,
    and an individual neighbour as

    * neighbor_set[index].attribute_name
    * neighbor_set.attribute_name[index]
    * [neighbor.attribute_name for neighbor in neighbor_set]

    The objective of this class is to hold the list of neighbor ids, weight and scales without
    having to pass around all the time three separate objects.

    Attributes are created on the fly, so there is no need to modify this class to hold new attributes.

    Notes:
         Remember that NNPZ keep a list of target objects per reference object and not the other way around!
         Meaning, it is normally used as reference_neighbors[target_index].weight
    """
    def __init__(self):
        self.__neighbors = []
        self.index = []

    def __len__(self):
        """
        Returns: Number of objects in the set
        """
        return len(self.__neighbors)

    def append(self, index, **kwargs):
        """
        Add a new neighbor to the set
        Args:
            index:
                Catalog index of the new neighbor
            **kwargs:
                A list of key=value pairs for the attributes of the neighbors for this new object
        """
        for key, value in kwargs.items():
            getattr(self, key).append(value)
        self.index.append(index)
        self.__neighbors.append(Neighbor(self, len(self.index) - 1))

    def __getitem__(self, i):
        """
        Returns: The neighbor at position i
        """
        return self.__neighbors[i]

    def __getattr__(self, key):
        """
        Args:
            key:
                Attribute name

        Returns: A list containing the values for all neighbors for the given attribute name
        Notes: If it doesn't exist, a new column is created
        """
        if key.startswith('_'):
            return object.__getattr__(self, key)
        if key not in dir(self):
            setattr(self, key, [None] * len(self.index))
        return getattr(self, key)

    def __iter__(self):
        """
        Iterate over the neighbors contained inside the set
        """
        for n in self.__neighbors:
            yield n


class Neighbor(object):
    """
    Wraps the attributes contained within NeighborSet, accessing them by the index that identifies the target object
    """
    def __init__(self, neighbor_set, position):
        assert isinstance(neighbor_set, NeighborSet)
        self.__set = neighbor_set
        self.__position = position

    def __index__(self):
        """
        Returns: The catalog index of this object
        Notes:
            This allow to use a neighbor object as a neighbor ID, this is, as a key to lists
        """
        return self.index

    def __getattr__(self, key):
        """
        Args:
            key:
                Attribute name

        Returns: The values for the given attribute name
        """
        if key.startswith('_'):
            return getattr(super(Neighbor, self), key)
        return getattr(self.__set, key)[self.__position]

    def __setattr__(self, key, value):
        """
        Set, or create, an attribute value
        """
        if key.startswith('_'):
            return super(Neighbor, self).__setattr__(key, value)
        getattr(self.__set, key)[self.__position] = value
