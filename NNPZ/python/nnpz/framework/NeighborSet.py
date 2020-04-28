class NeighborSet(object):
    def __init__(self):
        self.__neighbors = []
        self.index = []

    def __len__(self):
        return len(self.__neighbors)

    def append(self, index, **kwargs):
        for key, value in kwargs.items():
            getattr(self, key).append(value)
        self.index.append(index)
        self.__neighbors.append(Neighbor(self, len(self.index) - 1))

    def __getitem__(self, i):
        return self.__neighbors[i]

    def __getattr__(self, key):
        if key.startswith('_'):
            return object.__getattr__(self, key)
        if key not in dir(self):
            setattr(self, key, [None] * len(self.index))
        return getattr(self, key)

    def __iter__(self):
        for n in self.__neighbors:
            yield n


class Neighbor(object):
    def __init__(self, neighbor_set, position):
        assert isinstance(neighbor_set, NeighborSet)
        self.__set = neighbor_set
        self.__position = position

    def __index__(self):
        return self.index

    def __getattr__(self, key):
        if key.startswith('_'):
            return getattr(super(Neighbor, self), key)
        return getattr(self.__set, key)[self.__position]

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super(Neighbor, self).__setattr__(key, value)
        getattr(self.__set, key)[self.__position] = value
