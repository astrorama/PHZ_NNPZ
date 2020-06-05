import pytest
from nnpz.framework.NeighborSet import NeighborSet, Neighbor


###############################################################################

def test_populate():
    ns = NeighborSet()
    assert len(ns) == 0

    for idx in range(10):
        ns.append(idx)
    assert len(ns) == 10

    for idx in range(10):
        assert ns[idx].index == idx
        assert ns.index[idx] == idx


###############################################################################

def test_populate_with_extras():
    ns = NeighborSet()

    for idx in range(10):
        ns.append(idx, extra_attribute=idx ** 2)
    assert len(ns) == 10

    for idx in range(10):
        assert ns.extra_attribute[idx] == idx ** 2
        assert ns[idx].extra_attribute == idx ** 2


###############################################################################

def test_populate_with_extras_modify():
    ns = NeighborSet()

    for idx in range(10):
        ns.append(idx, extra_attribute=idx ** 2)

    for idx in range(10):
        if idx % 2 == 0:
            ns[idx].extra_attribute = idx * 3

    for idx in range(10):
        if idx % 2 == 0:
            assert ns.extra_attribute[idx] == idx * 3
        else:
            assert ns.extra_attribute[idx] == idx ** 2


###############################################################################

def test_populate_and_add():
    ns = NeighborSet()

    for idx in range(10):
        ns.append(idx, extra_attribute=idx ** 2)

    for idx in range(10):
        ns[idx].another_attribute = idx + 5

    for idx in range(10):
        assert ns.another_attribute[idx] == idx + 5


###############################################################################

def test_populate_and_add_2():
    ns = NeighborSet()

    for idx in range(10):
        ns.append(idx, extra_attribute=idx ** 2)

    for idx in range(10):
        ns.another_attribute[idx] = idx + 5

    for idx in range(10):
        assert ns[idx].another_attribute == idx + 5
