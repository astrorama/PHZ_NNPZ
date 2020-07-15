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
