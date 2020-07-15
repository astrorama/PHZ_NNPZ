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

"""
Created on: 02/05/2018
Author: Alejandro Alvarez Ayllon
"""

from nnpz.flags.NnpzFlag import NnpzFlag


def test_emptyFlag():
    flags = NnpzFlag()
    assert flags.isClear()
    assert not (flags & NnpzFlag.AlternativeWeightFlag)

def test_setFlag():
    flags = NnpzFlag()
    flags |= NnpzFlag.AlternativeWeightFlag
    assert not flags.isClear()
    assert flags & NnpzFlag.AlternativeWeightFlag
