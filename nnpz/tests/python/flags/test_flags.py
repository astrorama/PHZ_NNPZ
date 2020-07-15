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
