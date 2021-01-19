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
import logging

import numpy as np
import pytest
from nnpz import NnpzFlag
from nnpz.framework import ProgressListener
from nnpz.framework.NeighborSet import NeighborSet
from nnpz.framework.ReferenceSampleWeightCalculator import ReferenceSampleWeightCalculator
from nnpz.weights import InverseEuclideanWeight, WeightPhotometryProvider


class DummyPhotometryProvider(WeightPhotometryProvider):
    """
    Dummy implementation of WeightPhotometryProvider
    """

    def __init__(self, filter_list):
        self.__filter_list = filter_list
        self.__dtype = [(filter_name, np.float32) for filter_name in filter_list]
        self.__sample = np.zeros(2, dtype=self.__dtype)
        for i, filter_name in enumerate(filter_list):
            self.__sample[filter_name][0] = ord(filter_name[0])
            self.__sample[filter_name][1] = i
        assert self.__sample.shape == (2,)

    def __call__(self, ref_i, cat_i, flags):
        photo = self.__sample.copy()
        for filter_name in self.__filter_list:
            photo[filter_name][0] *= 1 + ref_i
            photo[filter_name][1] *= np.sqrt(1 + ref_i)
        return photo


@pytest.fixture
def filter_list():
    return list('ugriz')


@pytest.fixture
def photometry_provider(filter_list):
    return DummyPhotometryProvider(filter_list + ['X'])


@pytest.fixture
def target_data(filter_list):
    data = np.zeros((5, len(filter_list), 2), dtype=np.float32)
    ids = np.arange(5)
    for i, filter_name in enumerate(filter_list):
        data[:, i, 0] = ids + np.random.normal(0, 0.25, size=5)
        data[:, i, 1] = np.sqrt(ids)
    return data


def test_ref_weight_calculator(filter_list, photometry_provider, target_data):
    """
    Test the ReferenceSampleWeightCalculator. Note that the filter_list is a subset
    of the bands provided by the photometry provider. The weighting must use this strict
    subset, but pass down the pipeline the full one.
    """
    weight_calculator = InverseEuclideanWeight()
    calculator = ReferenceSampleWeightCalculator(photometry_provider, weight_calculator,
                                                 weight_calculator, filter_list=filter_list)
    affected = {
        0: NeighborSet(indexes=[0, 1, 2]),
        1: NeighborSet(indexes=[1, 2, 3]),
        2: NeighborSet(indexes=[2, 3, 4]),
    }
    result_flags = [NnpzFlag() for _ in target_data]
    calculator.computeWeights(
        affected, target_data, result_flags,
        progress_listener=ProgressListener(len(affected), logger=logging.getLogger(__name__))
    )

    for ref_i, targets in affected.items():
        for t in targets:
            target_i = t.index
            expected_photo = photometry_provider(ref_i, target_i, None)
            assert target_i in affected[ref_i].index
            assert 'X' in t.matched_photo.dtype.names
            assert np.array_equal(t.matched_photo, expected_photo)
            assert t.weight > 0
