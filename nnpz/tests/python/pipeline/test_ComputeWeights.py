#
#  Copyright (C) 2022 Euclid Science Ground Segment
#
#  This library is free software; you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation;
#  either version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with this library;
#  if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA
#
from _Nnpz import WeightCalculator
from nnpz.pipeline.compute_weights import ComputeWeights

from ..photometry.fixtures import *


class MockReferenceSample:
    class MockIndex:
        def __init__(self, weights):
            self.__weights = weights

        def get_weight_for_index(self, idxs):
            return self.__weights

    def __init__(self, weights):
        self.index = MockReferenceSample.MockIndex(weights)


###############################################################################

def test_computeWeights(reference_photometry: Photometry, target_photometry: Photometry):
    weights = ComputeWeights(dict(
        target_system=target_photometry.system,
        reference_system=reference_photometry.system,
        reference_sample=MockReferenceSample(weights=np.ones_like(reference_photometry.ids)),
        weight_calculator=WeightCalculator('Euclidean', 'Euclidean')
    ))
    neighbor_photo = np.repeat(reference_photometry.values.value[np.newaxis],
                               len(target_photometry), axis=0) * u.uJy
    neighbor_scales = np.ones((len(target_photometry), len(reference_photometry)), dtype=np.float64)
    out_weights = np.zeros_like(neighbor_scales, dtype=np.float32)
    out_flags = np.zeros(len(target_photometry), dtype=np.uint32)
    weights(target_photometry, reference_photometry.ids, neighbor_photo, neighbor_scales,
            out_weights=out_weights, out_flags=out_flags)

    assert out_weights[0].argmax() == 0 and out_weights[0].argmin() == 4
    assert out_weights[1].argmax() == 1 and out_weights[1].argmin() == 4
    assert out_weights[2].argmax() == 2 and out_weights[1].argmin() in {0, 4}
    assert out_weights[3].argmax() == 3 and out_weights[3].argmin() == 0
    assert out_weights[4].argmax() == 4 and out_weights[3].argmin() == 0

    np.testing.assert_allclose(out_weights[0],
                               [82.671295, 2.754853, 1.356582, 0.899841, 0.673187], rtol=1e-4)
    np.testing.assert_allclose(out_weights[1],
                               [2.816, 48.375763, 2.542062, 1.302916, 0.875916], rtol=1e-4)
    np.testing.assert_allclose(out_weights[2],
                               [1.311178, 2.573811, 64.65588, 2.778595, 1.362332], rtol=1e-4)
    np.testing.assert_allclose(out_weights[3],
                               [0.883841, 1.320545, 2.610245, 103.75939, 2.737767], rtol=1e-4)
    np.testing.assert_allclose(out_weights[4],
                               [0.66692, 0.88868, 1.331383, 2.652985, 334.6473], rtol=1e-4)


###############################################################################

def test_computeWeightsWithAbsolute(reference_photometry: Photometry,
                                    target_photometry: Photometry):
    weight_values = np.array([1., 2., 3., 4., 5.])
    weights = ComputeWeights(dict(
        target_system=target_photometry.system,
        reference_system=reference_photometry.system,
        reference_sample=MockReferenceSample(weights=weight_values),
        weight_calculator=WeightCalculator('Euclidean', 'Euclidean')
    ))
    neighbor_photo = np.repeat(reference_photometry.values.value[np.newaxis],
                               len(target_photometry), axis=0) * u.uJy
    neighbor_scales = np.ones((len(target_photometry), len(reference_photometry)), dtype=np.float64)
    out_weights = np.zeros_like(neighbor_scales, dtype=np.float32)
    out_flags = np.zeros(len(target_photometry), dtype=np.uint32)
    weights(target_photometry, reference_photometry.ids, neighbor_photo, neighbor_scales,
            out_weights=out_weights, out_flags=out_flags)

    np.testing.assert_allclose(out_weights[0],
                               [82.671295, 2.754853, 1.356582, 0.899841, 0.673187] * weight_values,
                               rtol=1e-4)
    np.testing.assert_allclose(out_weights[1],
                               [2.816, 48.375763, 2.542062, 1.302916, 0.875916] * weight_values,
                               rtol=1e-4)
    np.testing.assert_allclose(out_weights[2],
                               [1.311178, 2.573811, 64.65588, 2.778595, 1.362332] * weight_values,
                               rtol=1e-4)
    np.testing.assert_allclose(out_weights[3],
                               [0.883841, 1.320545, 2.610245, 103.75939, 2.737767] * weight_values,
                               rtol=1e-4)
    np.testing.assert_allclose(out_weights[4],
                               [0.66692, 0.88868, 1.331383, 2.652985, 334.6473] * weight_values,
                               rtol=1e-4)

###############################################################################
