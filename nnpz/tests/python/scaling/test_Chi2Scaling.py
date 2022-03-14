#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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

import numpy as np
import pytest
from _Nnpz import scaling_factory, ScaleFunctionParams
from nnpz.utils.distances import chi2


###############################################################################

@pytest.fixture
def target_object():
    return np.array([(1., 0.5), (2., 0.2), (3., 0.6)])


###############################################################################

@pytest.fixture
def reference_scaling():
    return np.linspace(0.2, 4., 5)


###############################################################################

@pytest.fixture
def exact_reference_objects(target_object, reference_scaling):
    reference = np.zeros((len(reference_scaling), 3, 2))
    for i, s in enumerate(reference_scaling):
        reference[i, :, 0] = target_object[:, 0] * s
        reference[i, :, 1] = target_object[:, 1] * s
    return reference


###############################################################################

def test_simple_uniform_prior(target_object, exact_reference_objects, reference_scaling):
    """
    The reference objects are just the target object exactly scaled in all bands, so
    the ScaledChi2Distance should return a distance of 0, and a recovered scale.
    We use a uniform prior (equivalent to looking in color-space).
    """
    scaling_method = scaling_factory('uniform', ScaleFunctionParams(10, 1e-4))
    computed_s = np.asarray(list(
        map(lambda reference: scaling_method(reference, target_object), exact_reference_objects)))
    computed_d = chi2(computed_s[:, np.newaxis, np.newaxis] * exact_reference_objects,
                      target_object)
    np.testing.assert_allclose(computed_s, 1. / reference_scaling, rtol=1e-3)
    np.testing.assert_allclose(computed_d, 0, atol=1e-6)

    ###############################################################################


def test_simple_delta_prior(target_object, exact_reference_objects):
    """
    In this case the prior is a delta function at one (equivalent to looking in flux space)
    """
    scaling_method = scaling_factory('delta 1', ScaleFunctionParams(10, 1e-4))
    computed_s = np.asarray(list(
        map(lambda reference: scaling_method(reference, target_object), exact_reference_objects)))
    computed_d = chi2(computed_s[:, np.newaxis, np.newaxis] * exact_reference_objects,
                      target_object)
    np.testing.assert_allclose(computed_s, 1.)

    reference_d = chi2(exact_reference_objects, target_object)

    np.testing.assert_allclose(computed_d, reference_d)

###############################################################################
