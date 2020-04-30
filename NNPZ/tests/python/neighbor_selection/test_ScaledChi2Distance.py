from nnpz.neighbor_selection.brute_force_methods import ScaledChi2Distance, Chi2Distance
import numpy as np
import pytest


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
    We use an uniform prior (equivalent to looking in color space).
    """
    distance_method = ScaledChi2Distance(lambda a: 1, a_min=1e-6, a_max=1e6)
    computed_d, computed_s = distance_method(exact_reference_objects[:, 0, :], exact_reference_objects[:, 1, :],
                                             target_object[0, :], target_object[1, :])
    assert np.allclose(computed_s, 1. / reference_scaling, rtol=1e-3)
    assert np.allclose(computed_d, 0, atol=1e-6)


###############################################################################

def test_simple_delta_prior(target_object, exact_reference_objects, reference_scaling):
    """
    In this case the prior is a delta function at one (equivalent to looking in flux space)
    """
    distance_method = ScaledChi2Distance(lambda a: 1, a_min=1, a_max=1)
    computed_d, computed_s = distance_method(exact_reference_objects[:, 0, :], exact_reference_objects[:, 1, :],
                                             target_object[0, :], target_object[1, :])
    assert np.allclose(computed_s, 1.)

    reference_method = Chi2Distance()
    reference_d, _ = reference_method(exact_reference_objects[:, 0, :], exact_reference_objects[:, 1, :],
                                      target_object[0, :], target_object[1, :])

    assert np.allclose(computed_d, reference_d)
