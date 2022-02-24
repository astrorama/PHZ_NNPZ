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

"""
Created on: 14/12/17
Author: Nikolaos Apostolakos
"""

from collections import OrderedDict

import astropy.units as u
import numpy as np
import pytest
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.projection.source_independent_ebv import SourceIndependentGalacticEBV


@pytest.fixture
def sed():
    return np.array(
        [[0., 1.], [1., 1.], [2., 1.], [3., 1.], [4., 1], [5., 1], [6., 1], [7., 1], [8., 1],
         [9., 1], [10., 1]])


@pytest.fixture
def reddening():
    return np.array(
        [[0., 0.1], [1., 0.2], [2., 0.3], [3., 0.4], [4., 0.5], [5., 0.6], [6., 0.7], [7., 0.8],
         [8., 0.9], [9., 1.0], [10., 1.1]])


@pytest.fixture
def filter_1():
    return np.array(
        [[0., 1.], [1., 0.], [2., 0.], [3., 0.], [4., 0], [5., 0], [6., 0], [7., 0], [8., 0],
         [9., 0], [10., 0]])


@pytest.fixture
def filter_2():
    return np.array(
        [[0., 0.], [1., 0.], [2., 1.], [3., 2.], [4., 3], [5., 4], [6., 3], [7., 2], [8., 1],
         [9., 0], [10., 0]])


@pytest.fixture
def photometry():
    return np.array(
        [[[1., 0.1], [2., 0.2]], [[3., 0.3], [4., 0.4]], [[5., 0.5], [6., 0.6]]]) * u.uJy


@pytest.fixture
def system(filter_1, filter_2) -> PhotometricSystem:
    return PhotometricSystem(OrderedDict(f1=filter_1, f2=filter_2))


###############################################################################

def test_compute_k_x(system: PhotometricSystem, sed: np.ndarray, reddening: np.ndarray,
                     filter_1: np.ndarray):
    """
    Test the preProcess() method
    """
    ebv_0 = 1.
    dereddener = SourceIndependentGalacticEBV(system=system, reddening_curve=reddening,
                                              reference_sed=sed, ebv_0=ebv_0)
    k_x = dereddener._compute_k_x(sed, reddening, filter_1, ebv_0)
    np.testing.assert_almost_equal(k_x, 0.1)

    # ebv dependency: for 1 point no dependency
    ebv_0 = -100.
    k_x = dereddener._compute_k_x(sed, reddening, filter_1, ebv_0)
    np.testing.assert_almost_equal(k_x, 0.1)
    ebv_0 = 1.

    # full band
    filter = np.array(
        [[0., 1.], [1., 1.], [2., 1.], [3., 1.], [4., 1], [5., 1], [6., 1], [7., 1], [8., 1],
         [9., 1], [10., 1]])
    red = np.array(
        [[0., 0.2], [1., 0.3], [2., 0.4], [3., 0.5], [4., 0.6], [5., 0.7], [6., 0.8], [7., 0.9],
         [8., 1.], [9., 1.1], [10., 1.2]])
    k_x = dereddener._compute_k_x(sed, red, filter, ebv_0)
    np.testing.assert_almost_equal(k_x, 0.661124127639338)

    # ebv dependency
    ebv_0 = 0.02
    k_x = dereddener._compute_k_x(sed, red, filter, ebv_0)
    np.testing.assert_almost_equal(k_x, 0.699217123241736)
    ebv_0 = 0.1
    k_x = dereddener._compute_k_x(sed, red, filter, ebv_0)
    np.testing.assert_almost_equal(k_x, 0.69608587697567)
    ebv_0 = 1.

    # Reddening curve dependency
    red = np.array(
        [[0., 0.1], [1., 0.2], [2., 0.3], [3., 0.4], [4., 0.5], [5., 0.6], [6., 0.7], [7., 0.8],
         [8., 0.9], [9., 1.0], [10., 1.1]])
    k_x = dereddener._compute_k_x(sed, red, filter, ebv_0)
    np.testing.assert_almost_equal(k_x, 0.561124127639338)

    # filter & SED dependency
    factor = (np.random.rand(11) + 0.5)
    sed_f = np.array(sed)
    sed_f[:, 1] = sed_f[:, 1] * factor
    filter_f = np.array(filter)
    filter_f[:, 1] = filter_f[:, 1] / factor
    k_1 = dereddener._compute_k_x(sed, red, filter, ebv_0)
    k_2 = dereddener._compute_k_x(sed_f, red, filter_f, ebv_0)
    np.testing.assert_almost_equal(k_1, k_2)

    filter = np.array(
        [[0., 0.], [1., 0.], [2., 1.], [3., 2.], [4., 3], [5., 4], [6., 3], [7., 2], [8., 1],
         [9., 0], [10., 0]])
    k_x = dereddener._compute_k_x(sed, red, filter, ebv_0)
    np.testing.assert_almost_equal(k_x, 0.588500880507632)


###############################################################################

def test_compute_ks(sed: np.ndarray, reddening: np.ndarray, system: PhotometricSystem):
    ebv_0 = 1.

    dereddener = SourceIndependentGalacticEBV(system, reddening_curve=reddening,
                                              reference_sed=sed, ebv_0=ebv_0)
    ks = dereddener._compute_ks(sed, reddening, ebv_0)

    assert len(ks) == 2
    assert 'f1' in ks
    assert 'f2' in ks
    np.testing.assert_almost_equal(ks['f1'], 0.1)
    np.testing.assert_almost_equal(ks['f2'], 0.588500880507632)


###############################################################################

def test_init(sed: np.ndarray, reddening: np.ndarray, system: PhotometricSystem):
    ebv_0 = 1.
    dereddener = SourceIndependentGalacticEBV(system, reddening_curve=reddening,
                                              reference_sed=sed, ebv_0=ebv_0)
    assert len(dereddener._k_x) == 2
    assert 'f1' in dereddener._k_x
    assert 'f2' in dereddener._k_x
    np.testing.assert_almost_equal(dereddener._k_x['f1'], 0.1)
    np.testing.assert_almost_equal(dereddener._k_x['f2'], 0.588500880507632)


###############################################################################

def test_unapply_reddening(sed: np.ndarray, reddening: np.ndarray, system: PhotometricSystem):
    ebv_0 = 1.

    dereddener = SourceIndependentGalacticEBV(system, reddening_curve=reddening,
                                              reference_sed=sed, ebv_0=ebv_0)

    np.testing.assert_almost_equal(dereddener._k_x['f1'], 0.1)

    corrected = dereddener._remove_reddening(1., 'f1', 1.)
    np.testing.assert_almost_equal(corrected, 1.0964782)

    # ebv dependency
    corrected = dereddener._remove_reddening(1., 'f1', 2.)
    np.testing.assert_almost_equal(corrected, 1.2022644)

    # flux dependency
    corrected = dereddener._remove_reddening(1., 'f1', 2.)
    corrected_2 = dereddener._remove_reddening(10., 'f1', 2.)
    corrected_3 = dereddener._remove_reddening(100., 'f1', 2.)

    np.testing.assert_almost_equal(corrected_2 / corrected, 10)
    np.testing.assert_almost_equal(corrected_3 / corrected_2, 10)


###############################################################################

def test_de_redden_data(sed: np.ndarray, reddening: np.ndarray, system: PhotometricSystem,
                        photometry: u.uJy):
    ebv_0 = 0.02
    dereddener = SourceIndependentGalacticEBV(system, reddening_curve=reddening,
                                              reference_sed=sed, ebv_0=ebv_0)
    np.testing.assert_almost_equal(dereddener._k_x['f1'], 0.1)
    np.testing.assert_almost_equal(dereddener._k_x['f2'], 0.599769741601381)

    corrected = dereddener.deredden(photometry, np.array([0.05, 0.01, 0.02]))
    expected = np.array([[[1.0046158, 0.1], [2.0560108, 0.2]], [[3.0027644, 0.3], [4.0221575, 0.4]],
                         [[5.0092188, 0.5], [6.0666565, 0.6]]])
    # expected were computed with a hand calculator with only 7 digit...
    np.testing.assert_almost_equal(expected, corrected.value, 5)


###############################################################################

def test_redden_data(sed: np.ndarray, reddening: np.ndarray, system: PhotometricSystem,
                     photometry: u.uJy):
    """
    Reddening must be symmetrical with de-reddening
    """
    ebv_0 = 0.02

    dereddener = SourceIndependentGalacticEBV(system, reddening_curve=reddening,
                                              reference_sed=sed, ebv_0=ebv_0)
    de_reddened_photo = dereddener.deredden(photometry, ebv=np.array([0.05, 0.01, 0.02]))

    and_back = dereddener.redden(de_reddened_photo, ebv=np.array([0.05, 0.01, 0.02]))
    np.testing.assert_almost_equal(photometry.value, and_back.value, 5)

###############################################################################
