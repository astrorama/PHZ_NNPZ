"""
Created on: 14/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from platform import processor

import pytest
import numpy as np

from nnpz.photometry.SourceIndependantGalacticUnReddening import SourceIndependantGalacticUnReddening


@pytest.fixture
def sed():
    return np.array([[0.,1.],[1.,1.],[2.,1.],[3.,1.],[4.,1],[5.,1],[6.,1],[7.,1],[8.,1],[9.,1],[10.,1]])


@pytest.fixture
def reddening():
    return np.array([[0.,0.1],[1.,0.2],[2.,0.3],[3.,0.4],[4.,0.5],[5.,0.6],[6.,0.7],[7.,0.8],[8.,0.9],[9.,1.0],[10.,1.1]])


@pytest.fixture
def filter_1():
    return np.array([[0.,1.],[1.,0.],[2.,0.],[3.,0.],[4.,0],[5.,0],[6.,0],[7.,0],[8.,0],[9.,0],[10.,0]])


@pytest.fixture
def filter_2():
    return np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,3],[5.,4],[6.,3],[7.,2],[8.,1],[9.,0],[10.,0]])


@pytest.fixture
def filter_map(filter_1, filter_2):
    return {'f1': filter_1, 'f2': filter_2}


@pytest.fixture
def photometry():
    return np.array([[[1., 0.1], [2., 0.2]], [[3., 0.3], [4., 0.4]], [[5., 0.5], [6., 0.6]]])

###############################################################################


def test_compute_k_x(sed, reddening, filter_1):
    """Test the preProcess() method"""
    ebv_0 = 1.
    dereddener = SourceIndependantGalacticUnReddening({}, [], galactic_reddening_curve=reddening, ref_sed=sed,
                                                      ebv_0=ebv_0)
    k_x = dereddener._compute_k_x(sed, reddening, filter_1, ebv_0)
    np.testing.assert_almost_equal(k_x, 0.1)

    # ebv dependency: for 1 point no dependency
    ebv_0 = -100.
    k_x = dereddener._compute_k_x(sed, reddening, filter_1, ebv_0)
    np.testing.assert_almost_equal(k_x, 0.1)
    ebv_0 = 1.

    # full band
    filter = np.array([[0.,1.],[1.,1.],[2.,1.],[3.,1.],[4.,1],[5.,1],[6.,1],[7.,1],[8.,1],[9.,1],[10.,1]])
    red = np.array([[0.,0.2],[1.,0.3],[2.,0.4],[3.,0.5],[4.,0.6],[5.,0.7],[6.,0.8],[7.,0.9],[8.,1.],[9.,1.1],[10.,1.2]])
    k_x = dereddener._compute_k_x(sed,red,filter,ebv_0)
    np.testing.assert_almost_equal(k_x,0.661124127639338)

    # ebv dependency
    ebv_0=0.02
    k_x = dereddener._compute_k_x(sed,red,filter,ebv_0)
    np.testing.assert_almost_equal(k_x,0.699217123241736)
    ebv_0=0.1
    k_x = dereddener._compute_k_x(sed,red,filter,ebv_0)
    np.testing.assert_almost_equal(k_x,0.69608587697567)
    ebv_0=1.

    # Reddening curve dependency
    red = np.array([[0.,0.1],[1.,0.2],[2.,0.3],[3.,0.4],[4.,0.5],[5.,0.6],[6.,0.7],[7.,0.8],[8.,0.9],[9.,1.0],[10.,1.1]])
    k_x = dereddener._compute_k_x(sed,red,filter,ebv_0)
    np.testing.assert_almost_equal(k_x,0.561124127639338)

    # filter & SED dependency
    factor = (np.random.rand(11)+0.5)
    sed_f=np.array(sed)
    sed_f[:,1]=sed_f[:,1]*factor
    filter_f=np.array(filter)
    filter_f[:,1]=filter_f[:,1]/factor
    k_1 = dereddener._compute_k_x(sed,red,filter,ebv_0)
    k_2 = dereddener._compute_k_x(sed_f,red,filter_f,ebv_0)
    np.testing.assert_almost_equal(k_1,k_2)

    filter = np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,3],[5.,4],[6.,3],[7.,2],[8.,1],[9.,0],[10.,0]])
    k_x = dereddener._compute_k_x(sed,red,filter,ebv_0)
    np.testing.assert_almost_equal(k_x,0.588500880507632)


def test_compute_ks(sed, reddening, filter_map):
    ebv_0 = 1.

    dereddener = SourceIndependantGalacticUnReddening(filter_map, [], galactic_reddening_curve=reddening, ref_sed=sed,
                                                      ebv_0=ebv_0)
    ks = dereddener._compute_ks(filter_map, sed, reddening, ebv_0)

    assert len(ks) == 2
    assert 'f1' in ks
    assert 'f2' in ks
    np.testing.assert_almost_equal(ks['f1'], 0.1)
    np.testing.assert_almost_equal(ks['f2'], 0.588500880507632)


def test_init(sed, reddening, filter_map):
    ebv_0 = 1.
    dereddener = SourceIndependantGalacticUnReddening(filter_map, [], galactic_reddening_curve=reddening, ref_sed=sed,
                                                      ebv_0=ebv_0)

    assert len(dereddener._k_x) == 2
    assert 'f1' in dereddener._k_x
    assert 'f2' in dereddener._k_x
    np.testing.assert_almost_equal(dereddener._k_x['f1'], 0.1)
    np.testing.assert_almost_equal(dereddener._k_x['f2'], 0.588500880507632)


def test_unapply_reddening(sed, reddening, filter_map):
    ebv_0 = 1.

    dereddener = SourceIndependantGalacticUnReddening(filter_map, [], galactic_reddening_curve=reddening, ref_sed=sed,
                                                      ebv_0=ebv_0)

    np.testing.assert_almost_equal(dereddener._k_x['f1'], 0.1)

    corrected = dereddener._unapply_reddening(1., 'f1', 1.)
    np.testing.assert_almost_equal(corrected, 1.0964782)

    # ebv dependency
    corrected = dereddener._unapply_reddening(1., 'f1', 2.)
    np.testing.assert_almost_equal(corrected, 1.2022644)

    # flux dependency
    corrected = dereddener._unapply_reddening(1., 'f1', 2.)
    corrected_2 = dereddener._unapply_reddening(10., 'f1', 2.)
    corrected_3 = dereddener._unapply_reddening(100., 'f1', 2.)

    np.testing.assert_almost_equal(corrected_2 / corrected, 10)
    np.testing.assert_almost_equal(corrected_3 / corrected_2, 10)


def test_de_redden_data(sed, reddening, filter_map, photometry):
    filter_order = ['f1', 'f2']
    ebv_0 = 0.02
    dereddener = SourceIndependantGalacticUnReddening(filter_map, filter_order, galactic_reddening_curve=reddening,
                                                      ref_sed=sed, ebv_0=ebv_0)
    np.testing.assert_almost_equal(dereddener._k_x['f1'], 0.1)
    np.testing.assert_almost_equal(dereddener._k_x['f2'], 0.599769741601381)

    corrected = dereddener.de_redden_data(photometry, [0.05, 0.01, 0.02])
    expected = np.array([[[1.0046158,0.1],[2.0560108,0.2]],[[3.0027644,0.3],[4.0221575,0.4]],[[5.0092188,0.5],[6.0666565,0.6]]])
    # expected were computed with a hand calculator with only 7 digit...
    np.testing.assert_almost_equal(expected, corrected, 5)

    # filter order
    filter_order = ['f2', 'f1']
    dereddener = SourceIndependantGalacticUnReddening(filter_map, filter_order, galactic_reddening_curve=reddening,
                                                      ref_sed=sed, ebv_0=ebv_0)
    np.testing.assert_almost_equal(dereddener._k_x['f1'], 0.1)
    np.testing.assert_almost_equal(dereddener._k_x['f2'], 0.599769741601381)
    corrected = dereddener.de_redden_data(photometry, [0.05, 0.01, 0.02])
    expected = np.array([[[1.0280054,0.1],[2.0092316,0.2]],[[3.0166181,0.3],[4.0036858,0.4]],[[5.0555471,0.5],[6.0110626,0.6]]])
    # expected were computed with a hand calculator with only 7 digit...
    np.testing.assert_almost_equal(expected, corrected, 5)


def test_redden_data(sed, reddening, filter_map, photometry):
    """
    Reddening must be symmetrical with de-reddening
    """
    filter_order = ['f1', 'f2']
    ebv_0 = 0.02

    dereddener = SourceIndependantGalacticUnReddening(filter_map, filter_order, galactic_reddening_curve=reddening,
                                                      ref_sed=sed,ebv_0=ebv_0)
    de_reddened_photo = dereddener.de_redden_data(photometry, [0.05, 0.01, 0.02])

    and_back = dereddener.redden_data(de_reddened_photo, [0.05, 0.01, 0.02])
    np.testing.assert_almost_equal(photometry, and_back, 5)

    filter_order = ['f2', 'f1']
    dereddener = SourceIndependantGalacticUnReddening(filter_map, filter_order, galactic_reddening_curve=reddening,
                                                      ref_sed=sed, ebv_0=ebv_0)
    de_reddened_photo = dereddener.de_redden_data(photometry, [0.05, 0.01, 0.02])

    and_back = dereddener.redden_data(de_reddened_photo, [0.05, 0.01, 0.02])
    np.testing.assert_almost_equal(photometry, and_back, 5)

################################################
