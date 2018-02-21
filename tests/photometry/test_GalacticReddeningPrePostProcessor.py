"""
Created on: 21/02/2018
Author: Florian Dubath
"""

from __future__ import division, print_function

from platform import processor
import types
import pytest
from mock import Mock
import numpy as np

from nnpz.photometry import PhotometryPrePostProcessorInterface, GalacticReddeningPrePostProcessor


def test_computeBpc():
    """Test the computeBpc() method"""
     # Given
    pre_post_processor = Mock(spec_set=PhotometryPrePostProcessorInterface)
    pre_post_processor.preProcess.side_effect = lambda x: x
    pre_post_processor.postProcess.side_effect = lambda x,y,z: x

    arr = [[i+4498,1] for i in range(504)]
    arr[0][1]=0
    arr[1][1]=0
    arr[-1][1]=0
    arr[-2][1]=0
    b_filter = np.array(arr )

    arr = [[i+5998,1] for i in range(1504)]
    arr[0][1]=0
    arr[1][1]=0
    arr[-1][1]=0
    arr[-2][1]=0
    r_filter = np.array(arr)
    galactic_reddening_curve = np.array([[1000,0],[5500,0],[5600,17.47425],[9000,17.47425]])

    sed =  np.array([[i+1000,1] for i in range(9001)])
    # we have b =500
    #         b_r=500
    #         r=1500
    #         r_r=300
    # so -0.04*log(b_r*r/(b*r_r)) = -0.0279588
    expected = -0.0279588

    # When
    processor=GalacticReddeningPrePostProcessor.GalacticReddeningPrePostProcessor( pre_post_processor, b_filter, r_filter, galactic_reddening_curve, 1.0)
    bpc = processor.computeBpc(sed)

    # Then
    assert bpc == pytest.approx(expected)


###############################################################################

def test_preProcess():
    """Test the preProcess() method"""

    pre_post_processor = Mock(spec_set=PhotometryPrePostProcessorInterface)
    pre_post_processor.preProcess.side_effect = lambda x: x
    pre_post_processor.postProcess.side_effect = lambda x,y,z: x

    b_filter = np.array([[5000,1],[5500,1]] )
    r_filter = np.array([[6000,1],[7500,1]] )

    galactic_reddening_curve = np.array([[1000,0],[5599,0],[5600,1],[9200,1]])

    sed =  np.array([[i+1000,1.] for i in range(8001)])


    def mock_computeBpc(self,sed):
        return 1.0
    # When
    processor=GalacticReddeningPrePostProcessor.GalacticReddeningPrePostProcessor( pre_post_processor, b_filter, r_filter, galactic_reddening_curve, 2.5/1.018)
    processor.computeBpc = types.MethodType(mock_computeBpc, processor)

    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result[:,0], sed[:,0])

    assert np.array_equal(result[:4600,1], sed[:4600,1])
    assert np.array_equal(result[4600:,1], sed[4600:,1]/10.)


    # Check the bpc dependency
    def mock_computeBpc_2(self,sed):
        return 2.0

    processor.computeBpc = types.MethodType(mock_computeBpc_2, processor)
    # Changing bpc from 1 to 2 should devide the exponant by 2.
    # It was tunned to be -1 so we expect a factor 1/sqrt(10)
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result[:,0], sed[:,0])

    assert np.array_equal(result[:4600,1], sed[:4600,1])
    assert np.array_equal(result[4600:,1], sed[4600:,1]/np.sqrt(10.))


    # Check the provided E(B-V) dependency
    processor=GalacticReddeningPrePostProcessor.GalacticReddeningPrePostProcessor( pre_post_processor, b_filter, r_filter, galactic_reddening_curve, 5./1.018)
    processor.computeBpc = types.MethodType(mock_computeBpc, processor)

    # Changing E(B-V) by a factor 2 should multiply the exponant by 2.
    # It was tunned to be -1 so we expect a factor 1/100
    result = processor.preProcess(sed)

    # Then
    assert np.array_equal(result[:,0], sed[:,0])

    assert np.array_equal(result[:4600,1], sed[:4600,1])
    assert np.array_equal(result[4600:,1], sed[4600:,1]/100.)




###############################################################################
