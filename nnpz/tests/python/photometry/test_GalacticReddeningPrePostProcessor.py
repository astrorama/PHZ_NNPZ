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

"""
Created on: 21/02/2018
Author: Florian Dubath
"""

try:
    from mock import Mock
except ModuleNotFoundError:
    from unittest.mock import Mock

import numpy as np
from nnpz.photometry.calculator import PhotometryPrePostProcessorInterface
from nnpz.photometry.calculator.ebv_processor import EBVPrePostProcessor


###############################################################################

def test_computeAbsorption():
    pre_post_processor = Mock(spec_set=PhotometryPrePostProcessorInterface)
    pre_post_processor.preProcess.side_effect = lambda x: x
    pre_post_processor.postProcess.side_effect = lambda x, y, z: x
    ebv = 1.
    red = np.array(
        [[0., 0.1], [1., 0.2], [2., 0.3], [3., 0.4], [4., 0.5], [5., 0.6], [6., 0.7], [7., 0.8],
         [8., 0.9], [9., 1.0], [10., 1.1]])
    processor = EBVPrePostProcessor(pre_post_processor, ebv, red)

    sed = np.array(
        [[0., 1.], [1., 1.], [2., 1.], [3., 1.], [4., 3], [5., 1], [6., 1], [7., 1], [8., 1],
         [9., 1], [10., 1]])

    actual = processor._computeAbsorption(sed)

    expected = np.array([[0, 0.91201083935591], [1, 0.831763771102671], [2, 0.758577575029184],
                         [3, 0.691830970918937], [4, 1.89287203344058], [5, 0.575439937337157],
                         [6, 0.524807460249773], [7, 0.478630092322638], [8, 0.436515832240166],
                         [9, 0.398107170553497], [10, 0.363078054770101]])

    np.testing.assert_almost_equal(actual, expected)

    # ebv dep
    ebv = 0.1
    processor = EBVPrePostProcessor(pre_post_processor, ebv, red)
    actual = processor._computeAbsorption(sed)
    expected = np.array([[0, 0.990831944892768], [1, 0.981747943019984], [2, 0.972747223776965],
                         [3, 0.963829023623971], [4, 2.86497775806431], [5, 0.946237161365793],
                         [6, 0.93756200692588], [7, 0.928966386779936], [8, 0.920449571753171],
                         [9, 0.91201083935591], [10, 0.903649473722301]])
    np.testing.assert_almost_equal(actual, expected)


###############################################################################

def test_preProcess():
    pre_post_processor = Mock(spec_set=PhotometryPrePostProcessorInterface)
    pre_post_processor.preProcess.side_effect = lambda x: x
    pre_post_processor.postProcess.side_effect = lambda x, y, z: x
    ebv = 1.
    red = np.array(
        [[0., 0.1], [1., 0.2], [2., 0.3], [3., 0.4], [4., 0.5], [5., 0.6], [6., 0.7], [7., 0.8],
         [8., 0.9], [9., 1.0], [10., 1.1]])
    processor = EBVPrePostProcessor(pre_post_processor, ebv, red)

    sed = np.array(
        [[0., 1.], [1., 1.], [2., 1.], [3., 1.], [4., 3], [5., 1], [6., 1], [7., 1], [8., 1],
         [9., 1], [10., 1]])
    actual = processor.preProcess(sed)
    expected = np.array([[0, 0.91201083935591], [1, 0.831763771102671], [2, 0.758577575029184],
                         [3, 0.691830970918937], [4, 1.89287203344058], [5, 0.575439937337157],
                         [6, 0.524807460249773], [7, 0.478630092322638], [8, 0.436515832240166],
                         [9, 0.398107170553497], [10, 0.363078054770101]])

    # check that preprocess actually redden the sed
    np.testing.assert_almost_equal(actual, expected)


###############################################################################

def test_postProcess():
    pre_post_processor = Mock(spec_set=PhotometryPrePostProcessorInterface)
    pre_post_processor.preProcess.side_effect = lambda x: x
    pre_post_processor.postProcess.side_effect = lambda x, y: x
    ebv = 1.
    red = np.array(
        [[0., 0.1], [1., 0.2], [2., 0.3], [3., 0.4], [4., 0.5], [5., 0.6], [6., 0.7], [7., 0.8],
         [8., 0.9], [9., 1.0], [10., 1.1]])
    processor = EBVPrePostProcessor(pre_post_processor, ebv, red)

    values = [1, 2, 3, 4, 5, 6, 7, 8]
    actual = [processor.postProcess(v, 'f1') for v in values]
    # check that the postprocessing has no effect
    np.testing.assert_almost_equal(actual, values)

###############################################################################
