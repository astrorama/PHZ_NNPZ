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

#
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

from nnpz.pipeline.correct_photometry import CorrectPhotometry

from ..photometry.fixtures import *


###############################################################################

def test_correctedPhotometry(reference_provider_fixture: PhotometryProvider,
                             reference_photometry: Photometry, target_photometry: Photometry):
    # Given
    ebv_corr = reference_provider_fixture.get_ebv_correction_factors()
    shift_corr = reference_provider_fixture.get_shift_correction_factors()

    # When
    corrected = CorrectPhotometry(dict(reference_system=reference_photometry.system,
                                       reference_ebv_correction=ebv_corr,
                                       reference_filter_variation_correction=shift_corr))
    nn_ids = np.tile(np.arange(0, 5), (5, 1))
    nn_photo = np.tile(reference_photometry.values[np.newaxis], (len(target_photometry), 1, 1, 1))

    # Then
    Y_idx, g_idx, vis_idx = target_photometry.system.get_band_indexes(['Y', 'g', 'vis'])

    # The first one is not shifted at all
    phot = corrected(target_photometry, nn_ids, nn_photo)
    np.testing.assert_array_equal(reference_photometry.values, phot[0])

    ###########################################################
    # The second one shifts VIS +1500, g and Y remain untouched
    ###########################################################

    np.testing.assert_allclose(phot[1, :, Y_idx, 0], reference_photometry.values[:, Y_idx, 0])
    np.testing.assert_allclose(phot[1, :, g_idx, 0], reference_photometry.values[:, g_idx, 0])

    # For the first neighbor, there is no correction
    np.testing.assert_allclose(phot[1, 0, vis_idx, 0].value,
                               reference_photometry.values[0, vis_idx, 0].value)
    # For the second neighbor, corr = 1 * 1500**2 + 1
    np.testing.assert_allclose(phot[1, 1, vis_idx, 0].value,
                               (1 * 1500 ** 2 + 1) * reference_photometry.values[
                                   1, vis_idx, 0].value)
    # For the third neighbor, corr = 2 * 1500**2 + 1
    np.testing.assert_allclose(phot[1, 2, vis_idx, 0].value,
                               (2 * 1500 ** 2 + 1) * reference_photometry.values[
                                   2, vis_idx, 0].value)
    # For the fourth neighbor, corr = 3 * 1500**2 + 1
    np.testing.assert_allclose(phot[1, 3, vis_idx, 0].value,
                               (3 * 1500 ** 2 + 1) * reference_photometry.values[
                                   3, vis_idx, 0].value)

    ###########################################################
    # The third one shift VIS +1500 and g +1000
    ###########################################################

    np.testing.assert_allclose(phot[2, :, Y_idx, 0], reference_photometry.values[:, Y_idx, 0])

    # For the first neighbor, there is no correction
    np.testing.assert_allclose(phot[2, 0, g_idx, 0].value,
                               reference_photometry.values[0, g_idx, 0].value)
    # For the second neighbor, corr = 1 * 1000 + 1
    np.testing.assert_allclose(phot[2, 1, g_idx, 0].value,
                               (1 * 1000 + 1) * reference_photometry.values[1, g_idx, 0].value)
    # For the third neighbor, corr = 2 * 1000 + 1
    np.testing.assert_allclose(phot[2, 2, g_idx, 0].value,
                               (2 * 1000 + 1) * reference_photometry.values[2, g_idx, 0].value)
    # For the fourth neighbor, corr = 3 * 1000**2 + 1
    np.testing.assert_allclose(phot[2, 3, g_idx, 0].value,
                               (3 * 1000 + 1) * reference_photometry.values[3, g_idx, 0].value)

    ###########################################################
    # The fourth one shift VIS +1500, g +1000 and Y -999 (but Y is constant!)
    ###########################################################
    np.testing.assert_allclose(phot[4, :, Y_idx, 0], reference_photometry.values[:, Y_idx, 0])
