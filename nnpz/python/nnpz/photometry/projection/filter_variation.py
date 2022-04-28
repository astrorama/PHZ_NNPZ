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
import astropy.units as u
import numpy as np


@u.quantity_input
def correct_filter_variation(ref_photo: u.uJy, corr_coef: np.ndarray, shift: np.ndarray,
                             out: u.uJy):
    """
    Correct for filter variation. i.e., project from the reference colorspace (nominal
    average filter transmission) to the target colorspace (actual average filter transmission,
    which depends on the location on the FOV).
    Args:
        ref_photo: u.uJy
            Reference objects photometry for a given band.
        corr_coef: np.ndarray
            Correction coefficients computed when building the reference photometry.
        shift: np.ndarray
            Filter shift for the band (delta between the nominal average transmission and the
            actual one).
        out: u.uJy
            If specified, store the computation here.

    Returns:
        out if it was None, otherwise a newly allocated array.
    """
    if out is None:
        out = np.copy(ref_photo)
    shift = shift[:, np.newaxis]
    shift_corr = corr_coef[:, :, 0] * shift * shift + corr_coef[:, :, 1] * shift + 1
    out *= shift_corr[:, :, np.newaxis]
    return out
