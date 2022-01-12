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
def correct_ebv(ref_photo: u.uJy, corr_coef: np.ndarray, ebv: float, out: u.uJy = None):
    if out is None:
        out = np.copy(ref_photo)
    corr = (10 ** (-0.4 * corr_coef * ebv[:, np.newaxis]))[:, :, np.newaxis]
    out *= corr
    return out
