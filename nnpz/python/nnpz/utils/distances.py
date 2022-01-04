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
import numpy as np

from nnpz.photometry.photometry import Photometry


def euclidean(ref: Photometry, target: Photometry, out: np.ndarray) -> np.ndarray:
    assert ref.system == target.system
    x = ref.values[:, :, 0] - target.values[:, 0]
    return np.sqrt(np.sum(x * x, axis=1, out=out), out=out)


def chi2(ref: Photometry, target: Photometry, out: np.ndarray) -> np.ndarray:
    assert ref.system == target.system
    nom = ref.values[:, :, 0] - target.values[:, 0]
    den = ref.values[:, :, 1] * ref.values[:, :, 1] + target.values[:, 1] * target.values[:, 1]
    return np.sum(nom * nom / den, axis=1, out=out)
