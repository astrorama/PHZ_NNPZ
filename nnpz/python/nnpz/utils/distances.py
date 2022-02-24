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


def euclidean(ref: np.ndarray, target: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    """
    Euclidean distance. Ignores errors.
    """
    x = ref[:, :, 0] - target[:, 0]
    return np.sqrt(np.sum(x * x, axis=1, out=out), out=out)


def chi2(ref: np.ndarray, target: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    """
    χ2 distance. Takes errors into account.
    Warnings:
        If *both* reference and target errors are 0, the results will be +Inf (or indeterminate
        if the Euclidean distance is 0).
        Use Euclidean instead if that is the case.
    """
    nom = ref[:, :, 0] - target[:, 0]
    den = ref[:, :, 1] * ref[:, :, 1] + target[:, 1] * target[:, 1]
    return np.sum(nom * nom / den, axis=1, out=out)
