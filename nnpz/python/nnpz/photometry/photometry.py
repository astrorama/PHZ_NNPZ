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
from typing import List, Union

import numpy as np
from astropy import units as u

from nnpz.exceptions import InvalidDimensionsException
from nnpz.photometry.colorspace import ColorSpace
from nnpz.photometry.photometric_system import PhotometricSystem


class Photometry:
    def __init__(self, ids: np.ndarray, values: u.Quantity,
                 system: PhotometricSystem, colorspace: ColorSpace, copy=False):
        if not values.unit.is_equivalent(u.uJy):
            raise ValueError('Photometry values must be in flux units (uJy or equivalent)')

        if len(values.shape) > 3:
            raise InvalidDimensionsException()
        elif values.shape[-1] != 2:
            raise InvalidDimensionsException('flux axis must have a size of two')
        elif values.shape[-2] != len(system.bands):
            raise InvalidDimensionsException(
                'unexpected photometry shape: expected {}, got {}'.format(len(system.bands),
                                                                          values.shape[1])
            )

        if not np.all(np.isfinite(values)):
            raise ValueError('NaN or Inf values found on the photometry')

        self.ids = np.array(ids, copy=copy)
        self.values = u.Quantity(values, copy=copy)
        self.system = system
        self.colorspace = colorspace
        # Make sure we don't corrupt the data by mistake
        self.ids.flags.writeable = False
        self.values.flags.writeable = False

    def subsystem(self, bands: List[str]):
        cols = [self.system.bands.index(b) for b in bands]
        return Photometry(ids=self.ids, values=self.values[:, cols],
                          system=self.system[bands], colorspace=self.colorspace)

    def get_fluxes(self, bands: Union[List[str], str], return_error=False) -> np.ndarray:
        if isinstance(bands, str):
            c = self.system.bands.index(bands)
        else:
            c = [self.system.bands.index(b) for b in bands]
        if return_error:
            return self.values[:, c]
        else:
            return self.values[:, c, 0]

    @property
    def unit(self):
        return self.values.unit

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, rows) -> 'Photometry':
        if not isinstance(rows, (slice, list, int, np.ndarray)):
            raise IndexError('Photometry can only be directly accessed by row')
        return Photometry(ids=self.ids[rows], values=self.values[rows],
                          system=self.system, colorspace=self.colorspace[rows])

    def __str__(self) -> str:
        return """Photometry with {} objects, in {}
    Photometric system: {}
""".format(len(self), self.unit, str(self.system))
