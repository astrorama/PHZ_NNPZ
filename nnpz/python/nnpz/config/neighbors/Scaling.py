#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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
import os
from typing import Any, Callable, Dict, Union

from astropy.table import Table
from nnpz.config import ConfigManager
from nnpz.scaling import Chi2Scaling
from scipy.interpolate import interpolate


class Scaling(ConfigManager.ConfigHandler):
    def __init__(self):
        self.__initialized = False
        self.__scaler = None

    @staticmethod
    def get_scale_prior(prior: Union[str, Callable[[float], float]]) -> Callable[[float], float]:
        if hasattr(prior, '__call__'):
            return prior
        if prior == 'uniform':
            return lambda a: 1
        if os.path.exists(prior):
            table = Table.read(prior, format='ascii')
            return interpolate.interp1d(table.columns[0], table.columns[1], kind='linear')
        raise Exception('Unknown prior')

    def __setup_scale_prior(self, args: Dict[str, Any]):
        scale_prior = args.get('scale_prior', None)
        batch_size = args.get('batch_size')
        if scale_prior:
            self.__scaler = Chi2Scaling(prior=self.get_scale_prior(scale_prior),
                                        batch_size=batch_size,
                                        max_iter=args.get('scale_max_iter', 20),
                                        rtol=args.get('scale_rtol', 1e-4))

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.__initialized:
            self.__setup_scale_prior(args)
        return {
            'scaler': self.__scaler
        }


ConfigManager.add_handler(Scaling)
