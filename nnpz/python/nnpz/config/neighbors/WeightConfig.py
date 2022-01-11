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
from nnpz.config import ConfigManager
from Weights import WeightCalculator


class WeightConfig(ConfigManager.ConfigHandler):
    def __init__(self):
        self.__calculator = None

    def __createCalculator(self, args):
        self._checkParameterExists('weight_method', args)
        method = args['weight_method']
        alternative = args.get('weight_method_alternative', 'Euclidean')
        self.__calculator = WeightCalculator(method, alternative)

    def parseArgs(self, args):
        if self.__calculator is None:
            self.__createCalculator(args)
        return {
            'weight_calculator': self.__calculator,
        }


ConfigManager.addHandler(WeightConfig)
