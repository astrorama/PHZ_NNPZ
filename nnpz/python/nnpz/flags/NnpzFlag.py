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

NO_FLAG = 0b00

# These flags are to be used by PHZ_Pipeline
ALTERNATIVE_WEIGHT_FLAG = 0b00001
MAG_CUTOUT = 0b00010
SNR_CUTOUT = 0b00100
MISSING_BANDS = 0b01000

assert ALTERNATIVE_WEIGHT_FLAG | MAG_CUTOUT | SNR_CUTOUT | MISSING_BANDS == ALTERNATIVE_WEIGHT_FLAG ^ MAG_CUTOUT ^ SNR_CUTOUT ^ MISSING_BANDS

FLAGS = [ALTERNATIVE_WEIGHT_FLAG]
FLAG_NAMES = ['AlternativeWeightFlag']
