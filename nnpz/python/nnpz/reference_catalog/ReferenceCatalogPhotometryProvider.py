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
from nnpz.io import CatalogReader, catalog_properties as prop


class CatalogPhotometryProvider(object):
    """
    As PhotometryProvider, but wrapping a regular catalog instead
    """

    def __init__(self, reader: CatalogReader):
        self.__reader = reader

    def getData(self, *filter_list):
        return self.__reader.get(prop.Photometry(filter_list))
