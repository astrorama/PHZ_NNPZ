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

import os
from glob import glob

from nnpz.reference_sample import IndexProvider


class BaseProvider:
    """
    Base for different set of providers. A provider is a set of data file providers.

    Args:
        index:
            An index provider
        key:
            The key to use on the index
        data_pattern:
            Full path to the data files, as a pattern string (i.e. '.../blah_data_{}.npy')
            The brackets will be replaced with the file identification.
        data_limit:
            Partition data on files of maximum this size, in bytes.
        extra_data:
            Additional metadata for the provider
        overwrite:
            If true, remove the existing datafiles and clean the index if already set
    """

    def __init__(self, index: IndexProvider, key: str, data_pattern: str, data_limit: int,
                 extra_data: dict, overwrite: bool = False):
        self._index = index
        self._key = key
        self._data_pattern = data_pattern
        self._data_limit = data_limit
        self.extra = extra_data

        if overwrite:
            self._clean_existing()

    def _clean_existing(self):
        for path in glob(self._data_pattern.format('*')):
            os.unlink(path)
        self._index.clear(self._key)

    @property
    def data_pattern(self):
        return self._data_pattern
