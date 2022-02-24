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
from typing import Set

from nnpz.reference_sample import IndexProvider


def locate_existing_data_files(pattern: str) -> Set[str]:
    """
    Returns a set with the indices of the existing data files following the pattern
    """
    result = set()
    i = 1
    while os.path.exists(pattern.format(i)):
        result.add(i)
        i += 1
    return result


def validate_data_files(pattern: str, index: IndexProvider, key: str) -> Set[str]:
    """
    Cross-check the existing data files and those referenced by the index

    Returns:
        A set of existing file ids
    """
    existing_files = locate_existing_data_files(pattern)
    index_files = index.get_files(key)
    if not existing_files.issuperset(index_files):
        missing_files = index_files.difference(existing_files)
        missing_files = list(map(pattern.format, missing_files))
        raise FileNotFoundError(
            'Missing {} data files: {}'.format(key, ', '.join(missing_files))
        )
    return existing_files
