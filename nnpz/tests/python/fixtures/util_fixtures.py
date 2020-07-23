#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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

"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import pytest
import tempfile
import shutil
import os


@pytest.fixture()
def temp_dir_fixture(request):
    """Creates a temporary directory to be used from a test function.

    Returns: The path to the newly created directory

    Raises:
        OSError: If the directory already exists or it cannot be created

    The directory is generated under the system temp directory and is prefixed
    with "nnpz_test_". After the function test using the fixure terminates, the
    directory is automatically being removed from the drive."""


    # Create the temporary directory
    d = tempfile.mkdtemp(prefix='nnpz_test_')

    # Add to the request a finalizer which deletes the directory after the
    # tests are done
    def fin():
        if os.path.isdir(d):
            shutil.rmtree(d)
    request.addfinalizer(fin)

    return d
