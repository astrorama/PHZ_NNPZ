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