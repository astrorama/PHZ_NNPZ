"""
Created on: 21/02/2018
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import os
import sys


def getAuxiliaryPath(filename):

    try:
        from ElementsKernel import Auxiliary
        return Auxiliary.getAuxiliaryPath(filename, False)
    except ImportError:
        # GitHub version handling

        path = os.path.abspath(os.path.dirname(__file__)) # Remove the filename
        path = os.path.dirname(path) # Remove the utis package
        path = os.path.dirname(path) # Remove the nnpz package

        # First check if the user is running using the full source code of the
        # NNPZ, in which case the data are in the auxdir
        auxdir_path = os.path.join(path, 'auxdir', filename)
        if os.path.exists(auxdir_path):
            return auxdir_path

        # check if the file has been installed in a directory by unziping the egg
        etc_path = os.path.join(path, 'etc', 'nnpz', filename)
        if os.path.exists(etc_path):
            return etc_path

        # If we reached here the file should be in the etc directory
        return os.path.join(sys.prefix, 'etc', 'nnpz', filename)