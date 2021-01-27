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

import numpy as np
import pytest
from astropy.table import Table
from nnpz.exceptions import IdMismatchException, CorruptedFileException
from nnpz.reference_catalog.ReferenceCatalog import ReferenceCatalog


@pytest.fixture()
def ref_cat_fixture():
    return Table({
        'ID': [1, 2, 3, 3],
        'PDZ': [[0.29, 0.57, 0.29, 0.], [0., 0.57, 0.29, 0.29], [0., 0.29, 0.29, 0.57], [0., 0., 0., 0.]],
    }), Table({
        'BINS': [0., 1., 2., 3.]
    })


###############################################################################

def testReferenceCatalog(ref_cat_fixture):
    """
    Test a regular reference catalog
    """
    ref_cat = ReferenceCatalog(ref_cat_fixture[0]['ID'], ref_cat_fixture[0]['PDZ'], ref_cat_fixture[1]['BINS'])
    assert (len(ref_cat.getIds()) == 4)
    pdz = ref_cat.getPdzData(1)
    assert (np.allclose(pdz[:, 0], ref_cat_fixture[1]['BINS']))
    assert (np.allclose(pdz[:, 1], ref_cat_fixture[0]['PDZ'][0]))


###############################################################################

def testBadId(ref_cat_fixture):
    """
    Retrieve an ID that does not exist
    """
    ref_cat = ReferenceCatalog(ref_cat_fixture[0]['ID'], ref_cat_fixture[0]['PDZ'], ref_cat_fixture[1]['BINS'])
    with pytest.raises(IdMismatchException):
        ref_cat.getPdzData(55)


###############################################################################

def testCorruptedCatalog(ref_cat_fixture):
    """
    There are two entries with the same ID
    """
    ref_cat = ReferenceCatalog(ref_cat_fixture[0]['ID'], ref_cat_fixture[0]['PDZ'], ref_cat_fixture[1]['BINS'])
    with pytest.raises(CorruptedFileException):
        ref_cat.getPdzData(3)
