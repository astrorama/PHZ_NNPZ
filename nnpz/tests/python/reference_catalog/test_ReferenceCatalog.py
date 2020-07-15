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
