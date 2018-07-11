"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import pytest
import os
from astropy.table import Table
import numpy as np

from nnpz import ReferenceSample
from nnpz.exceptions import *

from ..util_fixtures import temp_dir_fixture
from .fixtures import *

def pdzEqual(a, b):
    """Compare two pdz, taking into account their normalization"""
    norm_a = a[:, 1] / np.trapz(a[:, 1], a[:, 0])
    norm_b = b[:, 1] / np.trapz(b[:, 1], b[:, 0])
    return np.allclose(norm_a, norm_b) and np.all(a[:, 0] == b[:, 0])

###############################################################################

def test_createNew_dirExists(temp_dir_fixture):
    """Tests that if the directory exists an exception is raised"""

    # Given
    dir_name = os.path.join(temp_dir_fixture, 'ref_sample')
    os.makedirs(dir_name)

    # Then
    with pytest.raises(OSError):
        ReferenceSample.createNew(dir_name)

###############################################################################

def test_createNew_success(temp_dir_fixture):
    """Tests that all the files of the reference sample are constructed correctly"""

    # Given
    dir_name = os.path.join(temp_dir_fixture, 'ref_sample')

    # When
    result = ReferenceSample.createNew(dir_name)

    # Then

    # Check the result object is correct
    assert not result is None

    # Check that the result contains nothing
    assert result.size() == 0

    # Check the directory was creted correctly
    assert os.path.isdir(dir_name)

    # Check that the empty index file is created
    index_file = os.path.join(dir_name, 'index.bin')
    assert os.path.exists(index_file)
    assert os.path.getsize(index_file) == 0

    # Check that the empty SED file is created
    sed_file = os.path.join(dir_name, 'sed_data_1.bin')
    assert os.path.exists(sed_file)
    assert os.path.getsize(sed_file) == 0

    # Check that the empty PDZ file is created
    pdz_file = os.path.join(dir_name, 'pdz_data_1.bin')
    assert os.path.exists(pdz_file)
    assert os.path.getsize(pdz_file) == 0

###############################################################################

def test_constructor_missingDir(temp_dir_fixture):
    """Test the constructor when the whole directory is missing"""

    # Given
    dir_name = os.path.join(temp_dir_fixture, 'missing')

    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(dir_name)

###############################################################################

def test_constructor_missingIndexFile(reference_sample_dir_fixture):
    """Test the constructor when the index file is missing"""

    # Given
    os.remove(os.path.join(reference_sample_dir_fixture, 'index.bin'))

    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(reference_sample_dir_fixture)

###############################################################################

def test_constructor_missingSedDataFile(reference_sample_dir_fixture):
    """Test the constructor when the SED data file is missing"""

    # Given
    os.remove(os.path.join(reference_sample_dir_fixture, 'sed_data_1.bin'))

    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(reference_sample_dir_fixture)

###############################################################################

def test_constructor_missingPdzDataFile(reference_sample_dir_fixture):
    """Test the constructor when the PDZ data file is missing"""

    # Given
    os.remove(os.path.join(reference_sample_dir_fixture, 'pdz_data_1.bin'))

    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(reference_sample_dir_fixture)

###############################################################################

def test_size(reference_sample_dir_fixture, sed_list_fixture):
    """Test the size() method works correctly"""

    # Given
    expected = 0
    for key in sed_list_fixture:
        expected += len(sed_list_fixture[key])
    expected += 2

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    size = sample.size()

    # Then
    assert size == expected

###############################################################################

def test_getIds(reference_sample_dir_fixture, sed_list_fixture):
    """Test the getIds() method works correctly"""

    # Given
    expected = []
    for key in sed_list_fixture:
        expected = expected + [i for i,_ in sed_list_fixture[key]]
    expected = expected + [100, 101]

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    ids = sample.getIds()

    # Then
    assert np.all(ids == expected)

###############################################################################

def test_getSedData_wrongId(reference_sample_dir_fixture):
    """Test the case where the ID is not in the index"""

    # Given
    wrong_id = 200

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(IdMismatchException):
        sample.getSedData(wrong_id)

###############################################################################

def test_getSedData_corruptedFile(reference_sample_dir_fixture):
    """Test the case where the ID in the index and in the SED data file differ"""

    # Given
    with open(os.path.join(reference_sample_dir_fixture, 'index.bin'), 'rb+') as f:
        f.seek(0)
        correct_id = np.fromfile(f, count=1, dtype=np.int64)[0]
        wrong_id = correct_id + 1
        f.seek(0)
        np.asarray([wrong_id], dtype=np.int64).tofile(f)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(CorruptedFileException):
        sample.getSedData(wrong_id)

###############################################################################

def test_getSedData_dataUnset(reference_sample_dir_fixture):
    """Test the case where the SED data are not set yet"""

    # Given
    unset_sed_ids = [100, 101]

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    sed_data = [sample.getSedData(i) for i in unset_sed_ids]

    # Then
    assert sed_data[0] is None
    assert sed_data[1] is None

###############################################################################

def test_getSedData_withData(reference_sample_dir_fixture, sed_list_fixture):
    """Test the case where the SED data exist"""

    # Given
    id_list = []
    expected_data = []
    for key in sed_list_fixture:
        id_list += [i for i,_ in sed_list_fixture[key]]
        expected_data += [d for _,d in sed_list_fixture[key]]

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    sed_data = [sample.getSedData(i) for i in id_list]

    # Then
    for i in range(len(id_list)):
        assert sed_data[i].shape == expected_data[i].shape
        assert np.all(sed_data[i] == expected_data[i])

###############################################################################

def test_getPdzData_wrongId(reference_sample_dir_fixture):
    """Test the case where the ID is not in the index"""

    # Given
    wrong_id = 200

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(IdMismatchException):
        sample.getPdzData(wrong_id)

###############################################################################

def test_getPdzData_corruptedFile(reference_sample_dir_fixture):
    """Test the case where the ID in the index and in the PDZ data file differ"""

    # Given
    with open(os.path.join(reference_sample_dir_fixture, 'index.bin'), 'rb+') as f:
        f.seek(0)
        correct_id = np.fromfile(f, count=1, dtype=np.int64)[0]
        wrong_id = correct_id + 1
        f.seek(0)
        np.asarray([wrong_id], dtype=np.int64).tofile(f)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(CorruptedFileException):
        sample.getPdzData(wrong_id)

##############################################################################

def test_getPdzData_dataUnset(reference_sample_dir_fixture):
    """Test the case where the PDZ data are not set yet"""

    # Given
    unset_pdz_ids = [100, 101]

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    pdz_data = [sample.getPdzData(i) for i in unset_pdz_ids]

    # Then
    assert pdz_data[0] is None
    assert pdz_data[1] is None

###############################################################################

def test_getSedData_withData(reference_sample_dir_fixture, pdz_list_fixture, redshift_bins_fixture):
    """Test the case where the PDZ data exist"""

    # Given
    id_list = []
    expected_data = []
    for key in pdz_list_fixture:
        id_list += [i for i,_ in pdz_list_fixture[key]]
        expected_data += [d for _,d in pdz_list_fixture[key]]

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    pdz_data = [sample.getPdzData(i) for i in id_list]

    # Then
    for i in range(len(id_list)):
        assert pdz_data[i].shape == (len(expected_data[i]), 2)
        assert np.all(pdz_data[i][:,0] == redshift_bins_fixture)
        assert np.all(pdz_data[i][:,1] == expected_data[i])

###############################################################################

def test_createObject_duplicateId(reference_sample_dir_fixture):
    """Tests the createObject with an ID already in the iindex"""

    # Given
    existing_id = 100

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(DuplicateIdException):
        sample.createObject(existing_id)

###############################################################################

def test_createObject_success(reference_sample_dir_fixture):
    """Tests the createObject() successful case"""

    # Given
    new_id = 200

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    old_size = sample.size()
    assert not new_id in sample.getIds()
    sample.createObject(new_id)

    # Then
    # First check that the sample object beaves as expected
    assert sample.size() == old_size + 1
    assert new_id in sample.getIds()
    assert sample.getSedData(new_id) is None
    assert sample.getPdzData(new_id) is None
    # Then check it is appended at the end of the index file
    indexfile = os.path.join(reference_sample_dir_fixture, 'index.bin')
    filesize = os.path.getsize(indexfile)
    with open(indexfile, 'rb') as f:
        f.seek(filesize - 28)
        assert np.fromfile(f, count=1, dtype=np.int64) == new_id
        assert np.fromfile(f, count=1, dtype=np.int16) == 0
        assert np.fromfile(f, count=1, dtype=np.int64) == -1
        assert np.fromfile(f, count=1, dtype=np.int16) == 0
        assert np.fromfile(f, count=1, dtype=np.int64) == -1

###############################################################################

def test_addSedData_wrongId(reference_sample_dir_fixture):
    """Test the case where the ID is not in the index"""

    # Given
    wrong_id = 200
    sed_data = np.asarray([(1,2),(3,4),(5,6)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(IdMismatchException):
        sample.addSedData(wrong_id, sed_data)

###############################################################################

def test_addSedData_alreadySet(reference_sample_dir_fixture, sed_list_fixture):
    """Test the case where the data are already set"""

    # Given
    obj_id = sed_list_fixture[1][-1][0]
    sed_data = np.asarray([(1,2),(3,4),(5,6)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(AlreadySetException):
        sample.addSedData(obj_id, sed_data)

###############################################################################

def test_addSedData_wrongDimensions(reference_sample_dir_fixture):
    """Test the case where the data have wrong dimensions"""

    # Given
    obj_id = 100
    sed_data = np.asarray([(1,2,3), (4,5,6), (7,8,9)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(InvalidDimensionsException):
        sample.addSedData(obj_id, sed_data)

###############################################################################

def test_addSedData_wrongWavelength(reference_sample_dir_fixture):
    """Test the case where the data have non increasing wavelength values"""

    # Given
    obj_id = 100
    sed_data = np.asarray([(1,2),(5,4),(3,6)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(InvalidAxisException):
        sample.addSedData(obj_id, sed_data)

###############################################################################

def test_addSedData_success(reference_sample_dir_fixture):
    """Test successful call of the addSedData()"""

    # Given
    obj_id = 100
    expected_data = np.asarray([(1,2),(3,4),(5,6)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    assert obj_id in sample.missingSedList()
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)

    # Then
    assert not sed_data is None
    assert np.all(sed_data == expected_data)

    # Given
    obj_id = 101
    expected_data = np.asarray([(1,2),(3,4),(5,6),(7,8)], dtype=np.float32)

    # When
    assert obj_id in sample.missingSedList()
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)

    # Then
    assert not sed_data is None
    assert np.all(sed_data == expected_data)
    assert len(sample.missingSedList()) == 0

###############################################################################

def test_addSedData_newDataFile(reference_sample_dir_fixture):
    """Test the case when the addSedData() method creates a new file"""

    # Given
    obj_id = 100
    expected_data = np.asarray([(1,2),(3,4),(5,6)], dtype=np.float32)
    last_size = os.path.getsize(os.path.join(reference_sample_dir_fixture, 'sed_data_2.bin'))

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    sample._ReferenceSample__data_file_limit = last_size - 1
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)

    # Then
    assert not sed_data is None
    assert np.all(sed_data == expected_data)
    assert os.path.exists(os.path.join(reference_sample_dir_fixture, 'sed_data_3.bin'))
    assert os.path.getsize(os.path.join(reference_sample_dir_fixture, 'sed_data_3.bin')) == 0

    # Given
    obj_id = 101
    expected_data = np.asarray([(1,2),(3,4),(5,6),(7,8)], dtype=np.float32)
    last_size = os.path.getsize(os.path.join(reference_sample_dir_fixture, 'sed_data_2.bin'))

    # When
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)

    # Then
    assert not sed_data is None
    assert np.all(sed_data == expected_data)
    assert os.path.getsize(os.path.join(reference_sample_dir_fixture, 'sed_data_2.bin')) == last_size
    assert os.path.getsize(os.path.join(reference_sample_dir_fixture, 'sed_data_3.bin')) == 8 + 4 + 8 * len(expected_data)

###############################################################################

def test_addSedData_notInOrder(reference_sample_dir_fixture):
    """Thest the case when the addSedData() is not called in the same order as the
    objects are stored in the index file"""

    # Given
    obj_id = 101
    expected_data = np.asarray([(1,2),(3,4),(5,6)], dtype=np.float32)
    sample = ReferenceSample(reference_sample_dir_fixture)

    # When
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)

    # Then
    assert not sed_data is None
    assert np.all(sed_data == expected_data)
    assert set(sample.missingSedList()) == set([100])

    # Given
    obj_id = 100
    expected_data = np.asarray([(1,2),(3,4),(5,6),(7,8)], dtype=np.float32)

    # When
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)

    # Then
    assert not sed_data is None
    assert np.all(sed_data == expected_data)
    assert len(sample.missingSedList()) == 0

###############################################################################

def test_missingSedList(reference_sample_dir_fixture):
    """Test calling the missingSedList() method"""

    # Given
    provider = ReferenceSample(reference_sample_dir_fixture)

    # When
    missing_ids = provider.missingSedList()

    # Then
    assert set(missing_ids) == set([100, 101])

    # When
    sed_data = np.asarray([(1,2),(3,4),(5,6)], dtype=np.float32)
    provider.addSedData(100, sed_data)
    missing_ids = provider.missingSedList()

    # Then
    assert set(missing_ids) == set([101])

    # When
    provider.addSedData(101, sed_data)
    missing_ids = provider.missingSedList()

    # Then
    assert len(missing_ids) == 0

    # When
    provider.createObject(200)
    missing_ids = provider.missingSedList()

    # Then
    assert set(missing_ids) == set([200])

###############################################################################

def test_addPdzData_wrongId(reference_sample_dir_fixture):
    """Test the case where the ID is not in the index"""

    # Given
    wrong_id = 200
    pdz_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(IdMismatchException):
        sample.addPdzData(wrong_id, pdz_data)

###############################################################################

def test_addPdzData_alreadySet(reference_sample_dir_fixture, pdz_list_fixture):
    """Test the case where the PDZ is already set"""

    # Given
    obj_id = pdz_list_fixture[1][-1][0]
    pdz_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(AlreadySetException):
        sample.addPdzData(obj_id, pdz_data)

###############################################################################

def test_addPdzData_invalidDimensions(reference_sample_dir_fixture):
    """Test the case where the PDZ array has wrong dimensions"""

    # Given
    obj_id = 100
    pdz_data = np.asarray([(1,10,1),(2,20,2),(5,50,5),(6,60,6),(8,80,8),(9,90,9)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(InvalidDimensionsException):
        sample.addPdzData(obj_id, pdz_data)

###############################################################################

def test_addPdzData_decreasingWavelength(reference_sample_dir_fixture):
    """Test the case where the PDZ is already set"""

    # Given
    obj_id = 100
    pdz_data = np.asarray([(1,10),(2,20),(6,50),(5,60),(8,80),(9,90)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(InvalidAxisException):
        sample.addPdzData(obj_id, pdz_data)

###############################################################################

def test_addPdzData_differentWavelengthFromExisting(reference_sample_dir_fixture):
    """Test the case where the PDZ is already set"""

    # Given
    obj_id = 100
    pdz_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(7,80),(9,90)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(InvalidAxisException):
        sample.addPdzData(obj_id, pdz_data)

    # Given
    pdz_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90),(10,100)], dtype=np.float32)

    # Then
    with pytest.raises(InvalidAxisException):
        sample.addPdzData(obj_id, pdz_data)

###############################################################################

def test_addPdzData_success(reference_sample_dir_fixture):
    """Test successful call of the addPdzData()"""

    # Given
    obj_id = 100
    expected_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    assert set(sample.missingPdzList()) == set([100, 101])
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)

    # Then
    assert not pdz_data is None
    assert pdzEqual(pdz_data, expected_data)

    # Given
    obj_id = 101
    expected_data = np.asarray([(1,100),(2,200),(5,500),(6,600),(8,800),(9,900)], dtype=np.float32)

    # When
    assert set(sample.missingPdzList()) == set([101])
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)

    # Then
    assert not pdz_data is None
    assert pdzEqual(pdz_data, expected_data)
    assert len(sample.missingPdzList()) == 0

###############################################################################

def test_addPdzData_newDataFile(reference_sample_dir_fixture):
    """Test the case when the addPdzData() method creates a new file"""

    # Given
    obj_id = 100
    expected_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)
    last_size = os.path.getsize(os.path.join(reference_sample_dir_fixture, 'pdz_data_2.bin'))

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    sample._ReferenceSample__data_file_limit = last_size - 1
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)

    # Then
    assert not pdz_data is None
    assert pdzEqual(pdz_data, expected_data)
    assert os.path.exists(os.path.join(reference_sample_dir_fixture, 'pdz_data_3.bin'))
    assert os.path.getsize(os.path.join(reference_sample_dir_fixture, 'pdz_data_3.bin')) == 4 + 4 * len(expected_data)

    # Given
    obj_id = 101
    expected_data = np.asarray([(1,100),(2,200),(5,500),(6,600),(8,800),(9,900)], dtype=np.float32)
    last_size = os.path.getsize(os.path.join(reference_sample_dir_fixture, 'pdz_data_2.bin'))

    # When
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)

    # Then
    assert not pdz_data is None
    assert pdzEqual(pdz_data, expected_data)
    assert os.path.getsize(os.path.join(reference_sample_dir_fixture, 'pdz_data_2.bin')) == last_size
    assert os.path.getsize(os.path.join(reference_sample_dir_fixture, 'pdz_data_3.bin')) == 4 + 4 * len(expected_data) + 8 + 4 * len(expected_data)

###############################################################################

def test_addPdzData_notInOrder(reference_sample_dir_fixture):
    """Test the case when the addPdzData() is not called in the same order as the
    objects are stored in the index file"""

    # Given
    obj_id = 101
    expected_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)
    sample = ReferenceSample(reference_sample_dir_fixture)

    # When
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)

    # Then
    assert not pdz_data is None
    assert pdzEqual(pdz_data, expected_data)
    assert set(sample.missingPdzList()) == set([100])

    # Given
    obj_id = 100
    expected_data = np.asarray([(1,100),(2,200),(5,500),(6,600),(8,800),(9,900)], dtype=np.float32)

    # When
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)

    # Then
    assert not pdz_data is None
    assert pdzEqual(pdz_data, expected_data)
    assert len(sample.missingPdzList()) == 0

###############################################################################

def test_addPdzData_newFileWrongWavelength(reference_sample_dir_fixture):
    """Test the case when a PDZ is added to a new data file and it has wrong wavelengths"""

    # Given
    obj_id = 100
    expected_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)
    last_size = os.path.getsize(os.path.join(reference_sample_dir_fixture, 'pdz_data_2.bin'))
    sample = ReferenceSample(reference_sample_dir_fixture)
    sample._ReferenceSample__data_file_limit = last_size - 1
    sample.addPdzData(obj_id, expected_data)

    # When
    obj_id = 101
    expected_data = np.asarray([(1,100),(2,200),(5,500),(6,600),(7,800),(9,900)], dtype=np.float32)

    # Then
    with pytest.raises(InvalidAxisException):
        sample.addPdzData(obj_id, expected_data)

###############################################################################

def test_missingPdzList(reference_sample_dir_fixture):
    """Test calling the missingPdzList() method"""

    # Given
    provider = ReferenceSample(reference_sample_dir_fixture)

    # When
    missing_ids = provider.missingPdzList()

    # Then
    assert set(missing_ids) == set([100, 101])

    # When
    pdz_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)
    provider.addPdzData(100, pdz_data)
    missing_ids = provider.missingPdzList()

    # Then
    assert set(missing_ids) == set([101])

    # When
    provider.addPdzData(101, pdz_data)
    missing_ids = provider.missingPdzList()

    # Then
    assert len(missing_ids) == 0

    # When
    provider.createObject(200)
    missing_ids = provider.missingPdzList()

    # Then
    assert set(missing_ids) == set([200])

###############################################################################

def test_iterate_ids(reference_sample_dir_fixture, sed_list_fixture):
    """Test iteration of the sample for the ID values"""

    # Given
    id_list = []
    for key in sed_list_fixture:
        id_list += [i for i,_ in sed_list_fixture[key]]
    id_list += [100, 101]

    # When
    provider = ReferenceSample(reference_sample_dir_fixture)

    # Then
    for obj, expected in zip(provider.iterate(), id_list):
        assert hasattr(obj, 'id')
        assert obj.id == expected

###############################################################################

def test_iterate_seds(reference_sample_dir_fixture, sed_list_fixture):
    """Test iteration of the sample for the SED values"""

    # Given
    sed_list = []
    for key in sed_list_fixture:
        sed_list += [s for _,s in sed_list_fixture[key]]
    sed_list += [None, None]

    # When
    provider = ReferenceSample(reference_sample_dir_fixture)

    # Then
    for obj, expected in zip(provider.iterate(), sed_list):
        assert hasattr(obj, 'sed')
        if obj.sed is None:
            assert expected == None
        else:
            assert np.all(obj.sed == expected)

###############################################################################

def test_iterate_pdzs(reference_sample_dir_fixture, redshift_bins_fixture, pdz_list_fixture):
    """Test iteration of the sample for the PDZ values"""

    # Given
    pdz_list = []
    for key in pdz_list_fixture:
        pdz_list += [np.stack((redshift_bins_fixture,p), axis=-1) for _,p in pdz_list_fixture[key]]
    pdz_list += [None, None]

    # When
    provider = ReferenceSample(reference_sample_dir_fixture)

    # Then
    for obj, expected in zip(provider.iterate(), pdz_list):
        assert hasattr(obj, 'pdz')
        if obj.pdz is None:
            assert expected == None
        else:
            assert np.all(obj.pdz == expected)

###############################################################################

def test_normalizePdz(reference_sample_dir_fixture):
    """When adding a PDZ, it should be normalized before stored"""

    # Given
    provider = ReferenceSample(reference_sample_dir_fixture)
    obj_id = 200
    original_pdz_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)
    original_integral = np.trapz(original_pdz_data[:, 1], original_pdz_data[:, 0])
    assert original_integral != 1

    # When
    provider.createObject(obj_id)
    provider.addPdzData(obj_id, original_pdz_data)

    # Then
    stored_pdz = provider.getPdzData(obj_id)
    assert np.trapz(stored_pdz[:, 1], stored_pdz[:, 0]) == 1
    assert pdzEqual(stored_pdz, original_pdz_data)

###############################################################################
