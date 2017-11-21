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

from tests.util_fixtures import temp_dir_fixture
from .fixtures import *

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
    sed_file = os.path.join(dir_name, 'sed_data.bin')
    assert os.path.exists(sed_file)
    assert os.path.getsize(sed_file) == 0
    
    # Check that the empty PDZ file is created
    pdz_file = os.path.join(dir_name, 'pdz_data.bin')
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
    os.remove(os.path.join(reference_sample_dir_fixture, 'sed_data.bin'))
    
    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(reference_sample_dir_fixture)
    
###############################################################################
    
def test_constructor_missingPdzDataFile(reference_sample_dir_fixture):
    """Test the constructor when the PDZ data file is missing"""
    
    # Given
    os.remove(os.path.join(reference_sample_dir_fixture, 'pdz_data.bin'))
    
    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(reference_sample_dir_fixture)
    
###############################################################################
    
def test_size(reference_sample_dir_fixture, sed_list_fixture):
    """Test the size() method works correctly"""
    
    # Given
    expected = len(sed_list_fixture) + 2
    
    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    size = sample.size()
    
    # Then
    assert size == expected
    
###############################################################################
    
def test_getIds(reference_sample_dir_fixture, sed_list_fixture):
    """Test the getIds() method works correctly"""
    
    # Given
    expected = [i for i,_ in sed_list_fixture] + [100, 101]
    
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
    id_list = [i for i,_ in sed_list_fixture]
    expected_data = [d for _,d in sed_list_fixture]
    
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
    id_list = [i for i,_ in pdz_list_fixture]
    expected_data = [d for _,d in pdz_list_fixture]
    
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
        f.seek(filesize - 8 * 3)
        fromfile = np.fromfile(f, count=3, dtype=np.int64)
        assert fromfile[0] == new_id
        assert fromfile[1] == -1
        assert fromfile[2] == -1
    
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
    obj_id = sed_list_fixture[-1][0]
    sed_data = np.asarray([(1,2),(3,4),(5,6)], dtype=np.float32)
    
    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    
    # Then
    with pytest.raises(AlreadySetException):
        sample.addSedData(obj_id, sed_data)
    
###############################################################################
    
def test_addSedData_previousNotSet(reference_sample_dir_fixture):
    """Test the case where the data of the previous object are not set"""
    
    # Given
    obj_id = 101
    sed_data = np.asarray([(1,2),(3,4),(5,6)], dtype=np.float32)
    
    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    
    # Then
    with pytest.raises(InvalidPositionException):
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
    assert sample.firstIdMissingSedData() == obj_id
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)
    
    # Then
    assert not sed_data is None
    assert np.all(sed_data == expected_data)
    
    # Given
    obj_id = 101
    expected_data = np.asarray([(1,2),(3,4),(5,6),(7,8)], dtype=np.float32)
    
    # When
    assert sample.firstIdMissingSedData() == obj_id
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)
    
    # Then
    assert not sed_data is None
    assert np.all(sed_data == expected_data)
    assert sample.firstIdMissingSedData() is None
    
###############################################################################

def test_firstIdMissingSedData(reference_sample_dir_fixture):
    """Test calling the firstIdMissingSedData() method"""
    
    # Given 
    sed_data = np.asarray([(1,2),(3,4),(5,6)], dtype=np.float32)
                    
    # When
    provider = ReferenceSample(reference_sample_dir_fixture)
    missing_id = provider.firstIdMissingSedData()
    
    # Then
    assert missing_id == 100
    
    # When
    provider.addSedData(100, sed_data)
    missing_id = provider.firstIdMissingSedData()
    
    # Then
    assert missing_id == 101
    
    # When
    provider.addSedData(101, sed_data)
    missing_id = provider.firstIdMissingSedData()
    
    # Then
    assert missing_id is None
    
    # When
    provider.createObject(200)
    missing_id = provider.firstIdMissingSedData()
    
    # Then
    assert missing_id == 200
        
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
    obj_id = pdz_list_fixture[-1][0]
    pdz_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)
    
    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    
    # Then
    with pytest.raises(AlreadySetException):
        sample.addPdzData(obj_id, pdz_data)
    
###############################################################################

def test_addPdzData_alreadySet(reference_sample_dir_fixture):
    """Test the case where the PDZ is already set"""
    
    # Given
    obj_id = 101
    pdz_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)
    
    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    
    # Then
    with pytest.raises(InvalidPositionException):
        sample.addPdzData(obj_id, pdz_data)
    
###############################################################################

def test_addPdzData_invalidDimensions(reference_sample_dir_fixture):
    """Test the case where the PDZ is already set"""
    
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
    assert sample.firstIdMissingPdzData() == obj_id
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)
    
    # Then
    assert not pdz_data is None
    assert np.all(pdz_data == expected_data)
    
    # Given
    obj_id = 101
    expected_data = np.asarray([(1,100),(2,200),(5,500),(6,600),(8,800),(9,900)], dtype=np.float32)
    
    # When
    assert sample.firstIdMissingPdzData() == obj_id
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)
    
    # Then
    assert not pdz_data is None
    assert np.all(pdz_data == expected_data)
    assert sample.firstIdMissingPdzData() is None
    
###############################################################################

def test_firstIdMissingPdzData(reference_sample_dir_fixture):
    """Test calling the firstIdMissingPdzData() method"""
    
    # Given 
    pdz_data = np.asarray([(1,10),(2,20),(5,50),(6,60),(8,80),(9,90)], dtype=np.float32)
                    
    # When
    provider = ReferenceSample(reference_sample_dir_fixture)
    missing_id = provider.firstIdMissingPdzData()
    
    # Then
    assert missing_id == 100
    
    # When
    provider.addPdzData(100, pdz_data)
    missing_id = provider.firstIdMissingPdzData()
    
    # Then
    assert missing_id == 101
    
    # When
    provider.addPdzData(101, pdz_data)
    missing_id = provider.firstIdMissingPdzData()
    
    # Then
    assert missing_id is None
    
    # When
    provider.createObject(200)
    missing_id = provider.firstIdMissingPdzData()
    
    # Then
    assert missing_id == 200
        
###############################################################################
    