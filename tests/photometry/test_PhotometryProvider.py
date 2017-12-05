"""
Created on: 15/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import pytest
import os
from astropy.table import Table
import numpy as np

from nnpz.exceptions import *
from nnpz.photometry import PhotometryProvider

from tests.util_fixtures import temp_dir_fixture
from .fixtures import photometry_data_fixture, photometry_dir_fixture

###############################################################################

def test_constructor_idMismatch(photometry_dir_fixture, photometry_data_fixture):
    """Tests that if the photometry files do not contain the same IDs an exception is raised"""
    
    # Given
    f = os.path.join(photometry_dir_fixture, next(iter(photometry_data_fixture)))
    t = Table.read(f)
    t['ID'][-1] += 1
    os.remove(f)
    t.write(f, format='fits')
    
    # Then
    with pytest.raises(IdMismatchException):
        PhotometryProvider(photometry_dir_fixture)

###############################################################################

def test_getFilenameList(photometry_dir_fixture, photometry_data_fixture):
    """Test that the returned filenames are correct"""
    
    # Given
    expected = photometry_data_fixture.keys()
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    result = provider.getFilenameList()
    
    # Then
    assert set(result) == set(expected)

###############################################################################

def test_getFileBandList_wrongName(photometry_dir_fixture):
    """Test an exception is raised for unknown filename"""
    
    # Given
    wrong_name = "wrong.fits"
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    
    # Then
    with pytest.raises(UnknownNameException):
        provider.getFileBandList(wrong_name)

###############################################################################

def test_getFileBandList_success(photometry_dir_fixture, photometry_data_fixture):
    """Test successful call of getFileBandList()"""
    
    for filename in photometry_data_fixture:
        columns = photometry_data_fixture[filename]
        
        # Given
        expected = list(columns.keys())
        expected.remove('ID') 
        
        # When
        provider = PhotometryProvider(photometry_dir_fixture)
        bands = provider.getFileBandList(filename)
        
        # Then
        assert bands == expected

###############################################################################

def test_getFullBandList(photometry_dir_fixture, photometry_data_fixture):
    """Test successful call of getFullBandList()"""
    
    # Given
    expected = set()
    for filename in photometry_data_fixture:
        columns = photometry_data_fixture[filename]
        
        for band in columns.keys():
            if band != 'ID':
                expected.add((filename, band))
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    bands = provider.getFullBandList()
    
    # Then
    assert set(bands) == expected

###############################################################################

def test_getBandsData_unknownBand(photometry_dir_fixture):
    """Test we get an exception if we pass a wrong band name"""
    
    # Given
    band_list = [('photo1.fits', 'A2'), 'unknown_band', 'B3']
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    
    # Then
    with pytest.raises(UnknownNameException):
        provider.getBandsData(*band_list)
    
    # Given
    band_list = [('photo1.fits', 'A2'), 'unknown_band', 'B3']
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    
    # Then
    with pytest.raises(UnknownNameException):
        provider.getBandsData(*band_list)
    
    # Given
    band_list = [('photo1.fits', 'A2'), ('photo1.fits', 'unknown_band'), 'B3']
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    
    # Then
    with pytest.raises(UnknownNameException):
        provider.getBandsData(*band_list)
    
    # Given
    band_list = [('photo1.fits', 'A2'), ('unknown_file', 'A1'), 'B3']
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    
    # Then
    with pytest.raises(UnknownNameException):
        provider.getBandsData(*band_list)

###############################################################################

def test_getBandsData_ambiguity(photometry_dir_fixture):
    """Test we get an exception if we pass a band that exists in two files"""
    
    # Given
    band_list = [('photo1.fits', 'A2'), 'A4', 'B3']
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    
    # Then
    with pytest.raises(AmbiguityException):
        provider.getBandsData(*band_list)

###############################################################################

def test_getBandsData_success(photometry_dir_fixture, photometry_data_fixture):
    """Test successful call of getBandsData()"""
    
    # Given
    band_list = [('photo1.fits', 'A2'), ('photo1.fits', 'A4'), 'B3', ('photo2.fits', 'A4')]
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    result = provider.getBandsData(*band_list)
    
    # Then
    assert result.shape == (4, 4)
    assert np.array_equal(result[0], photometry_data_fixture['photo1.fits']['A2'])
    assert np.array_equal(result[1], photometry_data_fixture['photo1.fits']['A4'])
    assert np.array_equal(result[2], photometry_data_fixture['photo2.fits']['B3'])
    assert np.array_equal(result[3], photometry_data_fixture['photo2.fits']['A4'])

###############################################################################

def test_getFileData(photometry_dir_fixture):
    """Test getBandsData() with wrong filename"""
    
    # Given
    filename = 'wrong.fits'
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    
    # Then
    with pytest.raises(UnknownNameException):
        provider.getFileData(filename)

###############################################################################

def test_getFileData_success(photometry_dir_fixture, photometry_data_fixture):
    """Test successful call of getFileData()"""
    
    # Given
    provider = PhotometryProvider(photometry_dir_fixture)
    
    for filename in photometry_data_fixture:
        columns = photometry_data_fixture[filename]
        
        # When
        result = provider.getFileData(filename)
        
        # Then
        assert result.shape == (len(columns)-1, 4)
        i = 0
        for band in columns:
            if band == 'ID':
                continue
            data = columns[band]
            assert np.array_equal(result[i,:], data)
            i += 1

###############################################################################

def test_validate_success(photometry_dir_fixture, photometry_data_fixture):
    """Test that when the IDs are correct it returns True"""
    
    # Given
    id_list = list(photometry_data_fixture['photo1.fits']['ID'])
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    result = provider.validate(id_list)
    
    # Then
    assert result == True

###############################################################################

def test_validate_wrongId(photometry_dir_fixture, photometry_data_fixture):
    """Test that when an ID is different it returns False"""
    
    # Given
    id_list = list(photometry_data_fixture['photo1.fits']['ID'])
    id_list[1] = 100
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    result = provider.validate(id_list)
    
    # Then
    assert result == False

###############################################################################

def test_validate_wrongLength(photometry_dir_fixture, photometry_data_fixture):
    """Test that when the length is different it returns False"""
    
    # Given
    id_list = list(photometry_data_fixture['photo1.fits']['ID'])
    id_list.append(100)
    
    # When
    provider = PhotometryProvider(photometry_dir_fixture)
    result = provider.validate(id_list)
    
    # Then
    assert result == False

###############################################################################
