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

"""
Created on: 10/11/17
Author: Nikolaos Apostolakos
"""

import tempfile

from nnpz import ReferenceSample
from nnpz.exceptions import *
from nnpz.reference_sample.MontecarloProvider import MontecarloProvider

from .fixtures import *


def pdzEqual(a, b):
    """
    Compare two pdz, taking into account their normalization
    """
    norm_a = a[:, 1] / np.trapz(a[:, 1], a[:, 0])
    norm_b = b[:, 1] / np.trapz(b[:, 1], b[:, 0])
    return np.allclose(norm_a, norm_b) and np.all(a[:, 0] == b[:, 0])


@pytest.fixture
def providers_with_mc():
    providers = dict(ReferenceSample.DEFAULT_PROVIDERS)
    providers['MontecarloProvider'] = [
        {'name': 'mc', 'data': 'mc_data_{}.npy'}
    ]
    return providers


###############################################################################

def test_createNew_dirExists(temp_dir_fixture):
    """
    Tests that if the directory exists an exception is raised
    """

    # Given
    dir_name = os.path.join(temp_dir_fixture, 'ref_sample')
    os.makedirs(dir_name)

    # Then
    with pytest.raises(OSError):
        ReferenceSample.createNew(dir_name)


###############################################################################

def test_createNew_success(temp_dir_fixture):
    """
    Tests that all the files of the reference sample are constructed correctly
    """

    # Given
    dir_name = os.path.join(temp_dir_fixture, 'ref_sample')

    # When
    result = ReferenceSample.createNew(dir_name)

    # Then

    # Check the result object is correct
    assert result is not None

    # Check that the result contains nothing
    assert len(result) == 0

    # Check the directory was created correctly
    assert os.path.isdir(dir_name)


###############################################################################

def test_constructor_missingDir(temp_dir_fixture):
    """
    Test the constructor when the whole directory is missing
    """

    # Given
    dir_name = os.path.join(temp_dir_fixture, 'missing')

    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(dir_name)


###############################################################################

def test_constructor_isFile(temp_dir_fixture):
    """
    Test the constructor when the path is actually a file
    """

    # Given
    file_name = os.path.join(temp_dir_fixture, 'file')
    with open(file_name, 'w') as fd:
        fd.write('CONTENT')

    # Then
    with pytest.raises(NotADirectoryError):
        ReferenceSample(file_name)


###############################################################################

def test_constructor_missingSedDataFile(reference_sample_dir_fixture):
    """
    Test the constructor when the SED data file is missing
    """

    # Given
    os.remove(os.path.join(reference_sample_dir_fixture, 'sed_data_1.npy'))

    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(reference_sample_dir_fixture)


###############################################################################

def test_constructor_missingPdzDataFile(reference_sample_dir_fixture):
    """
    Test the constructor when the PDZ data file is missing
    """

    # Given
    os.remove(os.path.join(reference_sample_dir_fixture, 'pdz_data_1.npy'))

    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(reference_sample_dir_fixture)


###############################################################################

def test_constructor_missingMcDataFile(reference_sample_dir_fixture, providers_with_mc):
    """
    Test the constructor when the MC data file is missing
    """

    # Given
    os.remove(os.path.join(reference_sample_dir_fixture, 'mc_data_1.npy'))

    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(reference_sample_dir_fixture, providers=providers_with_mc)


###############################################################################

def test_size(reference_sample_dir_fixture, sed_list_fixture):
    """
    Test the size() method works correctly
    """

    # Given
    expected = 0
    for key in sed_list_fixture:
        expected += len(sed_list_fixture[key])

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    assert len(sample) == expected


###############################################################################

def test_getIds(reference_sample_dir_fixture, sed_list_fixture):
    """
    Test the getIds() method works correctly
    """

    # Given
    expected = []
    for key in sed_list_fixture:
        expected = expected + [i for i, _ in sed_list_fixture[key]]

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    ids = sample.getIds()

    # Then
    assert np.all(sorted(ids) == sorted(expected))


###############################################################################

def test_getSedData_dataUnset(reference_sample_dir_fixture):
    """
    Test the case where the SED data are not set yet
    """

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
    """
    Test the case where the SED data exist
    """

    # Given
    id_list = []
    expected_data = []
    for key in sed_list_fixture:
        id_list += [i for i, _ in sed_list_fixture[key]]
        expected_data += [d for _, d in sed_list_fixture[key]]

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    sed_data = [sample.getSedData(i) for i in id_list]

    # Then
    for i in range(len(id_list)):
        assert sed_data[i].shape == expected_data[i].shape
        assert np.all(sed_data[i] == expected_data[i])


###############################################################################

def test_corruptedIndex(reference_sample_dir_fixture):
    """
    Corrupted index file
    """

    # Given
    with open(os.path.join(reference_sample_dir_fixture, 'index.npy'), 'rb+') as f:
        f.seek(0)
        correct_id = np.fromfile(f, count=1, dtype=np.int64)[0]
        wrong_id = correct_id + 1
        f.seek(0)
        np.asarray([wrong_id], dtype=np.int64).tofile(f)

    # Then
    with pytest.raises(CorruptedFileException):
        ReferenceSample(reference_sample_dir_fixture)


##############################################################################

def test_getPdzData_dataUnset(reference_sample_dir_fixture):
    """
    Test the case where the PDZ data are not set yet
    """

    # Given
    unset_pdz_ids = [100, 101]

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    pdz_data = [sample.getPdzData(i) for i in unset_pdz_ids]

    # Then
    assert pdz_data[0] is None
    assert pdz_data[1] is None


##############################################################################

def test_getMcData_dataUnset(reference_sample_dir_fixture, providers_with_mc):
    """
    Test the case where the MC data are not set yet
    """

    # Given
    unset_mc_ids = [100, 101]

    # When
    sample = ReferenceSample(reference_sample_dir_fixture, providers=providers_with_mc)
    mc_data = [sample.getData('mc', i) for i in unset_mc_ids]

    # Then
    assert mc_data[0] is None
    assert mc_data[1] is None


###############################################################################

def test_getMcData_withData(reference_sample_dir_fixture, mc_data_fixture, providers_with_mc):
    """
    Test the case where the MC data exist
    """

    # Given
    id_list = []
    expected_data = []
    for key in mc_data_fixture:
        id_list += mc_data_fixture[key][0]
        expected_data.extend([row for row in mc_data_fixture[key][1]])

    # When
    sample = ReferenceSample(reference_sample_dir_fixture, providers=providers_with_mc)
    data = [sample.getData('mc', i) for i in id_list]

    # Then
    for i in range(len(id_list)):
        assert data[i].shape == expected_data[i].shape
        assert all([np.allclose(data[i][c], expected_data[i][c]) for c in data[i].dtype.names])


###############################################################################

def test_getPdzData_withData(reference_sample_dir_fixture, pdz_list_fixture, redshift_bins_fixture):
    """
    Test the case where the PDZ data exist
    """
    sample = ReferenceSample(reference_sample_dir_fixture)

    for obj in pdz_list_fixture.values():
        for obj_id, expected in obj:
            pdz = sample.getPdzData(obj_id)
            assert pdz.shape == (len(expected), 2)
            assert np.array_equal(pdz[:, 0], redshift_bins_fixture)
            assert np.array_equal(pdz[:, 1], expected)


###############################################################################

def test_addSedData_alreadySet(reference_sample_dir_fixture, sed_list_fixture):
    """
    Test the case where the data are already set
    """

    # Given
    obj_id = sed_list_fixture[1][-1][0]
    sed_data = np.asarray([(1, 2), (3, 4), (5, 6)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(AlreadySetException):
        sample.addSedData(obj_id, sed_data)


###############################################################################

def test_addSedData_wrongDimensions(reference_sample_dir_fixture):
    """
    Test the case where the data have wrong dimensions
    """

    # Given
    obj_id = 100
    sed_data = np.asarray([(1, 2, 3), (4, 5, 6), (7, 8, 9)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(InvalidDimensionsException):
        sample.addSedData(obj_id, sed_data)


###############################################################################

def test_addSedData_wrongWavelength(reference_sample_dir_fixture):
    """
    Test the case where the data have non increasing wavelength values
    """

    # Given
    obj_id = 100
    sed_data = np.asarray([(1, 2), (5, 4), (3, 6)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(InvalidAxisException):
        sample.addSedData(obj_id, sed_data)


###############################################################################

def test_addSedData_success(reference_sample_dir_fixture):
    """
    Test successful call of the addSedData()
    """

    # Given
    obj_id = 100
    expected_data = np.asarray([(1, 2), (3, 4), (5, 6)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)

    # Then
    assert sed_data is not None
    assert np.all(sed_data == expected_data)

    # Given
    obj_id = 101
    expected_data = np.asarray([(1, 2), (3, 4), (5, 6), (7, 8)], dtype=np.float32)

    # When
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)

    # Then
    assert sed_data is not None
    assert np.all(sed_data == expected_data)


###############################################################################

def test_addSedData_newDataFile(reference_sample_dir_fixture):
    """
    Test the case when the addSedData() method creates a new file
    """

    sed_data_2_filename = os.path.join(reference_sample_dir_fixture, 'sed_data_2.npy')
    sed_data_3_filename = os.path.join(reference_sample_dir_fixture, 'sed_data_3.npy')

    # Given
    obj_id = 100
    expected_data = np.asarray([(1, 2), (3, 4), (5, 6)], dtype=np.float32)
    last_size_2 = os.path.getsize(sed_data_2_filename)
    assert not os.path.exists(sed_data_3_filename)

    # When
    with ReferenceSample(reference_sample_dir_fixture, max_file_size=last_size_2 - 1) as sample:
        sample.addSedData(obj_id, expected_data)
        sed_data = sample.getSedData(obj_id)

    # Then
    assert sed_data is not None
    assert np.all(sed_data == expected_data)
    assert os.path.exists(sed_data_3_filename)
    last_size_3 = os.path.getsize(sed_data_3_filename)

    # Given
    obj_id = 101
    expected_data = np.asarray([(1, 2), (5, 6), (7, 8)], dtype=np.float32)

    # When
    with ReferenceSample(reference_sample_dir_fixture, max_file_size=last_size_2 - 1) as sample:
        sample.addSedData(obj_id, expected_data)
        sed_data = sample.getSedData(obj_id)

    # Then
    assert sed_data is not None
    assert np.all(sed_data == expected_data)
    assert os.path.getsize(sed_data_2_filename) == last_size_2
    assert os.path.getsize(sed_data_3_filename) > last_size_3


###############################################################################

def test_addSedData_notInOrder(reference_sample_dir_fixture):
    """
    Test the case when the addSedData() is not called in the same order as the
    objects are stored in the index file
    """

    # Given
    obj_id = 101
    expected_data = np.asarray([(1, 2), (3, 4), (5, 6)], dtype=np.float32)
    sample = ReferenceSample(reference_sample_dir_fixture)

    # When
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)

    # Then
    assert not sed_data is None
    assert np.all(sed_data == expected_data)

    # Given
    obj_id = 100
    expected_data = np.asarray([(1, 2), (3, 4), (5, 6), (7, 8)], dtype=np.float32)

    # When
    sample.addSedData(obj_id, expected_data)
    sed_data = sample.getSedData(obj_id)

    # Then
    assert not sed_data is None
    assert np.all(sed_data == expected_data)


###############################################################################

def test_addPdzData_alreadySet(reference_sample_dir_fixture, pdz_list_fixture):
    """
    Test the case where the PDZ is already set
    """

    # Given
    obj_id = pdz_list_fixture[1][-1][0]
    pdz_data = np.asarray([(1, 10), (2, 20), (5, 50), (6, 60), (8, 80), (9, 90)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(AlreadySetException):
        sample.addPdzData(obj_id, pdz_data)


###############################################################################

def test_addPdzData_invalidDimensions(reference_sample_dir_fixture):
    """
    Test the case where the PDZ array has wrong dimensions
    """

    # Given
    obj_id = 100
    pdz_data = np.asarray([(1, 10, 1), (2, 20, 2), (5, 50, 5), (6, 60, 6), (8, 80, 8), (9, 90, 9)],
                          dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(InvalidDimensionsException):
        sample.addPdzData(obj_id, pdz_data)


###############################################################################

def test_addPdzData_decreasingBins(reference_sample_dir_fixture):
    """
    Test the case where the PDZ is already set
    """

    # Given
    obj_id = 100
    pdz_data = np.asarray([(1, 10), (2, 20), (6, 50), (5, 60), (8, 80), (9, 90)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(InvalidAxisException):
        sample.addPdzData(obj_id, pdz_data)


###############################################################################

def test_addPdzData_differentBins(reference_sample_dir_fixture):
    """
    Test the case where the PDZ is already set
    """

    # Given
    obj_id = 100
    pdz_data = np.asarray([(1, 10), (2, 20), (5, 50), (6, 60), (7, 80), (9, 90)], dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)

    # Then
    with pytest.raises(InvalidAxisException):
        sample.addPdzData(obj_id, pdz_data)

    # Given
    pdz_data = np.asarray([(1, 10), (2, 20), (5, 50), (6, 60), (8, 80), (9, 90), (10, 100)],
                          dtype=np.float32)

    # Then
    with pytest.raises(InvalidAxisException):
        sample.addPdzData(obj_id, pdz_data)


###############################################################################

def test_addPdzData_success(reference_sample_dir_fixture):
    """
    Test successful call of the addPdzData()
    """

    # Given
    obj_id = 100
    expected_data = np.asarray([(1, 10), (2, 20), (5, 50), (6, 60), (8, 80), (9, 90)],
                               dtype=np.float32)

    # When
    sample = ReferenceSample(reference_sample_dir_fixture)
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)

    # Then
    assert pdz_data is not None
    assert pdzEqual(pdz_data, expected_data)

    # Given
    obj_id = 101
    expected_data = np.asarray([(1, 100), (2, 200), (5, 500), (6, 600), (8, 800), (9, 900)],
                               dtype=np.float32)

    # When
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)

    # Then
    assert \
        pdz_data is not None
    assert pdzEqual(pdz_data, expected_data)


###############################################################################

def test_addPdzData_newDataFile(reference_sample_dir_fixture):
    """
    Test the case when the addPdzData() method creates a new file
    """

    pdz_data_2_filename = os.path.join(reference_sample_dir_fixture, 'pdz_data_2.npy')
    pdz_data_3_filename = os.path.join(reference_sample_dir_fixture, 'pdz_data_3.npy')

    # Given
    obj_id = 100
    expected_data = np.asarray(
        [(1, 10), (2, 20), (5, 50), (6, 60), (8, 80), (9, 90)],
        dtype=np.float32
    )
    last_size = os.path.getsize(pdz_data_2_filename)
    assert not os.path.exists(pdz_data_3_filename)

    # When
    with ReferenceSample(reference_sample_dir_fixture, max_file_size=last_size - 1) as sample:
        sample.addPdzData(obj_id, expected_data)
        pdz_data = sample.getPdzData(obj_id)

    # Then
    assert pdz_data is not None
    assert pdzEqual(pdz_data, expected_data)
    assert os.path.exists(pdz_data_3_filename)
    assert os.path.getsize(pdz_data_2_filename) == last_size
    last_size_3 = os.path.getsize(pdz_data_3_filename)

    # Given
    obj_id = 101
    expected_data = np.asarray(
        [(1, 100), (2, 200), (5, 500), (6, 600), (8, 800), (9, 900)],
        dtype=np.float32
    )

    # When
    with ReferenceSample(reference_sample_dir_fixture, max_file_size=last_size - 1) as sample:
        sample.addPdzData(obj_id, expected_data)
        pdz_data = sample.getPdzData(obj_id)

    # Then
    assert pdz_data is not None
    assert pdzEqual(pdz_data, expected_data)
    assert os.path.getsize(pdz_data_2_filename) == last_size
    assert os.path.getsize(pdz_data_3_filename) > last_size_3


###############################################################################

def test_addPdzData_notInOrder(reference_sample_dir_fixture):
    """
    Test the case when the addPdzData() is not called in the same order as the
    objects are stored in the index file
    """

    # Given
    obj_id = 101
    expected_data = np.asarray([(1, 10), (2, 20), (5, 50), (6, 60), (8, 80), (9, 90)],
                               dtype=np.float32)
    sample = ReferenceSample(reference_sample_dir_fixture)

    # When
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)

    # Then
    assert pdz_data is not None
    assert pdzEqual(pdz_data, expected_data)

    # Given
    obj_id = 100
    expected_data = np.asarray(
        [(1, 100), (2, 200), (5, 500), (6, 600), (8, 800), (9, 900)],
        dtype=np.float32
    )

    # When
    sample.addPdzData(obj_id, expected_data)
    pdz_data = sample.getPdzData(obj_id)

    # Then
    assert pdz_data is not None
    assert pdzEqual(pdz_data, expected_data)


###############################################################################

def test_addPdzData_newFileWrongBins(reference_sample_dir_fixture):
    """
    Test the case when a PDZ is added to a new data file and it has wrong bins
    """

    # Given
    obj_id = 100
    expected_data = np.asarray([(1, 10), (2, 20), (5, 50), (6, 60), (8, 80), (9, 90)],
                               dtype=np.float32)
    last_size = os.path.getsize(os.path.join(reference_sample_dir_fixture, 'pdz_data_2.npy'))
    sample = ReferenceSample(reference_sample_dir_fixture, max_file_size=last_size - 1)
    sample.addPdzData(obj_id, expected_data)

    # When
    obj_id = 101
    expected_data = np.asarray(
        [(1, 100), (2, 200), (5, 500), (6, 600), (7, 800), (9, 900)],
        dtype=np.float32
    )

    # Then
    with pytest.raises(InvalidAxisException):
        sample.addPdzData(obj_id, expected_data)


###############################################################################

def test_iterate_ids(reference_sample_dir_fixture, sed_list_fixture):
    """Test iteration of the sample for the ID values"""

    # Given
    expected_id_list = []
    for key in sed_list_fixture:
        expected_id_list += [i for i, _ in sed_list_fixture[key]]

    # When
    provider = ReferenceSample(reference_sample_dir_fixture)

    # Then
    seen = []
    for obj in provider.iterate():
        assert hasattr(obj, 'id')
        seen.append(obj.id)
    assert sorted(expected_id_list) == sorted(seen)


###############################################################################

def test_iterate_seds(reference_sample_dir_fixture, sed_list_fixture):
    """
    Test iteration of the sample for the SED values
    """

    # Given
    sed_list = {}
    for key in sed_list_fixture:
        for obj_id, sed in sed_list_fixture[key]:
            sed_list[obj_id] = sed

    # When
    provider = ReferenceSample(reference_sample_dir_fixture)

    # Then
    for obj in provider.iterate():
        assert hasattr(obj, 'sed')
        if obj.sed is None:
            assert sed_list[obj.id] is None
        else:
            assert np.all(obj.sed == sed_list[obj.id])


###############################################################################

def test_iterate_pdzs(reference_sample_dir_fixture, redshift_bins_fixture, pdz_list_fixture):
    """
    Test iteration of the sample for the PDZ values
    """

    # Given
    pdz_list = {}
    for key in pdz_list_fixture:
        for obj_id, pdz in pdz_list_fixture[key]:
            pdz_list[obj_id] = np.stack((redshift_bins_fixture, pdz), axis=-1)

    # When
    provider = ReferenceSample(reference_sample_dir_fixture)

    # Then
    for obj in provider.iterate():
        assert hasattr(obj, 'pdz')
        if obj.pdz is None:
            assert pdz_list[obj.id] is None
        else:
            assert np.array_equal(obj.pdz, pdz_list[obj.id])


###############################################################################

def test_normalizePdz(reference_sample_dir_fixture):
    """
    When adding a PDZ, it should be normalized before stored
    """

    # Given
    ref_sample = ReferenceSample(reference_sample_dir_fixture)
    obj_id = 200
    original_pdz_data = np.asarray([(1, 10), (2, 20), (5, 50), (6, 60), (8, 80), (9, 90)],
                                   dtype=np.float32)
    original_integral = np.trapz(original_pdz_data[:, 1], original_pdz_data[:, 0])
    assert original_integral != 1

    # When
    ref_sample.addPdzData(obj_id, original_pdz_data)

    # Then
    stored_pdz = ref_sample.getPdzData(obj_id)
    assert np.trapz(stored_pdz[:, 1], stored_pdz[:, 0]) == 1
    assert pdzEqual(stored_pdz, original_pdz_data)


###############################################################################

def test_importRefSample(reference_sample_dir_fixture):
    """
    Test the import of another reference sample
    """
    new_ref_dir = tempfile.mktemp(prefix='nnpz_test_new_ref')

    new_ref = ReferenceSample.createNew(new_ref_dir)
    new_ref.importDirectory(reference_sample_dir_fixture)

    old_ref = ReferenceSample(reference_sample_dir_fixture)
    assert len(old_ref) == len(new_ref)

    for obj_id in old_ref.getIds():
        nsed = new_ref.getSedData(obj_id)
        npdz = new_ref.getPdzData(obj_id)
        assert np.array_equal(nsed, old_ref.getSedData(obj_id))
        assert np.array_equal(npdz, old_ref.getPdzData(obj_id))


###############################################################################

def test_addProvider(reference_sample_dir_fixture):
    """
    Add a provider after creation
    """

    # Given
    expected_data = np.zeros((2, 100), dtype=[
        ('A', np.float), ('B', np.float), ('C', np.float), ('D', np.float)
    ])
    for c in expected_data.dtype.names:
        expected_data[c] = np.random.rand(*expected_data.shape)

    ref_sample = ReferenceSample(reference_sample_dir_fixture)
    with pytest.raises(KeyError):
        ref_sample.getProvider('mc2')

    # When
    ref_sample.addProvider(
        'MontecarloProvider', name='mc2',
        data_pattern='mc2_data_{}.npy',
        object_ids=[100, 101],
        data=expected_data, extra=dict(key='value')
    )

    # Then
    prov = ref_sample.getProvider('mc2')
    assert prov is not None
    assert isinstance(prov, MontecarloProvider)

    data = ref_sample.getData('mc2', 100)
    assert all(
        [np.allclose(expected_data[c][0], data[c].reshape(1, 100)) for c in data.dtype.names]
    )


###############################################################################

def test_addProviderExists(reference_sample_dir_fixture):
    """
    Add a provider after creation
    """

    # Given
    expected_data = np.zeros((2, 100), dtype=[
        ('A', np.float), ('B', np.float), ('C', np.float), ('D', np.float)
    ])
    for c in expected_data.dtype.names:
        expected_data[c] = np.random.rand(*expected_data.shape)

    ref_sample = ReferenceSample(reference_sample_dir_fixture)
    with pytest.raises(KeyError):
        ref_sample.getProvider('mc2')

    # When
    ref_sample.addProvider(
        'MontecarloProvider', name='mc2',
        data_pattern='mc2_data_{}.npy',
        object_ids=[100, 101],
        data=expected_data, extra=dict(key='value')
    )

    # Then
    with pytest.raises(AlreadySetException):
        ref_sample.addProvider(
            'MontecarloProvider', name='mc2',
            data_pattern='mc2_data_{}.npy',
            object_ids=[205, 206],
            data=expected_data,
            overwrite=False)


###############################################################################

def test_addProviderOverwrite(reference_sample_dir_fixture):
    """
    Add a provider after creation
    """

    # Given
    expected_data = np.zeros((2, 100), dtype=[
        ('A', np.float), ('B', np.float), ('C', np.float), ('D', np.float)
    ])
    for c in expected_data.dtype.names:
        expected_data[c] = np.random.rand(*expected_data.shape)

    ref_sample = ReferenceSample(reference_sample_dir_fixture)
    with pytest.raises(KeyError):
        ref_sample.getProvider('mc2')

    # When
    ref_sample.addProvider(
        'MontecarloProvider', name='mc2',
        data_pattern='mc2_data_{}.npy',
        object_ids=[100, 101],
        data=expected_data, extra=dict(key='value')
    )

    # Then
    ref_sample.addProvider(
        'MontecarloProvider', name='mc2',
        data_pattern='mc2_data_{}.npy',
        object_ids=[205, 206],
        data=expected_data,
        overwrite=True)


###############################################################################

def test_missingIndex(reference_sample_dir_fixture):
    """
    Open a directory where the SED index is missing
    """

    # When
    os.unlink(os.path.join(reference_sample_dir_fixture, 'index.npy'))

    # Then
    with pytest.raises(FileNotFoundException):
        ReferenceSample(reference_sample_dir_fixture)

###############################################################################
