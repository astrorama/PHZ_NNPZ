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

import os

import numpy as np
import pytest

from ..fixtures.util_fixtures import temp_dir_fixture


##############################################################################

@pytest.fixture()
def sed_list_fixture():
    """
    Returns a list of SEDs to be used for testing.

    The SEDs are the following:
      File   ID  Data
     ------ ---- --------------
       1     1   [(1., 100.), (1.5, 168.3)]
       1     3   [(167.7,4.2), (456.49,7.8)]
       1    12   [(1.,1.), (2.,2.)]
       1     7   [(56.,78.1), (57, 60.3)]
       2    34   [(6.,6.), (7.,7.)]
       2    56   [(8.,8.), (9.,9.)]
    """
    return {
        1: [
            [1, np.asarray([(1., 100.), (1.5, 168.3)], dtype=np.float32)],
            [3, np.asarray([(167.7, 4.2), (456.49, 7.8)], dtype=np.float32)],
            [12, np.asarray([(1., 1.), (2., 2.)], dtype=np.float32)],
            [7, np.asarray([(56., 78.1), (57, 60.3)], dtype=np.float32)]
        ],
        2: [
            [34, np.asarray([(6., 6.), (7., 7.)], dtype=np.float32)],
            [56, np.asarray([(8., 8.), (9., 9.)], dtype=np.float32)]
        ]
    }


##############################################################################

@pytest.fixture()
def sed_data_files_fixture(temp_dir_fixture, sed_list_fixture):
    """
    Creates SED data files to be used for testing.

    Returns: A map with keys the file indices and values the paths of the files

    The created files contains the SEDs of the sed_list_fixture.
    """

    result = {}
    for file_index, sed_data in sed_list_fixture.items():
        filename = os.path.join(temp_dir_fixture, 'sed_data_{}.npy'.format(file_index))
        array = np.concatenate(
            [a.reshape(1, -1, 2) for _, a in sed_data],
            axis=0
        )
        np.save(filename, array)
        result[file_index] = filename

    return result


##############################################################################

@pytest.fixture()
def redshift_bins_fixture():
    """
    Returns an array with the redsift bins of the PDZs to be used for testing
    """
    return np.asarray([1, 2, 5, 6, 8, 9], dtype=np.float32)


##############################################################################

@pytest.fixture(scope='session')
def pdz_list_fixture():
    """
    Returns a list of PDZs to be used for testing.
    """
    return {
        1: [
            [1, np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32)],
            [3, np.asarray([2, 3, 4, 5, 6, 7], dtype=np.float32)],
            [12, np.asarray([3, 4, 5, 6, 7, 8], dtype=np.float32)],
            [7, np.asarray([4, 5, 6, 7, 8, 9], dtype=np.float32)]
        ],
        2: [
            [34, np.asarray([11, 12, 13, 14, 15, 16], dtype=np.float32)],
            [56, np.asarray([12, 13, 14, 15, 16, 17], dtype=np.float32)]
        ]
    }


##############################################################################

@pytest.fixture()
def pdz_data_files_fixture(temp_dir_fixture, redshift_bins_fixture, pdz_list_fixture):
    """
    Creates PDZ data files to be used for testing.

    Returns: A map with keys the file indices and values he path to the newly created files

    The created files contain the PDZs of the pdz_list_fixture and the redshift
    bins of the redshift_bins_fixture.
    """
    result = {}
    for file_index, pdz_data in pdz_list_fixture.items():
        filename = os.path.join(temp_dir_fixture, 'pdz_data_{}.npy'.format(file_index))
        array = np.concatenate(
            [redshift_bins_fixture.reshape((1, -1))] + [a.reshape(1, -1) for _, a in pdz_data],
            axis=0
        )
        np.save(filename, array)
        result[file_index] = filename
    return result


##############################################################################

@pytest.fixture(scope='session')
def mc_data_fixture(pdz_list_fixture):
    """
    Generate on the fly some MC data
    """
    result = {}
    for i, pdz_data in pdz_list_fixture.items():
        data = np.zeros((len(pdz_data), 100), dtype=[
            ('A', np.float), ('B', np.float), ('C', np.float), ('D', np.float)
        ])
        for c in data.dtype.names:
            data[c] = np.random.rand(*data.shape)
        result[i] = ([obj for obj, _ in pdz_data], data)
    return result


##############################################################################

@pytest.fixture
def mc_data_files_fixture(temp_dir_fixture, mc_data_fixture):
    """
    Creates some MC data files to be used for testing

    Returns: A map with keys the file indices and values he path to the newly created files
    """
    result = {}
    for file_index, (_, mc_data) in mc_data_fixture.items():
        filename = os.path.join(temp_dir_fixture, 'mc_data_{}.npy'.format(file_index))
        np.save(filename, mc_data)
        result[file_index] = filename
    return result


##############################################################################

@pytest.fixture()
def reference_sample_dir_fixture(temp_dir_fixture,
                                 sed_data_files_fixture, sed_list_fixture,
                                 pdz_data_files_fixture, pdz_list_fixture,
                                 mc_data_files_fixture, mc_data_fixture):
    """
    Returns a directory which contains a reference sample
    """

    # Create the index files
    index_filename = os.path.join(temp_dir_fixture, 'index.npy')

    # Store on a dictionary each index individually
    index_dict = {
        'sed': dict(),
        'pdz': dict(),
        'mc': dict()
    }
    for file_index in sed_list_fixture:
        for pos, (obj_id, _) in enumerate(sed_list_fixture[file_index]):
            index_dict['sed'][obj_id] = (file_index, pos)

    for file_index in pdz_list_fixture:
        for pos, (obj_id, _) in enumerate(pdz_list_fixture[file_index]):
            index_dict['pdz'][obj_id] = (file_index, pos + 1)

    for file_index in mc_data_fixture:
        for pos, obj_id in enumerate(mc_data_fixture[file_index][0]):
            index_dict['mc'][obj_id] = (file_index, pos)

    # All known object IDS
    all_ids = set()
    for d in index_dict.values():
        all_ids.update(d.keys())

    # Allocate array with the full index
    fields = ['id']
    for prov in ['sed', 'pdz', 'mc']:
        fields.append(f'{prov}_file')
        fields.append(f'{prov}_offset')
    index = np.full(len(all_ids), -1, dtype=list(map(lambda c: (c, np.int64), fields)))

    # Fill and save it
    for i, obj_id in enumerate(all_ids):
        index['id'][i] = obj_id
        for prov in ['sed', 'pdz', 'mc']:
            index[f'{prov}_file'][i], index[f'{prov}_offset'][i] = index_dict[prov][obj_id]

    np.save(index_filename, index)
    return temp_dir_fixture

##############################################################################
