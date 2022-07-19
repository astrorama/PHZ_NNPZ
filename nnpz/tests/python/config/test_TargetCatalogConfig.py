#
#  Copyright (C) 2022 Euclid Science Ground Segment
#
#  This library is free software; you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation;
#  either version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with this library;
#  if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA
#
from tempfile import NamedTemporaryFile

from nnpz.config.ConfigManager import ConfigManager, _handler_map
# noinspection PyUnresolvedReferences
from nnpz.config.reference import ReferenceConfig, ReferenceSampleConfig, \
    ReferenceSamplePhotometryConfig
# noinspection PyUnresolvedReferences
from nnpz.config.target import TargetCatalogConfig
# noinspection PyUnresolvedReferences
from nnpz.photometry.photometry import Photometry

from ..reference_sample.fixtures import *


@pytest.fixture
def target_data_fixture():
    return Table(data={
        'ID': np.arange(10, dtype=int),
        'FLUX_VIS': np.random.uniform(0, 100, size=10),
        'FLUXERR_VIS': np.random.uniform(0, 1, size=10),
        'FLUX_Y': np.random.uniform(100, 200, size=10),
        'FLUXERR_Y': np.random.uniform(1, 2, size=10),
        'FLUX_EXT_G': np.random.uniform(200, 300, size=10),
        'FLUXERR_EXT_G': np.random.uniform(2, 3, size=10),
    })


@pytest.fixture
def target_file_fixture(target_data_fixture):
    temp = NamedTemporaryFile(suffix='.fits')
    target_data_fixture.write(temp.name, overwrite=True)
    return temp


@pytest.fixture
def config_file(photometry_file_fixture, reference_sample_dir_fixture, target_file_fixture):
    cfg = NamedTemporaryFile(mode='w+t', suffix='.conf')
    print(f"""
reference_sample_dir="{reference_sample_dir_fixture}"
reference_sample_phot_file="{photometry_file_fixture}"
reference_sample_phot_filters = [
    'Y', 'vis', 'g',
]
target_catalog="{target_file_fixture.name}"
target_catalog_filters=[
    ('FLUX_Y', 'FLUXERR_Y'), ('FLUX_VIS', 'FLUXERR_VIS'), ('FLUX_EXT_G', 'FLUXERR_EXT_G')
]
foo="abc"
""", file=cfg)
    cfg.flush()
    return cfg


###############################################################################

@pytest.fixture
def config_manager_fixture():
    _handler_map.clear()
    ConfigManager.add_handler(ReferenceConfig)
    ConfigManager.add_handler(ReferenceSampleConfig)
    ConfigManager.add_handler(ReferenceSamplePhotometryConfig)
    ConfigManager.add_handler(TargetCatalogConfig)


###############################################################################

def test_config_target(config_file, target_data_fixture, photometry_data_fixture,
                       config_manager_fixture):
    config = ConfigManager(config_file.name, extra_arguments=[])

    ref_photo: Photometry = config.get('reference_photometry')
    target_photo: Photometry = config.get('target_photometry')

    assert set(ref_photo.system.bands) == set(target_photo.system.bands)
    assert set(ref_photo.system.bands) == {'Y', 'g', 'vis'}

    target_values = target_photo.get_fluxes(['Y', 'g', 'vis'], return_error=True).value
    np.testing.assert_array_equal(target_values[:, 0, 0], target_data_fixture['FLUX_Y'])
    np.testing.assert_array_equal(target_values[:, 0, 1], target_data_fixture['FLUXERR_Y'])
    np.testing.assert_array_equal(target_values[:, 1, 0], target_data_fixture['FLUX_EXT_G'])
    np.testing.assert_array_equal(target_values[:, 1, 1], target_data_fixture['FLUXERR_EXT_G'])
    np.testing.assert_array_equal(target_values[:, 2, 0], target_data_fixture['FLUX_VIS'])
    np.testing.assert_array_equal(target_values[:, 2, 1], target_data_fixture['FLUXERR_VIS'])

    ref_values = ref_photo.get_fluxes(['Y', 'g', 'vis'], return_error=True).value
    np.testing.assert_array_equal(ref_values[:, 0], photometry_data_fixture['Y'])
    np.testing.assert_array_equal(ref_values[:, 1], photometry_data_fixture['g'])
    np.testing.assert_array_equal(ref_values[:, 2], photometry_data_fixture['vis'])


###############################################################################

def test_config_target_subset(config_file, target_data_fixture, photometry_data_fixture,
                              config_manager_fixture):
    config = ConfigManager(config_file.name, extra_arguments=['--enable_filters="vi?;Y"'])

    ref_photo: Photometry = config.get('reference_photometry')
    target_photo: Photometry = config.get('target_photometry')

    assert set(ref_photo.system.bands) == {'Y', 'vis', 'g'}
    assert set(target_photo.system.bands) == {'Y', 'vis'}

    target_values = target_photo.get_fluxes(['Y', 'vis'], return_error=True).value
    np.testing.assert_array_equal(target_values[:, 0, 0], target_data_fixture['FLUX_Y'])
    np.testing.assert_array_equal(target_values[:, 0, 1], target_data_fixture['FLUXERR_Y'])
    np.testing.assert_array_equal(target_values[:, 1, 0], target_data_fixture['FLUX_VIS'])
    np.testing.assert_array_equal(target_values[:, 1, 1], target_data_fixture['FLUXERR_VIS'])

    ref_values = ref_photo.get_fluxes(['Y', 'vis'], return_error=True).value
    np.testing.assert_array_equal(ref_values[:, 0], photometry_data_fixture['Y'])
    np.testing.assert_array_equal(ref_values[:, 1], photometry_data_fixture['vis'])

###############################################################################
