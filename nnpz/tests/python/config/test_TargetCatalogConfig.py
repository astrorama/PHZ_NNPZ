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

from ElementsKernel.Temporary import TempFile
from nnpz.config.ConfigManager import ConfigManager, _handler_map

# noinspection PyUnresolvedReferences
from ..reference_sample.fixtures import *


@pytest.fixture
def target_file_fixture():
    temp = NamedTemporaryFile(suffix='.fits')
    Table(data={
        'ID': np.arange(10, dtype=int),
        'FLUX_VIS': np.random.uniform(0, 100, size=10),
        'FLUXERR_VIS': np.random.uniform(0, 1, size=10),
        'FLUX_Y': np.random.uniform(100, 200, size=10),
        'FLUXERR_Y': np.random.uniform(1, 2, size=10),
        'FLUX_EXT_G': np.random.uniform(200, 300, size=10),
        'FLUXERR_EXT_G': np.random.uniform(2, 3, size=10),
    }).write(temp.name, overwrite=True)
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
""", file=cfg)
    cfg.flush()
    return cfg


###############################################################################

def test_config_target(config_file):
    _handler_map.clear()

    # noinspection PyUnresolvedReferences
    from nnpz.config.reference import ReferenceSampleConfig
    # noinspection PyUnresolvedReferences
    from nnpz.config.target import TargetCatalogConfig

    config = ConfigManager(config_file.name, extra_arguments=[])
    raise (Exception(str(config.get_available_object_list())))


###############################################################################

def test_config_target_subset():
    pass

###############################################################################
