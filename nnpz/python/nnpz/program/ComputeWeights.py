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

# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
from datetime import datetime

import astropy.units as u
import fitsio
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.neighbors.WeightConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.neighbors.WeightConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.pipeline.NeighborsCatalogConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.target
import numpy as np
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.utils.ArgumentParserWrapper import ArgumentParserWrapper

logger = Logging.getLogger(__name__)


def defineSpecificProgramOptions():
    """
    Program options. Returns a ArgumentParserWrapper that tricks Elements so we can
    capture anything extra and do the evaluation ourselves: NNPZ interpret flags
    as Python code
    """
    return ArgumentParserWrapper(description='Nearest Neighbor Photo-Z (photometry correction)')


def mainMethod(args):
    """
    Entry point for ComputeWeights

    Args:
        args: argparse.Namespace or similar
    """

    # Create the object which handles the user parameters
    conf_manager = ConfigManager(args.config_file, args.extra_arguments)

    # Reference photometry
    ref_photometry = conf_manager.getObject('reference_photometry')

    # Open the target catalog
    input_photometry = conf_manager.getObject('target_photometry')

    # Open the output catalog
    neighbor_catalog = conf_manager.getObject('neighbor_catalog')
    output_fits = fitsio.FITS(neighbor_catalog, mode='rw', clobber=False)
    output_hdu: fitsio.hdu.TableHDU = output_fits[1]

    assert 'NEIGHBOR_INDEX' in output_hdu.get_colnames()
    assert 'NEIGHBOR_SCALING' in output_hdu.get_colnames()
    assert 'NEIGHBOR_WEIGHTS' in output_hdu.get_colnames()
    assert 'NEIGHBOR_PHOTOMETRY' in output_hdu.get_colnames()

    # Chunks
    chunks: slice = conf_manager.getObject('target_idx_slices')

    # Weight calculator
    weight_calculator = conf_manager.getObject('weight_calculator')

    # Indexes
    ref_filter_indexes = ref_photometry.system.get_band_indexes(input_photometry.system.bands)

    # Process in chunks
    start = datetime.utcnow()
    for i, chunk in enumerate(chunks, start=1):
        logger.info('Processing chunk %d / %d', i, len(chunks))
        chunk_output = output_hdu[chunk]
        chunk_target = input_photometry[chunk]

        for i in range(len(chunk_output)):
            neighbor_photo = chunk_output['NEIGHBOR_PHOTOMETRY'][i, :, ref_filter_indexes] * u.uJy
            neighbor_photo *= chunk_output['NEIGHBOR_SCALING'][i, np.newaxis].T
            chunk_output['NEIGHBOR_WEIGHTS'][i], flag = weight_calculator(
                neighbor_photo, chunk_target.values[i])
            chunk_output[i]['FLAGS'] |= flag

        output_hdu.write(chunk_output, firstrow=chunk.start)

    end = datetime.utcnow()
    duration = end - start
    logger.info('Finished in %s (%.2f sources / second)', duration,
                len(input_photometry) / duration.total_seconds())

    del output_hdu
    output_fits.close()

    return 0
