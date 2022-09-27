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

from datetime import datetime
from typing import List

import astropy.units as u
import fitsio
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.pipeline.NeighborsCatalogConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.target
import numpy as np
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.pipeline.compute_weights import ComputeWeights
from nnpz.utils.ArgumentParserWrapper import ArgumentParserWrapper

logger = Logging.getLogger(__name__)


# noinspection PyPep8Naming
def defineSpecificProgramOptions():
    """
    Program options. Returns a ArgumentParserWrapper that tricks Elements so we can
    capture anything extra and do the evaluation ourselves: NNPZ interpret flags
    as Python code
    """
    return ArgumentParserWrapper(description='Nearest Neighbor Photo-Z (photometry correction)')


# noinspection PyPep8Naming
def mainMethod(args):
    """
    Entry point for ComputeWeights

    Args:
        args: argparse.Namespace or similar
    """

    # Create the object which handles the user parameters
    conf_manager = ConfigManager(args.config_file, args.extra_arguments)

    # Open the target catalog
    input_photometry = conf_manager.get('target_photometry')

    # Open the output catalog
    neighbor_catalog = conf_manager.get('neighbor_catalog')
    output_fits = fitsio.FITS(neighbor_catalog, mode='rw', clobber=False)
    output_hdu: fitsio.hdu.TableHDU = output_fits[1]
    output_colnames = output_hdu.get_colnames()
    output_header = output_hdu.read_header()

    assert 'NEIGHBOR_INDEX' in output_colnames
    assert 'NEIGHBOR_SCALING' in output_colnames
    assert 'NEIGHBOR_WEIGHTS' in output_colnames
    assert 'NEIGHBOR_PHOTOMETRY' in output_colnames

    neighbor_photo_idx = output_colnames.index('NEIGHBOR_PHOTOMETRY') + 1
    photo_unit = u.Unit(output_header.get(f'TUNIT{neighbor_photo_idx}'))
    assert photo_unit == u.uJy

    # Chunks
    chunks: List[slice] = conf_manager.get('target_idx_slices')

    # Weighter
    weighter = ComputeWeights(conf_manager)

    # Process in chunks
    start = datetime.utcnow()
    for i, chunk in enumerate(chunks, start=1):
        logger.info('Processing chunk %d / %d', i, len(chunks))
        chunk_target = input_photometry[chunk]
        chunk_workarea = output_hdu.read(['NEIGHBOR_PHOTOMETRY', 'NEIGHBOR_SCALING', 'FLAGS'],
                                         rows=range(chunk.start, chunk.stop))

        out_weights = np.zeros_like(chunk_workarea['NEIGHBOR_SCALING'], dtype=np.float32)
        out_flags = np.zeros(len(chunk_workarea), dtype=np.uint32)

        weighter(chunk_target, chunk_workarea['NEIGHBOR_INDEX'],
                 chunk_workarea['NEIGHBOR_PHOTOMETRY'] * photo_unit,
                 chunk_workarea['NEIGHBOR_SCALING'],
                 out_weights=out_weights, out_flags=out_flags)

        output_hdu.write([out_weights, out_flags], names=['NEIGHBOR_WEIGHTS', 'FLAGS'],
                         firstrow=chunk.start)

    end = datetime.utcnow()
    duration = end - start
    logger.info('Finished in %s (%.2f sources / second)', duration,
                len(input_photometry) / duration.total_seconds())

    del output_hdu
    output_fits.close()

    return 0
