#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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
from datetime import datetime
from typing import List

import fitsio
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.pipeline.NeighborsCatalogConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.target
import numpy as np
from ElementsKernel import Logging
from nnpz.config.ConfigManager import ConfigManager
from nnpz.pipeline.neighbor_finder import NeighborFinder
from nnpz.utils.ArgumentParserWrapper import ArgumentParserWrapper

logger = Logging.getLogger(__name__)


def defineSpecificProgramOptions():
    """
    Program options. Returns a ArgumentParserWrapper that tricks Elements so we can
    capture anything extra and do the evaluation ourselves: NNPZ interpret flags
    as Python code
    """
    return ArgumentParserWrapper(description='Nearest Neighbor Photo-Z (neighbor search)')


def mainMethod(args):
    """
    Entry point for ComputeNeighbors

    Args:
        args: argparse.Namespace or similar
    """

    # Create the object which handles the user parameters
    conf_manager = ConfigManager(args.config_file, args.extra_arguments)

    # Finder
    neighbor_finder = NeighborFinder(conf_manager)
    ref_data = conf_manager.getObject('reference_photometry')
    knn = conf_manager.getObject('neighbor_no')

    # Read the target catalog data
    input_photometry = conf_manager.getObject('target_photometry')
    id_col = conf_manager.getObject('target_id_column')

    # Prepare the output catalog
    neighbor_catalog = conf_manager.getObject('neighbor_catalog')
    neighbor_columns = [
        (id_col[0], id_col[1]),
        ('NEIGHBOR_INDEX', np.int64, knn),
        ('NEIGHBOR_SCALING', np.float32, knn),
        ('NEIGHBOR_WEIGHTS', np.float32, knn),
        ('FLAGS', np.uint32),
        ('NEIGHBOR_PHOTOMETRY', np.double, (knn, len(ref_data.system), 2)),
    ]

    output_fits = fitsio.FITS(neighbor_catalog, mode='rw', clobber=True)
    output_fits.create_table_hdu(
        dtype=neighbor_columns,
        units=[''] * 5 + [str(ref_data.unit)]
    )
    output_hdu = output_fits[-1]

    # Chunks
    chunks: List[slice] = conf_manager.getObject('target_idx_slices')

    # Process in chunks
    start = datetime.utcnow()
    for i, chunk in enumerate(chunks, start=1):
        logger.info('Processing chunk %d / %d', i, len(chunks))
        chunk_photometry = input_photometry[chunk]

        output = np.zeros(len(chunk_photometry), dtype=neighbor_columns)
        output[id_col[0]] = chunk_photometry.ids
        output['NEIGHBOR_WEIGHTS'] = 1.

        neighbor_finder(chunk_photometry, out=output)

        logger.info('Writing chunk into the output catalog')
        output_hdu.append(output)

    del output_hdu
    output_fits.close()

    end = datetime.utcnow()
    duration = end - start
    logger.info('Finished in %s (%.2f sources / second)', duration,
                len(input_photometry) / duration.total_seconds())
    return 0
