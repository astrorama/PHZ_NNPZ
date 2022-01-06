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

import fitsio
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.neighbors.GalacticUnreddenerConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.neighbors.NeighborSelectorConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.pipeline.NeighborsCatalogConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.reference
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.target
import numpy as np
from ElementsKernel import Logging
from nnpz.config.ConfigManager import ConfigManager
from nnpz.utils.ArgumentParserWrapper import ArgumentParserWrapper

logger = Logging.getLogger(__name__)


def defineSpecificProgramOptions():
    """
    Program options. Returns a ArgumentParserWrapper that tricks Elements so we can
    capture anything extra and do the evaluation ourselves: NNPZ interpret flags
    as Python code
    """
    return ArgumentParserWrapper(description='Nearest Neighbor Photo-Z')


def mainMethod(args):
    """
    Entry point for NNPZ

    Args:
        args: argparse.Namespace or similar
    """

    # Create the object which handles the user parameters
    conf_manager = ConfigManager(args.config_file, args.extra_arguments)

    # Read the reference sample data
    ref_data = conf_manager.getObject('reference_photometry')

    # Read the target catalog data
    input_photometry = conf_manager.getObject('target_photometry')
    id_col = conf_manager.getObject('target_id_column')

    # De-redden
    source_independent_ebv = conf_manager.getObject('source_independent_ebv')

    # Finder
    selector = conf_manager.getObject('neighbor_selector')
    knn = conf_manager.getObject('neighbor_no')
    selector.fit(ref_data, input_photometry.system)

    # Prepare the output catalog
    neighbor_catalog = conf_manager.getObject('neighbor_catalog')
    neighbor_columns = [
        (id_col[0], id_col[1]),
        ('NEIGHBOR_INDEX', np.int64, knn),
        ('NEIGHBOR_SCALING', np.float32, knn),
        # To be filled later
        ('NEIGHBOR_WEIGHTS', np.float, knn),
    ]

    # Reserve photometry columns, also to be filled later
    for filter_name in ref_data.system.bands:
        neighbor_columns.append((filter_name, ref_data.values.dtype, (knn, 2)))

    output_fits = fitsio.FITS(neighbor_catalog, mode='rw', clobber=True)
    output_fits.create_table_hdu(
        dtype=neighbor_columns,
        units=[''] * 4 + [str(ref_data.unit)] * len(ref_data.system)
    )
    output_hdu = output_fits[-1]

    # Chunk size
    chunk_size = conf_manager.getObject('target_chunk_size')
    nchunks, remainder = divmod(len(input_photometry), chunk_size)
    nchunks += remainder > 0

    # Process in chunks
    start = datetime.utcnow()
    for chunk in range(nchunks):
        logger.info('Processing chunk %d / %d', chunk + 1, nchunks)
        offset = chunk * chunk_size
        chunk_photometry = input_photometry[offset:offset + chunk_size]

        if source_independent_ebv:
            logger.info('Deredden')
            chunk_photometry.values = source_independent_ebv.deredden(chunk_photometry.values,
                                                                      ebv=chunk_photometry.colorspace.ebv)

        logger.info('Looking for neighbors')
        all_idx, all_scales = selector.query(chunk_photometry)
        output = np.zeros(len(chunk_photometry), dtype=neighbor_columns)

        output[id_col[0]] = chunk_photometry.ids
        output['NEIGHBOR_INDEX'] = all_idx
        output['NEIGHBOR_SCALING'] = all_scales
        output['NEIGHBOR_WEIGHTS'] = 1.

        for filter_name in ref_data.system.bands:
            input_area = ref_data.get_fluxes(filter_name, return_error=True).value
            output[filter_name] = input_area[all_idx]

        logger.info('Writing chunk into the output catalog')
        output_hdu.append(output)

    del output_hdu
    output_fits.close()

    end = datetime.utcnow()
    duration = end - start
    logger.info('Finished in %s (%.2f sources / second)', duration,
                len(input_photometry) / duration.total_seconds())
    return 0
