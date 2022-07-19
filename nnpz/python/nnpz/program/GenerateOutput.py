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

"""
Created on: 27/05/19
Author: Alejandro Alvarez Ayllon
"""
from datetime import datetime
from typing import List

import fitsio
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.output
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.pipeline.NeighborsCatalogConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.reference
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.io import OutputHandler
from nnpz.utils.ArgumentParserWrapper import ArgumentParserWrapper


# noinspection PyPep8Naming
def defineSpecificProgramOptions():
    """
    Program options. Returns a ArgumentParserWrapper that tricks Elements so we can
    capture anything extra and do the evaluation ourselves: NNPZ interpret flags
    as Python code
    """
    return ArgumentParserWrapper(description='Reconstruct PDZ from a NNPZ result catalog')


# noinspection PyPep8Naming
def mainMethod(args):
    """
    Entry point for ReconstructPDZ

    Args:
        args: argparse.Namespace or similar
    """
    logger = Logging.getLogger('ReconstructPDZ')

    conf_manager = ConfigManager(args.config_file, args.extra_arguments)

    # Open the output catalog
    neighbor_catalog = conf_manager.get('neighbor_catalog')
    neighbor_fits = fitsio.FITS(neighbor_catalog, mode='r')
    neighbor_hdu: fitsio.hdu.TableHDU = neighbor_fits[1]
    neighbor_colnames = neighbor_hdu.get_colnames()

    assert 'NEIGHBOR_INDEX' in neighbor_colnames
    assert 'NEIGHBOR_SCALING' in neighbor_colnames
    assert 'NEIGHBOR_WEIGHTS' in neighbor_colnames
    assert 'NEIGHBOR_PHOTOMETRY' in neighbor_colnames

    # Chunks
    chunks: List[slice] = conf_manager.get('target_idx_slices')

    # Prepare the output
    output: OutputHandler = conf_manager.get('output_handler')
    output.initialize()

    # Process in chunks
    start = datetime.utcnow()
    for i, chunk in enumerate(chunks, start=1):
        logger.info('Processing chunk %d / %d', i, len(chunks))
        chunk_neighbor = neighbor_hdu[chunk]
        output.write_output_for(chunk, chunk_neighbor)

    output.write_additional_hduls()

    end = datetime.utcnow()
    duration = end - start
    logger.info('Finished in %s (%.2f sources / second)', duration,
                (chunks[-1].stop if chunks else 0) / duration.total_seconds())

    return 0
