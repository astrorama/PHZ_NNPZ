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
from typing import List, Tuple

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
import numpy as np
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.io import OutputHandler
from nnpz.utils.ArgumentParserWrapper import ArgumentParserWrapper

logger = Logging.getLogger('GenerateOutput')


def patchOutputCatalog(conf_manager: ConfigManager, table: fitsio.hdu.TableHDU) -> Tuple[
    np.ndarray, List[str]]:
    """
    Patch a "regular" nnpz output catalog so it can be used as a neighbor catalog
    """
    knn = table['NEIGHBOR_IDS'][0].shape[0]
    id_col = conf_manager.get('target_id_column')
    neighbor_columns = [
        (id_col[0], id_col[1]),
        ('NEIGHBOR_INDEX', np.int64, knn),
        ('NEIGHBOR_SCALING', np.float32, knn),
        ('NEIGHBOR_WEIGHTS', np.float32, knn),
        ('FLAGS', np.uint32)
    ]
    neighbors = np.zeros(table.get_nrows(), dtype=neighbor_columns)
    # IDs => Index
    ref_ids = conf_manager.get('reference_ids')
    ids = table.read_column('NEIGHBOR_IDS')
    sorter = np.argsort(ref_ids)
    neighbors['NEIGHBOR_INDEX'] = sorter[np.searchsorted(ref_ids, ids, sorter=sorter)]
    # Copy the rest
    for col in ['NEIGHBOR_SCALING', 'NEIGHBOR_WEIGHTS', 'FLAGS']:
        ref_val = table[col][:]
        try:
            np.copyto(neighbors[col], ref_val, casting='same_kind')
        except TypeError:
            logger.warning('Failed a safe cast for %s from %s to %s, doing an unsafe cast!', col,
                           ref_val.dtype, neighbors[col].dtype)
            np.copyto(neighbors[col], table[col][:], casting='unsafe')
    return neighbors, neighbors.dtype.names


# noinspection PyPep8Naming
def defineSpecificProgramOptions():
    """
    Program options. Returns a ArgumentParserWrapper that tricks Elements, so we can
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
    conf_manager = ConfigManager(args.config_file, args.extra_arguments)

    # Open the output catalog
    neighbor_catalog = conf_manager.get('neighbor_catalog')
    neighbor_fits = fitsio.FITS(neighbor_catalog, mode='r')
    neighbor_hdu: fitsio.hdu.TableHDU = neighbor_fits[1]
    neighbor_colnames = neighbor_hdu.get_colnames()

    # Chunks
    chunks: List[slice] = conf_manager.get('target_idx_slices')

    if chunks[-1].stop != neighbor_hdu.get_nrows():
        logger.error('The intermediate and target catalogs have different lengths!')
        logger.error('Please, re-run with --input_size=%d', neighbor_hdu.get_nrows())
        return 1

    # Patch catalog if needed
    if 'NEIGHBOR_IDS' in neighbor_colnames and 'NEIGHBOR_INDEX' not in neighbor_colnames:
        logger.warning('Patching catalog in-memory!')
        neighbor_hdu, neighbor_colnames = patchOutputCatalog(conf_manager, neighbor_hdu)

    assert 'NEIGHBOR_INDEX' in neighbor_colnames
    assert 'NEIGHBOR_SCALING' in neighbor_colnames
    assert 'NEIGHBOR_WEIGHTS' in neighbor_colnames

    if 'NEIGHBOR_PHOTOMETRY' not in neighbor_colnames:
        logger.warning('NEIGHBOR_PHOTOMETRY not available on the input, '
                       'the computation of some properties may fail')

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
