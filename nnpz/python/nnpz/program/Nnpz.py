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
import logging
from datetime import datetime
from typing import List

# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.output
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.target
import numpy as np
from nnpz.config.ConfigManager import ConfigManager
from nnpz.io import OutputHandler
from nnpz.photometry.photometry import Photometry
from nnpz.pipeline.compute_weights import ComputeWeights
from nnpz.pipeline.correct_photometry import CorrectPhotometry
from nnpz.pipeline.neighbor_finder import NeighborFinder
from nnpz.utils.ArgumentParserWrapper import ArgumentParserWrapper

logger = logging.getLogger(__name__)


# noinspection PyPep8Naming
def defineSpecificProgramOptions():
    """
    Program options. Returns a ArgumentParserWrapper that tricks Elements so we can
    capture anything extra and do the evaluation ourselves: NNPZ interpret flags
    as Python code
    """
    return ArgumentParserWrapper(description='Nearest Neighbor Photo-Z (neighbor search)')


# noinspection PyPep8Naming
def mainMethod(args):
    """
    Entry point for NNPZ

    Args:
        args: argparse.Namespace or similar
    """
    # Create the object which handles the user parameters
    conf_manager = ConfigManager(args.config_file, args.extra_arguments)

    # Read the target catalog data
    input_photometry: Photometry = conf_manager.get('target_photometry')
    id_col: str = conf_manager.get('target_id_column')

    # 1. Find
    neighbor_finder: NeighborFinder = NeighborFinder(conf_manager)
    ref_data: Photometry = conf_manager.get('reference_photometry')
    knn: int = conf_manager.get('neighbor_no')

    # 2. Correct
    corrector: CorrectPhotometry = CorrectPhotometry(conf_manager)

    # 3. Weight
    weighter: ComputeWeights = ComputeWeights(conf_manager)

    # 4. Output
    output_handler: OutputHandler = conf_manager.get('output_handler')
    output_handler.initialize()

    # Dtype for work area
    neighbor_columns = [
        (id_col[0], id_col[1]),
        ('NEIGHBOR_INDEX', np.int64, knn),
        ('NEIGHBOR_SCALING', np.float32, knn),
        ('NEIGHBOR_WEIGHTS', np.float32, knn),
        ('FLAGS', np.uint32),
        ('NEIGHBOR_PHOTOMETRY', np.double, (knn, len(ref_data.system), 2)),
    ]

    # Chunks
    chunks: List[slice] = conf_manager.get('target_idx_slices')

    # Process in chunks
    start = datetime.utcnow()
    for i, chunk in enumerate(chunks, start=1):
        logger.info('Processing chunk %d / %d', i, len(chunks))

        logger.info('Finding neighbors')
        chunk_photometry = input_photometry[chunk]

        workarea = np.zeros(len(chunk_photometry), dtype=neighbor_columns)
        workarea[id_col[0]] = chunk_photometry.ids
        workarea['NEIGHBOR_WEIGHTS'] = 1.

        neighbor_finder(chunk_photometry, out=workarea)

        logger.info('Correcting photometry')
        chunk_ref_photo = workarea['NEIGHBOR_PHOTOMETRY'] * ref_data.unit
        corrector(chunk_photometry, workarea['NEIGHBOR_INDEX'], chunk_ref_photo,
                  out=chunk_ref_photo)

        logger.info('Weighting sources')
        weighter(chunk_photometry, chunk_ref_photo, workarea['NEIGHBOR_SCALING'],
                 out_weights=workarea['NEIGHBOR_WEIGHTS'], out_flags=workarea['FLAGS'])

        logger.info('Generating output')
        output_handler.write_output_for(chunk, workarea)

    logger.info('Appending additional HDUs')
    output_handler.write_additional_hduls()

    end = datetime.utcnow()
    duration = end - start
    logger.info('Finished in %s (%.2f sources / second)', duration,
                chunks[-1].stop if chunks else 0 / duration.total_seconds())
