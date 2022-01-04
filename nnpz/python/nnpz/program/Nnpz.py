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
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""

# Trigger the configuration of the NNPZ pipeline
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.nnpz
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.reference
from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.framework import AffectedSourcesFinder, ProgressListener
from nnpz.utils.ArgumentParserWrapper import ArgumentParserWrapper


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
    logger = Logging.getLogger('NNPZ')

    # Create the object which handles the user parameters
    conf_manager = ConfigManager(args.config_file, args.extra_arguments)

    # Read the reference sample data
    ref_data = conf_manager.getObject('reference_phot_data')

    # Read the target catalog data
    target_data = conf_manager.getObject('target_phot_data')
    target_ebv = conf_manager.getObject('target_ebv')

    # Get the neighbor selector and initialize it
    selector = conf_manager.getObject('neighbor_selector').initialize(ref_data)

    # Get the flag list
    result_flags = conf_manager.getObject('flag_list')

    # Get the output handler
    output = conf_manager.getObject('output_handler')

    # Check if the Galactic reddening handling is on if so create de-reddend data
    if conf_manager.getObject('apply_galactic_absorption'):
        logger.info('Using Galactic reddening correction.')
        # Get the instance for de-redden the targets
        galactic_absorption_corrector = conf_manager.getObject('galactic_absorption_unreddener')

        # Get de-reddened data
        de_reddened_target_data = galactic_absorption_corrector.de_redden_data(
            target_data, target_ebv
        )
    else:
        de_reddened_target_data = target_data

    # Construct the neighbor finder and build the affected sources map
    finder = AffectedSourcesFinder(selector)

    weight_calculator = conf_manager.getObject('weight_calculator')

    # Output
    logger.info('Allocating output')
    output.initialize(len(target_data))

    overall_progress_listener = ProgressListener(
        len(target_data), 'Finding neighbors... ', logger=logger
    )

    # Split the input in chunks to keep memory under control
    chunk_size = conf_manager.getObject('target_chunk_size')
    nchunks, remainder = divmod(len(de_reddened_target_data), chunk_size)
    if remainder > 0:
        nchunks += 1

    logger.info('Running over %d chunks', nchunks)

    for chunk in range(nchunks):
        offset = chunk * chunk_size
        logger.info('Chunk %d / %d', chunk + 1, nchunks)
        chunk_data = de_reddened_target_data[offset:offset + chunk_size]

        affected = finder.findAffected(chunk_data, result_flags, overall_progress_listener,
                                       offset=offset)

        # Compute the weights
        progress_listener = ProgressListener(
            len(affected), '\tComputing neighbor weights...', logger=logger
        )

        weight_calculator.computeWeights(affected, target_data, result_flags, progress_listener)

        # Loop through the maps and add the contributions to the output
        # Note that we iterate the affected map in increasing order of the reference
        # sample indices. This is done to use as much the cache of the disk, by accessing
        # the PDZs sequentially.
        progress_listener = ProgressListener(
            len(affected) - 1, '\tAdding contributions to output...', logger=logger
        )

        for progress, ref_i in enumerate(sorted(affected)):
            progress_listener(progress)
            for target in affected[ref_i]:
                output.addContribution(ref_i, target, result_flags[target.index])

    # Create the output catalog
    output_file = conf_manager.getObject('output_file')
    output.save(output_file)
    logger.info('Created file %s', output_file)
    return 0
