#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
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

from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.framework import ProgressListener, AffectedSourcesReconstructor
from nnpz.program.ArgumentParserWrapper import ArgumentParserWrapper

# Trigger the configuration of the NNPZ pipeline
import nnpz.config.reference
import nnpz.config.reconstruction


def defineSpecificProgramOptions():
    return ArgumentParserWrapper(description='Reconstruct PDZ from a NNPZ result catalog')


def mainMethod(args):
    logger = Logging.getLogger('ReconstructPDZ')

    conf_manager = ConfigManager(args.config_file, args.extra_arguments)

    # Read the reference sample data
    ref_ids = conf_manager.getObject('reference_ids')

    # Read NNPZ output
    nnpz_idx = conf_manager.getObject('nnpz_idx')
    nnpz_neighbors = conf_manager.getObject('nnpz_neighbors')
    nnpz_weights = conf_manager.getObject('nnpz_weights')

    progress_listener = ProgressListener(len(nnpz_idx) - 1, 'Reconstructing affected table from neighbors list...',
                                         logger=logger)
    affected, weights = AffectedSourcesReconstructor() \
        .reconstruct(ref_ids, nnpz_idx, nnpz_neighbors, nnpz_weights, progress_listener)

    # This stage is identical to nnpz, iterating the affected map in increasing order of the reference
    # sample indices.
    output = conf_manager.getObject('pdz_output_handler')

    progress_listener = ProgressListener(len(affected) - 1, 'Adding contributions to output...', logger=logger)
    for progress, ref_i in enumerate(sorted(affected)):
        progress_listener(progress)
        for target_i, w in zip(affected[ref_i], weights[ref_i]):
            output.addContribution(ref_i, target_i, w, None)

    # Create the output catalog
    output_file = conf_manager.getObject('pdz_output_file')
    output.save(output_file)
    logger.info('Created file {}'.format(output_file))
