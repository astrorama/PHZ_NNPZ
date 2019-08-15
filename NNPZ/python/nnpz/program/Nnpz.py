"""
Created on: 01/02/18
Author: Nikolaos Apostolakos
"""

from ElementsKernel import Logging
from nnpz.config import ConfigManager
from nnpz.framework import *
from nnpz.program.ArgumentParserWrapper import ArgumentParserWrapper

# Trigger the configuration of the NNPZ pipeline
import nnpz.config.nnpz
import nnpz.config.reference


def defineSpecificProgramOptions():
    return ArgumentParserWrapper(description='Nearest Neighbor Photo-Z')


def mainMethod(args):
    logger = Logging.getLogger(__name__)

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

    # Check if the Galactic reddening handling is on if so create de-reddend data
    if conf_manager.getObject('apply_galactic_absorption'):
        logger.info('Using Galactic reddening correction.')
        # Get the instance for de-redden the targets
        galacticAbsorptionCorrector = conf_manager.getObject('galactic_absorption_unreddener')

        # Get de-reddened data
        de_reddened_target_data = galacticAbsorptionCorrector.de_redden_data(target_data, target_ebv)
    else:
        de_reddened_target_data = target_data

    # Construct the neighbor finder and build the affected sources map
    finder = AffectedSourcesFinder(selector)
    progress_listener = ProgressListener(len(target_data) - 1, 'Finding neighbors... ', logger=logger)
    affected = finder.findAffected(de_reddened_target_data, result_flags, progress_listener)

    # Compute the weights
    progress_listener = ProgressListener(len(affected) - 1, 'Computing neighbor weights...', logger=logger)

    weight_calculator = conf_manager.getObject('weight_calculator')
    weights = weight_calculator.computeWeights(affected, target_data, result_flags, progress_listener)

    # Get the output handler
    output = conf_manager.getObject('output_handler')

    # Loop through the maps and add the contributions to the output
    # Note that we iterate the affected map in increasing order of the reference
    # sample indices. This is done to use as much the cache of the disk, by accessing
    # the PDZs sequentially.
    progress_listener = ProgressListener(len(affected) - 1, 'Adding contributions to output...', logger=logger)
    for progress, ref_i in enumerate(sorted(affected)):
        progress_listener(progress)
        for target_i, w in zip(affected[ref_i], weights[ref_i]):
            output.addContribution(ref_i, target_i, w, result_flags[target_i])

    # Create the output catalog
    output_file = conf_manager.getObject('output_file')
    output.save(output_file)
    logger.info('Created file {}'.format(output_file))
