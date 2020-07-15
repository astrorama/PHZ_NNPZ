"""
Created on: 17/12/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import argparse
import os
import numpy as np
from astropy.io import fits
from astropy import table

from ElementsKernel import Logging
from nnpz.framework import ProgressListener

from nnpz.photometry import *
from nnpz.utils import Fits
from nnpz import ReferenceSample

logger = Logging.getLogger('BuildPhotometry')


def defineSpecificProgramOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-dir', dest='sample_dir', type=str, metavar='DIR', required=True,
                        help='The reference sample directory')
    parser.add_argument('--filters', dest='filters', type=str, metavar='DIR | LIST_FILE', required=True,
                        help='The directory containing the filter transmissions')
    parser.add_argument('--out-type', dest='type', type=str, metavar=' | '.join(PhotometryTypeMap), required=True,
                        help='The type of the photometry to create')
    parser.add_argument('--out-file', dest='out', type=str, metavar='FILE', required=True,
                        help='The output FITS file to create the photometry in')
    parser.add_argument('--gal-ebv', dest='gal_ebv', type=float,
                        help='The E(B-V) value of the galactic absorption to apply to the SEDs')
    parser.add_argument('--parallel', dest='parallel', type=int, default=None,
                        help='Number of parallel processes to spawn')
    return parser


def mainMethod(args):
    # Validate the user inputs
    if not args.type in PhotometryTypeMap:
        logger.error('ERROR: Invalid photometry type. Available types:')
        for t in PhotometryTypeMap:
            logger.error('   ' + t + ' : ' + PhotometryTypeMap[t][1])
        return 1

    if os.path.exists(args.out):
        logger.error('ERROR: File ' + args.out + ' already exists')
        return 1
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.exists(out_dir):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)))

    # Open the reference sample for reading
    logger.info('')
    logger.info('Opening reference sample ' + args.sample_dir + '...')
    sample = ReferenceSample(args.sample_dir)
    logger.info('Successfully opened reference sample')

    # Read the filter transmissions
    logger.info('')
    logger.info('Reading filter transmissions from ' + args.filters + '...')
    if os.path.isdir(args.filters):
        filter_provider = DirectoryFilterProvider(args.filters)
    else:
        filter_provider = ListFileFilterProvider(args.filters)
    filter_name_list = filter_provider.getFilterNames()
    filter_map = {}
    for filter_name in filter_name_list:
        logger.info('    ' + filter_name)
        filter_map[filter_name] = filter_provider.getFilterTransmission(filter_name)
    logger.info('Successfully read filter transmissions')

    # Create the photometry builder to use for computing the photometry values
    pre_post_processor = PhotometryTypeMap[args.type][0]()
    if not args.gal_ebv is None:
        pre_post_processor = GalacticReddeningPrePostProcessor(pre_post_processor, args.gal_ebv)

    if args.parallel:
        phot_builder = ReferenceSamplePhotometryParallelBuilder(filter_provider, pre_post_processor, args.parallel)
    else:
        phot_builder = ReferenceSamplePhotometryBuilder(filter_provider, pre_post_processor)

    phot_builder.setFilters(filter_name_list)

    # Compute the photometry values
    logger.info('')
    logger.info('Computing photometry values...')
    logger.info('    Photometry type: ' + args.type)

    # Use the builder to compute the photometries and check if they were computed
    # for the full sample
    phot_map = phot_builder.buildPhotometry(sample.iterate(), ProgressListener(sample.size(), logger=logger))
    if len(phot_map[filter_name_list[0]]) != sample.size():
        logger.warning('    WARNING: Stopped because of reference sample object with missing SED')
    logger.info('Successfully computed photometry for ' + str(len(phot_map[filter_name_list[0]])) + ' objects')

    # Create the output
    logger.info('')
    logger.info('Creating output...')

    # The HDU list to store in the output file
    hdus = fits.HDUList()

    # Build a numpy array with the IDs of the objects for which we have photometry
    ids = np.zeros(len(phot_map[filter_name_list[0]]), dtype=np.int64)
    for obj, i in zip(sample.iterate(), range(len(ids))):
        ids[i] = obj.id

    # Build the photometry table HDU
    phot_col_names = ['ID'] + filter_name_list
    phot_col_data = [table.Column(ids)]
    for f in filter_name_list:
        phot_col_data.append(table.Column(phot_map[f]))
    phot_table = table.Table(phot_col_data, names=phot_col_names)
    phot_hdu = Fits.tableToHdu(phot_table)
    phot_hdu.header.set('EXTNAME', 'NNPZ_PHOTOMETRY')
    phot_hdu.header.set('PHOTYPE', args.type)
    hdus.append(phot_hdu)

    # Build the filter transmission HDUs
    for f in filter_name_list:
        data = filter_map[f]
        f_col_data = [
            table.Column(data[:, 0]),
            table.Column(data[:, 1])
        ]
        f_table = table.Table(f_col_data, names=['Wavelength', 'Transmission'])
        f_hdu = Fits.tableToHdu(f_table)
        f_hdu.header.set('EXTNAME', f)
        hdus.append(f_hdu)

    # Write the HDUs in the output file
    hdus.writeto(args.out)
    logger.info('Photometry created in ' + args.out)
