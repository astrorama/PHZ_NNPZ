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

from nnpz.photometry import PhotometryTypeMap, DirectoryFilterProvider, ListFileFilterProvider, \
    GalacticReddeningPrePostProcessor, ReferenceSamplePhotometryParallelBuilder, \
    ReferenceSamplePhotometryBuilder
from nnpz.utils.fits import tableToHdu
from nnpz import ReferenceSample


def defineSpecificProgramOptions():
    """
    Specific options for BuildPhotometry
    Returns: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-dir', dest='sample_dir', type=str, metavar='DIR', required=True,
                        help='The reference sample directory')
    parser.add_argument('--filters', dest='filters', type=str, metavar='DIR | LIST_FILE',
                        required=True,
                        help='The directory containing the filter transmissions')
    parser.add_argument('--out-type', dest='type', type=str, metavar=' | '.join(PhotometryTypeMap),
                        required=True,
                        help='The type of the photometry to create')
    parser.add_argument('--out-file', dest='out', type=str, metavar='FILE', required=True,
                        help='The output FITS file to create the photometry in')
    parser.add_argument('--gal-ebv', dest='gal_ebv', type=float,
                        help='The E(B-V) value of the galactic absorption to apply to the SEDs')
    parser.add_argument('--parallel', dest='parallel', type=int, default=None,
                        help='Number of parallel processes to spawn')
    return parser


def mainMethod(args):
    """
    Entry point for BuildPhotometry
    Args:
        args: argparse.Namespace or similar
    """
    logger = Logging.getLogger('BuildPhotometry')

    # Validate the user inputs
    if args.type not in PhotometryTypeMap:
        logger.error('ERROR: Invalid photometry type. Available types:')
        for photometry_type in PhotometryTypeMap:
            logger.error('   %s : %s', photometry_type, PhotometryTypeMap[photometry_type][1])
        return 1

    if os.path.exists(args.out):
        logger.error('ERROR: File %s already exists', args.out)
        return 1
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.exists(out_dir):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)))

    # Open the reference sample for reading
    logger.info('')
    logger.info('Opening reference sample %s...', args.sample_dir)
    sample = ReferenceSample(args.sample_dir)
    logger.info('Successfully opened reference sample')

    # Read the filter transmissions
    logger.info('')
    logger.info('Reading filter transmissions from %s...', args.filters)
    if os.path.isdir(args.filters):
        filter_provider = DirectoryFilterProvider(args.filters)
    else:
        filter_provider = ListFileFilterProvider(args.filters)
    filter_name_list = filter_provider.getFilterNames()
    filter_map = {}
    for filter_name in filter_name_list:
        logger.info('    %s', filter_name)
        filter_map[filter_name] = filter_provider.getFilterTransmission(filter_name)
    logger.info('Successfully read filter transmissions')

    # Create the photometry builder to use for computing the photometry values
    pre_post_processor = PhotometryTypeMap[args.type][0]()
    if args.gal_ebv is not None:
        pre_post_processor = GalacticReddeningPrePostProcessor(pre_post_processor, args.gal_ebv)

    if args.parallel:
        phot_builder = ReferenceSamplePhotometryParallelBuilder(filter_provider, pre_post_processor,
                                                                args.parallel)
    else:
        phot_builder = ReferenceSamplePhotometryBuilder(filter_provider, pre_post_processor)

    phot_builder.setFilters(filter_name_list)

    # Compute the photometry values
    logger.info('')
    logger.info('Computing photometry values...')
    logger.info('    Photometry type: %s', args.type)

    # Use the builder to compute the photometries and check if they were computed
    # for the full sample
    phot_map = phot_builder.buildPhotometry(sample.iterate(),
                                            ProgressListener(len(sample), logger=logger))
    if len(phot_map[filter_name_list[0]]) != len(sample):
        logger.warning('Stopped because of reference sample object with missing SED')
    logger.info(
        'Successfully computed photometry for %d objects',
        len(phot_map[filter_name_list[0]])
    )

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
    for filter_name in filter_name_list:
        phot_col_data.append(table.Column(phot_map[filter_name]))
    phot_table = table.Table(phot_col_data, names=phot_col_names)
    phot_hdu = tableToHdu(phot_table)
    phot_hdu.header.set('EXTNAME', 'NNPZ_PHOTOMETRY')
    phot_hdu.header.set('PHOTYPE', args.type)
    hdus.append(phot_hdu)

    # Build the filter transmission HDUs
    for filter_name in filter_name_list:
        data = filter_map[filter_name]
        f_col_data = [
            table.Column(data[:, 0]),
            table.Column(data[:, 1])
        ]
        f_table = table.Table(f_col_data, names=['Wavelength', 'Transmission'])
        f_hdu = tableToHdu(f_table)
        f_hdu.header.set('EXTNAME', filter_name)
        hdus.append(f_hdu)

    # Write the HDUs in the output file
    hdus.writeto(args.out)
    logger.info('Photometry created in %s', args.out)
    return 0
