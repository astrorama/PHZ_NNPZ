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
Created on: 17/12/17
Author: Nikolaos Apostolakos
"""

import argparse
import os
from typing import Tuple

import itertools
import numpy as np
from ElementsKernel import Logging
from astropy import table
from astropy.io import fits
from astropy.io.fits import BinTableHDU
from nnpz.photometry.calculator.fnu_ujy_processor import FnuuJyPrePostProcessor
from nnpz.photometry.filter_provider import DirectoryFilterProvider, ListFileFilterProvider
from nnpz.photometry.photometry_builder import PhotometryBuilder
from nnpz.reference_sample.ReferenceSample import ReferenceSample
from nnpz.utils.ProgressListener import ProgressListener

logger = Logging.getLogger('BuildPhotometry')

DEFAULT_SHIFTS = np.concatenate([np.arange(-100, 0), np.arange(1, 101)])


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
    parser.add_argument('--out-file', dest='out', type=str, metavar='FILE', required=True,
                        help='The output FITS file to create the photometry in')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Overwrite the output file')
    parser.add_argument('--gal-ebv', dest='gal_ebv', type=float, default=0.3,
                        help='The E(B-V) value used to compute the correction factor')
    parser.add_argument('--parallel', dest='parallel', type=int, default=None,
                        help='Number of parallel processes to spawn')
    parser.add_argument('--input-size', dest='input_size', type=int, default=None,
                        help='Limit the computation to this many sources')
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=5000,
                        help='Compute MC photometry this many objects at once')
    parser.add_argument('--shifts', type=float, nargs='+', default=DEFAULT_SHIFTS,
                        help='Use these shifts to compute the correction factors')
    return parser


def createPhotometryBuilder(gal_ebv, parallel, filter_provider, shifts):
    filter_names = filter_provider.getFilterNames()
    filter_trans = {fname: filter_provider.getFilterTransmission(fname) for fname in filter_names}
    pre_post_processor = FnuuJyPrePostProcessor(filter_trans)
    phot_builder = PhotometryBuilder(filter_provider, pre_post_processor, gal_ebv, shifts,
                                     ncores=parallel)
    phot_builder.setFilters(filter_names)
    return phot_builder


def buildPhotometry(args: argparse.Namespace, ref_sample: ReferenceSample) \
        -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build the photometry using the reference redshift (max of the PDZ)
    """
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
    phot_builder = createPhotometryBuilder(args.gal_ebv, args.parallel, filter_provider,
                                           args.shifts)

    # Compute the photometry values
    logger.info('')
    logger.info('Computing photometry values...')
    logger.info('    Photometry type: uJy')

    # Use the builder to compute the photometries and check if they were computed
    # for the full sample
    n_items = args.input_size if args.input_size is not None else len(ref_sample)
    progress = ProgressListener(n_items, logger=logger)
    phot_map, ebv_corr_map, shift_corr_map = phot_builder.buildPhotometry(
        itertools.islice(ref_sample.iterate(), args.input_size),
        progress
    )
    n_phot = len(phot_map[filter_name_list[0]])
    if n_phot != n_items:
        logger.warning('Stopped because of reference sample object with missing SED')
    logger.info('Successfully computed photometry for %d objects', n_phot)

    return n_phot, phot_map, ebv_corr_map, shift_corr_map, filter_map


def mainMethod(args):
    """
    Entry point for BuildPhotometry
    Args:
        args: argparse.Namespace or similar
    """

    if os.path.exists(args.out) and not args.overwrite:
        logger.error('ERROR: File %s already exists', args.out)
        return 1
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.exists(out_dir):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)))

    # Open the reference sample for reading
    logger.info('')
    logger.info('Opening reference sample %s...', args.sample_dir)
    ref_sample = ReferenceSample(args.sample_dir)
    logger.info('Successfully opened reference sample')

    # Build photometry
    n_phot, phot_map, ebv_corr_map, shift_corr_map, filter_map = buildPhotometry(args, ref_sample)

    # Create the output
    logger.info('')
    logger.info('Creating output...')

    # The HDU list to store in the output file
    hdus = fits.HDUList()

    # Build a numpy array with the IDs of the objects for which we have photometry
    ids = np.zeros(n_phot, dtype=np.int64)
    for obj, i in zip(ref_sample.iterate(), range(len(ids))):
        ids[i] = obj.id

    # Build the photometry table HDU
    phot_col_names = ['ID']
    phot_col_data = [table.Column(ids)]

    for filter_name, photo in phot_map.items():
        phot_col_names.append(filter_name)
        phot_col_data.append(table.Column(photo))

    for filter_name, corr in ebv_corr_map.items():
        phot_col_names.append(filter_name + '_EBV_CORR')
        phot_col_data.append(table.Column(corr))

    for filter_name, corr in shift_corr_map.items():
        phot_col_names.append(filter_name + '_SHIFT_CORR')
        phot_col_data.append(table.Column(corr))

    phot_table = table.Table(phot_col_data, names=phot_col_names)
    phot_hdu = BinTableHDU(phot_table)
    phot_hdu.header.set('EXTNAME', 'NNPZ_PHOTOMETRY')
    phot_hdu.header.set('PHOTYPE', 'F_nu_uJy')
    hdus.append(phot_hdu)

    # Build the filter transmission HDUs
    for filter_name, filter_trans in filter_map.items():
        f_hdu = createTransmissionHdu(filter_name, filter_trans)
        hdus.append(f_hdu)

    # Write the HDUs in the output file
    hdus.writeto(args.out, overwrite=args.overwrite)
    logger.info('Photometry created in %s', args.out)
    return 0


def createTransmissionHdu(filter_name, filter_trans):
    f_col_data = [
        table.Column(filter_trans[:, 0]),
        table.Column(filter_trans[:, 1])
    ]
    f_table = table.Table(f_col_data, names=['Wavelength', 'Transmission'])
    f_hdu = BinTableHDU(f_table)
    f_hdu.header.set('EXTNAME', filter_name)
    return f_hdu
