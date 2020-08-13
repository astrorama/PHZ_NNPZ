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
import itertools
import os
import numpy as np
from astropy.io import fits
from astropy import table

from ElementsKernel import Logging
from nnpz.framework import ProgressListener

from nnpz.photometry import PhotometryTypeMap, DirectoryFilterProvider, ListFileFilterProvider, \
    GalacticReddeningPrePostProcessor, ReferenceSamplePhotometryParallelBuilder, \
    ReferenceSamplePhotometryBuilder
from nnpz.utils.SedGenerator import SedGenerator
from nnpz.utils.fits import tableToHdu
from nnpz import ReferenceSample

logger = Logging.getLogger('BuildPhotometry')


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
    parser.add_argument('--mc-filters', dest='mc_filters', type=str, metavar='LIST_FILE',
                        required=False, default=None,
                        help='File with a list of filters that must be sampled at different '
                             'realizations of the PDZ')
    parser.add_argument('--mc-samples', dest='mc_samples', type=int, default=100,
                        help='Number of MC photometry samples to take')
    parser.add_argument('--mc-photo-index', dest='mc_photo_index', type=str, default='ph_index.npy',
                        help='MC Photometry index filename')
    parser.add_argument('--mc-data-pattern', dest='mc_photo_data', type=str,
                        default='ph_data_{}.npy',
                        help='MC Photometry data filename pattern')
    parser.add_argument('--out-type', dest='type', type=str, metavar=' | '.join(PhotometryTypeMap),
                        required=True,
                        help='The type of the photometry to create')
    parser.add_argument('--out-file', dest='out', type=str, metavar='FILE', required=True,
                        help='The output FITS file to create the photometry in')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Overwrite the output file')
    parser.add_argument('--gal-ebv', dest='gal_ebv', type=float,
                        help='The E(B-V) value of the galactic absorption to apply to the SEDs')
    parser.add_argument('--parallel', dest='parallel', type=int, default=None,
                        help='Number of parallel processes to spawn')
    return parser


def createPhotometryBuilder(phot_type, gal_ebv, parallel, filter_provider):
    pre_post_processor = PhotometryTypeMap[phot_type][0]()
    if gal_ebv is not None:
        pre_post_processor = GalacticReddeningPrePostProcessor(pre_post_processor, gal_ebv)

    if parallel:
        phot_builder = ReferenceSamplePhotometryParallelBuilder(
            filter_provider, pre_post_processor, parallel)
    else:
        phot_builder = ReferenceSamplePhotometryBuilder(filter_provider, pre_post_processor)

    phot_builder.setFilters(filter_provider.getFilterNames())
    return phot_builder


def buildPhotometry(args, ref_sample):
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
    phot_builder = createPhotometryBuilder(args.type, args.gal_ebv, args.parallel, filter_provider)

    # Compute the photometry values
    logger.info('')
    logger.info('Computing photometry values...')
    logger.info('    Photometry type: %s', args.type)

    # Use the builder to compute the photometries and check if they were computed
    # for the full sample
    phot_map = phot_builder.buildPhotometry(ref_sample.iterate(),
                                            ProgressListener(len(ref_sample), logger=logger))
    n_phot = len(phot_map[filter_name_list[0]])
    if n_phot != len(ref_sample):
        logger.warning('Stopped because of reference sample object with missing SED')
    logger.info('Successfully computed photometry for %d objects',
                len(phot_map[filter_name_list[0]]))

    return n_phot, phot_map, filter_map


def buildMontecarloPhotometry(args, ref_sample):
    """
    Build Montercarlo sampling of the photometry for different realizations of the PDZ
    """
    if os.path.isdir(args.mc_filters):
        filters_provider = DirectoryFilterProvider(args.mc_filters)
    else:
        filters_provider = ListFileFilterProvider(args.mc_filters)
    filter_name_list = filters_provider.getFilterNames()
    filter_map = {}
    logger.info('Loading filter transmissions for Montecarlo photometry')
    for filter_name in filter_name_list:
        logger.info('    %s', filter_name)
        filter_map[filter_name] = filters_provider.getFilterTransmission(filter_name)
    logger.info('Successfully read filter transmissions')

    # Create the photometry builder to use for computing the photometry values
    phot_builder = createPhotometryBuilder(args.type, args.gal_ebv, args.parallel, filters_provider)
    n_phot = 0
    progress_listener = ProgressListener(len(ref_sample), logger=logger)

    # Initialize the SED generator
    logger.info('Initializing MC SED generator')
    all_seds = SedGenerator()
    obj_idx = []
    for obj in ref_sample.iterate():
        pdz = obj.pdz
        obj_idx.append(obj.id)
        redshifted_sed = obj.sed
        # Reference SED corresponds to the maximum value of the PDZ
        ref_z = pdz[:, 0][pdz[:, 1].argmax()]
        # Un-shift SED
        sed = np.copy(redshifted_sed)
        sed[:, 0] /= 1 + ref_z
        sed[:, 1] *= (1 + ref_z) ** 2

        # Generate MC Z samples
        # Note: we use ABS becase due to some floating point errors we may get tiny negatives
        # (basically 0)
        normed_pdz = np.abs(pdz[:, 1]) / np.sum(np.abs(pdz[:, 1]))
        z_picks = np.random.choice(pdz[:, 0], args.mc_samples, p=normed_pdz)
        all_seds.add(sed, z_picks)

        n_phot += 1
        progress_listener(n_phot)

    # Build the photometry
    logger.info('Building the MC photometry, this can take a while...')
    phot_map = phot_builder.buildPhotometry(
        all_seds, progress_listener=ProgressListener(len(all_seds), logger=logger))

    means = dict()
    std = dict()
    nd_photo = []
    for filter_name, phot in phot_map.items():
        phot = phot.reshape(-1, args.mc_samples, 1)
        nd_photo.append(phot)
        means[filter_name + '_mean'] = np.mean(phot, axis=1)
        std[filter_name + '_std'] = np.std(phot, axis=1)

    # Merge all and generate the MC Provider
    nd_photo = np.concatenate(nd_photo, axis=-1)
    ref_sample.addProvider('MontecarloProvider', name='MontecarloPhotometry',
                           index_name=args.mc_photo_index, data_pattern=args.mc_photo_data,
                           object_ids=obj_idx, data=nd_photo,
                           overwrite=True,
                           extra=dict(names=filter_name_list))

    return n_phot, means, std, filter_map


def mainMethod(args):
    """
    Entry point for BuildPhotometry
    Args:
        args: argparse.Namespace or similar
    """

    # Validate the user inputs
    if args.type not in PhotometryTypeMap:
        logger.error('ERROR: Invalid photometry type. Available types:')
        for photometry_type in PhotometryTypeMap:
            logger.error('   %s : %s', photometry_type, PhotometryTypeMap[photometry_type][1])
        return 1

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
    n_phot, phot_map, filter_map = buildPhotometry(args, ref_sample)

    # MC Sampling
    if args.mc_filters:
        n_mc_phot, mc_phot_mean, mc_phot_std, mc_filter_map = buildMontecarloPhotometry(
            args, ref_sample
        )
    else:
        mc_phot_mean, mc_phot_std = {}, {}

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

    for mu_name, mu in mc_phot_mean.items():
        phot_col_names.append(mu_name)
        phot_col_data.append(table.Column(mu))

    for std_name, std in mc_phot_std.items():
        phot_col_names.append(std_name)
        phot_col_data.append(table.Column(std))

    phot_table = table.Table(phot_col_data, names=phot_col_names)
    phot_hdu = tableToHdu(phot_table)
    phot_hdu.header.set('EXTNAME', 'NNPZ_PHOTOMETRY')
    phot_hdu.header.set('PHOTYPE', args.type)
    hdus.append(phot_hdu)

    # Build the filter transmission HDUs
    for filter_name, filter_trans in filter_map.items():
        f_hdu = createTransmissionHdu(filter_name, filter_trans)
        hdus.append(f_hdu)

    for filter_name, filter_trans in mc_filter_map.items():
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
    f_hdu = tableToHdu(f_table)
    f_hdu.header.set('EXTNAME', filter_name)
    return f_hdu
