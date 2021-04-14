#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
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

import argparse

from ElementsKernel.Logging import getLogger
from astropy.table import Table
from nnpz.io.catalog_properties import Photometry
from nnpz.photometry import ListFileFilterProvider
from nnpz.photometry.SourceIndependantGalacticUnReddening import \
    SourceIndependantGalacticUnReddening

_logger = getLogger(__name__)


def defineSpecificProgramOptions():
    """
    Specific options for BuildPhotometry
    Returns: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filter-list', type=str, required=True,
                        help='Filter list')
    parser.add_argument('--galactic-reddening-curve', type=str, default=None,
                        help='Galactic reddening curve')
    parser.add_argument('--reference-sed', type=str, default=None,
                        help='Reference SED')
    parser.add_argument('--ebv0', type=float, default=0.02,
                        help='Reference E(B-V)_0 for which the K_X are computed')
    parser.add_argument('--target-ebv', type=str, default='GAL_EBV',
                        help='Column with the target EBV value')
    parser.add_argument('-i', '--input-size', type=int, default=None,
                        help='Process a subset of the input catalog')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Overwrite output catalog')
    parser.add_argument('input', type=str, metavar='INPUT',
                        help='Input catalog')
    parser.add_argument('output', type=str, metavar='OUTPUT',
                        help='Output catalog')
    return parser


def mainMethod(args):
    """
    Entry point for BuildPhotometry
    Args:
        args: argparse.Namespace or similar
    """
    filter_provider = ListFileFilterProvider(args.filter_list)
    filter_names = filter_provider.getFilterNames()
    filter_dict = dict([(fn, filter_provider.getFilterTransmission(fn)) for fn in filter_names])
    _logger.info('De-reddening filters')
    for fn in filter_names:
        _logger.info('\t%s', fn)

    catalog = Table.read(args.input)
    if args.input_size:
        catalog = catalog[:args.input_size]
    _logger.info('Processing %d objects', len(catalog))

    photometry_prop = Photometry(filter_names)
    photo = photometry_prop(catalog)
    ebv = catalog[args.target_ebv]

    dereddener = SourceIndependantGalacticUnReddening(
        filter_dict, filter_names, galactic_reddening_curve=args.galactic_reddening_curve,
        ref_sed=args.reference_sed, ebv_0=args.ebv0
    )
    _logger.info('De-reddening')
    deredden_photo = dereddener.de_redden_data(photo, ebv)

    _logger.info('Generating output catalog %s', args.output)
    for i, fn in enumerate(filter_names):
        catalog[fn] = deredden_photo[:, i, 0]

    catalog.write(args.output, overwrite=args.overwrite)
