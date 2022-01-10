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
from datetime import datetime

import astropy.units as u
import fitsio
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.pipeline.NeighborsCatalogConfig
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.reference
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import nnpz.config.target
import numpy as np
from ElementsKernel import Logging
from nnpz.config.ConfigManager import ConfigManager
from nnpz.photometry.projection.ebv import correct_ebv
from nnpz.photometry.projection.filter_variation import correct_filter_variation
from nnpz.utils.ArgumentParserWrapper import ArgumentParserWrapper

logger = Logging.getLogger(__name__)


def defineSpecificProgramOptions():
    """
    Program options. Returns a ArgumentParserWrapper that tricks Elements so we can
    capture anything extra and do the evaluation ourselves: NNPZ interpret flags
    as Python code
    """
    return ArgumentParserWrapper(description='Nearest Neighbor Photo-Z (photometry correction)')


def mainMethod(args):
    """
    Entry point for CorrectPhotometry

    Args:
        args: argparse.Namespace or similar
    """

    # Create the object which handles the user parameters
    conf_manager = ConfigManager(args.config_file, args.extra_arguments)

    # Read the reference correction factors
    ebv_corr_coefs = conf_manager.getObject('reference_ebv_correction')
    filter_corr_coefs = conf_manager.getObject('reference_filter_variation_correction')

    # Open the target catalog
    input_photometry = conf_manager.getObject('target_photometry')

    # Open the output catalog
    neighbor_catalog = conf_manager.getObject('neighbor_catalog')
    output_fits = fitsio.FITS(neighbor_catalog, mode='rw', clobber=False)
    output_hdu: fitsio.hdu.TableHDU = output_fits[1]

    assert 'NEIGHBOR_INDEX' in output_hdu.get_colnames()
    assert 'NEIGHBOR_SCALING' in output_hdu.get_colnames()

    # Chunk size
    chunk_size = conf_manager.getObject('target_chunk_size')
    nchunks, remainder = divmod(len(input_photometry), chunk_size)
    nchunks += remainder > 0

    # Process in chunks
    start = datetime.utcnow()
    for chunk in range(nchunks):
        logger.info('Processing chunk %d / %d', chunk + 1, nchunks)
        offset = chunk * chunk_size
        subset = slice(offset, offset + chunk_size)
        chunk_input = input_photometry[subset]
        chunk_output = output_hdu[subset]

        if 'shifts' in chunk_input.colorspace:
            chunk_filter_corr_coefs = filter_corr_coefs[chunk_output['NEIGHBOR_INDEX']]
            for filter_idx, filter_name in enumerate(chunk_input.system.bands):
                if filter_name not in chunk_input.colorspace.shifts.dtype.names:
                    continue
                logger.info('Correcting for %s filter variation', filter_name)
                correct_filter_variation(chunk_output[filter_name],
                                         corr_coef=chunk_filter_corr_coefs[:, :, filter_idx],
                                         shift=chunk_input.colorspace.shifts[filter_name],
                                         out=chunk_output[filter_name])

        if 'ebv' in chunk_input.colorspace:
            logger.info('Correcting for EBV')
            chunk_ebv_corr_coefs = ebv_corr_coefs[chunk_output['NEIGHBOR_INDEX']]
            for filter_idx, filter_name in enumerate(chunk_input.system.bands):
                correct_ebv(chunk_output[filter_name],
                            corr_coef=chunk_ebv_corr_coefs[:, :, filter_idx],
                            ebv=chunk_input.colorspace.ebv,
                            out=chunk_output[filter_name])

        output_hdu.write(chunk_output, firstrow=offset)

    end = datetime.utcnow()
    duration = end - start
    logger.info('Finished in %s (%.2f sources / second)', duration,
                len(input_photometry) / duration.total_seconds())

    del output_hdu
    output_fits.close()

    return 0
