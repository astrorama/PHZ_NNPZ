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
from ElementsKernel import Logging
from nnpz.config.ConfigManager import ConfigManager
from nnpz.pipeline.correct_photometry import CorrectPhotometry
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

    # Corrector
    corrector = CorrectPhotometry(conf_manager)

    # Open the target catalog
    input_photometry = conf_manager.getObject('target_photometry')

    # Open the input/output catalog
    neighbor_catalog = conf_manager.getObject('neighbor_catalog')
    output_fits = fitsio.FITS(neighbor_catalog, mode='rw', clobber=False)
    output_hdu: fitsio.hdu.TableHDU = output_fits[1]
    output_colnames = output_hdu.get_colnames()
    output_header = output_hdu.read_header()

    assert 'NEIGHBOR_INDEX' in output_colnames
    assert 'NEIGHBOR_SCALING' in output_colnames
    assert 'NEIGHBOR_PHOTOMETRY' in output_colnames

    neighbor_photo_idx = output_colnames.index('NEIGHBOR_PHOTOMETRY') + 1
    photo_unit = u.Unit(output_header.get(f'TUNIT{neighbor_photo_idx}'))
    assert photo_unit == u.uJy

    # Chunks
    chunks: slice = conf_manager.getObject('target_idx_slices')

    # Process in chunks
    start = datetime.utcnow()
    for i, chunk in enumerate(chunks, start=1):
        logger.info('Processing chunk %d / %d', i, len(chunks))
        chunk_input = input_photometry[chunk]
        chunk_output = output_hdu[chunk]
        chunk_ref_photo = chunk_output['NEIGHBOR_PHOTOMETRY'] * photo_unit

        corrector(chunk_input, chunk_output['NEIGHBOR_INDEX'], chunk_ref_photo)

        output_hdu.write_column('NEIGHBOR_PHOTOMETRY', chunk_ref_photo, firstrow=chunk.start)

    end = datetime.utcnow()
    duration = end - start
    logger.info('Finished in %s (%.2f sources / second)', duration,
                len(input_photometry) / duration.total_seconds())

    del output_hdu
    output_fits.close()

    return 0
