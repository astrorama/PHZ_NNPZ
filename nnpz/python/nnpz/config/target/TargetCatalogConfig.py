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
Created on: 28/02/18
Author: Nikolaos Apostolakos
"""

from typing import Dict, List, Tuple

import fitsio
import numpy as np
from ElementsKernel import Logging
from astropy import units as u
from nnpz.config import ConfigManager
from nnpz.config.reference import ReferenceConfig
from nnpz.photometry.colorspace import ColorSpace
from nnpz.photometry.photometry import Photometry

logger = Logging.getLogger('Configuration')


class TargetCatalogConfig(ConfigManager.ConfigHandler):
    """
    Configure the input catalog containing the target objects
    """

    def __init__(self):
        self.__target_photo = None
        self.__chunks = None
        self.__id_column = None

    def __readCatalog(self, table: fitsio.hdu.TableHDU,
                      filters: List[Tuple[str, str]], id_column: str,
                      missing_values: List[float],
                      rows: int) -> Tuple[np.ndarray, np.ndarray]:
        val_cols = list(map(lambda t: t[0], filters))
        err_cols = list(map(lambda t: t[1], filters))

        header = table.read_header()
        columns = table.get_colnames()
        col_units = [header.get('TUNIT{}'.format(columns.index(c) + 1)) for c in val_cols]
        if len(set(col_units)) > 1:
            raise ValueError(
                'Columns from the target column have multiple units! {}'.format(col_units))
        elif len(col_units) == 0 or set(col_units) == {None}:
            logger.warning('The input catalog has no units specified. Assuming uJy!')
            unit = u.uJy
        else:
            unit = u.Unit(col_units[0])

        logger.info('Reading target catalog in %s', unit)

        values = table.read_columns(val_cols, rows=rows)
        errors = table.read_columns(err_cols, rows=rows)
        dtype = values.dtype[0]
        values = values.view(dtype=dtype).reshape(len(values), -1)
        errors = errors.view(dtype=dtype).reshape(values.shape)

        ids = table.read_column(id_column, rows=rows)
        values = u.Quantity(np.stack([values, errors], axis=-1), unit, copy=False)

        # Note that we read in whatever unit the catalog is in, and then pass to uJy
        # so we don't need to do it each time the data is accessed
        if values.unit != u.uJy:
            logger.warning('Converting target catalog into uJy')
            values <<= u.uJy

        for nan_val in missing_values:
            values[values == nan_val] = np.nan

        return ids, values

    def __setupColorspace(self, table: fitsio.hdu.TableHDU, args: Dict,
                          bands: List[str], rows: np.ndarray) -> ColorSpace:
        catalog_ebv = args.get('target_catalog_gal_ebv', None)
        filter_shifts = args.get('target_catalog_filters_shifts', None)

        factors = {}
        if catalog_ebv:
            factors['ebv'] = table.read_column(catalog_ebv, rows=rows)

        if filter_shifts:
            dtype = [(b, float) for b in bands if b in filter_shifts]
            shifts = np.zeros(len(rows) if rows is not None else table.get_nrows(), dtype=dtype)
            for b, _ in dtype:
                shifts[b] = table.read_column(filter_shifts[b], rows=rows)
            factors['shifts'] = shifts
        return ColorSpace(**factors)

    def __createData(self, args):
        ref_photometry = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)[
            'reference_photometry']

        self._checkParameterExists('target_catalog', args)
        self._checkParameterExists('target_catalog_filters', args)

        target_cat = args['target_catalog']
        missing_phot_flags = args.get('missing_photometry_flags', [])
        self.__id_column = args.get('target_catalog_id_column', 'ID')

        target_filters = args['target_catalog_filters']
        ref_filters = args['reference_sample_phot_filters']
        if len(target_filters) != len(ref_filters):
            raise ValueError('Number of target columns does not match the reference bands')

        logger.info('Target catalog photometric columns: %s', target_filters)
        logger.info('Reading target catalog: %s', target_cat)

        fits = fitsio.FITS(target_cat)
        table = fits[1]

        input_rows = args.get('input_size', None)
        if input_rows is not None:
            logger.warning('Processing only %s objects from target catalog', input_rows)
            input_rows = np.arange(input_rows)

        ids, values = self.__readCatalog(table, target_filters, self.__id_column,
                                         missing_values=missing_phot_flags,
                                         rows=input_rows)
        self.__id_column = (self.__id_column, ids.dtype)

        target_system = ref_photometry.system[ref_filters]

        target_colorspace = self.__setupColorspace(table, args, target_system.bands,
                                                   rows=input_rows)
        self.__target_photo = Photometry(ids, values, system=target_system,
                                         colorspace=target_colorspace)

        chunk_size = args.get('input_chunk_size', min(1000, len(self.__target_photo)))
        nchunks, remainder = divmod(len(ids), chunk_size)
        self.__chunks = [slice(chunk * chunk_size, chunk * chunk_size + chunk_size) for chunk in
                         range(nchunks)]
        if remainder > 0:
            self.__chunks.append(slice(len(ids) - remainder, remainder))

    def parseArgs(self, args):
        if self.__target_photo is None:
            self.__createData(args)

        return {
            'target_photometry': self.__target_photo,
            'target_idx_slices': self.__chunks,
            'target_id_column': self.__id_column
        }


ConfigManager.addHandler(TargetCatalogConfig)
