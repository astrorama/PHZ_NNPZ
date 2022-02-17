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
import fitsio.hdu
import numpy as np
import pytest
from ElementsKernel.Temporary import TempFile
from nnpz.io.output_column_providers import CatalogCopy


@pytest.fixture
def catalog_fixture() -> fitsio.hdu.TableHDU:
    data = {
        'ID': np.arange(0, 10, dtype=int),
        'RA': np.random.rand(10),
        'DEC': np.random.rand(10),
    }
    fits = fitsio.FITS(TempFile().path(), mode='rw', clobber=True)
    fits.create_table_hdu(data=data)
    hdu: fitsio.hdu.TableHDU = fits[-1]
    hdu.append(data)
    return hdu


###############################################################################

def test_CatalogCopy(catalog_fixture: fitsio.hdu.TableHDU):
    cat_copy = CatalogCopy(columns=np.dtype([('ID', int), ('DEC', float)]), catalog=catalog_fixture)
    col_defs = cat_copy.get_column_definition()
    col_names = [col_name for col_name, _, _ in col_defs]
    assert len(col_names) == 2
    assert 'ID' in col_names
    assert 'DEC' in col_names
    assert 'RA' not in col_names

    output = np.zeros(catalog_fixture.get_nrows(), dtype=col_defs)
    cat_copy.generate_output(indexes=np.arange(0, 10), neighbor_info=None, output=output)

    np.testing.assert_array_equal(output['ID'], catalog_fixture['ID'][:])
    np.testing.assert_array_equal(output['DEC'], catalog_fixture['DEC'][:])

###############################################################################
