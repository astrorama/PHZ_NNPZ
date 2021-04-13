import numpy as np
import pytest
from astropy.io import fits
from nnpz.utils import numpy as npu


@pytest.fixture
def table_fixture():
    col_defs = [
        fits.Column(name='ID', format='K'),
    ]
    for b in 'ugriz':
        col_defs.append(fits.Column(name=b, format='E'))
    col_defs.append(fits.Column(name='ColumnZ', format='J'))
    col_defs.append(fits.Column(name='Float', format='E'))
    return fits.BinTableHDU.from_columns(col_defs, nrows=100)


def test_view(table_fixture):
    """
    View a subset of the fields
    """
    view = npu.recarray_flat_view(table_fixture.data, 'ugri')
    assert np.may_share_memory(view, table_fixture.data)
    assert view.shape == (100, 4)
    view[:, 0] = 25.  # u
    view[:, 1] = 68.  # g
    view[:, 2] = 42.  # r
    view[:, 3] = 12.  # i
    view[5, 1] = 14.  # g, row 5

    assert np.all(table_fixture.data['u'] == 25.)
    assert np.sum(table_fixture.data['g'] == 68.) == 99
    assert np.all(table_fixture.data['r'] == 42.)
    assert np.all(table_fixture.data['i'] == 12.)
    assert np.all(table_fixture.data['g'][5] == 14.)
    assert np.all(table_fixture.data['z'] == 0.)


def test_view_mix_types(table_fixture):
    """
    Can not mix types
    """
    with pytest.raises(TypeError):
        npu.recarray_flat_view(table_fixture.data, ['u', 'g', 'r', 'ID'])


def test_view_no_consecutive(table_fixture):
    """
    Fields must be consecutive
    """
    with pytest.raises(IndexError):
        npu.recarray_flat_view(table_fixture.data, 'zgri')

    with pytest.raises(IndexError):
        npu.recarray_flat_view(table_fixture.data, ['u', 'g', 'r', 'i', 'Float'])
