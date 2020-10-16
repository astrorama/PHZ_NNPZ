from astropy.table import Column
from nnpz.io.output_column_providers.McPdf2D import McPdf2D

from .fixtures import *


def test_pdf2d(sampler):
    pdf = McPdf2D(
        sampler, param_names=('P1', 'P2'),
        binning=(
            np.array([-0.5, 0.5, 1.5, 2.5, 3.5]),
            np.array([-0.5, 0.5, 1.5, 2.5, 3.5]) * 2,
        )
    )

    columns = pdf.getColumns()
    assert isinstance(columns, list)
    column = columns[0]
    assert isinstance(column, Column)
    assert column.name == 'MC_PDF_2D_P1_P2'
    assert column.shape == (2, 4 ** 2)
    column = column.reshape(-1, 4, 4)
    # First object can not have any sample from 2, and the weight is higher for 1
    assert (column[0][1, 1] >= column[0]).all()
    assert column[0][0, 0] != 0
    assert column[0][2, 2] == 0
    # Second object must have the most samples from 2, and more from 0 than from 1
    assert (column[1][2, 2] >= column[1]).all()
