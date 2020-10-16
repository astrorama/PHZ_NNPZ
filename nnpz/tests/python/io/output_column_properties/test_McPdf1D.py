from astropy.table import Column
from nnpz.io.output_column_providers.McPdf1D import McPdf1D

from .fixtures import *


def test_pdf1d(sampler):
    pdf = McPdf1D(sampler, param_name='P1', binning=np.array([-0.5, 0.5, 1.5, 2.5, 3.5]))

    columns = pdf.getColumns()
    assert isinstance(columns, list)
    column = columns[0]
    assert isinstance(column, Column)
    assert column.name == 'MC_PDF_1D_P1'
    assert column.shape == (2, 4)
    # First object can not have any sample from 2, and the weight is higher for 1
    assert column[0][3] == 0
    assert column[0][2] == 0
    assert column[0][1] > column[0][2]
    # Second object must have the most samples from 2, and more from 0 than from 1
    assert (column[1][2] >= column[1]).all()
    assert column[1][0] > column[1][1]
