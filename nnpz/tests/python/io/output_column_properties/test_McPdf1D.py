import numpy as np
import pytest
from astropy.table import Column
from nnpz.framework.NeighborSet import NeighborSet
from nnpz.io.output_column_providers.McPdf1D import McPdf1D


class MockProvider:
    def getData(self, obj_id):
        data = np.zeros(50, dtype=[('Test', np.float)])
        data['Test'] = obj_id
        return data


@pytest.fixture
def contributions():
    NS = NeighborSet()
    return [
        (0, NS.append(0, weight=0.1)),
        (0, NS.append(1, weight=0.2)),
        (1, NS.append(0, weight=0.3)),
        (1, NS.append(1, weight=0.001)),
        (2, NS.append(0, weight=0.00)),
        (2, NS.append(1, weight=0.8)),
    ]


def test_pdf1d(contributions):
    pdf = McPdf1D(
        2, n_neighbors=3, take_n=100, param_name='Test',
        binning=np.array([-0.5, 0.5, 1.5, 2.5, 3.5]),
        mc_provider=MockProvider(), ref_ids=np.arange(5)
    )
    for ref_i, contrib in contributions:
        pdf.addContribution(ref_i, contrib, None)
    columns = pdf.getColumns()
    assert isinstance(columns, list)
    column = columns[0]
    assert isinstance(column, Column)
    assert column.name == 'MC_PDF_1D_TEST'
    assert column.shape == (2, 4)
    # First object can not have any sample from 2, and the weight is higher for 1
    assert column[0][3] == 0
    assert column[0][2] == 0
    assert column[0][1] > column[0][2]
    # Second object must have the most samples from 2, and more from 0 than from 1
    assert (column[1][2] >= column[1]).all()
    assert column[1][0] > column[1][1]
