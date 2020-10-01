import numpy as np
import pytest
from astropy.table import Column
from nnpz.framework.NeighborSet import NeighborSet
from nnpz.io.output_column_providers.McPdf2D import McPdf2D


class MockProvider:
    def getData(self, obj_id):
        data = np.zeros(50, dtype=[('P1', np.float), ('P2', np.float)])
        random = np.random.multivariate_normal(
            (obj_id, obj_id * 2), cov=[[0.1, 0.], [0., 0.1]], size=50
        )
        data['P1'] = random[:, 0]
        data['P2'] = random[:, 1]
        return data


@pytest.fixture
def contributions():
    NS = NeighborSet()
    return [
        (0, NS.append(0, weight=0.1)),
        (0, NS.append(1, weight=0.2)),
        (1, NS.append(0, weight=0.6)),
        (1, NS.append(1, weight=0.001)),
        (2, NS.append(0, weight=0.00)),
        (2, NS.append(1, weight=0.8)),
    ]


def test_pdf2d(contributions):
    pdf = McPdf2D(
        2, n_neighbors=3, take_n=200, param_names=('P1', 'P2'),
        binning=(
            np.array([-0.5, 0.5, 1.5, 2.5, 3.5]),
            np.array([-0.5, 0.5, 1.5, 2.5, 3.5]) * 2,
        ),
        mc_provider=MockProvider(), ref_ids=np.arange(5)
    )
    for ref_i, contrib in contributions:
        pdf.addContribution(ref_i, contrib, None)
    columns = pdf.getColumns()
    assert isinstance(columns, list)
    column = columns[0]
    print(column[0])
    assert isinstance(column, Column)
    assert column.name == 'MC_PDF_2D_P1_P2'
    assert column.shape == (2, 4**2)
    column = column.reshape(-1, 4, 4)
    # First object can not have any sample from 2, and the weight is higher for 1
    assert (column[0][1, 1] >= column[0]).all()
    assert column[0][0, 0] != 0
    assert column[0][2, 2] == 0
    # Second object must have the most samples from 2, and more from 0 than from 1
    assert (column[1][2, 2] >= column[1]).all()
