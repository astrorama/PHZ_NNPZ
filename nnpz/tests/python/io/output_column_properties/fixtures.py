import numpy as np
import pytest
from nnpz.framework.NeighborSet import NeighborSet
from nnpz.io.output_column_providers.McSampler import McSampler


class MockProvider:
    def getData(self, obj_id):
        data = np.zeros(50, dtype=[('P1', np.float), ('P2', np.float)])
        random = np.random.multivariate_normal(
            (obj_id, obj_id * 2), cov=[[0.0, 0.], [0., 0.0]], size=50
        )
        data['P1'] = random[:, 0]
        data['P2'] = random[:, 1]
        return data


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def reference_ids():
    return np.arange(3)


@pytest.fixture
def contributions():
    NS = NeighborSet()
    return [
        (0, NS.append(0, weight=0.1)),
        (0, NS.append(1, weight=0.3)),
        (1, NS.append(0, weight=0.6)),
        (1, NS.append(1, weight=0.05)),
        (2, NS.append(0, weight=0.00)),
        (2, NS.append(1, weight=0.8)),
    ]


@pytest.fixture
def sampler(mock_provider, reference_ids, contributions):
    sampler = McSampler(
        2, n_neighbors=3, take_n=200, mc_provider=mock_provider,
        ref_ids=reference_ids
    )
    for ref_i, contrib in contributions:
        sampler.addContribution(ref_i, contrib, None)
    return sampler
