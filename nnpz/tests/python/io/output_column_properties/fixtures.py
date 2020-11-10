import numpy as np
import pytest
from nnpz.framework.NeighborSet import NeighborSet
from nnpz.io.output_column_providers.McSampler import McSampler


class MockProvider:

    def __init__(self):
        self.__data = np.zeros((3, 50), dtype=[('P1', np.float), ('P2', np.float), ('I1', np.int)])
        for i in range(len(self.__data)):
            random = np.random.multivariate_normal(
                (i, i * 2), cov=[[0.0, 0.], [0., 0.0]], size=50
            )
            self.__data['P1'][i] = random[:, 0]
            self.__data['P2'][i] = random[:, 1]
            self.__data['I1'][i] = i

    def getData(self, obj_id):
        return self.__data[obj_id]


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
