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


class MockOutputHandler:
    def __init__(self):
        self.__providers = []
        self.__provider_columns = {}
        self.__output = None

    def addColumnProvider(self, provider):
        self.__providers.append(provider)

    def initialize(self, nrows: int):
        dtype = []
        for prov in self.__providers:
            self.__provider_columns[prov] = []
            prov_def = prov.getColumnDefinition()
            dtype.extend(prov_def)
            for col in prov_def:
                self.__provider_columns[prov].append(col[0])
        self.__output = np.zeros(nrows, dtype=dtype)

        for prov in self.__providers:
            prov.setWriteableArea(self.__output)

    def addContribution(self, reference_sample_i, neighbor, flags):
        for p in self.__providers:
            p.addContribution(reference_sample_i, neighbor, flags)

    def getDataForProvider(self, provider):
        return self.__output[self.__provider_columns[provider]]


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def mock_output_handler():
    return MockOutputHandler()


@pytest.fixture
def reference_ids():
    return np.arange(3)


@pytest.fixture
def contributions():
    NS = NeighborSet()
    return [
        (0, NS.append(0, weight=0.10, position=0)),
        (0, NS.append(1, weight=0.30, position=0)),
        (1, NS.append(0, weight=0.60, position=1)),
        (1, NS.append(1, weight=0.05, position=1)),
        (2, NS.append(0, weight=0.00, position=2)),
        (2, NS.append(1, weight=0.80, position=2)),
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
