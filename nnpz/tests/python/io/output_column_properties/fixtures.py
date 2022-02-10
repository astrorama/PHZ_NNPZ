from typing import List, Union

import numpy as np
import pytest
from nnpz.io import OutputHandler
from nnpz.io.output_column_providers.McSampler import McSampler


class MockProvider:
    DTYPE = np.dtype([('P1', np.float32), ('P2', np.float32), ('I1', np.int32)])

    def __init__(self):
        self.__data = np.zeros((3, 50), dtype=self.DTYPE)
        for i in range(len(self.__data)):
            random = np.random.multivariate_normal(
                (i, i * 2), cov=[[0.0, 0.], [0., 0.0]], size=50
            )
            self.__data['P1'][i] = random[:, 0]
            self.__data['P2'][i] = random[:, 1]
            self.__data['I1'][i] = i

    def getDataForIndex(self, obj_idx: np.ndarray):
        data = self.__data[obj_idx[:, 0], obj_idx[:, 1]]
        data[obj_idx[:, 0] < 0] = 0
        return data

    def getData(self, obj_id: int):
        return self.__data[obj_id]

    def getDtype(self) -> np.dtype:
        return self.DTYPE

    def getNSamples(self) -> int:
        return self.__data.shape[1]


class MockOutputHandler:
    def __init__(self):
        self.__providers: List[OutputHandler.OutputColumnProviderInterface] = []
        self.__provider_columns = {}
        self.__output = None

    def add_column_provider(self, provider):
        self.__providers.append(provider)

    def initialize(self, nrows: int):
        dtype = []
        for prov in self.__providers:
            self.__provider_columns[prov] = []
            prov_def = prov.get_column_definition()
            for d in prov_def:
                if len(d) == 4:
                    n, t, _, s = d
                    dtype.append((n, t, s))
                else:
                    n, t, _ = d
                    dtype.append((n, t))
                self.__provider_columns[prov].append(n)
        self.__output = np.zeros(nrows, dtype=dtype)

    def write_output_for(self, indexes: Union[np.ndarray, slice], neighbor_info: np.ndarray):
        for p in self.__providers:
            p.generate_output(indexes, neighbor_info, output=self.__output)

    def get_data_for_provider(self, provider):
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
    return np.array(
        [(0, [0, 1, 2], [0.10, 0.60, 0.00]), (1, [0, 1, 2], [0.30, 0.05, 0.80])],
        dtype=[('ID', int), ('NEIGHBOR_INDEX', np.float32, 3), ('NEIGHBOR_WEIGHTS', np.float32, 3)])


@pytest.fixture
def sampler(mock_provider, reference_ids):
    sampler = McSampler(take_n=200, mc_provider=mock_provider, ref_ids=reference_ids)
    return sampler
