from typing import List, Union

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from nnpz.io import OutputHandler
from nnpz.io.output_column_providers.McSampler import McSampler
from nnpz.photometry.colorspace import RestFrame
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry


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

    def get_data_for_index(self, obj_idx: np.ndarray):
        data = self.__data[obj_idx[:, 0], obj_idx[:, 1]]
        data[obj_idx[:, 0] < 0] = 0
        return data

    def get_data(self, obj_id: int):
        return self.__data[obj_id]

    def get_dtype(self, param=None) -> np.dtype:
        if param is None:
            return self.DTYPE
        return self.DTYPE[param]

    def get_n_samples(self) -> int:
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
        units = []
        for prov in self.__providers:
            self.__provider_columns[prov] = []
            prov_def = prov.get_column_definition()
            for d in prov_def:
                if len(d) == 4:
                    n, t, unit, s = d
                    dtype.append((n, t, s))
                    units.append(unit)
                else:
                    n, t, unit = d
                    dtype.append((n, t))
                    units.append(unit)
                self.__provider_columns[prov].append(n)
        self.__output = Table(np.zeros(nrows, dtype=dtype), units=units)

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
    sampler = McSampler(take_n=200, mc_provider=mock_provider)
    return sampler


class DummyPhotometry:
    def __init__(self):
        self._data = np.zeros((2, 2),
                              dtype=[('A', np.float32), ('B', np.float32), ('C', np.float32)])
        self._data['A'][:, 0] = [0.5626045, 0.94242]
        self._data['B'][:, 0] = [1.7679665, 1.8930919]
        self._data['C'][:, 0] = [3.5187345, 2.606762]

    def getData(self, *filter_list):
        data = np.zeros((len(self._data), len(filter_list), 2), dtype=np.float32)
        for i, f in enumerate(filter_list):
            data[:, i, 0] = self._data[f][:, 0]
        return data


@pytest.fixture
def reference_photometry():
    return Photometry(
        ids=np.arange(2), values=np.array(
            [((0.5626045, 0.), (1.7679665, 0.), (3.5187345, 0.)),
             ((0.94242, 0.), (1.8930919, 0.), (2.606762, 0.))]) * u.uJy,
        system=PhotometricSystem(['A', 'B', 'C']), colorspace=RestFrame)


@pytest.fixture()
def reference_matched_photometry(reference_photometry):
    matched = np.zeros((1, 2, 3, 2), dtype=np.float32)
    matched[0, :, 0, 0] = reference_photometry.values[:, 0, 0] * np.array([1., 2.])
    matched[0, :, 1, 0] = reference_photometry.values[:, 1, 0] * np.array([1.5, 1.7])
    matched[0, :, 2, 0] = reference_photometry.values[:, 2, 0] * np.array([1.2, 1.5])
    return matched


@pytest.fixture
def target_photometry():
    return Photometry(
        ids=np.array([0]), values=np.array(
            [((10., 0.), (10., 0.), (10., 0.))]) * u.uJy,
        system=PhotometricSystem(['A', 'B', 'C']), colorspace=RestFrame)


class MockPdzProvider:
    def __init__(self, nbins=1500):
        self.__bins = np.arange(nbins)

    def get_redshift_bins(self):
        return self.__bins

    def get_pdz_for_index(self, index):
        output = np.zeros((len(index), len(self.__bins)))
        for i, idx in enumerate(index):
            output[i][idx - 1] = 1.
        return output


class MockReferenceSample:
    def __init__(self, **kwargs):
        self.__providers = kwargs

    def get_provider(self, key):
        return self.__providers[key]


@pytest.fixture
def reference_sample() -> MockReferenceSample:
    return MockReferenceSample(pdz=MockPdzProvider())


@pytest.fixture
def neighbor_info() -> np.ndarray:
    ninfo = np.zeros(50, dtype=[('NEIGHBOR_INDEX', int, 30), ('NEIGHBOR_WEIGHTS', float, 30)])
    ninfo['NEIGHBOR_INDEX'] = (1 + np.arange(30)[np.newaxis])
    ninfo['NEIGHBOR_INDEX'] *= (1 + np.arange(50)[:, np.newaxis])
    ninfo['NEIGHBOR_WEIGHTS'] = 10 ** -np.linspace(0, 10, 30)[np.newaxis]
    return ninfo
