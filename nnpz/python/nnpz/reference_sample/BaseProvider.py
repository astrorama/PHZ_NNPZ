import pathlib
from typing import Union

from nnpz.reference_sample import IndexProvider


class BaseProvider(object):
    """
    Base for different set of providers. A provider groups an index and a set of data file
    providers.
    Args:
        index_path:
            Full path to the index file
        data_pattern:
            Full path to the data files, as a pattern string (i.e. '.../blah_data_{}.npy')
            The brackets will be replaced with the file identification.
        data_limit:
            Partition data on files of maximum this size, in bytes.
        extra_data:
            Additional metadata for the provider
    """

    def __init__(self, index_path: Union[str, pathlib.Path], data_pattern: str, data_limit: int,
                 extra_data: dict):
        self._index_path = index_path
        self._data_pattern = data_pattern
        self._data_limit = data_limit
        self.extra = extra_data
        self._index = IndexProvider(self._index_path)

    def __len__(self):
        return len(self._index)

    @property
    def index_path(self):
        return self._index_path

    @property
    def data_pattern(self):
        return self._data_pattern

    def getIds(self):
        """
        Returns a list of long integers with the IDs
        """
        return self._index.getIds()
