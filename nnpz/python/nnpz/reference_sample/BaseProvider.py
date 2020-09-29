import os
from glob import glob

from nnpz.reference_sample import IndexProvider


class BaseProvider(object):
    """
    Base for different set of providers. A provider is a set of data file providers.

    Args:
        index:
            An index provider
        key:
            The key to use on the index
        data_pattern:
            Full path to the data files, as a pattern string (i.e. '.../blah_data_{}.npy')
            The brackets will be replaced with the file identification.
        data_limit:
            Partition data on files of maximum this size, in bytes.
        extra_data:
            Additional metadata for the provider
        overwrite:
            If true, remove the existing datafiles and clean the index if already set
    """

    def __init__(self, index: IndexProvider, key: str, data_pattern: str, data_limit: int,
                 extra_data: dict, overwrite: bool = False):
        self._index = index
        self._key = key
        self._data_pattern = data_pattern
        self._data_limit = data_limit
        self.extra = extra_data

        if overwrite:
            self._clean_existing()

    def _clean_existing(self):
        for path in glob(self._data_pattern.format('*')):
            os.unlink(path)
        self._index.clear(self._key)

    @property
    def data_pattern(self):
        return self._data_pattern
