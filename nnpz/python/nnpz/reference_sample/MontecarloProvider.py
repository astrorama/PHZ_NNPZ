from nnpz.reference_sample import MontecarloDataProvider
from nnpz.reference_sample.BaseProvider import BaseProvider
from nnpz.reference_sample.util import validate_data_files


class MontecarloProvider(BaseProvider):
    """
    A Montecarlo provider contains a set of N samples, each with D dimensions.
    Two examples would be:
        * Physical Parameters, where each dimension is a different physical parameter
        * Intermediate Bands for galaxies, where each dimension corresponds to the photometry for a
          given IB filter

    Args:
        *args:
            Forwarded to BaseProvider
        **kwargs:
            Forwarded to BaseProvider
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_files = validate_data_files(self._data_pattern, self._index, 'MontecarloProvider')

        self._data_map = {}
        for data_file in data_files:
            self._data_map[data_file] = MontecarloDataProvider(
                self._data_pattern.format(data_file)
            )

    def flush(self):
        """
        Write the changes to disk.
        """
        self._index.flush()
        for data_prov in self._data_map.values():
            data_prov.flush()
