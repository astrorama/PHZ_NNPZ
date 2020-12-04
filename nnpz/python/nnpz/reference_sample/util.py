import os

from nnpz.exceptions import FileNotFoundException
from nnpz.reference_sample import IndexProvider


def locate_existing_data_files(pattern):
    """
    Returns a set with the indices of the existing data files following the pattern
    """
    result = set()
    i = 1
    while os.path.exists(pattern.format(i)):
        result.add(i)
        i += 1
    return result


def validate_data_files(pattern: str, index: IndexProvider, key: str):
    """
    Cross-check the existing data files and those referenced by the index

    Returns:
        A set of existing file ids
    """
    existing_files = locate_existing_data_files(pattern)
    index_files = index.getFiles(key)
    if not existing_files.issuperset(index_files):
        missing_files = index_files.difference(existing_files)
        missing_files = list(map(pattern.format, missing_files))
        raise FileNotFoundException(
            'Missing {} data files: {}'.format(key, ', '.join(missing_files))
        )
    return existing_files
