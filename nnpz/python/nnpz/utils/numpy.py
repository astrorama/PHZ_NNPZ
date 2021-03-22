from typing import List

import numpy as np


def recarray_flat_view(data: np.recarray, fields: List[str]):
    """
    Return a view as a flat array into a set of fields on a structured array
    Args:
        data: np.recarray
            Original structured array
        fields: List of field names
    Returns: np.array
        A flat view of the selected set of fields
    Raises:
        TypeError:
            If the set of fields have different types
        IndexError:
            If the fields are not consecutive on the recarray
    """
    selected = [data.dtype.fields[c] for c in fields]
    dtypes = [f[0] for f in selected]
    offsets = np.array([f[1] for f in selected])
    sizes = np.array([data.dtype[f].itemsize for f in fields])
    if len(set(dtypes)) > 1:
        raise TypeError('All fields must have the same type')

    consecutive = (offsets[:-1] + sizes[:-1] == offsets[1:]).all()
    if not consecutive:
        raise IndexError('All fields must be consecutive')

    view = np.ndarray(data.shape, dtype=(dtypes[0], len(dtypes)), buffer=data, offset=offsets[0],
                      strides=data.strides)
    assert np.may_share_memory(view, data)
    return view
