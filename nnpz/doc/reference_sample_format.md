Reference sample format
=======================

NNPZ reference sample consist on a set of collection of arrays
stored on disk as [.npy files](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).

Different attributes are stored as independent set of files, each one comprised of

1. An index file
2. An ensemble of data files

Data is partitioned due to the big size of the reference sample.

NNPZ handles, by default, a set of files for the SED, and a set of files for the PDZ.

Index file
----------

An index file is an array of 64 bits ints and shape `(n.entries, 3)`, where the
first column corresponds to the object ID, the second to the file index, and the third
to the offset within that file (as an array index).

Example:

```python
In [1]: import numpy as np
In [2]: import matplotlib.pyplot as plt

In [3]: sed_idx = np.load('sed_index.npy')
In [4]: sed_idx[sed_idx[:,0] == 1574081264730001]
Out[4]: array([[1574081264730001,                1,            49972]])
# Associated SED is on sed_data_1.npy, position 49972
```

SED data files
--------------

Each SED data file is an array of single precision floats and three axes:

1. The first axis corresponds to the number of objects on the array
2. The second axis corresponds to the number of knots for the stored SEDs
3. The third axis contains the wavelength (position 0) and the flux (position 1)

Units are not supported by the format and they are assumed to be
&#x212B; for the wavelength and erg/s/cm<sup>`2`</sup>/&#x212B; for the flux density.

Note that different SEDs may have different resolutions. They will be organized so
SEDs with the same number of knots are kept together on the same set of files.

By default, the index is named `sed_index.npy` and the data files `sed_data_XX.npy`.

Example:

```python
In [5]: sed_data = np.load('sed_data_1.npy')
In [6]: sed = sed_data[49972]
In [7]: plt.plot(sed[:,0], sed[:,1])
```

PDZ data files
--------------

The PDZ data files store the PDZs of the reference sample, which are produced
using higher quality photometry. The files are named `pdz_data_XX.npy` and the
index `pdz_index.npy`. The data is an array with two axes:

1. The first axis corresponds to the number of objects on the array, plus
   one extra for the bin values
2. The second axis corresponds to the number of bins

Unlike SEDs, PDZs are all assumed to have the same number of bins. The corresponding
Z for each bin is stored on the very first position of every array.

Example:

```python
In [8]: pdz_data = np.load('pdz_data_1.npy')
In [9]: pdz_bins =  pdz_data[0]
# Let's assume we know we want the PDZ on the position 49973
In [10]: pdz = pdz_data[49973]
In [11]: plt.plot(pdz_bins, pdz)
```
