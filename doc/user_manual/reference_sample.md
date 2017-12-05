Reference Sample
================

The NNPZ uses a custom binary format for storing the reference sample data. A
detailed description of the format can be found
[here](../reference_sample_format.md). Te NNPZ library contains the class
`ReferenceSample` to facilitate handling of the reference sample via Python. The
following sections describe how to create and use a reference sample. Note that
all code assumes that you have already included the `ReferenceSample` class with
the following code:

```python
from nnpz import ReferenceSample
```

Reading a reference sample directory
------------------------------------

To access the data in an already existing reference sample directory you just
need to create a new `ReferenceSample` instance:

```python
# Accessing an existing reference sample directory
sample = ReferenceSample('/path/to/your/reference/sample/dir')
```

To visit all the objects of the reference sample, you can call the `iterate()`
method, which returns an iterable:

```python
# This will print the IDs of all the objects in the reference sample
for obj in sample.iterate():
    print(obj.id)
```

The objects of the iteration provide the following members:

- `id`: The identifier of the object in the reference sample (a 64 bit integer)
- `sed`: The data of the SED template of the object. It is a 2D numpy array
    where the first axis represents the knots of the SED and the second has
    always size 2, representing the wavelength (expressed in &#x212B;) and the
    flux value (expressed in erg/s/cm<sup>2</sup>/&#x212B;) respectively.
- `pdz`: The data of the PDZ of the object. It is a 2D numpy array where the
    first axis represents the knots of the PDZ and the second has always size 2,
    representing the redshift and the probability (in the range [0,1]).