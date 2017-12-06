Reference Sample
================

- [Accessing an existing reference sample](#accessing-an-existing-reference-sample)
- [Creating a new reference sample](#creating-a-new-reference-sample)

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

Accessing an existing reference sample
--------------------------------------

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
    
The `id` member is the unique identifier of the object in the reference sample
and is always set, but the reference sample might not contain the SED template
or the PDZ of an object. In this case, the members `sed` and `pdz` will be set
to `None` accordingly. Note that because of the sequential way the data are
stored in the file, after meeting the first missing SED or PDZ, it is guaranteed
that all following will also be missing.

The second way to access the reference sample data is by using an objects ID as
shown at the following example:

```python
# Get the SED and PDZ data of the object with ID 192415
sed = sample.getSedData(192415)
pdz = sample.getPdzData(192415)
```

The returned objects are numpy arrays, following the same format as the members
of the iteration objects.

**TIP**: If you want to access the values of the two axes separately, use numpy
array slicing to avoid any unnecessary copying. For example, the following code
plots the sed using matplotlib, which gets the X and Y axes as separate lists:

```python
import matplotlib.pyplot as plt
plt.plot(sed[:,0], sed[:,1])
```

If you do not know the ID of the reference sample object you are interested
with, a list with all object IDs can be retrieved by using the `getIds()`
method:

```python
# Get the IDs of all objects in the reference sample
ids = sample.getIds()
```

The order of the returned IDs is guaranteed to be the same as the order the
objects are being iterated wen calling the `iterate()` method.


Creating a new reference sample
-------------------------------

If you want to create a reference sample in a new directory you can use the
provided factory method:

```python
# Creating a new reference sample directory
sample = ReferenceSample.createNew('/your/new/reference/sample/directory')
```

The above command will create the given directory as well as the binary files
defined by the NNPZ reference sample format. When first created the reference
sample will contain no objects. For convenience, the factory method returns a
`ReferenceSample` instance, which can be used to populate the reference sample
with objects, as described in the next secions.