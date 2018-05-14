Reference Sample
================

- [Accessing an existing reference sample](#accessing-an-existing-reference-sample)
- [Creating a new reference sample](#creating-a-new-reference-sample)
- [Adding data to a reference sample](#adding-data-to-a-reference-sample)

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

You can check how many objects contains the reference sample by calling the
`size()` method. You can get a list with all the object ids by calling the
`getIds()` method:

```python
# Print all object IDs if they are not too many
if sample.size() < 10:
    print(sample.getIds())
else:
    print('Too many objects to print their IDs!')
```

To visit all the objects of the reference sample, you can call the `iterate()`
method, which returns an iterable which you can directly use in a python loop:

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
that all following will also be missing. The order of the iteration is guaranteed
to be the same as the order of the IDs returned by the `getIds()` method.

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


Adding data to a reference sample
---------------------------------

Adding a new object to the reference sample is done by calling the
`createObject()` method, which gets as parameter the ID of the new object:

```python
# Add ten new objects in the sample
for obj_id in range(10):
    sample.createObject(obj_id)
```

The given ID must be a (64 bit signed) integer. Note that the new objects are
always added at the end of the reference sample and they will be the last ones
accessed during an iteration.
 
The newly created objects are not associated yet with any SED or PDZ data.
Accessing these data for the newly added IDs will return `None`. Adding SED data
of an object is done by using the `addSedData()` method:

```python
# Add the SED data for the object with ID 1
sample.addSedData(1, [(1,1), (2,2), (3,3)])
```

The first argument of the method is the ID of the object to add the SED for and
the second is the daa of the SED as a two dimensional array-like object which
can be converted to a numpy array (in the example a list of tuples is used).

Even though the ID is passed as an argument to the `addSedData()` method, due to
the sequential way the data are stored in the binary files, the SED data cannot
be set in a random order, but sequentially (the ID argument is only used to
avoid corruption of the reference sample by wrong calls). The ID of the first
object which does not have the SED set (and is the next to call the
`addSedData()` for) can be retrieved with the following call:

```python
# Get the ID of the first object not having the SED set yet
obj_id = sample.firstIdMissingSedData()
```

If the SEDs of all the objects in the reference sample are already set, the
method returns `None`.

The method to add the PDZs of the objects is identical with adding the SEDs, but
it uses the `addPdzData()` and `firstIdMissingPdzData()` methods instead:

```python
# Add the PDZ data for the next object
obj_id = sample.firstIdMissingPdzData()
sample.addPdzData(1, [(1,1), (2,2), (3,3)])
```

The only difference is that the PDZs of all the objects of the reference sample
must have the same X axis, which is defined the the X axis of the first PDZ
added. Note that the X axis still needs to be passed to the `addPdzData()`
method for ALL the objects. This is done to avoid corrupting by mistake the
reference sample with PDZs with X axis of correct length but wrong values.