.. _reference-sample-api:

Reference Sample API
********************

|NNPZ| uses `numpy's format <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>`_
for storing the reference sample data. A
detailed description of the format can be found in the :ref:`developer manual <dev-reference-sample>`.
The |NNPZ| library contains the class ``ReferenceSample`` to facilitate handling
of the reference sample via Python. The following sections describe how to
create and use a reference sample. Note that all code assumes that you have
already included the ``ReferenceSample`` class with the following code:

.. code:: python

  from nnpz.reference_sample import ReferenceSample


Accessing an existing reference sample
======================================

To access the data in an already existing reference sample directory you just
need to create a new ``ReferenceSample`` instance:

.. code:: python

  # Accessing an existing reference sample directory
  sample = ReferenceSample('/path/to/your/reference/sample/dir')

You can check how many objects contains the reference sample by calling the
``len(sample)`` method. You can get a list with all the object ids by calling the
``get_ids()`` method:

.. code:: python

  # Print all object IDs if they are not too many
  if len(sample) < 10:
      print(sample.get_ids())
  else:
      print('Too many objects to print their IDs!')

To visit all the objects of the reference sample, you can call the ``iterate()``
method, which returns an iterable which you can directly use in a python loop:

.. code:: python

  # This will print the IDs of all the objects in the reference sample
  for obj in sample.iterate():
    print(obj.id)

The objects of the iteration provide the following members:

id
  The identifier of the object in the reference sample (a 64 bit integer)

sed
  The data of the |SED| template of the object. It is a 2D numpy array
  where the first axis represents the knots of the |SED| and the second has
  always size 2, representing the wavelength (expressed in |AA|) and the
  flux value (expressed in erg/s/cm²/|AA|) respectively.

pdz
  The data of the |PDZ| of the object. It is a 2D numpy array where the
  first axis represents the knots of the |PDZ| and the second has always size 2,
  representing the redshift and the probability (in the range [0,1]).

The ``id`` member is the unique identifier of the object in the reference sample
and is always set, but the reference sample might not contain the |SED| template
or the |PDZ| of an object. In this case, the members ``sed`` and ``pdz`` will be set
to ``None`` accordingly. The order of the iteration is guaranteed
to be the same as the order of the IDs returned by the ``get_ids()`` method.

The second way to access the reference sample data is by using an objects ID as
shown at the following example:

.. code:: python

  # Get the SED and PDZ data of the object with ID 192415
  sed = sample.get_sed_data(192415)
  pdz = sample.get_pdz_data(192415)

The returned objects are numpy arrays, following the same format as the members
of the iteration objects.

Since |NNPZ| 1.0, the recommended method to get the data is

.. code:: python

  # Note that the names for the providers depend on the configuration
  sed = sample.get_data('SedProvider', 192415)
  pdz = sample.get_data('PdzProvider', 192415)
  pp = sample.get_data('MontecarloProvider', 192415)

This allows for greater flexibility, since multiple providers of the same
type can be registered under different names.

Creating a new reference sample
===============================

If you want to create a reference sample in a new directory you can use the
provided factory method:

.. code:: python

  # Creating a new reference sample directory
  # The method accepts a parameter provider, a dictionary where the key
  # is the provider name, and the value the provider setting
  # Defaults to:
  #  {
  #    'PdzProvider': {'name': 'pdz', 'data': 'pdz_data_{}.npy'},
  #    'SedProvider': {'name': 'sed', 'data': 'sed_data_{}.npy'}
  #  }
  sample = ReferenceSample.create('/your/new/reference/sample/directory')

The above command will create the given directory as well as the binary files
defined by the NNPZ reference sample format. When first created the reference
sample will contain no objects. For convenience, the factory method returns a
```ReferenceSample`` instance, which can be used to populate the reference sample
with objects, as described in the next sections.


Adding data to a reference sample
---------------------------------

Adding a new object to the reference sample is done under demand by the
individual providers. Since |NNPZ| 1.0 there is no need to add and then
associate data separately.

.. code:: python

  # Add ten new objects in the sample
  # The data layout is provider dependent. For PDZs, the first entry
  # must be the binning of the PDZ
  # The binning *must* be present on all calls to add_data, even if already
  # initialized
  pdz_prov = sample.get_provider('PdzProvider')
  pdz_prov.add_data(object_ids=np.arange(10), data=np.zeros((11, 200)))

  pp_prov = sample.get_provider('MontecarloProvider')
  pp_prov.add_data(object_ids=np.arange(10), ...)

The given ID must be a (64 bit signed) integer. Note that the new objects are
always added at the end of the reference sample and they will be the last ones
accessed during an iteration.

The only difference is that the |PDZ| of all the objects of the reference sample
must have the same X axis, which is defined the the X axis of the first PDZ
added. Note that the X axis still needs to be passed to the ``addPdzData()``
method for ALL the objects. This is done to avoid corrupting by mistake the
reference sample with PDZs with X axis of correct length but wrong values.
