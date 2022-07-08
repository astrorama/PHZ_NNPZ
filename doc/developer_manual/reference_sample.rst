.. _dev-reference-sample:

Reference Sample
****************

The reference sample consists of a collection of ``.npy`` files plus
a photometry file stored in the |FITS| format.

The main ``.npy`` file is ``index.npy``.

``index.npy``
=============

The index file contains information to easily retrieve the data files.
The file contains a numpy ``recarray`` containing 64-bits integer values.

The only strictly required field is ``id``, containing a unique identifier
for the object. For each registered provider at the time of creation,
a pair containing the file ID and file offset is present.
For instance, if there are |PDZ| and |PP| data registered, the fields will be

pdz_file
  File ID, interpolated into the string contained on the ``data`` configuration:
  i.e. ``pdz_data_{}.npy`` => ``pdz_data_1.npy``
pdz_offset
  Offset inside the SED data file
pp_file
  See ``pdz_file``
pp_offset
  See ``pdz_offset``

If the |PDZ| or |PP| data files do not yet contain data for a given object,
the corresponding file index column contains the value 0 and the
corresponding location column contains the value -1. Note that the order the
objects are stored in the |SED| and |PDZ| data do not need to match the order in
the index file, but it is strongly recommended for performance
(sequential I/O).

SED template data files
=======================

The |SED| template data of the reference sample are stored in numpy arrays
with the shape (number of objects, number of knots, 2). The last axis
corresponds to wavelength (index 0), and flux density (index 1).

Note that units are not supported by the format and they
are assumed to be |AA| for the wavelength and erg/s/cm²/|AA| for the flux density.

When a new template is added to the file, it is always added at the end. This
way the location of the templates is persistent for the lifetime of the file.
Removal or modification of templates is not supported. In this cases a new file
has to be created.

Note that |SED| templates with a different number of knots must be kept on
separate files (e.g. file 1 stores SEDs with 1000 knots, and
file 2 SEDs with 1500 knots).

|NNPZ| will keep the size of the data files under a configurable limit, which
defaults to 1 GiB.


PDZ data files
==============

The |PDZ| data are stored in numpy arrays with the shape
(number of objects, number of knots). All stored |PDZ| must have been computed
using the same binning, which is stored on position 0 of each individual file.

|NNPZ| will keep the size of the data files under a configurable limit, which
defaults to 1 GiB.


.. _reference-photometry-format:

Photometry files format
***********************

NNPZ stores the photometry of the reference sample in FITS files, with the
conventions as described in the following sections.

Primary HDU
===========

The photometry files contain only binary tables. For this reason the primary HDU
of the FITS file is ignored. When NNPZ produces photometry files this HDU will
contain an empty array.

Photometry HDU
==============

The first extension HDU is a binary table, which contain the photometry values.

The extension name (``EXTNAME`` header keyword) is always set to the string
``NNPZ_PHOTOMETRY``. This name is used by NNPZ to detect photometry files, so it
should never be changed.

The header also contains the keyword ``PHOTYPE`` which indicates the type of the
photometry values stored in the file. Since |NNPZ| 1.0, the only supported type is
``F_nu_uJy``.

The first column of the table is always named ``ID`` and contains 64 bit signed
integers, which match the identifiers of the reference sample objects. The rest
of the columns contain 32 bit floating point numbers and represent the
photometry values, EBV correction factors, and filter variation correction factors.
The names of the bands are extracted from the column names.
The bands can optionally have an error associated with them, in which case there
must be a column with the same name and the postfix ``_ERR``. For example, if the
table contain a column named ``g``, the error column must be named ``g_ERR``.
If EBV and filter variation correction are available, their column names
would be ``g_EBV_CORR`` and ``g_SHIFT_CORR`` respectively.

Filter transmission HDUs
========================

The rest of the HDUs in the fits file contain the filter transmissions of the
bands. They are binary tables with two columns, the first of which contains the
wavelength values (expressed in |AA|) and the second the filter transmission
(a number in the range [0,1]). Both columns contain 32 bit floating point
numbers. The extension names (``EXTNAME`` header keyword) is the same with the
band name and it is used for identifying the filter transmissions (the order
does not matter).
