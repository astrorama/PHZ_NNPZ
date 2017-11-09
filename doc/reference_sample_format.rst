.. include:: substitutions.txt

.. _reference_sample_format:

Reference sample file format
============================

Due to the big size of the reference SED data and PDZs a custom binary format is
designed. Having in mind that the amount of data will not fit in the RAM memory,
the design is trying to optimize the random SED access during the execution and
to facilitate appending new templates.

Reference sample files organization
-----------------------------------

To facilitate access and appending, the reference sample information is stored
in four different files. The first stores the SED template data, the second
stores the PDZ data, the third the photometries via the reference set of filters
and the last one acts as an index for accessing all the previous ones. The files
are grouped inside a directory.

SED template data file
----------------------

The SED template data are stored in a binary file following a custom format. The
data of each template are stored sequentially in the following way:

.. image:: images/SED_data_file_format.png

- The first 8 bytes of the template is its identifier (a long signed integer)
- The next 4 bytes contain the number of points (n) the template consists of (as
  an unsigned integer)
- The length is followed by the template data, organized in pairs of single
  precision decimal values (4 bytes), representing the wavelength and the flux
  density of the template.

This format allows to perform a fast read of any template knowing its initial
position in the file. Note that units are not supported by the format and they
are assumed to be |AA| for the wavelength and erg/s/cm\ :sup:`2`/|AA| for the
flux density.

When a new template is added to the file, it is always added at the end. This
way the location of the templates is persistent for the lifetime of the file.
Removal or modification of templates is not supported. In this cases a new file
has to be created.

Note that the SED templates can have different number of points from each other.

PDZ data file
-------------

The PDZ data file stores the PDZs of the reference sample, which are produced
using higher quality photometry. Like the SED data file, it follows a custom
binary format. One big difference between the two formats is that all the PDZs
in the file share the same redshift bins. The file is organized the following
way:

.. image:: images/PDZ_file_format.png

- The first 4 bytes contain the number of points (n) that each PDZ consists of
  (as an unsigned integer)
- The length is followed by n single precision decimal values (4 bytes each)
  representing the redshift bin values
- This concludes the header part of the file, which contains information common
  to all PDZs
- After the header, the PDZ data are stored sequentially
- The first 8 bytes of the PDZ is the SED identifier (a long signed integer)
- The ID is followed by n single precision decimal values (4 bytes each),
  representing the PDZ value for each redshift bin.

Photometry file
---------------

The photometry file contains the photometries of the SED templates as seen
through a reference set of filters. It is a FITS table, where the first column
represents the ID of the corresponding template (long integer type of 8 bytes),
followed by as many columns as the reference filters, which keep the flux
density of each filter, expressed in |mu|\Jansky. All flux columns are double
precision floating point (8 bytes).

Index file
----------

The index file contains information to easily retrieve the template and PDZ
data. It is a FITS table containing a row for each template, with the following
columns:

- **ID:** 
  The identifier of the template (long integer of 8 bytes)
- **SED_POS:** 
  The position of the template in the SED template data file, from the beginning
  of the file
- **PDZ_POS:** 
  The position of the PDZ data in the PDZ file from the beginning of the file

If the PDZ data file does not yet contain data for a given template (for example
if the SED file is created first and the PDZ file in a second run) the
corresponding column contain the value -1. Because of the appending rules of the
data binary files, if the index file contains a position different than -1, all
previous positions must also be different than -1. Similarly, if a position is
marked as -1, all the consecutive positions must be -1.

Note that the SED_POS column cannot contain -1 entries. This forces to provide
the SED data for creating a new row in the index file.

File validation rules
---------------------

To guarantee the consistency of the four files at any moment the following must
be true:

- All the values in the SED_POS column of the index file must be smaller than
  the size of the SED template data file
- For each row of the index file, reading a long integer from this position of
  the SED template data file must return the value of the ID column
- All the values in the PHZ_POS column of the index file must be smaller than
  the size of the PDZ data file
- For each row of the index file with PDZ_POS not -1, reading a long integer
  from this position of the PHZ data file must return the value of the ID column
- The rows of the photometry and index files with the same index must have the
  same ID values

Before a reference sample is used for the NNPZ, the following requirements must
also be met:

- The PHZ_POS column of the index file does not contain any -1 entries
- The index and photometry tables must contain the same number of rows