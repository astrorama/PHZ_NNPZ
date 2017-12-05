Reference sample file format
============================

Due to the big size of the reference sample SED and PDZ data, NNPZ uses a custom
binary format for storing the reference sample. Having in mind that the amount
of data will not fit in the RAM memory, the design is trying to optimize the
random access of the reference sample objects during the execution and to
facilitate appending new templates and PDZs.

Reference sample files organization
-----------------------------------

To facilitate access and appending, the reference sample information is stored
in three different files. The first file stores the SED template data, the
second stores the PDZ data and the third one acts as an index for accessing the
previous ones. All these files are grouped inside a directory, which represents
the reference sample.

SED template data file
----------------------

The SED template data of the reference sample are stored in a binary file
following a custom format. The file is named `sed_data.bin`. The data of each
template are stored sequentially in the following way:

![](images/SED_data_file_format.png)

- The first 8 bytes of the template is its identifier (a long signed integer)
- The next 4 bytes contain the number of points (n) the template consists of (as
  an unsigned integer)
- The length is followed by the template data, organized in pairs of single
  precision decimal values (4 bytes), representing the wavelength and the flux
  density of the template.

This format allows to perform a fast read of any template knowing its initial
position in the file. Note that units are not supported by the format and they
are assumed to be &#x212B; for the wavelength and
erg/s/cm<sup>`2`</sup>/&#x212B; for the flux density.

When a new template is added to the file, it is always added at the end. This
way the location of the templates is persistent for the lifetime of the file.
Removal or modification of templates is not supported. In this cases a new file
has to be created.

Note that the SED templates can have different number of points from each other.

PDZ data file
-------------

The PDZ data file stores the PDZs of the reference sample, which are produced
using higher quality photometry. The file is named `pdz_data.bin`. Like the SED
data file, it follows a custom binary format. One big difference between the two
formats is that all the PDZs in the file share the same redshift bins. The file
is organized the following way:

![](images/PDZ_file_format.png)

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

Index file
----------

The index file contains information to easily retrieve the SED template and PDZ
data. The file is named `index.bin` and it also follows a custom binary format.
The information of each reference sample object is stored sequentially in the
following way:

![](images/Index_file_format.png)

- The first 8 bytes contain the object identifier as a long signed integer
- The next 8 bytes contain the position of the SED of the object in the SED data
  file, as a long signed integer
- The last 8 bytes contain the position of the PDZ of the object in the PDZ data
  file, as a long signed integer

If the PDZ or SED data file does not yet contain data for a given template (for
example if the SED file is created first and the PDZ file in a second run or
vice versa), the corresponding column contain the value -1. Because of the
appending rules of the data binary files, if the index file contains a position
different than -1, all previous positions must also be different than -1.
Similarly, if a position is marked as -1, all the consecutive positions must be
-1.

File validation rules
---------------------

To guarantee the consistency of the files at any moment, the following must be
true:

- All rows after the first one with SED_POS set to -1 must also have SED_POS set
  to -1
- All the values in the SED_POS column of the index file must be smaller than
  the size of the SED template data file
- For each row of the index file with SED_POS not -1, reading a long integer
  from this position of the SED template data file must return the value of the
  ID column
- All rows after the first one with PDZ_POS set to -1 must also have SED_POS set
  to -1
- All the values in the PHZ_POS column of the index file must be smaller than
  the size of the PDZ data file
- For each row of the index file with PDZ_POS not -1, reading a long integer
  from this position of the PHZ data file must return the value of the ID column

