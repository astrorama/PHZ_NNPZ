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
in three different types of files. The first type stores the SED template data,
the second stores the PDZ data and the third one acts as an index for accessing
the previous ones. All these files are grouped inside a directory, which
represents the reference sample.

SED template data files
-----------------------

The SED template data of the reference sample are stored in binary files
following a custom format. Because of the high volume of the SED template data,
there are multiple of such files, to limit the size of each file to around 1GB.
The files are named `sed_data_XX.bin`, where the XX is the number of the file.
The data of each template are stored sequentially in the following way:

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

The way that the files are split is to fully write to a file the SED template
which makes the file exceeding the 1GB size. Then the next template is stored at
the beginning of the next file.

PDZ data files
--------------

The PDZ data files store the PDZs of the reference sample, which are produced
using higher quality photometry. The files are named `pdz_data_XX.bin` and the
splitting of the files follows the same rules like the SED template files. The
files follow a custom binary format, like the SED template data files. One big
difference between the two formats is that all the PDZs in the file share the
same redshift bins. The file is organized the following way:

![](images/PDZ_file_format.png)

- The first 4 bytes contain the number of points (n) that each PDZ consists of
  (as an unsigned integer)
- The length is followed by n single precision decimal values (4 bytes each)
  representing the redshift bin values
- This concludes the header part of the file, which contains information common
  to all PDZs. Note that this information is duplicated to all of the PDZ data
  files.
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
- The next 2 bytes contain the file index in which the SED is located, as a sort
  unsigned integer
- The next 8 bytes contain the position of the SED of the object in the SED data
  file, as a long signed integer
- The next 2 bytes contain the file index in which the PDZ is located, as a sort
  unsigned integer
- The last 8 bytes contain the position of the PDZ of the object in the PDZ data
  file, as a long signed integer

If the PDZ or SED data file does not yet contain data for a given template (for
example if the SED file is created first and the PDZ file in a second run or
vice versa), the corresponding file index column contains the value 0 and the
corresponding location column contains the value -1. Note that the order the
objects are stored in the SED and PDZ data does not need to match the order in
the index file. This allows for having intermediate objects with missing data.

File validation rules
---------------------

To guarantee the consistency of the files at any moment, the following must be
true:

- All the values in the SED_FILE column of the index file must map to an
  existing file
- All the values in the SED_POS column of the index file must be smaller than
  the size of the corresponding SED template data file
- For each row of the index file with SED_POS not -1, reading a long integer
  from this position of the corresponding SED template data file must return the
  value of the ID column
- All the values in the PDZ_FILE column of the index file must map to an
  existing file
- All the values in the PDZ_POS column of the index file must be smaller than
  the size of the PDZ data file
- For each row of the index file with PDZ_POS not -1, reading a long integer
  from this position of the corresponding PDZ data file must return the value of
  the ID column

