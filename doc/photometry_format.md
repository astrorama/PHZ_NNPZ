Photometry files format
=======================

The NNPZ stores the photometry of the reference sample in FITS files, with the
conventions as described in the following sections.

Primary HDU
-----------

The photometry files contain only binary tables. For this reason the primary HDU
of the FITS file is ignored. When NNPZ produces photometry files this HDU will
contain an empty array.

Photometry HDU
--------------

The first extension HDU is a binary table, which contain the photometry values.

The extension name (`EXTNAME` header keyword) is always set to the string
`NNPZ_PHOTOMETRY`. This name is used by NNPZ to detect photometry files, so it
should never be changed.

The header also contains the keyword `PHOTYPE` which indicates the type of the
photometry values stored in the file. The different photometry types are the
following:

- `Photons`: The photometry values are photon count rates, expressed in
    counts/s/cm<sup>2</sup>
- `F_nu`: The photometry values are energy flux densities expressed in
    erg/s/cm^2/Hz
- `F_nu_uJy`: The photometry values are energy flux densities expressed in
    &mu;Jy
- `F_lambda`: The photometry values are energy fluxes densities expressed in
    erg/s/cm^2/&#x212B;
- `MAG_AB`: The photometry values are AB magnitudes

The first column of the table is always named `ID` and contains 64 bit signed
integers, which match the identifiers of the reference sample objects. The rest
of the columns contain 32 bit floating point numbers and represent the
photometry values. The names of the bands are extracted from the column names.
The bands can optionally have an error associated with them, in which case there
must be a column with the same name and the postfix `_ERR`. For example, if the
table contain a column named `g`, the error column must be named `g_ERR`.

Filter transmission HDUs
------------------------

The rest of the HDUs in the fits file contain the filter transmissions of the
bands. They are binary tables with two columns, the first of which contains the
wavelength values (expressed in &#x212B;) and the second the filter transmission
(a number in the range [0,1]). Both columns contain 32 bit floating point
numbers. The extension names (`EXTNAME` header keyword) is the same with the
band name and it is used for identifying the filter transmissions (the order
does not matter).