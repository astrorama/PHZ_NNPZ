.. _release-notes:

Release notes
*************

1.0.0 - 2022-?-?
==================
* Split NNPZ functionality into smaller utilities (`#24883 <https://redmine.isdc.unige.ch/issues/24883>`_)
    - Fixes performance with sampling of physical parameters (`#24693 <https://redmine.isdc.unige.ch/issues/24693>`_)

0.14.0 - 2022-01-17
===================
* Migrate to EDEN 3.0
* Use Elements 6.0.1
* Remove support for Python 2
* Clean old, untested and unused code
* Add support for the correction factors for EBV and filter variation (`#16215 <https://redmine.isdc.unige.ch/issues/16215>`_)
* Fix mean Z when all neighbors have 0 weight
* Use a more fine grained grid for the interpolation

0.13.0 - 2021-07-02
===================
* More fixes for empty input catalogs  (`#25505 <https://redmine.isdc.unige.ch/issues/25505>`_)
* Fixes for handling existing reference samples where the index is not sorted

0.12.5 - 2021-06-0
===================
* Scale the neighbor photometry properly even in the presence of NaNs

0.12.4 - 2021-06-02
===================
* Fix McSampler when all neighbors have weight 0
* Fix McSampler when more points than available datapoints are requested

0.12.3 - 2021-05-26
===================
* Create an empty output for an empty input (`#25505 <https://redmine.isdc.unige.ch/issues/25505>`_)

0.12.2 - 2021-04-20
===================
* Use a read-only index when the filesystem is read-only

0.12.0 - 2021-04-14
===================
* Removed unused code
* Improvements of memory consumption (`#23871 <https://redmine.isdc.unige.ch/issues/23871>`_)
* Allow disabling output mean photo
* Issue only a warning when both neighbor and PDZ output configuration are disabled
* Uniform photometry output name configurable

0.11.2 - 2021-04-20
===================
* Fallback to read-only index

0.11.1 - 2021-02-03
===================
* Replace an assert by a warning

0.11.0 - 2021-02-01
===================
* Use a filter transmission as interpolation grid
* Avoid recomputing the filter normalization too many times
* Use float32 for PP
* Use -99 for filling value on thd McSliceAggregate
* Depend on Elements 5.12
* Do not keep all data files mmaped


0.10.0 - 2020-11-25
===================
* Fix binning in slice aggregate
* Check ref.sample IDs without the sort
* Add support for direct import of PDZ data
* Improvements of output for Montecarlo sampling
* Improve memory consumption (`#23070 <https://redmine.isdc.unige.ch/issues/23070>`_)

0.9.0 - 2020-09-22
==================
* Add free scaling of reference photometry (`#20771 <https://redmine.isdc.unige.ch/issues/20771>`_)
* Add NnpzDereddenPhotometry
* Go back to a single index file
* Fix BuildPhotometry when no MC is requested
* MC provider, and BuildPhotometry with MC
* New reference sample format
* Fix several warnings from pylint
* Add (C) to all files
* Several fixes for the QA

0.8 - 2020-03-27
================
* Update Elements dependency to 5.10.0
* Recursive DirectoryFilterProvider (`#20220 <https://redmine.isdc.unige.ch/issues/20220>`_)
* ``NnpzBuildPhotometry`` can run with multiple threads (``--parallel`` parameter)
* Add support for string columns to utils/Fits
* Interpolate filter and redshifted SED to a common grid (fixes loss of resolution at high redshift)

0.6 - 2019-11-21
================
* Update Elements dependency to 5.8

0.4 - 2019-06-14
================
* Changed the method used to compute the photometry

0.2 - 2019-01-14
================
* First stable release
