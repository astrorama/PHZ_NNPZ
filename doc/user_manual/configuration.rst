.. _configuration:

Configuration file
******************

NNPZ can receive a configuration file via the command line argument
``--config-file``. **Any** parameter can be overridden via command line
just appending it to the end of the call. For instance:

.. code:: bash

  nnpz --config-file myconfig.conf "reference_sample_dir=/data/nnpz/reference"

.. note::

  The NNPZ configuration file is evaluated as Python code, so assigments,
  generator and the like can be used freely, although it is not recommended
  so the configuration stays legible.

General
=======
log_level
  Logging level. It can be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR'

Reference Sample
================

reference_sample_dir
  The directory containing the reference sample files

reference_sample_providers
  A dictionary with the data providers. Defaults to

  .. code:: python

    reference_sample_providers =  {
      'PdzProvider': {'name': 'pdz', 'data': 'pdz_data_{}.npy'},
      'SedProvider': {'name': 'sed', 'data': 'sed_data_{}.npy'},
      'MontecarloProvider': {'name': 'pp', 'data': 'pp_data_{}.npy'}
    }

reference_sample_phot_file
  The file containing the photometry of the reference sample

reference_sample_phot_filters
  The filters of the reference sample photometry to be used. They must
  have the same order with the ones defined by the target_filters option. e.g:

  .. code:: python

    reference_sample_phot_filters = ['u', 'g', 'r', 'i', 'z', 'vis', 'Y', 'J', 'H']

reference_sample_out_mean_phot_filters (optional)
  A list of filters from the reference sample photometry file to compute
  the weighted mean photometry for


Target Catalog
==============

The Target Catalog Section handles the configuration related with the catalog
to compute the photometric redshift for.

target_catalog (required)
  The input photometric catalog, in FITS or ASCII format

target_catalog_filters (required)
  The columns of the target catalog containing the photometry and error values.
  They must have the same order defined by the reference_catalog_filters option.
  e.g

  .. code:: python

    target_catalog_filters = [
        ('u_obs', 'u_obs_err'),
        ('g_obs', 'g_obs_err'),
        ('r_obs', 'r_obs_err'),
        ('i_obs', 'i_obs_err'),
        ('z_obs', 'z_obs_err'),
        ('vis_obs', 'vis_obs_err'),
        ('Y_obs', 'Y_obs_err'),
        ('J_obs', 'J_obs_err'),
        ('H_obs', 'H_obs_err')
    ]

target_catalog_gal_ebv (optional)
  The column of the target catalog containing the galactic E(B-V)

dust_map_sed_bpc (optional)
  The band pass correction for the SED used for defining the dust column density map

target_catalog_filters_shifts (optional)
  A dictionary where the key is the band name, and the value is the column on the
  target catalog containing the average wavelength of the part of the filter that
  influenced the measure. e.g

  .. code:: python

    target_catalog_filters_mean = {
        'vis': 'SHIFT_TRANS_WAVE_VIS',
        'Y': 'SHIFT_TRANS_WAVE_Y',
        'J': 'SHIFT_TRANS_WAVE_J',
        'H': 'SHIFT_TRANS_WAVE_H'
    }

missing_photometry_flags (optional)
  A list containing all the values indicating missing data. Note that NaN
  is implicitly translated to missing data.

input_size (optional)
  Defines the number of rows of the input catalog to process

input_chunk_size (optional)
  Define the chunk size to use when processing the input catalog. A smaller
  chunk size will reduce the memory footprint.


NNPZ Algorithm
==============

This section contains the options related with the NNPZ algorithm configuration.

Neighbors selection
-------------------

neighbor_method (required)
  The method to be used for selecting the neighbors. It can be one of:

  - KDTree
      Fastest method, finding the neighbors using Euclidean distance.
      WARNING: All errors are ignored when this method is used.
  - BruteForce
      Finds the neighbors using :math:`\chi^2` distance

      .. warning::

        This method is slower than the alternatives.
  - Combined
      This method first finds a batch of neighbors in Euclidean
      distance using a KDTree and then it finds the closest neighbors inside
      the batch, by using :math:`\chi^2` distance.

neighbors_no (required)
  The number of neighbors to be used

scale_prior
  If ``BruteForce`` is enabled, ``scale_prior`` can be used to allow the scaling of the reference
  fluxes. It can be either the string ``'uniform'`` (equivalent to angular distance), a Python callable
  that receives the scale value and returns its prior probability, or the path to a file containing
  the prior curve. An example of a callable would be:

.. code:: python

  # Log-normal prior
  scale_prior = lambda a: np.exp(-np.log(a)**2/0.2**2)

scale_max_iter
  Maximum number of iterations to perform for the minimization of the posterior of the scale factor

scale_rtol
  Tolerance (relative) for termination.

balanced_kdtree (optional)
  By default, the KDTree will be built using the median, creating a more compact
  tree. It takes longer, but queries are faster.
  In rare cases, depending on the data distribution (i.e a lot of zeros),
  the algorithm may perform very poorly. In those cases, this could be disabled.

batch_size (required if neighbor_method is Combined, or if there is a scale prior)
  The size of the batch size when the 'Combined' method is used


Weight calculation
------------------

weight_method (required)
  The method to be used for computing the weights of the neighbors. It can
  be one of:

  - Euclidean
      The inversed Euclidean distance (errors are ignored)
  - Chi2
      The inversed :math:`\chi^2`
  - Likelihood
      The likelihood computed as :math:`e^{-\chi^2 / 2}`

weight_method_alternative (optional)
  If weight_method yields a value of 0 for *all* neighbors of a given object,
  this method will be used instead to recompute alternative weights.
  It supports the same values as weight_method


Output
======

The Output Section handles the configuration related with the produced output.

output_file (required)
  The file to store the output catalog

copy_input_columns (optional)
  If True, the target catalog columns are copied to the output
  If False, at least 'target_catalog_id_column' is copied to the output

neighbor_info_output (optional)
  Add to the output catalog the reference sample neighbor IDs and weights

pdz_output (required)
  If False, the output catalog will skip the PDZ columns. They can later be
  computed using NnpzConstructPDZ. Note that this flag being False only makes sense if
  neighbor_info_output is True, otherwise all information is lost.

pdz_quantiles (optional)
  A list with the PDZ quantiles to be added to the output

pdz_mc_samples (optional)
  The number of Monte Carlo samples to get from the PDZ for each source

flags_in_separate_columns  (optional)
  When True, the catalog will contain a column for each flag. When False,
  the flags wil be compressed by 8 per column.

pdz_point_estimates (optional)
  A list of point estimates to add to the output. Supported: median, mean
  and mode.
