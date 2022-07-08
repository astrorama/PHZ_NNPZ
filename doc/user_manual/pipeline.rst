.. _pipeline-description:

Pipeline description
********************

.. image:: /_static/PipelineNNPZ.png

The image above displays a typical setup of |NNPZ|. The exact
computations depend on the configuration, but in all cases it involves:

1. Looking for neighbors
2. Compute their weights
3. Constructing the output from the neighbors and the weights

The photometry correction stage is optional and depends on the configuration.

On the following description of the pipeline, ``this type of text`` refers
to the relevant settings in the :ref:`configuration file <configuration>`. However,
not all settings are described here, only the most relevant set.

Inputs
======

* Input catalog (``target_catalog, target_catalog_filters``)
* :ref:`Reference sample <reference-sample>` (``reference_sample_dir``)

  * Photometry file (filter transmission curves are embedded in additional HDUs) (``reference_sample_phot_file, reference_sample_phot_filters``)
  * Index file (``reference_sample_index``)
  * Provider files (``reference_sample_providers``)

The supported providers are:
  * ``PdzProvider``, which handles |PDZ| data
  * ``MontecarloProvider``, which handles a set of random samples from a multidimensional space (e.g. |PP|)
  * ``SedProvider``, whcih handles |SED| data. Note that these are not used since NNPZ 1.0

The input catalog can be either a FITS file, or an ASCII table.

If NNPZ has to correct for filter variations, then the input catalog needs
to contain a column with the shift of average effective wavelength for the bands to be
corrected (``target_catalog_filters_shifts``), and, as a part of the reference
sample, the correction factors.

The average effective wavelength for a given band is defined as:

.. math::

  \frac{\int_{\lambda}{r_{(\lambda)}\lambda}d\lambda}{\int_{\lambda}r_{(\lambda)d\lambda}}

Where :math:`r_{(\lambda)}` is the filter response function.


.. _dereddening:

Galactic de-reddening
=====================

If the input catalog has a column with the :math:`E(B-V)` extinction (``target_catalog_gal_ebv``),
NNPZ can perform an approximate de-reddening of the source photometry applying
the procedure described in (schlegel1998_) given an extinction law,
a filter curve and a reference spectrum.

The basic idea is to compute :math:`K_X`, the normalized to :math:`E(B-V)`
correction where :math:`X` is the given filter, and correct the observed fluxes:

.. math::

  K_X = \frac{-2.5\,\log_{10}\left( \frac{\int_{\lambda} \, f_\mathrm{ref}(\lambda) \, 10^{-E(B-V)_0\,k(\lambda)/2.5} \, r_X(\lambda) \, \mathrm{d} \lambda}{\int_{\lambda} \, f_\mathrm{ref}(\lambda) \,  r_X(\lambda) \, \mathrm{d} \lambda} \right)}{E(B-V)_0} \, ,

.. math::

  F_{X,\mathrm{corr}} =  F_{X, \mathrm{obs}} \, 10^{+K_X\,E(B-V)/2.5}\,  .

:math:`f_\mathrm{ref}(\lambda)` is a reference template. The natural choice is
to pick a reference spectrum that is as close as possible to our data.
The :math:`E(B-V)_0` factor correspond to the typical extinction range where
the correction is applied. NNPZ uses :math:`E(B-V)_0=0.02`.
:math:`r_X(\lambda)` is the filter response function of the band :math:`X`,
and :math:`k(\lambda)` is the shape of the curve.

.. image:: /_static/f99.png

The bulk of Euclid galaxies are :math:`z\sim1`, :math:`{\rm vis}\sim24` galaxies.
The selected reference |SED| is shown in the following figure:

.. image:: /_static/spectrum_ref.png


Neighbor search
===============

NNPZ supports three type of neighbor search (``neighbor_method,neigbors_no``):

* Purely in Euclidean space (``KDTree``), without taking into account the uncertainties

  .. math::

    d_{E} = \sqrt{\sum_{band}{\rm{flux_{ref}} - \rm{flux_{target}}}}

* Purely via :math:`\chi^2` (``BruteForce``), which takes into account the errors, but requires
  a brute force search (all target vs all reference objects)

  .. math::

    d_{\chi^2}(a) = \sum_{band}\frac{a \times \rm{flux_{ref}} - \rm{flux_{target}}}{(a \times \rm{error_{ref})^2} + \rm{error_{target}}^2}

  Note this distance depends on a scaling term :math:`a`. For the basic usage,
  we can assume its value is 1 and ignore it. For details, consult the
  documentation about the :ref:`scaling`.

* A combination of both (``Combined``), which selects first a candidate set of
  ``batch_size`` reference objects via Euclidean distance, and then looks for
  the final set of neighbors via :math:`\chi^2`.

The last option usually offers a good compromise between run time and accuracy
if ``batch_size`` is large enough (2000 is recommended).

.. _photometry-recomputation:

(Optional) Photometry re-computation
====================================

If the input catalog has information about the reddening (``target_catalog_gal_ebv``)
and/or the average filter transmission (``target_catalog_filters_shifts``),
|NNPZ| will "project" each reference object into the color-space of the target
object.

.. image:: /_static/nnpz_spaces.png

For the correction to be possible, the reference sample photometry file must
contain the pre-computed correction factors, which are stored
on the columns named ``{filter}_EBV_CORR`` for the galactic reddening,
and ``{filter}_SHIFT_CORR`` for the filter shifts.

For the galactic reddening a single value per reference object
is required. The correction is applied by the simple formula

.. math::

  f_{\rm{band},\rm{projected}} = 10^{-0.4 * \rm{EBV\_CORR} * \rm{EBV}} \times f_{\rm{band},\rm{restframe}}


For more information about the galactic reddening methodology, we recommend
reading the `Phosphoros Documentation <https://phosphoros.readthedocs.io/en/latest/user_manual/methodology.html#galactic-absorption>`_.

For the filter variation, two values per reference object are required, since
the correction is based on a quadratic fit.

.. math::

  f_{\rm{band},\rm{projected}} = (\rm{SHIFT\_CORR[0]} \times \Delta\lambda^2 + \rm{SHIFT\_CORR[1]} \times \Delta\lambda) \times f_{\rm{band},\rm{restframe}}

Weight computation
==================

NNPZ supports three different types of weighting method (``weight_method``):

* The inverse of the Euclidean distance, ignoring errors (``Euclidean``)
* The inverse of the :math:`\chi^2` distance (``Chi2``)
* The likelihood :math:`e^{-\chi^2 / 2}` (``Likelihood``)

If the weighting method yields a value of 0 for *all* neighbors of a given
object, an alternative method can be used (``weight_method_alternative``).
For instance, one may use the likelihood as the primary weighting, and
the inverse of the Euclidean as a fallback.

Note that both :math:`\chi^2` and likelihood weights take into account the
:ref:`scaling factor<scaling>` :math:`a`.

Output
======

The target object |PDZ| is computed as the weight-average of the |PDZ|
of each neighbor, and added to the output catalog if ``pdz_output``
is set to True.

1. Point estimates can be requested with ``pdz_point_estimates`` (median, mean, mode)
2. Quantiles can be requested with ``pdz_quantiles`` (i.e. 0.25, 0.5, 0.75)
3. And Monte Carlo samples with ``pdz_mc_samples``

The neighbor list and weights can be obtained if ``neighbor_info_output``
is set to True.

All columns from the input catalog will be copied if ``copy_input_columns``
is set to True.

Generally there will be as single "flag" column (as an integer) unless
``flags_in_separate_columns`` is set to true, in which case multiple
boolean columns will be generated instead.

.. _schlegel1998: https://ui.adsabs.harvard.edu/abs/1998ApJ...500..525S/abstract
