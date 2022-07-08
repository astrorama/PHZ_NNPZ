.. _reference-sample:

Reference Sample
****************

.. toctree::
  :titlesonly:

  scaling
  api

The reference sample contains a set of objects with known properties - i.e
|PDZ| or |PP|. Objects from the target catalog are then compared with these objects,
and the most similar are used to do a prediction of the properties of the
target object (`Nearest Neighbors Algorithm`_).

NNPZ can use a different input format: a custom set of files containing
|SED|, |PDZ| and |PP|, and a FITS file containing the photometries.

The reference sample can be build by |Phosphoros|, using template fitting,
or directly via a :ref:`Python API <reference-sample-api>`.
The photometries can be build using |Phosphoros|' command ``PhosphorosBuildPhotometry``.

Since |NNPZ| 1.0, only :math:`F_\nu` (|mu| Jy) is supported for the photometry.

.. math::

  F_{\nu} [{\rm Jy}] = 10^{29} \frac{\int_{\lambda}\textrm{SED}_{(\lambda)}r_{(\lambda)}d\lambda}{\int_{\lambda}r_{(\lambda)}\frac{c}{\lambda^2}d\lambda}

.. _`Nearest Neighbors Algorithm`: https://en.wikipedia.org/wiki/Nearest_neighbor_search
