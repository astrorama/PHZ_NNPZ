.. _quickstart:

Quickstart Guide
****************

* See :ref:`install`

You can download a example_ file containing a reference sample, a
photometry file and a target catalog.

The reference sample contains 50 000 sources with delta |PDZ|, sampled every 0.02z,
and a corresponding down-sampled |SED|, since this is just a quick start guide,
and for keeping both size and run time low.

The set of filters are u, g, r, i, z, VIS, Y, J and H.

.. image:: /_static/filters.png

The target catalog contains 10 000 sources, with the photometry, filter mean
transmission (so filter shift correction can be applied), and the "true" redshift.

A pre-filled configuration file ``nnpz.conf`` is also provided as a starting
point.

Once you have downloaded the file, uncompress it, and just run nnpz:

.. code:: bash

  tar xJf "Quickstart.tar.xz"
  cd Quickstart
  nnpz --config-file nnpz.conf

Once the run is finished, you will have a file called ``NNPZRun.fits`` with
the results of the run. You can compare the columns

* ``REDSHIFT_MODE`` for the mode of the resulting |PDZ|
* ``REDSHIFT_MEDIAN`` for the median of the resulting |PDZ|

with the "true" redshift ``z``. For convenience all columns from the input
catalog are copied to the result (:code:`copy_input_columns = True`)

This is the result you can expect:

.. image:: /_static/quickstart_z.png

Note that because of the downsampling of the |PDZ| the quality of
the results are limited, but they can be useful to get familiar with nnpz
nevertheless.

.. _example: /_static/QuickStart.tar.xz
