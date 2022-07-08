.. _nnpz-main:

Nearest Neighbors Photometric Redshift (Z)
******************************************

This is the NNPZ |release| main documentation page, last updated |today|.

NNPZ (Nearest-Neighbor Photometric Redshift) is a machine-learning algorithm,
which consists in a k-nearest neighbor method in photometric space, designed to
produce a Probability Density Function of the Redshift (Z).

It can perform the search in Euclidean space, or taking into account the
uncertainties. However, the latter can only be done
with a brute force approach (all reference objects are compared with all target
objects). For efficiency, an initial search in Euclidean distance can be done
to prune the number of candidates.

.. toctree::
  :titlesonly:

  install
  user_manual/index
  developer_manual/index
  release_notes
  API Reference <nnpz/modules>
