.. _install:

Installation guide
******************

.. _install-from-sources:

From sources
============

Installing from sources is for advanced users only, or for developers, as
it requires some familiarity with Elements, and, of course, access to
Euclid's Gitlab.

1. External dependencies
------------------------

The list of external dependencies are shown on the following table.
Note that here we also list the dependencies required to compile Elements.

+-----------------+------------+---------------------------------------------------------+
| Name            | Version    | Note                                                    |
+=================+============+=========================================================+
| CMake           | latest     | The underlying building tool                            |
+-----------------+------------+---------------------------------------------------------+
| Python          | 3.x        |                                                         |
+-----------------+------------+---------------------------------------------------------+
| C++ compiler    | latest     | clang++ from XCode for MacOSX or GNU C++ for Linux      |
+-----------------+------------+---------------------------------------------------------+
| Boost           | latest     | including the "devel" part                              |
+-----------------+------------+---------------------------------------------------------+
| log4cpp         | latest     | including the "devel" part                              |
+-----------------+------------+---------------------------------------------------------+

.. note::
  We strongly recommend to use the system package manager when possible (i.e
  `dnf`, `yum`, `apt`...), as Elements will be able to find these dependencies
  out-of-the-box. In macOS, MacPorts_ should work by default. Homebrew or conda
  can be used, but they will need manual configuration.

2. User configuration
---------------------
Edit your configuration (.bashrc or equivalent for other shell) and define:

.. parsed-literal::

  export CMAKE_PROJECT_PATH=$HOME/Work/Projects
  export CMAKE_PREFIX_PATH=$CMAKE_PROJECT_PATH/Elements/|elements-version|/cmake

This is required so Elements knows where to look for Elements projects.
Of course, you can also do this on a separate script file and source it
whenever required.

3. Elements |elements-version|
------------------------------

.. parsed-literal::

  cd $CMAKE_PROJECT_PATH
  mkdir -p Elements/|elements-version|
  wget https://github.com/astrorama/Elements/archive/|elements-version|.tar.gz
  tar xzf |elements-version|.tar.gz --strip-components 1 -C Elements/|elements-version|
  cd Elements/|elements-version|
  make -j
  make install


4. NNPZ
-------

You need to download the archive with the source code for NNPZ. As it is
in Euclid's Gitlab, and it requires login, it is more convenient to do so
with your browser: |latest-archive|

.. parsed-literal::

  cd $CMAKE_PROJECT_PATH
  mkdir -p PHZ_NNPZ/|release|
  tar xzf ~/Download/PHZ_NNPZ-|release|.tar.gz --strip-components 1 -C PHZ_NNPZ/|release|
  cd PHZ_NNPZ/|release|
  make -j
  make install

5. Running NNPZ
---------------

By default, Elements projects are installed on a folder called `InstallArea`,
not system-wide. To make sure nnpz can find the required auxiliary files -
i.e for the de-reddening -, you need to configure the environment:

.. parsed-literal::

  export ELEMENTS_AUX_PATH=$CMAKE_PROJECT_PATH/PHZ_NNPZ/|release|/InstallArea/<..>/auxdir
  ./InstallArea/<..>/scripts/nnpz --version

Note that ``<..>`` depends on the OS and on the compiler used (i.e. ``x86_64-fc31-gcc93-dbg``).

If everything is fine, you will get the version of nnpz. Now you can start
:ref:`creating your configuration <configuration>`.

.. _Conda: https://docs.conda.io/en/latest/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Anaconda: https://www.anaconda.com/distribution/
.. _MacPorts: https://www.macports.org/
