# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup (

    name='NNPZ',
    version='0.1',
    packages=find_packages(),

    # Declare your packages' dependencies here, for eg:
    install_requires=['astropy', 'numpy'],

    # Fill in these to make your Egg ready for upload to
    # PyPI
    author='Nikolaos Apostolakos <nikoapos@gmail.com>',
    author_email='',

    #summary = 'Just another Python package for the cheese shop',
    url='https://github.com/nikoapos/NNPZ',
    license='MIT',
    long_description='Nearest Neighbors Photometric Redshift',

    # Setup the scripts of the project
    scripts=['bin/NnpzBuildPhotometry'],

    # Install the auxiliary data
    #zip_safe = False,
    data_files=[('etc/nnpz', ['auxdir/F99_3.1.dat', 'auxdir/GCPD_Johnson.B.dat',
                              'auxdir/GCPD_Johnson.V.dat', 'auxdir/GalacticExtinctionCurves.list'])]

)

