"""
Created on: 20/03/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

from nnpz.photometry import *

PhotometryTypeMap = {
    'Photons' : (PhotonPrePostProcessor, 'Photon count rate in counts/s/cm^2'),
    'F_nu' : (FnuPrePostProcessor, 'Energy flux density in erg/s/cm^2/Hz'),
    'F_nu_uJy' : (FnuuJyPrePostProcessor, 'Energy flux density in uJy'),
    'MAG_AB' : (MagAbPrePostProcessor, 'AB magnitude'),
    'F_lambda' : (FlambdaPrePostProcessor, 'Energy flux density in erg/s/cm^2/A')
}