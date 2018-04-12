"""
Created on: 19/02/2018
Author: Florian Dubath
"""

from __future__ import division, print_function

import os
import numpy as np

from nnpz.photometry import PhotometryPrePostProcessorInterface, ListFileFilterProvider
from nnpz.utils import Auxiliary

class GalacticReddeningPrePostProcessor(PhotometryPrePostProcessorInterface):
    """Pre/Post processor for including the galactic absorption.

    This processor is a decorator around another PrePostProcessor, which apply
    galactic reddening to the sed beforhand.
    """

    __fp = ListFileFilterProvider(Auxiliary.getAuxiliaryPath('GalacticExtinctionCurves.list'))

    def __init__(self, pre_post_processor, p_14_ebv, b_filter=None, r_filter=None, galactic_reddening_curve=None):
        """Initialize a GalacticReddeningPrePostProcessor by decorating the
        provided pre/post processor

        Args:
            pre_post_processor: The pre/post processor to be decorated.
                Must implement the PhotometryPrePostProcessorInterface interface
            p_14_ebv: The P14 E(B-V) float value along the line of sight
            b_filter: The blue filter used to compute the SED band pass correction.
                The filter is a 2D numpy arrays, where the first axis
                represents the knots and the second axis has size 2 with the
                first element being the wavelength (expressed in Angstrom) and
                the second the filter transmission (in the range [0,1])
            r_filter: The red filter used to compute the SED band pass correction.
                The filter is a 2D numpy arrays, where the first axis
                represents the knots and the second axis has size 2 with the
                first element being the wavelength (expressed in Angstrom) and
                the second the filter transmission (in the range [0,1])
            galactic_reddening_curve: The galactic reddening curve.
                The curve is a 2D numpy arrays, where the first axis
                represents the knots and the second axis has size 2 with the
                first element being the wavelength (expressed in Angstrom) and
                the second the rescaled galactic absorption value

        Note that the b_filter, r_filter and galactic_reddening_curve parameters
        are optional. If they are not given, the default behavior is to use the
        Johnson B and V filters and the F99 extinction curve.
        """
        self.__processor = pre_post_processor
        self.__p_14_ebv = p_14_ebv

        self.__b_filter = self.__fp.getFilterTransmission('b_filter') if b_filter is None else b_filter
        self.__r_filter = self.__fp.getFilterTransmission('r_filter') if r_filter is None else r_filter
        self.__reddening_curve = fp.getFilterTransmission('extinction_curve') if galactic_reddening_curve is None else galactic_reddening_curve

    def __truncateSed(self, sed, range):
        """Truncates the given SED at the given range"""

        min_i = np.searchsorted(sed[:, 0], range[0])
        if min_i > 0:
            min_i -= 1
        max_i = np.searchsorted(sed[:, 0], range[1])
        max_i += 1
        return sed[min_i:max_i+1, :]

    def computeBpc(self,sed):
        """ COmpute the band Passs Correction for the provided SED"""
        reddening_factor = np.power(10,-0.04*self.__reddening_curve[:,1])
        blue_range = (self.__b_filter[0][0], self.__b_filter[-1][0])
        sed_truncated_blue = self.__truncateSed(sed, blue_range)
        interp_blue_filter = np.interp(sed_truncated_blue[:,0], self.__b_filter[:,0], self.__b_filter[:,1], left=0, right=0)
        interp_blue_absorption = np.interp(sed_truncated_blue[:,0], self.__reddening_curve[:,0], reddening_factor, left=0, right=0)

        # Compute b
        b = np.trapz(sed_truncated_blue[:,1] * interp_blue_filter, x=sed_truncated_blue[:,0])
        # Compute b_r
        b_r = np.trapz(sed_truncated_blue[:,1] * interp_blue_filter*interp_blue_absorption, x=sed_truncated_blue[:,0])

        red_range = (self.__r_filter[0][0], self.__r_filter[-1][0])
        sed_truncated_red = self.__truncateSed(sed, red_range)
        interp_red_filter = np.interp(sed_truncated_red[:,0], self.__r_filter[:,0], self.__r_filter[:,1], left=0, right=0)
        interp_red_absorption = np.interp(sed_truncated_red[:,0], self.__reddening_curve[:,0], reddening_factor, left=0, right=0)

        # Compute r
        r = np.trapz(sed_truncated_red[:,1] * interp_red_filter, x=sed_truncated_red[:,0])
        # Compute r_r
        r_r = np.trapz(sed_truncated_red[:,1] * interp_red_filter*interp_red_absorption, x=sed_truncated_red[:,0])

        # Compute the SED bpc
        bpc_sed = -25*np.log10(b_r*r/(b*r_r))
        return bpc_sed

    def preProcess(self, sed):
        """Redden the SED according to the galactic absorption law and the
        E(B-V) parameter then redirects the call to the FnuPrePostProcessor
        """
        # Get the bpc
        bpc_sed = self.computeBpc(sed)

        # Compute A_lambda
        interp_absorption = np.interp(sed[:,0], self.__reddening_curve[:,0], self.__reddening_curve[:,1], left=0, right=0)
        a_lambda =  interp_absorption * self.__p_14_ebv * 1.018 / bpc_sed
        absorption_factor = np.power(10, -0.4* a_lambda)
        # Compute the reddened sed: build an array with the same shape as the SED
        # with 1s in the first column and the absorption factor in the second,
        # so the element-wise multiplication do not affect the lambda
        mult = np.ones((absorption_factor.shape[0], 2))
        mult[:,1]=absorption_factor
        reddened_sed = np.multiply(sed,mult)

        # apply pre-processing
        return self.__processor.preProcess(reddened_sed)

    def postProcess(self, intensity, filter_name, filter_trans):
        """Returns the flux as computed from the decorated pre/post processor.
        """
        return self.__processor.postProcess(intensity, filter_name, filter_trans)
