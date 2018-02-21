"""
Created on: 19/02/2018
Author: Florian Dubath
"""

from __future__ import division, print_function

import numpy as np

from nnpz.photometry import PhotometryPrePostProcessorInterface

class GalacticReddeningPrePostProcessor(PhotometryPrePostProcessorInterface):
    """Pre/Post processor for including the galactic absorption.

    This processor is a decorator around another PrePostProcessor, which apply
    galactic reddening to the sed beforhand.
    """

    def __init__(self, pre_post_processor, b_filter, r_filter, galactic_reddening_curve, p_14_ebv):
        """Initialize a GalacticReddeningPrePostProcessor by decorating the
        provided pre/post processor

        Args:
            pre_post_processor: The pre/post processor to be decorated
            b_filter: The blue filter used to compute the SED band pass correction
            r_filter: The red filter used to compute the SED band pass correction
            galactic_reddening_curve: The galactic reddening curve
            p_14_ebv: The P14 E(B-V) value along the line of sight
        """
        self.__processor = pre_post_processor;
        self.__b_filter = b_filter;
        self.__r_filter = r_filter;
        self.__reddening_curve = galactic_reddening_curve
        self.__p_14_ebv = p_14_ebv

    def __truncateSed(self, sed, range):
        """Truncates the given SED at the given range"""

        min_i = np.searchsorted(sed[:, 0], range[0])
        if min_i > 0:
            min_i -= 1
        max_i = np.searchsorted(sed[:, 0], range[1])
        max_i += 1
        return sed[min_i:max_i+1, :]


    def preProcess(self, sed):
        """Redden the SED according to the galactic absorption law and the
        E(B-V) parameter then redirects the call to the FnuPrePostProcessor
        """
        blue_range = (b_filter[0][0], b_filter[-1][0])
        sed_truncated_blue = self.__truncateSed(sed, blue_range)
        interp_blue_filter = np.interp(sed_truncated_blue[:,0], self.__b_filter[:,0], self.__b_filter[:,1], left=0, right=0)
        interp_blue_absorption = np.interp(sed_truncated_blue[:,0], self.__reddening_curve[:,0], self.__reddening_curve[:,1], left=0, right=0)
        # Compute b
        b = np.trapz(sed_truncated_blue[:,1] * interp_blue_filter, x=sed_truncated_blue[:,0])
        # Compute b_r
        b_r = np.trapz(sed_truncated_blue[:,1] * interp_blue_filter*np.power(10, -0.04*interp_blue_absorption), x=sed_truncated_blue[:,0])

        red_range = (r_filter[0][0], r_filter[-1][0])
        sed_truncated_red = self.__truncateSed(sed, red_range)
        interp_red_filter = np.interp(sed_truncated_red[:,0], self.__r_filter[:,0], self.__r_filter[:,1], left=0, right=0)
        interp_red_absorption = np.interp(sed_truncated_red[:,0], self.__reddening_curve[:,0], self.__reddening_curve[:,1], left=0, right=0)
        # Compute r
        r = np.trapz(sed_truncated_red[:,1] * interp_red_filter, x=sed_truncated_red[:,0])
        # Compute r_r
        r_r = np.trapz(sed_truncated_red[:,1] * interp_red_filter*np.power(10,-0.04*interp_red_absorption), x=sed_truncated_red[:,0])

        # Compute the SED bpc
        bpc_sed = -0.04*np.log10(b_r*r/(b*r_r))

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
