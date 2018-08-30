"""
Created on: 10/07/2018
Author: Florian Dubath
"""

from __future__ import division, print_function

import os
import numpy as np

from nnpz.photometry import ListFileFilterProvider
from nnpz.utils import Auxiliary

class SourceIndependantGalacticUnReddening():
    """Source independant galactic absorption un-reddening.

    it apply the source independant galactic absorption correction on the 
    fluxes following the equation 23 & 24 of 
    J. Coupon private communication 9/07/2018 
    """

    __fp = ListFileFilterProvider(Auxiliary.getAuxiliaryPath('GalacticExtinctionCurves.list'))

    def __init__(self, 
                 filter_map, 
                 filter_order,
                 galactic_reddening_curve=None,
                 ref_sed=None,
                 ebv_0=0.02
                 ):
        """Initialize a SourceIndependantGalacticUnReddening

        Args:
            filter_map: A dictionary with keys the filter names and values the
                filter transmissions as 2D numpy arrays, where the first axis
                represents the knots and the second axis has size 2 with the
                first element being the wavelength (expressed in Angstrom) and
                the second the filter transmission (in the range [0,1])
            
            filter_order: An ordered list of the filter names 
                
            galactic_reddening_curve: The galactic reddening curve.
                The curve is a 2D numpy arrays, where the first axis
                represents the knots and the second axis has size 2 with the
                first element being the wavelength (expressed in Angstrom) and
                the second the rescaled galactic absorption value
                
            ref_sed: the typical (reference) SED for which the K_X are computed.
                The curve is a 2D numpy arrays, where the first axis
                represents the knots and the second axis has size 2 with the
                first element being the wavelength (expressed in Angstrom) and
                the second the sed flux value
                
            ebv_0: Reference E(B-V)_0 for which the K_X are computed. 
                Default value =0.02
                

        Note that the galactic_reddening_curve and ref_sed parameters
        are optional. If they are not given, the default behavior is to use the
        F99 extinction curve and J.Coupon ref SED from the auxiliary data.
        """
        
        
        # we use the knots of the reddening curve and resample the other curve 
        # according to it:
        reddening_curve = SourceIndependantGalacticUnReddening.__fp.getFilterTransmission('extinction_curve') if galactic_reddening_curve is None else galactic_reddening_curve
        ref_galactic_sed = SourceIndependantGalacticUnReddening.__fp.getFilterTransmission('typical_galactic_sed') if ref_sed is None else ref_sed
        ref_galactic_sed_ressampled=np.array(reddening_curve)
        ref_galactic_sed_ressampled[:,1]=np.interp(reddening_curve[:,0], ref_galactic_sed[:,0], ref_galactic_sed[:,1], left=0, right=0)
        
        self._k_x=self._compute_ks(filter_map,ref_galactic_sed_ressampled,reddening_curve,ebv_0)
        self._filter_order=filter_order
        
    def _compute_ks(self,filter_map,ref_galactic_sed,reddening_curve,ebv_0):
        ks={}
        for filter_name in filter_map:
            filter_transmission=filter_map[filter_name]
            filter_transmission_ressampled=np.array(reddening_curve)
            filter_transmission_ressampled[:,1]=np.interp(reddening_curve[:,0], filter_transmission[:,0], filter_transmission[:,1], left=0, right=0)
            ks[filter_name]=self._compute_k_x(ref_galactic_sed,
                             reddening_curve,
                             filter_transmission_ressampled,
                             ebv_0)
        return ks
        
    def _compute_k_x(self,sed,reddening,filter_curve,ebv_0):
        f_r_lambda = sed[:,1]*filter_curve[:,1]
        denominator=np.trapz(f_r_lambda, x=reddening[:,0])
        
        f_k_r_lambda = np.power(10,-ebv_0*reddening[:,1]/2.5)*f_r_lambda
        numerator=np.trapz(f_k_r_lambda, x=reddening[:,0])
        
        k_x=-2.5*np.log10(numerator/denominator)/ebv_0
        return k_x
    
    def _unapply_reddening(self,f_x_obs,filter_name,ebv):
        return f_x_obs*10**(+self._k_x[filter_name]*ebv/2.5)



    
    def de_redden_data(self, target_data, target_ebv):
        """Returns a data structure with unereddened fluxes .
        """
        data = np.zeros(target_data.shape, dtype=np.float32)
        
        #Copy the errors which are unaffected
        data[:,:,1]=target_data[:,:,1]
        
        for source_id in range(target_data.shape[0]):
            ebv=target_ebv[source_id]
            
            for filter_id in range(len(self._filter_order)):
                filter_name = self._filter_order[filter_id]
                data[source_id,filter_id,0]=self._unapply_reddening(target_data[source_id,filter_id,0],filter_name,ebv)
        return data
    
    
    
