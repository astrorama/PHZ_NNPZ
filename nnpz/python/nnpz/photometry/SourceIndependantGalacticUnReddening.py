#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under the terms of
# the GNU Lesser General Public License as published by the Free Software Foundation;
# either version 3.0 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this library;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301 USA
#

"""
Created on: 10/07/2018
Author: Florian Dubath
"""

from __future__ import division, print_function

import numpy as np
from ElementsKernel.Auxiliary import getAuxiliaryPath
from nnpz.photometry import ListFileFilterProvider


class SourceIndependantGalacticUnReddening(object):
    """
    Source independent galactic absorption un-reddening and reddening.

    For performance reasons, instead of reddening the reference objects to match
    the extinction that affected the target object, an approximate de-reddening
    is applied instead to the later.

    The procedure is described in Schlegel et al. (1998)
    who provided a recipe to compute a global correction per filter, given an extinction
    law, a filter curve and a reference spectrum.

    The basic idea is to compute $K_X$, the normalised to E(B-V) correction where $X$ is
    the given filter, and correct the observed fluxes

    See the NNPZ IB document, appendix B, for the equations.
    https://www.overleaf.com/read/tbbdxyxbfntw
    """

    __fp = ListFileFilterProvider(getAuxiliaryPath('GalacticExtinctionCurves.list'))

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

            filter_order: An ordered list of the filter names *for de-reddening*

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
        if galactic_reddening_curve is None:
            galactic_reddening_curve = self.__fp.getFilterTransmission('extinction_curve')
        if ref_sed is None:
            ref_sed = self.__fp.getFilterTransmission('typical_galactic_sed')
        ref_galactic_sed_ressampled = np.array(galactic_reddening_curve)
        ref_galactic_sed_ressampled[:, 1] = np.interp(
            galactic_reddening_curve[:, 0], ref_sed[:, 0], ref_sed[:, 1], left=0, right=0
        )

        self._k_x = self._compute_ks(filter_map, ref_galactic_sed_ressampled,
                                     galactic_reddening_curve, ebv_0)
        self._filter_order = filter_order

    def _compute_ks(self, filter_map, ref_galactic_sed, reddening_curve, ebv_0):
        ks = {}
        for filter_name in filter_map:
            filter_transmission = filter_map[filter_name]
            filter_transmission_ressampled = np.array(reddening_curve)
            filter_transmission_ressampled[:, 1] = np.interp(reddening_curve[:, 0],
                                                             filter_transmission[:, 0],
                                                             filter_transmission[:, 1], left=0,
                                                             right=0)
            ks[filter_name] = self._compute_k_x(ref_galactic_sed,
                                                reddening_curve,
                                                filter_transmission_ressampled,
                                                ebv_0)
        return ks

    @staticmethod
    def _compute_k_x(sed, reddening, filter_curve, ebv_0):
        f_r_lambda = sed[:, 1] * filter_curve[:, 1]
        denominator = np.trapz(f_r_lambda, x=reddening[:, 0])

        f_k_r_lambda = np.power(10, -ebv_0 * reddening[:, 1] / 2.5) * f_r_lambda
        numerator = np.trapz(f_k_r_lambda, x=reddening[:, 0])

        k_x = -2.5 * np.log10(numerator / denominator) / ebv_0
        return k_x

    def _unapply_reddening(self, f_x_obs, filter_name, ebv):
        return f_x_obs * 10 ** (+self._k_x[filter_name] * ebv / 2.5)

    def _apply_reddening(self, f_x, filter_name, ebv):
        return f_x / 10 ** (+self._k_x[filter_name] * ebv / 2.5)

    def de_redden_data(self, target_data, target_ebv):
        """
        Returns a data structure with unreddened fluxes .
        """
        data = np.zeros(target_data.shape, dtype=np.float32)

        # Copy the errors which are unaffected
        data[:, :, 1] = target_data[:, :, 1]

        for source_id in range(target_data.shape[0]):
            ebv = target_ebv[source_id]

            for filter_id in range(len(self._filter_order)):
                filter_name = self._filter_order[filter_id]
                data[source_id, filter_id, 0] = self._unapply_reddening(
                    target_data[source_id, filter_id, 0], filter_name, ebv)
        return data

    def redden_data(self, target_data, target_ebv):
        """
        Applies the inverse of de_redden_data
        """
        data = np.zeros(target_data.shape, dtype=np.float32)
        data[:, :, 1] = target_data[:, :, 1]

        for source_id in range(target_data.shape[0]):
            ebv = target_ebv[source_id]

            for filter_id in range(len(self._filter_order)):
                filter_name = self._filter_order[filter_id]
                data[source_id, filter_id, 0] = self._apply_reddening(
                    target_data[source_id, filter_id, 0],
                    filter_name, ebv)
        return data
