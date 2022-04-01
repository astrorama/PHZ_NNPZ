#
# Copyright (C) 2012-2022 Euclid Science Ground Segment
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

import numpy as np
from ElementsKernel.Auxiliary import getAuxiliaryPath
from astropy import units as u
from nnpz.photometry.filter_provider import ListFileFilterProvider
from nnpz.photometry.photometric_system import PhotometricSystem


class SourceIndependentGalacticEBV:
    """
    Source independent galactic absorption un-reddening and reddening.

    For performance reasons, instead of reddening the reference objects to match
    the extinction that affected the target object, an approximate de-reddening
    is applied to the later.

    The procedure is described in Schlegel et al. (1998)
    who provided a recipe to compute a global correction per filter, given an extinction
    law, a filter curve and a reference spectrum.

    The basic idea is to compute $K_X$, the normalised to E(B-V) correction where $X$ is
    the given filter, and correct the observed fluxes

    See the NNPZ IB document, appendix B, for the equations.
    https://www.overleaf.com/read/tbbdxyxbfntw


    Args:
        system: PhotometricSystem
            Set of filters and their transmissions

        reddening_curve: The galactic reddening curve.
            The curve is a 2D numpy arrays, where the first axis
            represents the knots and the second axis has size 2 with the
            first element being the wavelength (expressed in Angstrom) and
            the second the rescaled galactic absorption value

        reference_sed: the typical (reference) SED for which the K_X are computed.
            The curve is a 2D numpy arrays, where the first axis
            represents the knots and the second axis has size 2 with the
            first element being the wavelength (expressed in Angstrom) and
            the second the sed flux value

        ebv_0: Reference E(B-V)_0 for which the K_X are computed.
            Default value = 0.02
    """

    __fp = ListFileFilterProvider(getAuxiliaryPath('GalacticExtinctionCurves.list'))

    def __init__(self, system: PhotometricSystem,
                 reddening_curve: np.ndarray = None,
                 reference_sed: np.ndarray = None, ebv_0: float = 0.02):
        self.__system = system

        if reddening_curve is None:
            reddening_curve = self.__fp.get_filter_transmission('extinction_curve')
        if reference_sed is None:
            reference_sed = self.__fp.get_filter_transmission('typical_galactic_sed')
        ref_sed_resampled = np.array(reddening_curve)
        ref_sed_resampled[:, 1] = np.interp(
            reddening_curve[:, 0], reference_sed[:, 0], reference_sed[:, 1], left=0, right=0
        )

        self._k_x = self._compute_ks(ref_sed_resampled, reddening_curve, ebv_0)

    def _compute_ks(self, ref_galactic_sed: np.ndarray,
                    reddening_curve: np.ndarray, ebv_0: float):
        ks = {}
        for filter_name in self.__system.bands:
            transmission = self.__system.get_transmission(filter_name)
            transmission_resampled = np.array(reddening_curve)
            transmission_resampled[:, 1] = np.interp(reddening_curve[:, 0],
                                                     transmission[:, 0], transmission[:, 1],
                                                     left=0, right=0)
            ks[filter_name] = self._compute_k_x(ref_galactic_sed,
                                                reddening_curve,
                                                transmission_resampled,
                                                ebv_0)
        return ks

    @staticmethod
    def _compute_k_x(sed: np.ndarray, reddening: np.ndarray, filter_curve: np.ndarray,
                     ebv_0: float):
        f_r_lambda = sed[:, 1] * filter_curve[:, 1]
        denominator = np.trapz(f_r_lambda, x=reddening[:, 0])

        f_k_r_lambda = np.power(10, -ebv_0 * reddening[:, 1] / 2.5) * f_r_lambda
        numerator = np.trapz(f_k_r_lambda, x=reddening[:, 0])

        k_x = -2.5 * np.log10(numerator / denominator) / ebv_0
        return k_x

    def _add_reddening(self, f_x: np.ndarray, filter_name: str, ebv: np.ndarray):
        f_x /= 10 ** (+self._k_x[filter_name] * ebv / 2.5)
        return f_x

    def _remove_reddening(self, f_x_obs: np.ndarray, filter_name: str, ebv: np.ndarray):
        f_x_obs *= 10 ** (+self._k_x[filter_name] * ebv / 2.5)
        return f_x_obs

    @u.quantity_input
    def redden(self, photometry: u.uJy, ebv: np.ndarray, out: u.uJy = None):
        if out is None:
            out = photometry.copy()
        elif out is not photometry:
            np.copyto(out, photometry)
        for i, filter_name in enumerate(self.__system.bands):
            self._add_reddening(out[:, i, 0], filter_name, ebv)
        return out

    @u.quantity_input
    def deredden(self, photometry: u.uJy, ebv: np.ndarray, out: u.uJy = None):
        if out is None:
            out = photometry.copy()
        elif out is not photometry:
            np.copyto(out, photometry)
        for i, filter_name in enumerate(self.__system.bands):
            self._remove_reddening(out[:, i, 0], filter_name, ebv)
        return out

    def __str__(self) -> str:
        return 'E(B-V)'
