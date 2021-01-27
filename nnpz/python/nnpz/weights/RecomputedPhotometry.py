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
Created on: 20/03/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import itertools

import numpy as np
from ElementsKernel import Logging
from ElementsKernel.Auxiliary import getAuxiliaryPath
from nnpz.photometry import (PhotometryTypeMap, GalacticReddeningPrePostProcessor,
                             PhotometryCalculator, ListFileFilterProvider)
from nnpz.weights import WeightPhotometryProvider
from scipy.interpolate import interp1d

logger = Logging.getLogger(__name__)


class RecomputedPhotometry(WeightPhotometryProvider):
    """
    RecomputedPhotometry calculates the photometry of a reference source as if it were seen
    through the same part of the detector as the target source.
    """

    def __init_filters_and_curve(self):
        """
        GalacticReddeningPrePostProcessor loads these files *each time* it is instantiated
        unless we give them to it. So we pre-load them here if needed.
        """
        provider = ListFileFilterProvider(getAuxiliaryPath('GalacticExtinctionCurves.list'))
        self.__reddening_curve = provider.getFilterTransmission('extinction_curve')

    def __oversampleFilter(self, transmission: np.ndarray, n: int, kind: str):
        """
        Oversample a filter transmission `n` times
        """
        # Wavelength new sampling points
        nlambda = np.interp(np.arange(len(transmission) * n),
                            np.arange(len(transmission)) * n, transmission[:, 0])
        # Interpolate transmission over the new sampling points
        ntrans = interp1d(transmission[:, 0], transmission[:, 1], kind=kind)(nlambda)

        return np.column_stack([nlambda, ntrans])

    def __init__(self, ref_sample, filter_order, filter_trans_map, phot_type, ebv_list=None,
                 filter_trans_mean_lists=None,
                 oversample_filter: int = 1, oversample_kind: str = 'linear'):
        """
        Constructor.
        Args:
            ref_sample: A ReferenceSample instance
            filter_order: A list with the filters in the order they are expected to be returned
            filter_trans_map: A map filter_name => [average filter transmissions]
            phot_type: Photometry type
            ebv_list: None, or a 1D array with the (E(B-V) corresponding to each entry in the
                target catalog
            filter_trans_mean_list: A map with the filter_name as key, and a list/array with the
                filter mean corresponding to each entry in the target catalog
            oversample_filter: Number of times to oversample the filter transmissions.
                This may be needed if their resolution is not enough to cover the resolution of
                all the SEDs
            oversample_kind: See scipy.interpolate.interp1d
        """
        self.__ref_sample = ref_sample
        self.__filter_order = filter_order
        self.__filter_trans_map = dict(filter_trans_map)
        self.__ebv_list = ebv_list
        self.__current_ref_i = None
        self.__current_ref_sed = None

        # The classes that implement PhotometryPrePostProcessorInterface use the norm of the
        # filter transmission, which is the integration of their transmission over the wavelength
        # This integration is *invariant* even when filter shifts are applied, so we can save
        # quite a lot of computation if we just initialize it here
        self.__phot_pre_post = PhotometryTypeMap[phot_type][0](filter_trans_map)

        if self.__ebv_list is not None:
            self.__init_filters_and_curve()

        if oversample_filter and oversample_filter > 1:
            logger.info('Re-sampling filter transmissions for the recomputed photometry')
            for fname, ftrans in self.__filter_trans_map.items():
                self.__filter_trans_map[fname] = self.__oversampleFilter(ftrans, oversample_filter,
                                                                         oversample_kind)

        self.__filter_shifts = dict(itertools.product(self.__filter_trans_map.keys(), [None]))
        if filter_trans_mean_lists is not None:
            for filter_name, transmissions in self.__filter_trans_map.items():
                trans_mean = np.average(transmissions[:, 0], weights=transmissions[:, 1])
                if filter_name in filter_trans_mean_lists:
                    src_trans_mean = filter_trans_mean_lists[filter_name]
                    not_nan_mean = np.isfinite(src_trans_mean)
                    shifts = np.zeros(src_trans_mean.shape)
                    shifts[not_nan_mean] = src_trans_mean[not_nan_mean] - trans_mean
                    self.__filter_shifts[filter_name] = shifts

    def __call__(self, ref_i, cat_i, flags):
        """
        Re-compute the photometry of the reference sample as if it were seen through the same
        part of the detector as the target.
        Args:
            ref_i: The index of the reference sample for which re-compute the photometry
            cat_i: The index of the target to use for the re-computation
            flags: The flags objects to update

        Returns:
            A 2D numpy array of single precision floating point numbers. The
            first dimension represents each filter, and the second one has always
            size 2, representing the photometry value, and error (always 0 in this case).
        """
        # Retrieve the SED of the reference sample object
        if ref_i != self.__current_ref_i:
            ref_id = self.__ref_sample.getIds()[ref_i]
            self.__current_ref_sed = self.__ref_sample.getSedData(ref_id)
            self.__current_ref_i = ref_i

        # Create a map with the shifted filters
        filter_map = {}
        for filter_name, transmission in self.__filter_trans_map.items():
            filter_map[filter_name] = np.array(transmission, copy=True)
            if self.__filter_shifts[filter_name] is not None:
                filter_map[filter_name][:, 0] += self.__filter_shifts[filter_name][cat_i]

        # Create the photometry provider
        pre_post_proc = self.__phot_pre_post
        if self.__ebv_list is not None:
            ebv = self.__ebv_list[cat_i]
            pre_post_proc = GalacticReddeningPrePostProcessor(
                pre_post_proc, ebv, self.__reddening_curve
            )
        phot_calc = PhotometryCalculator(filter_map, pre_post_proc)

        # Compute the photometry
        phot_map = phot_calc.compute(self.__current_ref_sed)
        phot = np.zeros((len(self.__filter_order), 2), dtype=np.float32)
        for i, filter_name in enumerate(self.__filter_order):
            phot[i][0] = phot_map[filter_name]

        return phot
