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
from ElementsKernel.Auxiliary import getAuxiliaryPath
from nnpz import NnpzFlag, ReferenceSample
from nnpz.photometry import (GalacticReddeningPrePostProcessor, ListFileFilterProvider,
                             PhotometryCalculator, PhotometryTypeMap)
from nnpz.reference_sample import PhotometryProvider
from nnpz.weights import WeightPhotometryProvider


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

    def __init__(self, ref_sample: ReferenceSample, ref_phot: PhotometryProvider,
                 ebv_list: np.ndarray = None, filter_trans_mean_lists: dict = None):
        """
        Constructor.
        Args:
            ref_sample: A ReferenceSample instance
            ref_phot: A PhotometrtyProvider instance
            ebv_list: None, or a 1D array with the (E(B-V) corresponding to each entry in the
                target catalog
            filter_trans_mean_lists: A map with the filter_name as key, and a list/array with the
                filter mean corresponding to each entry in the target catalog
        """
        self.__ref_sample = ref_sample
        self.__ref_photo = ref_phot
        self.__ebv_list = ebv_list
        self.__phot_pre_post = PhotometryTypeMap[ref_phot.getType()][0]
        self.__current_ref_i = None
        self.__current_ref_sed = None

        if self.__ebv_list is not None:
            self.__init_filters_and_curve()

        self.__filter_shifts = dict()
        if filter_trans_mean_lists:
            for filter_name, src_trans_mean in filter_trans_mean_lists.items():
                transmission = self.__ref_photo.getFilterTransmission(filter_name)
                trans_mean = np.average(transmission[:, 0], weights=transmission[:, 1])
                not_nan_mean = np.isfinite(src_trans_mean)
                shifts = np.zeros(src_trans_mean.shape)
                shifts[not_nan_mean] = src_trans_mean[not_nan_mean] - trans_mean
                self.__filter_shifts[filter_name] = shifts

    def __call__(self, ref_i: int, cat_i: int, flags: NnpzFlag):
        """
        Re-compute the photometry of the reference sample as if it were seen through the same
        part of the detector as the target.
        Args:
            ref_i: The index of the reference sample for which re-compute the photometry
            cat_i: The index of the target to use for the re-computation
            flags: The flags objects to update

        Returns:
            A map with keys the filter names and values the photometry values
        """
        # Retrieve the SED of the reference sample object
        if ref_i != self.__current_ref_i:
            ref_id = self.__ref_sample.getIds()[ref_i]
            self.__current_ref_sed = self.__ref_sample.getSedData(ref_id)
            self.__current_ref_i = ref_i

        # Create a map with the shifted filters
        filter_map = {}
        for filter_name in self.__ref_photo.getFilterList():
            transmission = self.__ref_photo.getFilterTransmission(filter_name)
            filter_map[filter_name] = np.array(transmission, copy=True)
            if filter_name in self.__filter_shifts:
                filter_map[filter_name][:, 0] += self.__filter_shifts[filter_name][cat_i]

        # Create the photometry provider
        pre_post_proc = self.__phot_pre_post()
        if self.__ebv_list is not None:
            ebv = self.__ebv_list[cat_i]
            pre_post_proc = GalacticReddeningPrePostProcessor(
                pre_post_proc, ebv, self.__reddening_curve
            )
        phot_calc = PhotometryCalculator(filter_map, pre_post_proc)
        return phot_calc.compute(self.__current_ref_sed)
