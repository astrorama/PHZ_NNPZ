"""
Created on: 20/03/18
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function

import numpy as np

from nnpz.weights import WeightPhotometryProvider
from nnpz.photometry import (PhotometryTypeMap, GalacticReddeningPrePostProcessor,
                             PhotometryCalculator)


class RecomputedPhotometry(WeightPhotometryProvider):


    def __init__(self, ref_sample, filter_order, filter_trans_map, phot_type, ebv_list=None, filter_shift=None):

        self.__ref_sample = ref_sample
        self.__filter_order = filter_order
        self.__filter_trans_map = filter_trans_map
        self.__ebv_list = ebv_list
        self.__filter_shift = filter_shift
        self.__phot_pre_post = PhotometryTypeMap[phot_type][0]
        self.__current_ref_i = None
        self.__current_ref_sed = None


    def __call__(self, ref_i, cat_i):

        # Retrieve the SED of the reference sample object
        if ref_i != self.__current_ref_i:
            ref_id = self.__ref_sample.getIds()[ref_i]
            self.__current_ref_sed = self.__ref_sample.getSedData(ref_id)
            self.__current_ref_i = ref_i

        # Create a map with the shifted filters
        filter_map = {}
        for filter, transmission in self.__filter_trans_map.iteritems():
            shift = self.__filter_shift[filter][cat_i] if filter in self.__filter_shift else 0
            filter_map[filter] = transmission + shift

        # Create the photometry provider
        pre_post_proc = self.__phot_pre_post()
        if self.__ebv_list is not None:
            ebv = self.__ebv_list[cat_i]
            pre_post_proc = GalacticReddeningPrePostProcessor(pre_post_proc, ebv)
        phot_calc = PhotometryCalculator(filter_map, pre_post_proc)

        # Compute the photometry
        phot_map = phot_calc.compute(self.__current_ref_sed)
        phot = np.zeros((len(self.__filter_order), 2), dtype=np.float32)
        for i, f in enumerate(self.__filter_order):
            phot[i][0] = phot_map[f]

        return phot