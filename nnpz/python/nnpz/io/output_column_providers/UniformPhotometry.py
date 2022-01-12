#
# Copyright (C) 2012-2021 Euclid Science Ground Segment
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
from typing import Dict, Tuple

import numpy as np
from nnpz.io import OutputHandler
from nnpz.photometry.photometry import Photometry


class UniformPhotometry(OutputHandler.OutputColumnProviderInterface):
    """
    Generate a list of columns with an uniform photometry:
    \\f[
    g_o^* = g_o \\left( \\frac{g_r^*}{g_r} \\right)
    \\f]

    Where \\f$g_r^*\\f$ is the reference modeled flux for the objective band, and \\f$g_r\\f$ the
    reference modeled flux for the observed band (including any correction as reddening).
    \\f$g_o\\f$ is the observed flux for the target object

    Args:
        target_phot:
            Target photometry
        ref_phot:
            Reference photometry
        filter_map:
            A dictionary where the keys are the output names, and the value:
                1. Objective band
                2. Observed band
            The measured flux is corrected by the ratio objective/observed from the
            reference objects. Note that 1 and 2 can be the same, but not necessarily
    """

    def __init__(self, target_phot: Photometry, ref_phot: Photometry,
                 filter_map: Dict[Tuple[str, str], Tuple[str, str]]):
        self.__target_phot = target_phot
        self.__ref_phot = ref_phot

        obj_filters = {t[0] for t in filter_map.values()}
        obs_filters = {t[1] for t in filter_map.values()}

        self.__ref_filters = list(obj_filters.union(obs_filters))
        self.__ref_filters_idx = dict(
            zip(self.__ref_filters, ref_phot.system.get_band_indexes(self.__ref_filters)))
        self.__filter_map = filter_map

    def get_column_definition(self):
        col_defs = []
        for output_names, input_names, in self.__filter_map.items():
            t_obs_err = input_names[3]
            col_defs.append((output_names[0], np.float32, self.__ref_phot.unit))
            if t_obs_err:
                col_defs.append((output_names[1], np.float32, self.__ref_phot.unit))
        return col_defs

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray, output: np.ndarray):
        neighbor_indexes = neighbor_info['NEIGHBOR_INDEX']
        neighbor_weight = neighbor_info['NEIGHBOR_WEIGHTS']
        # Neighbor photometry on the target color space
        ref_target_colorspace = neighbor_info['NEIGHBOR_PHOTOMETRY']
        target_photo = self.__target_phot[indexes]
        with np.errstate(divide='ignore', invalid='ignore'):
            for (col, col_err), (obj, obs, _, _) in self.__filter_map.items():
                target_obs = target_photo.get_fluxes(obs)
                # Neighbor photometry on the restframe color space
                ref_restframe = self.__ref_phot.get_fluxes(obj)[neighbor_indexes]
                # Compute the ratio between the rest frame and the target colorspace
                obs_idx = self.__ref_filters_idx[obs]
                ratio = (ref_restframe / ref_target_colorspace[:, :, obs_idx, 0]).value
                # Normalize using the weights
                ratio *= neighbor_weight
                ratio /= np.sum(neighbor_weight, axis=-1)[..., np.newaxis]
                ratio = ratio.sum(axis=-1)
                # Correct the target photometry using the ratio (effectively projecting the
                # observed flux into the rest frame color space)
                output[col] = target_obs
                np.multiply(output[col], ratio, out=output[col])
