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
from typing import Dict, List, Tuple

import numpy as np
from astropy.table import Column
from nnpz.io import OutputHandler
from nnpz.reference_sample import PhotometryProvider


class UniformPhotometry(OutputHandler.OutputColumnProviderInterface):
    """
    Generate a list of columns with an uniform photometry:
    \f[
    g_o^* = g_o \left( \frac{g_r^*}{g_r} \right)
    \f]

    Where \f$g_r^*\f$ is the reference modeled flux for the objective band, and \f$g_r\f$ the
    reference modeled flux for the observed band (including any correction as reddening).
    \f$g_o\f$ is the observed flux for the target object

    Args:
        catalog_photo:
            Target catalog photometry
        reference_photo:
            A PhotometryProvider instance
        filter_map:
            A dictionary where the keys are the output names, and the value a quadruplet:
            1. Objective ideal band, must match a band on the reference
            2. Observed ideal band, must match a band on the reference
            3. Measured flux, must match a column on the target
            4. Measured flux error, must match a column on the target
            The measured flux (3) is corrected by the ratio objective/observed from the
            reference objects. Note that 1 and 2 can be the same, but not necessarily
    """

    def __init__(self, catalog_photo: np.ndarray, reference_photo: PhotometryProvider,
                 filter_map: Dict[Tuple[str, str], Tuple[str, str, str, str]]):
        self.__catalog_photo = catalog_photo
        obj_filters = set([t[0] for t in filter_map.values()])
        obs_filters = set([t[1] for t in filter_map.values()])
        self.__ref_filters = list(obj_filters.union(obs_filters))
        self.__ref_filters_idx = dict([(f, i) for i, f in enumerate(self.__ref_filters)])
        self.__ref_photo = reference_photo.getData(*self.__ref_filters)
        self.__filter_map = filter_map
        self.__total_ratios = np.zeros(
            (len(self.__catalog_photo), len(filter_map)), dtype=np.float64
        )
        self.__total_weights = np.zeros((len(self.__catalog_photo), 1), dtype=np.float64)

    def addContribution(self, reference_sample_i, neighbor, flags):
        original = self.__ref_photo[reference_sample_i, :, 0]
        matched = neighbor.matched_photo[0]

        for r, (r_obj, r_obs, _, _) in enumerate(self.__filter_map.values()):
            ratio = original[self.__ref_filters_idx[r_obj]] / matched[r_obs]
            self.__total_ratios[neighbor.index, r] += ratio * neighbor.weight
        self.__total_weights[neighbor.index] += neighbor.weight

    def getColumns(self):
        columns = []
        ratios = self.__total_ratios / self.__total_weights

        for r, (output_names, input_names) in enumerate(self.__filter_map.items()):
            r_obj, r_obs, t_obs, t_obs_err = input_names
            columns.append(
                Column(
                    self.__catalog_photo[t_obs] * ratios[:, r],
                    name=output_names[0], dtype=np.float32
                )
            )
            if t_obs_err:
                columns.append(
                    Column(
                        self.__catalog_photo[t_obs_err],
                        name=output_names[1], dtype=np.float32
                    )
                )
        return columns
