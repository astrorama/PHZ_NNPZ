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
import os
from typing import Tuple

import cupy as cp
import numpy as np
from ElementsKernel import Logging, Path
from nnpz.exceptions import InvalidDimensionsException, UninitializedException
from nnpz.neighbor_selection import SelectorInterface
from nnpz.photometry.photometric_system import PhotometricSystem
from nnpz.photometry.photometry import Photometry

logger = Logging.getLogger(__name__)

cuda_path = Path.getPath('cuda', file_type='auxiliary')
with open(os.path.join(cuda_path, 'bruteforce.cu'), 'r') as bruteforce_cu_fd:
    bruteforce_module = cp.RawModule(code=bruteforce_cu_fd.read(), backend='nvcc')


class GPUBruteForceSelector(SelectorInterface):
    """
    Look for neighbors using a bruteforce method (i.e. compare with every single reference
    object)

    Args:
        k: int
            Number of neighbors
    """

    def __init__(self, k: int, threads: int = 128, scaling: bool = False):
        self.__k = k
        self.__system = None
        self.__unit = None
        self.__reference = None
        self.__threads = threads
        self.__chi2_kernel = bruteforce_module.get_function('chi2_bruteforce')
        self.__scaling = scaling

    def fit(self, train: Photometry, system: PhotometricSystem):
        """
        See Also:
            SelectorInterface.fit
        """
        logger.info('Copying the reference to device')
        self.__reference = cp.asarray(train.subsystem(system.bands).values)
        logger.info('Done copying the reference to the device')
        self.__system = system
        self.__unit = train.unit

    def query(self, target: Photometry) -> Tuple[np.ndarray, np.ndarray]:
        """
        See Also:
            SelectorInterface.query
        """
        if self.__reference is None:
            raise UninitializedException()
        if target.system != self.__system:
            raise InvalidDimensionsException()
        assert target.unit == self.__unit

        nreference = len(self.__reference)
        ntarget = len(target)
        nbands = len(self.__system)

        # Prepare device data
        device_photo = cp.asarray(target.values.value, dtype=cp.float64)

        # Prepare output for the distances
        device_distances = cp.empty((ntarget, self.__k), dtype=cp.float64)
        device_closest = cp.empty((ntarget, self.__k), dtype=cp.int32)
        device_scaling = cp.ones((ntarget, self.__k), dtype=cp.float64)

        # Blocks per grid
        nblocks = (len(target) + self.__threads - 1) // self.__threads

        # Compute distance matrix
        self.__chi2_kernel((nblocks,), (self.__threads,),
                           (self.__reference, device_photo, device_distances, device_scaling,
                            device_closest, self.__k, nbands, nreference, ntarget, self.__scaling))

        # Get K best
        neighbors = cp.asnumpy(device_closest)
        scales = cp.asnumpy(device_scaling)

        return neighbors, scales
