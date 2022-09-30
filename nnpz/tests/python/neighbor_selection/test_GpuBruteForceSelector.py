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
from nnpz.exceptions import InvalidDimensionsException, UninitializedException
from nnpz.utils.distances import chi2

from .fixtures import *

try:
    import cupy as cp
    from nnpz.neighbor_selection.gpu_bruteforce import GPUBruteForceSelector

    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No GPU available', allow_module_level=True)
except ImportError:
    pytest.skip('No CUDA available', allow_module_level=True)


###############################################################################

def test_GpuBruteForceNotInitialized(target_values: Photometry):
    """
    Querying before initializing must throw
    """
    bf_selector = GPUBruteForceSelector(k=4)
    with pytest.raises(UninitializedException):
        bf_selector.query(target_values)


###############################################################################

def test_GpuBruteForceInvalidDimensions(reference_values: Photometry, target_values: Photometry):
    """
    Querying with an invalid dimensionality must throw
    """
    bf_selector = GPUBruteForceSelector(k=4)
    bf_selector.fit(reference_values, reference_values.system)
    with pytest.raises(InvalidDimensionsException):
        bf_selector.query(target_values.subsystem(['x', 'y']))


###############################################################################

def test_GpuBruteForceClosest(reference_values: Photometry, target_values: Photometry):
    bf_selector = GPUBruteForceSelector(k=4)
    bf_selector.fit(reference_values, reference_values.system)

    idx, scales = bf_selector.query(target_values)
    # Only the center is a hit
    distances = chi2(reference_values.values[idx[0]], target_values.values[0])
    assert (len(idx) == len(scales))
    assert (len(idx) == 1)

    # Order is not defined! Only that 0 should be there
    assert 0 in distances


###############################################################################

def test_GpuBruteForceClosestNaN(reference_values: Photometry, target_values: Photometry):
    # Search removing the 'z' band
    ref_subset = reference_values.subsystem(['x', 'y'])
    target_subset = target_values.subsystem(['x', 'y'])
    bf_selector = GPUBruteForceSelector(k=4)
    bf_selector.fit(ref_subset, ref_subset.system)
    idx, scales = bf_selector.query(target_subset)

    # Search all bands with Inf
    target_values.values = np.copy(target_values.values)
    target_values.values[:, 2, 1] = np.inf
    bf_selector_nan = GPUBruteForceSelector(k=4)
    bf_selector_nan.fit(reference_values, reference_values.system)
    idx_nan, scales_nan = bf_selector_nan.query(target_values)

    # Must be equal
    np.testing.assert_array_equal(idx_nan, idx)
    np.testing.assert_array_equal(scales_nan, scales)

###############################################################################
