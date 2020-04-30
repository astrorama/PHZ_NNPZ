"""
Created on: 27/04/2018
Author: Alejandro Alvarez Ayllon
"""
import numpy as np
from ElementsKernel import Logging

from nnpz import NnpzFlag

log = Logging.getLogger('ReferenceSampleWeightCalculator')


def _apply_weight_calculator(fcalculator, target_list, ref_obj, target_data, result_flags):
    target_obj = [target_data[t.index] for t in target_list]
    target_flags = [result_flags[t.index] for t in target_list]
    nan_masks = [np.logical_not(np.isnan(target_obj_i[:, 0])) for target_obj_i in target_obj]
    return [fcalculator(ref_obj_i[mask_i, :], target_obj_i[mask_i, :], flag) for
            ref_obj_i, target_obj_i, mask_i, flag in
            zip(ref_obj, target_obj, nan_masks, target_flags)]


class ReferenceSampleWeightCalculator(object):

    def __init__(self, weight_phot_provider, weight_calculator, weight_calculator_alt, scaling):
        """
        Constructor
        Args:
            weight_phot_provider: An object implementing WeightPhotometryProvider
            weight_calculator: An object implementing WeightCalculatorInterface
            scaling: An object that implement photometry rescaling
        """
        self._weight_phot_provider = weight_phot_provider
        self._weight_calculator = weight_calculator
        self._weight_calculator_alt = weight_calculator_alt
        self._scaling = scaling

    def computeWeights(self, affected, target_data, result_flags, progress_listener):
        """
        Args:
            affected: The output of AffectedSourcesFinder.findAffected
            target_data: Target catalog data
            result_flags: A list of NnpzFlag, one per entry on the target catalog
            progress_listener: An object implementing ProgressListener

        Returns:
            A map where the keys are the indices of the reference sample objects
            and values are lists of the computed weight of the reference sample object per each
            object in the target catalog
        """
        # Note that we iterate the affected map in increasing order of the reference
        # sample indices. This is done to use as much the cache of the disk, by accessing
        # the SEDs sequentially.

        # All weights of a target object are 0 if their sum is 0
        weight_sum_per_target = np.zeros(target_data.shape[0], dtype=np.float32)

        # A map where the key is a tuple (ref index, target_index), and the value
        # an alternative weight precomputed if the preferred one was 0
        alt_weights = {}

        # A map to keep track of the references that are a neighbor of a given target
        alt_neighbors = {}

        filters_shape = target_data.shape[1:]
        for progress, ref_i in enumerate(sorted(affected)):
            progress_listener(progress)

            # Affected targets
            target_list = affected[ref_i]

            # Get the reference sample object photometry to use for the weight calculation
            # This array has one row per target object, with one column for the target ID, plus one column per band
            ref_photometry = np.ndarray((len(target_list),) + filters_shape, dtype=np.float32)
            for i, target in enumerate(target_list):
                flag = result_flags[target.index]
                ref_photometry[i, :, :] = self._weight_phot_provider(ref_i, target, flag)

                # Re-apply the scaling if needed
                if self._scaling:
                    target.scale = self._scaling(
                        ref_photometry[i:i + 1, :, 0], ref_photometry[i:i + 1, :, 1],
                        target_data[target.index, :, 0], target_data[target.index, :, 1])
            ref_photometry[i, :, :] *= target_list.scale

            weights = _apply_weight_calculator(
                self._weight_calculator, target_list, ref_photometry, target_data, result_flags
            )

            for target, w in zip(target_list, weights):
                target.weight = w
                weight_sum_per_target[target.index] += target.weight

            # Since we have the photometry calculated here already, and
            # computing it is expensive, for weights that are 0 with the preferred method
            # we apply an alternative method, and keep it for later, in case all weights
            # for a given target are 0
            if self._weight_calculator_alt:
                # Objects with zero weight
                target_zero_weights = list(filter(lambda t: t.weight == 0, target_list))
                # Select the reference (recomputed) photometries for those objects
                ref_obj_zero_weights = [
                    ref_obj for ref_obj, target_obj in zip(ref_photometry, target_list) if target_obj.weight == 0
                ]
                new_weights = _apply_weight_calculator(
                    self._weight_calculator_alt, target_zero_weights, ref_obj_zero_weights, target_data, result_flags
                )
                for target, w in zip(target_zero_weights, new_weights):
                    alt_weights[(ref_i, target.index)] = w
                    if target.index not in alt_neighbors:
                        alt_neighbors[target.index] = list()
                    alt_neighbors[target.index].append((ref_i, target))

        # For target objects with *all* their weights being 0, override with the alternative
        all_zero_mask = (weight_sum_per_target == 0)
        all_zero_i = np.arange(len(weight_sum_per_target))[all_zero_mask]

        if len(all_zero_i) > 0:
            log.debug('{} objects with all weights set to 0, using alternative weight'.format(len(all_zero_i)))
            log.debug(all_zero_i)

        for target_i in all_zero_i:
            # Some target entries may not even have any neighbor
            for ref_i, target in alt_neighbors.get(target_i, []):
                target.weight = alt_weights[(ref_i, target_i)]
            result_flags[target_i] |= NnpzFlag.AlternativeWeightFlag
