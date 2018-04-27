"""
Created on: 27/04/2018
Author: Alejandro Alvarez Ayllon
"""
import numpy as np


class ReferenceSampleWeightCalculator(object):

    def __init__(self, weight_phot_provider, weight_calculator):
        """
        Constructor
        Args:
            weight_phot_provider: An object implementing WeightPhotometryProvider
            weight_calculator: An object implementing WeightCalculatorInterface
        """
        self._weight_phot_provider = weight_phot_provider
        self._weight_calculator = weight_calculator

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
        weights = {}
        for progress, ref_i in enumerate(sorted(affected)):
            progress_listener(progress)

            target_i_list = affected[ref_i]
            target_obj = target_data[target_i_list]
            target_flags = [result_flags[i] for i in target_i_list]

            # Get the reference sample object photometry to use for the weight calculation
            ref_obj = np.ndarray(target_obj.shape, dtype=np.float32)
            for i, target_i in enumerate(target_i_list):
                flag = result_flags[target_i]
                ref_obj[i, :, :] = self._weight_phot_provider(ref_i, target_i, flag)

            # Compute the weights. Note that we mask out the bands which are marked as NaN
            nan_masks = [np.logical_not(np.isnan(target_obj_i[:, 0])) for target_obj_i in target_obj]
            weights[ref_i] = [self._weight_calculator(ref_obj_i[mask_i, :], target_obj_i[mask_i, :], flag) for
                              ref_obj_i, target_obj_i, mask_i, flag in
                              zip(ref_obj, target_obj, nan_masks, target_flags)]

        return weights
