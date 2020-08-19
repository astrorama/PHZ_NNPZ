from collections import namedtuple

import numpy as np

SedIter = namedtuple('Iter', ['sed'])


class SedGenerator(object):
    """
    Generates lazily a list of SEDs at different redshift positions
    """

    def __init__(self):
        self.__seds = []
        self.__zs = []

    def add(self, sed, z):
        """
        Add a new pair of SED / set of redshift to the generator
        Args:
            sed:
                A 2D numpy array, where the first axis corresponds to the number of knots,
                and the second always to 2: wavelength and flux
            z:
                A 1D iterable with different redshifts samples
        """
        self.__seds.append(sed)
        self.__zs.append(z)

    def __iter__(self):
        """
        Generator

        Returns:
            It yields a SedIter that mimics the ReferenceSample iterator, so it can be
            used as a drop-in replacement for the photometry computation
        """
        for sed, zs in zip(self.__seds, self.__zs):
            for z in zs:
                yield SedIter(sed=np.stack([sed[:, 0] * (z + 1), sed[:, 1] / (1 + z) ** 2], axis=1))

    def __len__(self):
        return len(self.__seds) * len(self.__zs[0])
