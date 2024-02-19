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

import astropy.units as u
import numpy as np
from nnpz.io import OutputHandler
from nnpz.io.output_column_providers.McSampler import McSampler

def it_mode(x):
    N=len(x)
    M=int((len(x)+1)/2)
    hs=1e10
    j0=-1
    k0=-1
    for j in range(M-1):
        if (x[j+M]-x[j] < hs):
            hs=x[j+M]-x[j]
            j0=j
            k0=j+M
    return(j0,k0)
    
def mode(xx):
    rd=np.sort(xx).astype(float)
    
    unique, count = np.unique(rd,  return_counts=True)
    if np.max(count)>1:
        # We have repeated values: get the maximum repeated value
        return unique[np.argmax(count)]
    else: 
        # no repeated values apply HSM algo from Bickel D.R. and Fruehwirth R. (2006). On a Fast, Robust Estimator of the Mode: Comparisons to Other Robust Estimators with Applications. https://arxiv.org/pdf/math/0505419.pdf
        k0=10
        j0=0
        while (k0>j0+2):
            (j0,k0)=it_mode(rd)
            rd=rd[j0:k0+1]
        return(np.mean(rd))

class McPdf1DMode(OutputHandler.OutputColumnProviderInterface):
    """
    Compute the Mode for a given parameter using a weighted random sample
    from the reference objects nearest to a given object

    See Also:
        nnpz.io.output_column_providers.McSampler
    Args:
        sampler: McSampler
            The handler that takes care of handling the weighted random sampling
        param_name:
            The parameter to compute the mode for
    """

    def __init__(self, sampler: McSampler, param_name: str):
        super(McPdf1DMode, self).__init__()
        self.__sampler = sampler
        self.__param_name = param_name
        self.__column = 'PHZ_PP_MODE_{}'.format(self.__param_name.upper())

    def get_column_definition(self):
        return [
            (self.__column, np.float32, u.dimensionless_unscaled)
        ]

    def generate_output(self, indexes: np.ndarray, neighbor_info: np.ndarray,
                        output: np.ndarray):
        samples = self.__sampler.get_samples()
        param_samples = samples[self.__param_name]
        output_col = output[self.__column]
        for i in range(len(output)):
            output_col[i] = mode(param_samples[i])
