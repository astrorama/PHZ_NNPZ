#
#  Copyright (C) 2022 Euclid Science Ground Segment
#
#  This library is free software; you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation;
#  either version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with this library;
#  if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA
#

"""
NNPZ has been (re)designed as a pipeline process, where the input catalog
is cut into manageable "chunks", and for each chunk:

1. The best k neighbor candidates are chosen (NeighborFinder)
2. The photometry of those reference objects are projected into the target object color-spaces
   (CorrectPhotometry)
3. The weight for the neighbors are computed using the corrected photometry (ComputeWeights)

This allows delegating *most* of the tightest loops directly to numpy, as the same
operation can be applied between arrays, instead of target per target object.

These stages can be easily called separately, or chained together.

See Also:
    nnpz.program
"""
