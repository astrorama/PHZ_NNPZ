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
from typing import Any, Dict

import numpy as np
from nnpz.config import ConfigManager
from nnpz.config.output.OutputHandlerConfig import OutputHandlerConfig
from nnpz.config.reference import ReferenceConfig
from nnpz.io.output_column_providers.McCounter import McCounter
from nnpz.io.output_column_providers.McPdf1D import McPdf1D
from nnpz.io.output_column_providers.McPdf2D import McPdf2D
from nnpz.io.output_column_providers.McSampler import McSampler
from nnpz.io.output_column_providers.McSamples import McSamples
from nnpz.io.output_column_providers.McSliceAggregate import McSliceAggregate
from nnpz.io.output_hdul_providers.McCounterBins import McCounterBins
from nnpz.io.output_hdul_providers.McPdf1DBins import McPdf1DBins
from nnpz.io.output_hdul_providers.McPdf2DBins import McPdf2DBins
from nnpz.io.output_hdul_providers.McSliceAggregateBins import McSliceAggregateBins


class McPdfConfig(ConfigManager.ConfigHandler):
    """
    Configuration handler for the generation of 1D PDF computed from a weighted random
    sample taken from the reference objects sampled n-dimensional PDF
    """

    def __init__(self):
        self.__added = False
        self.__output = None
        self.__ref_sample = None
        self.__take_n = None
        self.__samplers = {}

    def __parse_args_common(self, args: Dict[str, Any]):
        output_config = ConfigManager.get_handler(OutputHandlerConfig).parse_args(args)
        ref_config = ConfigManager.get_handler(ReferenceConfig).parse_args(args)

        self.__output = output_config['output_handler']
        self.__ref_sample = ref_config['reference_sample']
        self.__take_n = args.get('mc_pdf_take_n', 100)

    def __add_common(self, args: Dict[str, Any]):
        for cfg_key in ['mc_1d_pdf', 'mc_2d_pdf', 'mc_samples', 'mc_count', 'mc_slice_aggregate']:
            cfg = args.get(cfg_key, {})
            for provider_name, _ in cfg.items():
                if provider_name in self.__samplers:
                    continue
                provider = self.__ref_sample.get_provider(provider_name)
                sampler = McSampler(take_n=self.__take_n, mc_provider=provider)
                self.__samplers[provider_name] = sampler
                self.__output.add_column_provider(sampler)

    def __add_mc_1d_pdf(self, args: Dict[str, Any]):
        mc_1d_pdf = args.get('mc_1d_pdf', None)
        if not mc_1d_pdf:
            return

        for provider_name, parameters in mc_1d_pdf.items():
            sampler = self.__samplers[provider_name]
            provider = sampler.get_provider()
            for param, binning in parameters:
                # Histogram values
                self.__output.add_column_provider(McPdf1D(
                    sampler, param_name=param, binning=binning
                ))
                # Histogram binning
                self.__output.add_extension_table_provider(
                    McPdf1DBins(param, binning, unit=provider.get_unit(param)))

    def __add_mc_2d_pdf(self, args: Dict[str, Any]):
        mc_2d_pdf = args.get('mc_2d_pdf', None)
        if not mc_2d_pdf:
            return

        for provider_name, parameters in mc_2d_pdf.items():
            sampler = self.__samplers[provider_name]
            provider = sampler.get_provider()
            for param1, param2, binning1, binning2 in parameters:
                # Histogram values
                self.__output.add_column_provider(McPdf2D(
                    sampler, param_names=(param1, param2),
                    binning=(binning1, binning2)
                ))
                # Histogram binning
                self.__output.add_extension_table_provider(
                    McPdf2DBins((param1, param2), (binning1, binning2),
                                units=[provider.get_unit(param1), provider.get_unit(param2)])
                )

    def __add_samples(self, args: Dict[str, Any]):
        mc_samples = args.get('mc_samples', None)
        if not mc_samples:
            return

        for provider_name, parameters in mc_samples.items():
            sampler = self.__samplers[provider_name]
            for parameter_set in parameters:
                self.__output.add_column_provider(McSamples(sampler, parameter_set))

    def __add_counters(self, args: Dict[str, Any]):
        mc_counters = args.get('mc_count', None)
        if not mc_counters:
            return

        for provider_name, parameters in mc_counters.items():
            sampler = self.__samplers[provider_name]
            provider = sampler.get_provider()
            for parameter, bins in parameters:
                pdtype = provider.get_dtype(parameter)
                if not np.issubdtype(pdtype, np.int) and not np.issubdtype(pdtype, np.bool):
                    raise Exception('Can only count integer types, got {}'.format(pdtype))
                if not np.issubdtype(bins.dtype, np.int) and not np.issubdtype(bins.dtype, np.bool):
                    raise Exception('The binning must be integers, got {}'.format(bins.dtype))
                bins = np.sort(bins)
                self.__output.add_column_provider(McCounter(sampler, parameter, bins))
                self.__output.add_extension_table_provider(
                    McCounterBins(parameter, bins, unit=provider.get_unit(parameter)))

    def __add_slicers(self, args: Dict[str, Any]):
        mc_slicers = args.get('mc_slice_aggregate', None)
        if not mc_slicers:
            return

        for provider_name, slice_cfgs in mc_slicers.items():
            sampler = self.__samplers[provider_name]
            provider = sampler.get_provider()
            for slice_cfg in slice_cfgs:
                target, sliced, binning, aggs = slice_cfg
                for suffix, agg in aggs.items():
                    self.__output.add_column_provider(McSliceAggregate(
                        sampler, target, sliced, suffix, agg, binning,
                        unit=provider.get_unit(target)
                    ))
                    self.__output.add_extension_table_provider(McSliceAggregateBins(
                        target, sliced, suffix, binning, unit=provider.get_unit(sliced)
                    ))

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.__added:
            self.__parse_args_common(args)
            self.__add_common(args)
            self.__add_mc_1d_pdf(args)
            self.__add_mc_2d_pdf(args)
            self.__add_samples(args)
            self.__add_counters(args)
            self.__add_slicers(args)
            self.__added = True
        return {}


ConfigManager.add_handler(McPdfConfig)
