import numpy as np
from nnpz.config import ConfigManager
from nnpz.config.nnpz import NeighborSelectorConfig, OutputHandlerConfig, TargetCatalogConfig
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
        self.__mc_1d_pdf = []
        self.__mc_2d_pfd = []
        self.__args = []
        self.__samplers = {}

    def __parse_args_common(self, args):
        neighbor_config = ConfigManager.getHandler(NeighborSelectorConfig).parseArgs(args)
        output_config = ConfigManager.getHandler(OutputHandlerConfig).parseArgs(args)
        ref_config = ConfigManager.getHandler(ReferenceConfig).parseArgs(args)
        target_config = ConfigManager.getHandler(TargetCatalogConfig).parseArgs(args)

        self.__catalog_size = target_config['target_size']
        self.__neighbor_no = neighbor_config['neighbor_no']
        self.__output = output_config['output_handler']
        self.__ref_ids = ref_config['reference_ids']
        self.__ref_sample = ref_config['reference_sample']
        self.__take_n = args.get('mc_pdf_take_n', 100)

    def __add_common(self, args):
        for cfg_key in ['mc_1d_pdf', 'mc_2d_pdf', 'mc_samples', 'mc_count', 'mc_slice_aggregate']:
            cfg = args.get(cfg_key, {})
            for provider_name, _ in cfg.items():
                if provider_name in self.__samplers:
                    continue
                provider = self.__ref_sample.getProvider(provider_name)
                sampler = McSampler(
                    catalog_size=self.__catalog_size, n_neighbors=self.__neighbor_no,
                    take_n=self.__take_n, mc_provider=provider, ref_ids=self.__ref_ids
                )
                self.__samplers[provider_name] = sampler
                self.__output.addColumnProvider(sampler)

    def __add_mc_1d_pdf(self, args):
        mc_1d_pdf = args.get('mc_1d_pdf', None)
        if not mc_1d_pdf:
            return

        for provider_name, parameters in mc_1d_pdf.items():
            sampler = self.__samplers[provider_name]
            for param, binning in parameters:
                # Histogram values
                self.__output.addColumnProvider(McPdf1D(
                    sampler, param_name=param, binning=binning
                ))
                # Histogram binning
                self.__output.addExtensionTableProvider(McPdf1DBins(param, binning))

    def __add_mc_2d_pdf(self, args):
        mc_2d_pdf = args.get('mc_2d_pdf', None)
        if not mc_2d_pdf:
            return

        for provider_name, parameters in mc_2d_pdf.items():
            sampler = self.__samplers[provider_name]
            for param1, param2, binning1, binning2 in parameters:
                # Histogram values
                self.__output.addColumnProvider(McPdf2D(
                    sampler, param_names=(param1, param2),
                    binning=(binning1, binning2)
                ))
                # Histogram binning
                self.__output.addExtensionTableProvider(
                    McPdf2DBins((param1, param2), (binning1, binning2))
                )

    def __add_samples(self, args):
        mc_samples = args.get('mc_samples', None)
        if not mc_samples:
            return

        for provider_name, parameters in mc_samples.items():
            sampler = self.__samplers[provider_name]
            for parameter_set in parameters:
                self.__output.addColumnProvider(McSamples(sampler, parameter_set))

    def __add_counters(self, args):
        mc_counters = args.get('mc_count', None)
        if not mc_counters:
            return

        for provider_name, parameters in mc_counters.items():
            sampler = self.__samplers[provider_name]
            for parameter, bins in parameters:
                pdtype = sampler.getProvider().getDtype(parameter)
                if not np.issubdtype(pdtype, np.int) and not np.issubdtype(pdtype, np.bool):
                    raise Exception('Can only count integer types, got {}'.format(pdtype))
                if not np.issubdtype(bins.dtype, np.int) and not np.issubdtype(bins.dtype, np.bool):
                    raise Exception('The binning must be an integer type, got {}', bins.dtype)
                bins = np.sort(bins)
                self.__output.addColumnProvider(McCounter(sampler, parameter, bins))
                self.__output.addExtensionTableProvider(McCounterBins(parameter, bins))

    def __add_slicers(self, args):
        mc_slicers = args.get('mc_slice_aggregate', None)
        if not mc_slicers:
            return

        for provider_name, slice_cfgs in mc_slicers.items():
            sampler = self.__samplers[provider_name]
            for slice_cfg in slice_cfgs:
                target, sliced, binning, aggs = slice_cfg
                for suffix, agg in aggs.items():
                    self.__output.addColumnProvider(McSliceAggregate(
                        sampler, target, sliced, suffix, agg, binning
                    ))
                    self.__output.addExtensionTableProvider(McSliceAggregateBins(
                        target, sliced, suffix, binning
                    ))

    def parseArgs(self, args):
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


ConfigManager.addHandler(McPdfConfig)
