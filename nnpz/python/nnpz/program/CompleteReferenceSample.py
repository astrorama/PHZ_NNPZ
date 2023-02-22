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

"""
Created on: 26/01/2023
Author: dubathf
"""
import ElementsKernel.Logging as Logging
import argparse
from nnpz.reference_sample import ReferenceSample
from nnpz.reference_sample.MontecarloProvider import MontecarloProvider
from astropy.table import Table
from os.path import join
import numpy as np

logger = Logging.getLogger('CompleteReferenceSample')

def defineSpecificProgramOptions():
    parser = argparse.ArgumentParser()

    parser.add_argument('-rd','--result-dir', type=str, required=True,
                        help='Path to the folder containing the result of the compute redshift run')
    parser.add_argument('-pc','--phz-cat-file', type=str, default='phz_cat.fits',
                        help='Name of the phz catalog caontaining the ids (default="phz_cat.fits")')
    parser.add_argument('-pf','--posterior-folder', type=str, default='posteriors',
                        help='Name of the folder containing the sampling (default="posteriors")')
    parser.add_argument('-if','--index-file', type=str, default='Index_File_posterior.fits',
                        help='Name of the sampling\'s index file (default="Index_File_posterior.fits")')
    parser.add_argument('-rs','--reference-sample-dir', type=str,  required=True,
                        help='Path of the reference sample dir to be completed')
    return parser

def mainMethod(args):
    """
    Entry point for CompleteReferenceSample
    """
    logger.info('Open the phz_cat to get the mapping between the IDs')
    phz_cat = Table.read(join(args.result_dir, args.phz_cat_file))
    ids = Table()
    ids['PHZ_ID']=phz_cat['ID']
    ids['REF_SAMPLE_ID']=np.array(range(len(ids)))
    del phz_cat
    
    logger.info('Open sampling index catalog')
    index_sampling = Table.read(join(args.result_dir, args.posterior_folder , args.index_file))
    
    logger.info('Open the existing reference sample to be completed')
    ref_sample = ReferenceSample.ReferenceSample(args.reference_sample_dir) 
    
    logger.info('Create the new data provider')
    pp_provider = ref_sample.add_provider('MontecarloProvider', name='pp', data_pattern = 'pp_data_{}.npy', overwrite = True)
    
    logger.info('Add the sampling data to the reference sample')
    current_sample_file = ''
    sample_table=Table()
    id_tot = str(len(ids))
    for row in ids:   
        target_file = index_sampling['FILE_NAME'][index_sampling['OBJECT_ID']==row['PHZ_ID']][0]
        # open the right sampling file
        if target_file!= current_sample_file:
            
            current_sample_file=target_file
            sample_table=Table.read(join(args.result_dir, args.posterior_folder , current_sample_file))
        
        # extract the data
        
        samples = sample_table[sample_table['OBJECT_ID']==row['PHZ_ID']]
        sample_t=Table()
        for col in sample_table.colnames:
            if col != 'OBJECT_ID':
                sample_t[col] = samples[col]     
        pp_provider.add_data(np.array([row['REF_SAMPLE_ID']]),np.array(sample_t).reshape(1, len(samples[col])))
        
        if row['REF_SAMPLE_ID']%100==0:
            id_curr = str(row['REF_SAMPLE_ID'])
            logger.info(f'Progress: {id_curr}/{id_tot}')
            
    logger.info('Writing the reference sample')   
    ref_sample.flush()
    
    logger.info('Completed') 
                      
    return 0
