"""
Created on: 03/11/17
Author: Nikolaos Apostolakos
"""

from __future__ import division, print_function


def getLogger(name):
    try:
        from ElementsKernel import Logging
        return Logging.getLogger(name)
    except ImportError:
        import logging
        return logging.getLogger(name)
    
    
def enableStdErrLogging(level='INFO'):
    try:
        from ElementsKernel import Logging
        Logging.setLevel(level)
    except ImportError:
        import logging
        if not [h for h in logging.getLogger().handlers if h.get_name() == 'nnpz_log_handler']:
            handler = logging.StreamHandler()
            handler.set_name('nnpz_log_handler')
            logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(level)
        
        