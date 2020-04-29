'''
Just a collection of useful functions
'''

import yaml
from yaml import Loader, Dumper

import os
import shutil

import glob

data_path = os.path.expandvars('$CMSSW_BASE/src/tW_scattering/data/')

def loadConfig():
    with open(data_path+'config.yaml') as f:
        config = yaml.load(f, Loader=Loader)
    return config

def getName( DAS ):
    split = DAS.split('/')
    if split[-1].count('AOD'):
        return '__'.join(DAS.split('/')[1:3])
    else:
        return '__'.join(DAS.split('/')[-3:-1])
        #return'dummy'

def finalizePlotDir( path ):
    path = os.path.expandvars(path)
    if not os.path.isdir(path):
        os.makedirs(path)
    shutil.copy( os.path.expandvars( '$CMSSW_BASE/src/tW_scattering/Tools/php/index.php' ), path )
    
