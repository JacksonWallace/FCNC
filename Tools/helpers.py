'''
Just a collection of useful functions
'''

#import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

#from yaml import Loader, Dumper

import os
import shutil

import glob

data_path = os.path.expandvars('$TWHOME/data/')

def loadConfig():
    with open(data_path+'config.yaml') as f:
        config = load(f, Loader=Loader)
    return config

def dumpConfig(cfg):
    with open(data_path+'config.yaml', 'w') as f:
        dump(cfg, f, Dumper=Dumper, default_flow_style=False)
    return True

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
    shutil.copy( os.path.expandvars( '$TWHOME/Tools/php/index.php' ), path )
    
def addRowToCutFlow( output, df, cfg, name, selection, processes=['TTW', 'TTX', 'diboson', 'ttbar', 'tW_scattering'] ):
    '''
    add one row with name and selection for each process to the cutflow accumulator
    '''
    for process in processes:
        if selection is not None:
            output[process][name] += ( sum(df['weight'][ (df['dataset']==process) & selection ].flatten() )*cfg['lumi'] )
        else:
            output[process][name] += ( sum(df['weight'][ (df['dataset']==process) ].flatten() )*cfg['lumi'] )
            
def getCutFlowTable(cache, processes=['tW_scattering', 'TTW', 'ttbar'], lines=['skim', 'twoJet', 'oneBTag']):
    '''
    Takes a cache and returns a formated cut-flow table of processes.
    Lines and processes have to follow the naming of the coffea processor output.
    '''
    res = {}
    for proc in processes:
        res[proc] = {line: cache.get('simple_output')[proc][line] for line in lines}
    df = pd.DataFrame(res)
    df = df.reindex(['skim', 'twoJet', 'oneBTag']) # restores the proper order
    print (df[processes])
    return df