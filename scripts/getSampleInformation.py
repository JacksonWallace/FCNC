'''
takes DAS name, checks for local availability, reads norm, x-sec

e.g.
/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/NANOAODSIM
-->
TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1

'''

import yaml
from yaml import Loader, Dumper

import os

import uproot
import glob
from coffea.processor.dataframe import LazyDataFrame

from Tools.helpers import *

data_path = os.path.expandvars('$TWBASE/data/')

#def loadConfig():
#    with open(data_path+'config.yaml') as f:
#        config = yaml.load(f, Loader=Loader)
#    return config
#
#def getName( DAS ):
#    split = DAS.split('/')
#    if split[-1].count('AOD'):
#        return '__'.join(DAS.split('/')[1:3])
#    else:
#        return'dummy'

def readSampleNames( sampleFile ):
    with open( sampleFile ) as f:
        samples = [ tuple(line.split()) for line in f.readlines() ]
    return samples
    
def readSumWeight( samplePath ):
    # for central nanoAOD
    sumWeight = 0.
    nEvents = 0
    files = glob.glob(samplePath + '/*.root')
    for f in files:
        rfile = uproot.open(f)
        try:
            tree = rfile['Runs']
            df = LazyDataFrame(tree)
            sumWeight += float(df['genEventSumw_'])
            nEvents += int(df['genEventCount_'])
        except KeyError:
            tree = rfile['Events']
            df = LazyDataFrame(tree)
            sumWeight += float(df['genWeight'].sum())
            nEvents += len(df['genWeight'])

    return sumWeight, nEvents


def main():

    config = loadConfig()

    # get list of samples
    sampleList = readSampleNames( data_path+'samples.txt' )

    with open(data_path+'samples.yaml') as f:
        samples = yaml.load(f, Loader=Loader)

    # initialize
    if not samples:
        samples = {}

    for sample in sampleList:
        name = getName(sample[0])
        print (sample[0], name)
        if not sample[0] in samples.keys():
            samplePath = os.path.join(config['meta']['localNano'], getName(sample[0]) )
            if os.path.isdir( samplePath ):
                pass
            elif os.path.isdir( sample[0] ):
                samplePath = sample[0]
            else:
                raise NotImplementedError
            sumWeight, nEvents = readSumWeight(samplePath)
            samples.update({str(sample[0]): {'sumWeight': sumWeight, 'nEvents': nEvents, 'xsec': float(sample[1]), 'name':name, 'path':samplePath}})
        #sample['sumWeight'] = sumWeight
        #sample[

    with open(data_path+'samples.yaml', 'w') as f:
        yaml.dump(samples, f, Dumper=Dumper)
    # check if they are in the yaml file
    
    

    return samples


if __name__ == '__main__':
    samples = main()



