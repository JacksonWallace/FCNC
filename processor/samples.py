import os
import re
import glob
from Tools.helpers import *
cfg = loadConfig()

version = cfg['meta']['version']
tag = version.replace('.','p')

data_path = os.path.join(cfg['meta']['localSkim'], tag)

# All samples
groups = {
    'tW_scattering': ['/tW_scattering[-_]'],
    'TTX':           ['/TTZToLLNuNu[-_]', '/ST_tWll[-_]', '/ST_tWnunu[-_]', '/TH[W,Q][-_]', '/TT[T,W,Z][T,W,Z][-_]', '/tZq[-_]', '/ttHToNonbb[-_]'],
    'ttbar':         ['/TTJets_SingleLept', '/TTJets_DiLept', '/ST_[s,t]-channel', '/ST_tW'],
    'TTW':           ['/TTWJets'],
    'wjets':         ['/W[1-4]JetsToLNu[-_]'],
    'diboson':       ['/[W,Z][W,Z]To', '/[W,Z][W,Z][W,Z][-_]']
}

# Selection for single lep
groups_1l = {
    'tW_scattering': ['/tW_scattering[-_]'],
    'TTX':           ['/TTZToLLNuNu[-_]', '/ST_tWnunu[-_]', '/TH[W,Q][-_]', '/TT[T,W,Z][T,W,Z][-_]', '/tZq[-_]', '/ttHToNonbb[-_]'],
    'ttbar':         ['/TTJets_SingleLept', '/ST_[s,t]-channel', '/ST_tW'],
    'TTW':           ['/TTWJets'],
    'wjets':         ['/W[1-4]JetsToLNu[-_]'],
}

# Selection for dilep = no W+jets
groups_2l = {
    'tW_scattering': ['/tW_scattering[-_]'],
    'TTX':           ['/TTZToLLNuNu[-_]', '/ST_tWll[-_]', '/ST_tWnunu[-_]', '/TH[W,Q][-_]', '/TT[T,W,Z][T,W,Z][-_]', '/tZq[-_]', '/ttHToNonbb[-_]'],
    'ttbar':         ['/TTJets_SingleLept', '/TTJets_DiLept', '/ST_[s,t]-channel', '/ST_tW'],
    'TTW':           ['/TTWJets'],
    'diboson':       ['/[W,Z][W,Z]To', '/[W,Z][W,Z][W,Z][-_]']
}

# Selection for trilep - no W+jets and single lepton tt/t
groups_3l = {
    'tW_scattering': ['/tW_scattering[-_]'],
    'TTX':           ['/TTZToLLNuNu[-_]', '/ST_tWll[-_]', '/ST_tWnunu[-_]', '/TH[W,Q][-_]', '/TT[T,W,Z][T,W,Z][-_]', '/tZq[-_]', '/ttHToNonbb[-_]'],
    'ttbar':         ['/TTJets_DiLept', '/ST_tW'],
    'TTW':           ['/TTWJets'],
    'diboson':       ['/[W,Z][W,Z]To', '/[W,Z][W,Z][W,Z][-_]']
}


samples = glob.glob(data_path + '/*')
fileset = { group: [] for group in groups.keys() }
fileset_1l = { group: [] for group in groups_1l.keys() }
fileset_2l = { group: [] for group in groups_2l.keys() }
fileset_3l = { group: [] for group in groups_3l.keys() }

for sample in samples:

    for group in groups.keys():
        for process in groups[group]:
            if re.search(process, sample):
                fileset[group] += glob.glob(sample+'/*.root')

    for group in groups_1l.keys():
        for process in groups_1l[group]:
            if re.search(process, sample):
                fileset_1l[group] += glob.glob(sample+'/*.root')

    for group in groups_2l.keys():
        for process in groups_2l[group]:
            if re.search(process, sample):
                fileset_2l[group] += glob.glob(sample+'/*.root')

    for group in groups_3l.keys():
        for process in groups_3l[group]:
            if re.search(process, sample):
                fileset_3l[group] += glob.glob(sample+'/*.root')

fileset_small = { sample: fileset[sample][:2] for sample in fileset.keys() }

