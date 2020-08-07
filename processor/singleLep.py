'''
Simple processor using coffea.
[x] weights
[ ] Missing pieces: appropriate sample handling
[x] Accumulator caching

'''


import os
import time
import glob
import re
from functools import reduce
from klepto.archives import dir_archive

import numpy as np
from tqdm.auto import tqdm
import coffea.processor as processor
from coffea.processor.accumulator import AccumulatorABC
from coffea.analysis_objects import JaggedCandidateArray
from coffea import hist
import pandas as pd
import uproot_methods
import awkward


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Tools.helpers import *

## Event shape variables
def cosTheta(obj):
    return np.cos(obj.cross(obj).i0.phi - obj.cross(obj).i1.phi)

def Wij(obj):
    return obj.cross(obj).i0.pt * obj.cross(obj).i1.pt

def FWMT1(obj):
    '''
    First Fox-Wolfram Moment reduced to transverse plane
    '''
    return (Wij(obj)*cosTheta(obj)).sum()/ (np.maximum(obj.pt.sum(), np.ones(len(obj.pt)))**2)

def FWMT2(obj):
    '''
    Second Fox-Wolfram Moment reduced to transverse plane
    '''
    return (Wij(obj)*(3*cosTheta(obj)**2-np.ones(len(obj.pt)))/2.).sum() / (np.maximum(obj.pt.sum(), np.ones(len(obj.pt)))**2)
    
def sphericity(obj):
    '''
    Attempt to calculate sphericity.
    S^{a,b} = sum_i(p_i^a p_i^b) / sum(|p_i|**2)
    S = 3/2 * (l2 + l3)
    with l2 and l3 the eigenvalues of S^{a,b}, l1>l2>l3, l1+l2+l3=1
    S=1: isotropic event (l1=l2=l3=1/3)
    S=0: linear event (l2=l3=0)
    S is not infrared safe. There's a linearized version, too.
    Circularity is C = 2*l2/(l1+l2)

    Numpy lesson:
    This is how you would easily get a 3x3 matrix from two vectors.
    row = np.array([[1, 3, 2]])
    col = np.array([[1], [3], [2]])
    M = col.dot(row)
    '''
    x, y, z = obj.p4.x, obj.p4.y, obj.p4.z
    # calculate sphericity tensor by hand
    S = np.array([[(x*x).sum(), (x*y).sum(), (x*z).sum()],[(x*y).sum(),(y*y).sum(),(z*y).sum()], [(x*z).sum(), (z*y).sum(),(z*z).sum()]] / np.maximum(np.ones(len(obj.p4)),(obj.p4.p**2).sum()) )
    # S[:,:,0] shows you the first matrix
    # np.linalg.eig(S[:,:,0])[0][1:].sum() * 3/2. gives the sphericity for the first event

    # sorted eigenvalues, l0>l1>l2. S needs to be transposed for np.linalg.eig to work
    l = -np.sort(-np.linalg.eig(S.transpose())[0])
    return l[:,1:].sum(axis=1) * 3/2.
    #return (l[:,1] + l[:,2]) * 3/2. # not sure why sum won't work

def sphericityBasic(obj):
    # same as above, but only using very basic AwkwardArray elements
    x, y, z = obj.p4.fPt*np.cos(obj.p4.fPhi), obj.p4.fPt*np.sin(obj.p4.fPhi), obj.p4.fPt*np.sinh(obj.p4.fEta)
    psq = x*x+y*y+z*z
    S_ten = np.array([[(x*x).sum(), (x*y).sum(), (x*z).sum()],[(x*y).sum(),(y*y).sum(),(z*y).sum()], [(x*z).sum(), (z*y).sum(),(z*z).sum()]] / np.maximum(np.ones(len(obj.p4)),(psq).sum()) )
    l = -np.sort(-np.linalg.eig(S_ten.transpose())[0])
    return l[:,1:].sum(axis=1) * 3/2.

def mergeArray(a1, a2):
    a1_tags = awkward.JaggedArray(a1.starts, a1.stops, np.full(len(a1.content), 0, dtype=np.int64))
    a1_index = awkward.JaggedArray(a1.starts, a1.stops, np.arange(len(a1.content), dtype=np.int64))
    a2_tags = awkward.JaggedArray(a2.starts, a2.stops, np.full(len(a2.content), 1, dtype=np.int64))
    a2_index = awkward.JaggedArray(a2.starts, a2.stops, np.arange(len(a2.content), dtype=np.int64))
    tags = awkward.JaggedArray.concatenate([a1_tags, a2_tags], axis=1)
    index = awkward.JaggedArray.concatenate([a1_index, a2_index], axis=1)
    return awkward.JaggedArray(tags.starts, tags.stops, awkward.UnionArray(tags.content, index.content, [a1.content, a2.content]))

# This just tells matplotlib not to open any
# interactive windows.
matplotlib.use('Agg')

class exampleProcessor(processor.ProcessorABC):
    """Dummy processor used to demonstrate the processor principle"""
    def __init__(self):

        # we can use a large number of bins and rebin later
        dataset_axis        = hist.Cat("dataset",   "Primary dataset")
        pt_axis             = hist.Bin("pt",        r"$p_{T}$ (GeV)", 1000, 0, 1000)
        ht_axis             = hist.Bin("ht",        r"$H_{T}$ (GeV)", 500, 0, 5000)
        mass_axis           = hist.Bin("mass",      r"M (GeV)", 1000, 0, 2000)
        eta_axis            = hist.Bin("eta",       r"$\eta$", 60, -5.5, 5.5)
        multiplicity_axis   = hist.Bin("multiplicity",         r"N", 20, -0.5, 19.5)
        norm_axis            = hist.Bin("norm",         r"N", 25, 0, 1)

        self._accumulator = processor.dict_accumulator({
            "MET_pt" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "pt_spec_max" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "MT" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "HT" :          hist.Hist("Counts", dataset_axis, ht_axis),
            "ST" :          hist.Hist("Counts", dataset_axis, ht_axis),
            "mbj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mjj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlb_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlb_min" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlj_max" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "mlj_min" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "N_b" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_spec" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "FWMT1" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT2" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT3" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "FWMT4" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "S" :               hist.Hist("Counts", dataset_axis, norm_axis),
            "S_lep" :           hist.Hist("Counts", dataset_axis, norm_axis),
            'cutflow_wjets':      processor.defaultdict_accumulator(int),
            'cutflow_ttbar':      processor.defaultdict_accumulator(int),
            'cutflow_TTW':      processor.defaultdict_accumulator(int),
            'cutflow_TTX':      processor.defaultdict_accumulator(int),
            'cutflow_signal':   processor.defaultdict_accumulator(int),
            'totalEvents':   processor.defaultdict_accumulator(int),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        """
        Processing function. This is where the actual analysis happens.
        """
        output = self.accumulator.identity()
        dataset = df["dataset"]
        cfg = loadConfig()
        # We can access the data frame as usual
        # The dataset is written into the data frame
        # outside of this function

        output['totalEvents']['all'] += len(df['weight'])

        output['cutflow_wjets']['all events'] += sum(df['weight'][(df['dataset']=='wjets')].flatten())
        output['cutflow_ttbar']['all events'] += sum(df['weight'][(df['dataset']=='ttbar')].flatten())
        output['cutflow_TTW']['all events'] += sum(df['weight'][(df['dataset']=='TTW')].flatten())
        output['cutflow_TTX']['all events'] += sum(df['weight'][(df['dataset']=='TTX')].flatten())
        output['cutflow_signal']['all events'] += sum(df['weight'][(df['dataset']=='tW_scattering')].flatten())

        cutFlow = ((df['nLepton']==1) & (df['nVetoLepton']==1))

        output['cutflow_wjets']['singleLep']  += sum(df['weight'][(df['dataset']=='wjets')         & cutFlow].flatten())
        output['cutflow_ttbar']['singleLep']  += sum(df['weight'][(df['dataset']=='ttbar')         & cutFlow].flatten())
        output['cutflow_TTW']['singleLep']    += sum(df['weight'][(df['dataset']=='TTW')           & cutFlow].flatten())
        output['cutflow_TTX']['singleLep']    += sum(df['weight'][(df['dataset']=='TTX')           & cutFlow].flatten())
        output['cutflow_signal']['singleLep'] += sum(df['weight'][(df['dataset']=='tW_scattering') & cutFlow].flatten())

        cutFlow = ((df['nLepton']==1) & (df['nVetoLepton']==1) & (df['nGoodJet']>5))

        output['cutflow_wjets']['5jets']  += sum(df['weight'][(df['dataset']=='wjets')         & cutFlow].flatten())
        output['cutflow_ttbar']['5jets']  += sum(df['weight'][(df['dataset']=='ttbar')         & cutFlow].flatten())
        output['cutflow_TTW']['5jets']    += sum(df['weight'][(df['dataset']=='TTW')           & cutFlow].flatten())
        output['cutflow_TTX']['5jets']    += sum(df['weight'][(df['dataset']=='TTX')           & cutFlow].flatten())
        output['cutflow_signal']['5jets'] += sum(df['weight'][(df['dataset']=='tW_scattering') & cutFlow].flatten())

        cutFlow = ((df['nLepton']==1) & (df['nVetoLepton']==1) & (df['nGoodJet']>5) & (df['nGoodBTag']==2))

        output['cutflow_wjets']['btags']  += sum(df['weight'][(df['dataset']=='wjets')         & cutFlow].flatten())
        output['cutflow_ttbar']['btags']  += sum(df['weight'][(df['dataset']=='ttbar')         & cutFlow].flatten())
        output['cutflow_TTW']['btags']    += sum(df['weight'][(df['dataset']=='TTW')           & cutFlow].flatten())
        output['cutflow_TTX']['btags']    += sum(df['weight'][(df['dataset']=='TTX')           & cutFlow].flatten())
        output['cutflow_signal']['btags'] += sum(df['weight'][(df['dataset']=='tW_scattering') & cutFlow].flatten())

        # preselection of events
        loose_selection = ((df['nLepton']==1) & (df['nVetoLepton']==1))
        #df = df[((df['nLepton']==1) & (df['nGoodJet']>5) & (df['nGoodBTag']==2))]

        # And fill the histograms
        output['MET_pt'].fill(dataset=dataset, pt=df["MET_pt"][loose_selection].flatten(), weight=df['weight'][loose_selection]*cfg['lumi'])
        output['MT'].fill(dataset=dataset, pt=df["MT"][loose_selection].flatten(), weight=df['weight'][loose_selection]*cfg['lumi'])
        output['N_b'].fill(dataset=dataset, multiplicity=df["nGoodBTag"][loose_selection], weight=df['weight'][loose_selection]*cfg['lumi'] )
        output['N_jet'].fill(dataset=dataset, multiplicity=df["nGoodJet"][loose_selection], weight=df['weight'][loose_selection]*cfg['lumi'] )

        Jet = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt = df['Jet_pt'].content,
            eta = df['Jet_eta'].content,
            phi = df['Jet_phi'].content,
            mass = df['Jet_mass'].content,
            goodjet = df['Jet_isGoodJetAll'].content,
            bjet = df['Jet_isGoodBJet'].content,
            jetId = df['Jet_jetId'].content,
            puId = df['Jet_puId'].content,
        )

        lepton = JaggedCandidateArray.candidatesfromcounts(
            df['nLepton'],
            pt = df['Lepton_pt'].content,
            eta = df['Lepton_eta'].content,
            phi = df['Lepton_phi'].content,
            mass = df['Lepton_mass'].content,
            pdgId = df['Lepton_pdgId'].content,
        )
        
        alljet = Jet[(Jet['goodjet']==1)] # all jets with pt>25 and pt>60 in 2.7<|eta|<3.0 (noise suppression)
        b = Jet[Jet['bjet']==1]
        nonb = Jet[(Jet['goodjet']==1) & (Jet['bjet']==0)]
        spectator = Jet[(abs(Jet.eta)>2.0) & (abs(Jet.eta)<4.7) & (Jet.pt>25) & (Jet['puId']>=7) & (Jet['jetId']>=6)] # 40 GeV seemed good. let's try going lower

        event_selection = (Jet.counts>5) & (b.counts>=2) & (nonb.counts>=4) & (df['nLepton']==1) & (df['nVetoLepton']==1)
        bj_pair = b.cross(nonb)
        jj_pair = nonb.cross(nonb)
        lb_pair = lepton.cross(b)
        lj_pair = lepton.cross(nonb)
        output['mbj_max'].fill(dataset=dataset, mass=bj_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mjj_max'].fill(dataset=dataset, mass=jj_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlb_min'].fill(dataset=dataset, mass=lb_pair[event_selection].mass.min().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlb_max'].fill(dataset=dataset, mass=lb_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlj_min'].fill(dataset=dataset, mass=lj_pair[event_selection].mass.min().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlj_max'].fill(dataset=dataset, mass=lj_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])

        ht = Jet[Jet['goodjet']==1].pt.sum()
        st = Jet[Jet['goodjet']==1].pt.sum() + lepton.pt.sum() + df['MET_pt']
        output['HT'].fill(dataset=dataset, ht=ht[event_selection].flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['ST'].fill(dataset=dataset, ht=st[event_selection].flatten(), weight=df['weight'][event_selection]*cfg['lumi'])

        output['FWMT1'].fill(dataset=dataset, norm=FWMT1(alljet)[event_selection], weight=df['weight'][event_selection]*cfg['lumi'])
        output['FWMT2'].fill(dataset=dataset, norm=FWMT2(alljet)[event_selection], weight=df['weight'][event_selection]*cfg['lumi'])
        output['S'].fill(dataset=dataset, norm=sphericityBasic(alljet)[event_selection], weight=df['weight'][event_selection]*cfg['lumi'])

        all_obj = mergeArray(alljet, lepton)
        output['S_lep'].fill(dataset=dataset, norm=sphericityBasic(all_obj)[event_selection], weight=df['weight'][event_selection]*cfg['lumi'])

        # forward stuff
        output['N_spec'].fill(dataset=dataset, multiplicity=spectator[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])
        output['pt_spec_max'].fill(dataset=dataset, pt=spectator[event_selection & (spectator.counts>0)].pt.max().flatten(), weight=df['weight'][event_selection & (spectator.counts>0)]*cfg['lumi'])

        return output

    def postprocess(self, accumulator):
        return accumulator


def main():

    overwrite = True

    # load the config and the cache
    cfg = loadConfig()

    # Inputs are defined in a dictionary
    # dataset : list of files
    from samples import fileset, fileset_small, fileset_1l

    # histograms
    histograms = ["MET_pt", "N_b", "N_jet", "MT", "N_spec", "pt_spec_max", "HT", "ST"]
    histograms += ['mbj_max', 'mjj_max', 'mlb_min', 'mlb_max', 'mlj_min', 'mlj_max']
    histograms += ['FWMT1', 'FWMT2', 'S', 'S_lep']


    # initialize cache
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['singleLep']), serialized=True)
    if not overwrite:
        cache.load()

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and fileset == cache.get('fileset') and cache.get('simple_output'):
        output = cache.get('simple_output')

    else:
        # Run the processor
        fileset = fileset_small
        output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=exampleProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': 8, 'function_args': {'flatten': False}},
                                      chunksize=100000,
                                     )
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()

    # Make a few plots
    outdir = "./tmp_plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for name in histograms:
        print (name)
        histogram = output[name]

        ax = hist.plot1d(histogram,overlay="dataset", stack=True) # make density plots because we don't care about x-sec differences
        ax.set_yscale('linear')
        ax.figure.savefig(os.path.join(outdir, "{}.pdf".format(name)))
        ax.clear()

    return output

if __name__ == "__main__":
    output = main()



