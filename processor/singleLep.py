'''
Simple processor using coffea.
[x] weights
[x] Missing pieces: appropriate sample handling
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

from memory_profiler import profile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Tools.helpers import *

## Event shape variables

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
            "FWMT5" :           hist.Hist("Counts", dataset_axis, norm_axis),
            "S" :               hist.Hist("Counts", dataset_axis, norm_axis),
            "S_lep" :           hist.Hist("Counts", dataset_axis, norm_axis),
            'wjets':            processor.defaultdict_accumulator(int),
            'ttbar':            processor.defaultdict_accumulator(int),
            'TTW':              processor.defaultdict_accumulator(int),
            'TTX':              processor.defaultdict_accumulator(int),
            'tW_scattering':    processor.defaultdict_accumulator(int),
            'diboson':          processor.defaultdict_accumulator(int),
            'totalEvents':      processor.defaultdict_accumulator(int),
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

        met_pt = df["MET_pt"]
        met_phi = df["MET_phi"]

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
        
        alljet       = Jet[(Jet['goodjet']==1)] # all jets with pt>25 and pt>60 in 2.7<|eta|<3.0 (noise suppression)
        b            = Jet[Jet['bjet']==1]
        nonb         = Jet[(Jet['goodjet']==1) & (Jet['bjet']==0)]
        leading_nonb = nonb[:,:6] # first six non-b-tagged jets
        lead_light   = nonb[nonb.pt.argmax()]
        spectator    = Jet[(abs(Jet.eta)>2.0) & (abs(Jet.eta)<4.7) & (Jet.pt>25) & (Jet['puId']>=7) & (Jet['jetId']>=6)] # 40 GeV seemed good. let's try going lower
        
        bj_pair = b.cross(nonb)
        jj_pair = nonb.cross(nonb)
        lb_pair = lepton.cross(b)
        lj_pair = lepton.cross(nonb)
        ht = Jet[Jet['goodjet']==1].pt.sum()
        st = Jet[Jet['goodjet']==1].pt.sum() + lepton.pt.sum() + met_pt
        
        leading_lepton = lepton[lepton.pt.argmax()]

        ## calculate mt
        mt_lep_met = mt(leading_lepton.pt, leading_lepton.phi, met_pt, met_phi)

        ### define selections here, using the objects defined above
        singlelep = ((df['nLepton']==1) & (df['nVetoLepton']==1))
        sixjet    = (alljet.counts >= 6 )
        sevenjet  = (alljet.counts >= 7)
        two_b     = ( b.counts >= 2)
        eta_lead  = ((lead_light.eta>-.5).counts>0) & ((lead_light.eta<0.5).counts>0)
        
        # selections used for the histograms below
        event_selection = (Jet.counts>5) & (b.counts>=2) & (nonb.counts>=4) & (df['nLepton']==1) & (df['nVetoLepton']==1)
        tight_selection = (Jet.counts>5) & (b.counts>=2) & (nonb.counts>=4) & (df['nLepton']==1) & (df['nVetoLepton']==1) & (df['MET_pt']>50) & (ht>500) & (df['MT']>50) & (spectator.counts>=1) & (spectator.pt.max()>50) & (st>600) & (bj_pair.mass.max()>300) & (jj_pair.mass.max()>300)

        ### work on the cutflow
        addRowToCutFlow( output, df, cfg, 'skim',        None ) # entry point
        addRowToCutFlow( output, df, cfg, 'singlelep',   singlelep )
        addRowToCutFlow( output, df, cfg, 'sixjet',      singlelep & sixjet )
        addRowToCutFlow( output, df, cfg, 'sevenjet',    singlelep & sevenjet )
        addRowToCutFlow( output, df, cfg, 'twob',        singlelep & sevenjet & two_b )
        addRowToCutFlow( output, df, cfg, 'etalead',     singlelep & sevenjet & eta_lead ) # this is a weird cut
        addRowToCutFlow( output, df, cfg, 'everything',  singlelep & sevenjet & two_b & eta_lead ) # this is a weird cut
        
        ### fill all the histograms
        
        output['MET_pt'].fill(dataset=dataset, pt=df["MET_pt"][singlelep].flatten(), weight=df['weight'][singlelep]*cfg['lumi'])
        output['MT'].fill(dataset=dataset, pt=df["MT"][singlelep].flatten(), weight=df['weight'][singlelep]*cfg['lumi'])
        output['N_b'].fill(dataset=dataset, multiplicity=df["nGoodBTag"][singlelep], weight=df['weight'][singlelep]*cfg['lumi'] )
        output['N_jet'].fill(dataset=dataset, multiplicity=df["nGoodJet"][singlelep], weight=df['weight'][singlelep]*cfg['lumi'] )
        
        output['mbj_max'].fill(dataset=dataset, mass=bj_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mjj_max'].fill(dataset=dataset, mass=jj_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlb_min'].fill(dataset=dataset, mass=lb_pair[event_selection].mass.min().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlb_max'].fill(dataset=dataset, mass=lb_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlj_min'].fill(dataset=dataset, mass=lj_pair[event_selection].mass.min().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['mlj_max'].fill(dataset=dataset, mass=lj_pair[event_selection].mass.max().flatten(), weight=df['weight'][event_selection]*cfg['lumi'])

        output['HT'].fill(dataset=dataset, ht=ht[event_selection].flatten(), weight=df['weight'][event_selection]*cfg['lumi'])
        output['ST'].fill(dataset=dataset, ht=st[event_selection].flatten(), weight=df['weight'][event_selection]*cfg['lumi'])

        # forward stuff
        output['N_spec'].fill(dataset=dataset, multiplicity=spectator[event_selection].counts, weight=df['weight'][event_selection]*cfg['lumi'])
        output['pt_spec_max'].fill(dataset=dataset, pt=spectator[event_selection & (spectator.counts>0)].pt.max().flatten(), weight=df['weight'][event_selection & (spectator.counts>0)]*cfg['lumi'])


        ### event shape variables - neglect for now
        
        #output['FWMT1'].fill(dataset=dataset, norm=FWMT(leading_nonb)[1][tight_selection], weight=df['weight'][tight_selection]*cfg['lumi'])
        #output['FWMT2'].fill(dataset=dataset, norm=FWMT(leading_nonb)[2][tight_selection], weight=df['weight'][tight_selection]*cfg['lumi'])
        #output['FWMT3'].fill(dataset=dataset, norm=FWMT(leading_nonb)[3][tight_selection], weight=df['weight'][tight_selection]*cfg['lumi'])
        #output['FWMT4'].fill(dataset=dataset, norm=FWMT(leading_nonb)[4][tight_selection], weight=df['weight'][tight_selection]*cfg['lumi'])
        #output['FWMT5'].fill(dataset=dataset, norm=FWMT(leading_nonb)[5][tight_selection], weight=df['weight'][tight_selection]*cfg['lumi'])
        ##output['S'].fill(dataset=dataset, norm=sphericityBasic(alljet)[event_selection], weight=df['weight'][event_selection]*cfg['lumi'])

        #all_obj = mergeArray(alljet, lepton)
        #output['S_lep'].fill(dataset=dataset, norm=sphericityBasic(all_obj)[event_selection], weight=df['weight'][event_selection]*cfg['lumi'])
        
        
        return output

    def postprocess(self, accumulator):
        return accumulator


#@profile
def main():

    overwrite = True
    small = True

    # load the config and the cache
    cfg = loadConfig()

    cacheName = 'singleLep_small' if small else 'singleLep'
    
    # Inputs are defined in a dictionary
    # dataset : list of files
    from samples import fileset, fileset_small, fileset_1l

    # histograms
    histograms = ["MET_pt", "N_b", "N_jet", "MT", "N_spec", "pt_spec_max", "HT", "ST"]
    histograms += ['mbj_max', 'mjj_max', 'mlb_min', 'mlb_max', 'mlj_min', 'mlj_max']
    #histograms += ['FWMT1', 'FWMT2', 'FWMT3', 'FWMT4', 'FWMT5']
    #histograms += ['S', 'S_lep']

    # initialize cache
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches'][cacheName]), serialized=True)
    if not overwrite:
        cache.load()

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
        output = cache.get('simple_output')

    else:
        # Run the processor
        if small:
            fileset = fileset_small
            workers = 1
        else:
            fileset = fileset_1l
            workers = 6
        output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=exampleProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': workers, 'function_args': {'flatten': False}},
                                      chunksize=50000,
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
    
    # get a cutflow from the output
    df = getCutFlowTable(output, processes=['tW_scattering', 'TTW', 'ttbar'], lines=['skim', 'singlelep', 'sixjet', 'sevenjet'])



