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
from coffea import hist
import pandas as pd
import uproot_methods
import awkward


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tW_scattering.Tools.helpers import *

# This just tells matplotlib not to open any
# interactive windows.
matplotlib.use('Agg')

class exampleProcessor(processor.ProcessorABC):
    """Dummy processor used to demonstrate the processor principle"""
    def __init__(self):

        # we can use a large number of bins and rebin later
        dataset_axis        = hist.Cat("dataset",   "Primary dataset")
        pt_axis             = hist.Bin("pt",        r"$p_{T}$ (GeV)", 600, 0, 1000)
        eta_axis            = hist.Bin("eta",       r"$\eta$", 60, -5.5, 5.5)
        multiplicity_axis   = hist.Bin("multiplicity",         r"N", 20, -0.5, 19.5)

        self._accumulator = processor.dict_accumulator({
            "MET_pt" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "Jet_pt" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "Jet_pt_fwd" :      hist.Hist("Counts", dataset_axis, pt_axis),
            "Jet_eta" :         hist.Hist("Counts", dataset_axis, eta_axis),
            "GenJet_pt_fwd" :   hist.Hist("Counts", dataset_axis, pt_axis),
            "Spectator_pt" :    hist.Hist("Counts", dataset_axis, pt_axis),
            "Spectator_eta" :   hist.Hist("Counts", dataset_axis, eta_axis),
            "W_pt_notFromTop" : hist.Hist("Counts", dataset_axis, pt_axis),
            "Top_pt" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "Top_eta" :         hist.Hist("Counts", dataset_axis, eta_axis),
            "Antitop_pt" :      hist.Hist("Counts", dataset_axis, pt_axis),
            "Antitop_eta" :     hist.Hist("Counts", dataset_axis, eta_axis),
            "W_pt" :            hist.Hist("Counts", dataset_axis, pt_axis),
            "W_eta" :           hist.Hist("Counts", dataset_axis, eta_axis),
            "N_b" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            'cutflow_bkg':      processor.defaultdict_accumulator(int),
            'cutflow_signal':   processor.defaultdict_accumulator(int),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        """
        Processing function. This is where the actual analysis happens.
        """
        output = self.accumulator.identity()
        # We can access the data frame as usual
        # The dataset is written into the data frame
        # outside of this function

        output['cutflow_bkg']['all events'] += sum(df['weight'][(df['dataset']=='TTW')].flatten())
        output['cutflow_signal']['all events'] += sum(df['weight'][(df['dataset']=='tW_scattering')].flatten())

        output['cutflow_bkg']['singleLep'] += sum(df['weight'][(df['dataset']=='TTW') & (df['nLepton']==1)].flatten())
        output['cutflow_signal']['singleLep'] += sum(df['weight'][(df['dataset']=='tW_scattering') & (df['nLepton']==1)].flatten())

        # preselection of events
        selection = df['nLepton']>=1

        dataset = df["dataset"]

        # do some stuff with W bosons
        W_bosons = awkward.JaggedArray.zip(pt=df['W_pt'], eta=df['W_eta'], phi=df['W_phi'], fromTop=df['W_fromTop'])
        
        W_notFromTop = W_bosons[W_bosons['fromTop']==0]
        one_W_notFromTop = (W_notFromTop.counts==1)
        output['W_pt_notFromTop'].fill(dataset=dataset, pt=W_notFromTop['pt'][one_W_notFromTop].flatten(), weight=df['weight'][one_W_notFromTop] )

        # And fill the histograms
        output['MET_pt'].fill(dataset=dataset, pt=df["MET_pt"][selection].flatten(), weight=df['weight'][selection])
        output['Jet_pt'].fill(dataset=dataset, pt=df["Jet_pt"].max().flatten(), weight=df['weight']) # maximum jet pt
        output['Jet_eta'].fill(dataset=dataset, eta=df["Jet_eta"].flatten())

        ## Do some stuff with gen jets and particles
        GenJets = awkward.JaggedArray.zip(pt=df['GenJet_pt'], eta=df['GenJet_eta'], phi=df['GenJet_phi'], hadronFlavour=df['GenJet_hadronFlavour'])
        Forward_noB = (abs(GenJets['eta'])>2.0) & (GenJets['hadronFlavour']!=5)
        GenJets_fwd = GenJets[Forward_noB]
        hasGenJets_fwd = (GenJets_fwd.counts>0)

        output['GenJet_pt_fwd'].fill(dataset=dataset, pt=GenJets_fwd['pt'][hasGenJets_fwd].max().flatten(), weight=df['weight'][hasGenJets_fwd] )

        spectators = awkward.JaggedArray.zip(pt=df['Spectator_pt'], eta=df['Spectator_eta'], phi=df['Spectator_phi'], pdgId=df['Spectator_pdgId'])
        spectators = spectators[spectators['pt']>10]
        hasSpectator = (spectators.counts>0)

        output['Spectator_pt'].fill(dataset=dataset, pt=spectators['pt'][hasSpectator].max().flatten(), weight=df['weight'][hasSpectator] )
        output['Spectator_eta'].fill(dataset=dataset, eta=spectators['eta'][hasSpectator].max().flatten(), weight=df['weight'][hasSpectator] )

        scatter = awkward.JaggedArray.zip(pt=df['Scatter_pt'], eta=df['Scatter_eta'], phi=df['Scatter_phi'], pdgId=df['Scatter_pdgId'])
        top = scatter[scatter['pdgId']==6]
        antitop = scatter[scatter['pdgId']==-6]
        w = scatter[abs(scatter['pdgId'])==24]
        hasTop = (top.counts>0)
        hasAntitop = (antitop.counts>0)
        hasW = (w.counts>0)

        output['Top_pt'].fill(dataset=dataset, pt=top['pt'][hasTop].max().flatten(), weight=df['weight'][hasTop] )
        output['Top_eta'].fill(dataset=dataset, eta=top['eta'][hasTop].max().flatten(), weight=df['weight'][hasTop] )
        
        output['Antitop_pt'].fill(dataset=dataset, pt=antitop['pt'][hasAntitop].max().flatten(), weight=df['weight'][hasAntitop] )
        output['Antitop_eta'].fill(dataset=dataset, eta=antitop['eta'][hasAntitop].max().flatten(), weight=df['weight'][hasAntitop] )

        output['W_pt'].fill(dataset=dataset, pt=w['pt'][hasW].max().flatten(), weight=df['weight'][hasW] )
        output['W_eta'].fill(dataset=dataset, eta=w['eta'][hasW].max().flatten(), weight=df['weight'][hasW] )

        ## We can also do arbitrary transformations
        ## E.g.: Sum of MET and the leading jet PTs
        #new_variable = df["MET_pt"] + df["Jet_pt"].max()
        #output['new_variable'].fill(dataset=dataset,
        #                    new_variable=new_variable)

        # To apply selections, simply mask
        mask = abs(df["Jet_eta"]) > 2.4

        # And plot the leading jet pt for these events
        output['Jet_pt_fwd'].fill(dataset=dataset, pt=df["Jet_pt"][mask].max().flatten())

        # Example for jets / b-jets
        jets = awkward.JaggedArray.zip(pt=df['Jet_pt'], eta=df['Jet_eta'], phi=df['Jet_phi'], btag=df['Jet_btagDeepB'], jetid=df['Jet_jetId'])
        goodjets = jets[ (jets['pt']>30) & (abs(jets['eta'])<2.4) & (jets['jetid']>0) ]
        bjets = jets[ (jets['pt']>30) & (abs(jets['eta'])<2.4) & (jets['jetid']>0) & (jets['btag']>0.4184) ]
        output['N_b'].fill(dataset=dataset, multiplicity=bjets.counts, weight=df['weight'] )
        output['N_jet'].fill(dataset=dataset, multiplicity=goodjets.counts, weight=df['weight'] )


        return output

    def postprocess(self, accumulator):
        return accumulator


def main():

    overwrite = True

    # load the config and the cache
    cfg = loadConfig()

    # Inputs are defined in a dictionary
    # dataset : list of files
    fileset = {
        'tW_scattering': glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/tW_scattering__nanoAOD/merged/*.root"),
        "TTW":           glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/merged/*.root") \
                        + glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root"),
        #"ttbar":        glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root") # adding this is still surprisingly fast (20GB file!)
    }

    # histograms
    histograms = ["MET_pt", "Jet_pt", "Jet_eta", "Jet_pt_fwd", "W_pt_notFromTop", "GenJet_pt_fwd", "Spectator_pt", "Spectator_eta"]
    histograms+= ["Top_pt", "Top_eta", "Antitop_pt", "Antitop_eta", "W_pt", "W_eta", "N_b", "N_jet"]


    # initialize cache
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['simpleProcessor']), serialized=True)
    if not overwrite:
        cache.load()

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and fileset == cache.get('fileset') and cache.get('simple_output'):
        output = cache.get('simple_output')

    else:
        # Run the processor
        output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=exampleProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': 12, 'function_args': {'flatten': False}},
                                      chunksize=500000,
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
        if name == 'MET_pt':
            # rebin
            new_met_bins = hist.Bin('pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)
            histogram = histogram.rebin('pt', new_met_bins)
        if name == 'W_pt_notFromTop':
            # rebin
            new_pt_bins = hist.Bin('pt', r'$p_{T}(W) \ (GeV)$', 25, 0, 500)
            histogram = histogram.rebin('pt', new_pt_bins)

        ax = hist.plot1d(histogram,overlay="dataset", density=False, stack=True) # make density plots because we don't care about x-sec differences
        ax.set_yscale('linear') # can be log
        #ax.set_ylim(0,0.1)
        ax.figure.savefig(os.path.join(outdir, "{}.pdf".format(name)))
        ax.clear()

        ax = hist.plot1d(histogram,overlay="dataset", density=True, stack=False) # make density plots because we don't care about x-sec differences
        ax.set_yscale('linear') # can be log
        #ax.set_ylim(0,0.1)
        ax.figure.savefig(os.path.join(outdir, "{}_shape.pdf".format(name)))
        ax.clear()

    return output

if __name__ == "__main__":
    output = main()



