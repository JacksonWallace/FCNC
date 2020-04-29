'''
Simple processor using coffea.
[ ] weights
[ ] Missing pieces: appropriate sample handling
[ ] Accumulator caching

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
        dataset_axis    = hist.Cat("dataset", "Primary dataset")
        MET_pt_axis     = hist.Bin("MET_pt", r"$p_{T}^{miss}$ (GeV)", 600, 0, 1000)
        Jet_pt_axis     = hist.Bin("Jet_pt", r"$p_{T}$", 600, 0, 1000)
        Jet_eta_axis    = hist.Bin("Jet_eta", r"$\eta$", 60, -5.5, 5.5)
        W_pt_axis       = hist.Bin("W_pt", r"$p_{T}(W)$", 500, 0, 500)

        self._accumulator = processor.dict_accumulator({
            "MET_pt" :          hist.Hist("Counts", dataset_axis, MET_pt_axis),
            "Jet_pt" :          hist.Hist("Counts", dataset_axis, Jet_pt_axis),
            "Jet_pt_fwd" :      hist.Hist("Counts", dataset_axis, Jet_pt_axis),
            "Jet_eta" :         hist.Hist("Counts", dataset_axis, Jet_eta_axis),
            "Spectator_pt" :    hist.Hist("Counts", dataset_axis, Jet_pt_axis),
            "W_pt_notFromTop" : hist.Hist("Counts", dataset_axis, W_pt_axis),
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

        # preselection of events
        selection = df['nLepton']>=1
        #df = df[df['nLepton']==2]

        dataset = df["dataset"]

        # do some stuff with W bosons
        W_bosons = awkward.JaggedArray.zip(pt=df['W_pt'], eta=df['W_eta'], phi=df['W_phi'], fromTop=df['W_fromTop'])
        
        W_notFromTop = W_bosons[W_bosons['fromTop']==0]
        one_W_notFromTop = (W_notFromTop.counts==1)
        output['W_pt_notFromTop'].fill(dataset=dataset,
                            W_pt=W_notFromTop['pt'][one_W_notFromTop].flatten(), weight=df['weight'][one_W_notFromTop] )

        # And fill the histograms
        W_noTop_selection = df['W_fromTop']==0
        #output['W_noTop_pt'].fill(dataset=dataset,
        #                    wpt=df["W_pt"][W_noTop_selection].flatten(), weight=df['weight'][W_noTop_selection] )#, weight=df['weight'][W_noTop_selection])
        output['MET_pt'].fill(dataset=dataset,
                            MET_pt=df["MET_pt"][selection].flatten(), weight=df['weight'][selection])
        output['Jet_pt'].fill(dataset=dataset,
                            Jet_pt=df["Jet_pt"].max().flatten(), weight=df['weight']) # maximum jet pt
        output['Jet_eta'].fill(dataset=dataset,
                            Jet_eta=df["Jet_eta"].flatten())

        ## We can also do arbitrary transformations
        ## E.g.: Sum of MET and the leading jet PTs
        #new_variable = df["MET_pt"] + df["Jet_pt"].max()
        #output['new_variable'].fill(dataset=dataset,
        #                    new_variable=new_variable)

        # To apply selections, simply mask
        # Let's see events with MET > 100
        mask = abs(df["Jet_eta"]) > 2.4

        # And plot the leading jet pt for these events
        output['Jet_pt_fwd'].fill(dataset=dataset,
                            Jet_pt=df["Jet_pt"][mask].max().flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator


def main():

    overwrite = False

    # load the config and the cache
    cfg = loadConfig()

    # Inputs are defined in a dictionary
    # dataset : list of files
    fileset = {
        'tW_scattering': glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/tW_scattering__nanoAOD/merged/*.root"),
        "TTW":           glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/merged/*.root") \
                        + glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root")
    }

    # histograms
    histograms = ["MET_pt", "Jet_pt", "Jet_eta", "Jet_pt_fwd", "W_pt_notFromTop"]

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
                                      executor_args={'workers': 1, 'function_args': {'flatten': False}},
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
        histogram = output[name]
        if name == 'MET_pt':
            # rebin
            new_met_bins = hist.Bin('MET_pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)
            histogram = histogram.rebin('MET_pt', new_met_bins)
        if name == 'W_pt_notFromTop':
            # rebin
            new_pt_bins = hist.Bin('W_pt', r'$p_{T}(W) \ (GeV)$', 25, 0, 500)
            histogram = histogram.rebin('W_pt', new_pt_bins)

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



