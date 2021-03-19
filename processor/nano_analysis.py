import awkward1 as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np

# this is all very bad practice
from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import *
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.lepton_scalefactors import *

class nano_analysis(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        
        self.btagSF = btag_scalefactor(year)
        
        #self.SF = SF(year=year)
        
        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>=0
        
        ev = events[presel]
        dataset = ev.metadata['dataset']
        
        # load the config - probably not needed anymore
        cfg = loadConfig()
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
        ## Muons
        muon     = ev.Muon
        
        ## Electrons
        electron     = ev.Electron
        electron = electron[(abs(electron.eta+electron.deltaEtaSC) < 2.4) & (abs(electron.dxy) < 0.05) & (abs(electron.dz) < 0.1) & (electron.tightCharge == 2) & (electron.convVeto) & (electron.lostHits == 0) & (abs(electron.sip3d) < 4) & (electron.miniPFRelIso_all < 0.12) & (electron.pt > 20) & (abs(electron.eta) < 2.4)]
        electron = electron[(abs(electron.eta) < 0.8 & electron.mva = 0.52) | (0.8 < abs(electron.eta) < 1.479 & electron.mva = 0.11) | (1.479< abs(electron.eta) < 2.5 & electron.mva = -0.01)
        electron = electron[(abs(electron.eta) < 0.8 & electron.pt < 25 & electron.mvaFall17V2Iso_WP90 = 4.277+0.122*(electron.pt-25)) | (abs(electron.eta) < 0.8 & electron.pt > 25 & electron.mvaFall17V2Iso_WP90 = 4.277) | (0.8 < abs(electron.eta) < 1.479 & electron.pt < 25 & electron.mvaFall17V2Iso_WP90 = 3.152+0.060*(electron.pt-25)) | (0.8 < abs(electron.eta) < 1.479 & electron.pt > 25 & electron.mvaFall17V2Iso_WP90 = 3.152) | (1.479< abs(electron.eta) < 2.5 & electron.pt < 25 & electron.mvaFall17V2Iso_WP90 = 2.359+0.087*(electron.pt-25)) | (1.479< abs(electron.eta) < 2.5 & electron.pt < 25 & electron.mvaFall17V2Iso_WP90 = 2.359)]                   
        gen_electron = electron[electron.genPartIdx >= 0]
        
        is_flipped = (ev.GenPart[gen_electron.genPartIdx].pdgId/abs(ev.GenPart[gen_electron.genPartIdx].pdgId) != gen_electron.pdgId/abs(gen_electron.pdgId))
        flipped_electron = gen_electron[is_flipped]
        n_flips = ak.num(flipped_electron)
        
        #diflipped_electrons = choose(flipped_electrons, 2)
        
        #flip_0_idx = (ev.GenPart[diflipped_electrons['0'].genPartIdx].pdgId/abs(ev.GenPart[diflipped_electrons['0'].genPartIdx].pdgId) != diflipped_electrons['0'].pdgId/abs(diflipped_electrons['0'].pdgId))
        #flip_1_idx = (ev.GenPart[diflipped_electrons['1'].genPartIdx].pdgId/abs(ev.GenPart[diflipped_electrons['1'].genPartIdx].pdgId) != diflipped_electrons['1'].pdgId/abs(diflipped_electrons['1'].pdgId))
        
        #diflipped_electrons = diflipped_electrons[ak.any(flip_0_idx or flip_1_idx, axis=1)]
        
        #flipped_electron = cross(diflipped_electrons['0'][flip_0_idx], diflipped_electrons['1'][flip_1_idx])

        ## Merge electrons and muons - this should work better now in ak1
        
        dielectron = choose(electron, 2)
        SSelectron = ak.any((dielectron['0'].charge * dielectron['1'].charge)>0, axis=1)
         
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        
        leading_flipped_electron_idx = ak.singletons(ak.argmax(flipped_electron.pt, axis=1))
        leading_flipped_electron = electron[leading_flipped_electron_idx]
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        # define the weight
        weight = Weights( len(ev) )
        
        if not dataset=='MuonEG':
            # generator weight
            weight.add("weight", ev.genWeight)
            
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        dilep     = ((ak.num(electron) + ak.num(muon))==2)
        electr = ((ak.num(electron) >= 2))
        ss = (SSelectron)
        #flip2 = (ak.any(flip_0_idx, axis=1))
        flip = (n_flips >= 1)
        
        
        selection = PackedSelection()
        #selection.add('dilep',         dilep )
        selection.add('filter',        (filters) )
        selection.add('electr',        electr  )
        selection.add('ss',        ss)
        selection.add('flip',          flip)
        #selection.add('flip2',          flip2)
        
        bl_reqs = ['filter', 'electr', 'ss']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)
        
        f_reqs = bl_reqs + ['flip']
        f_reqs_d = { sel: True for sel in f_reqs }
        flip_sel = selection.require(**f_reqs_d)
        
        #f2_reqs = bl_reqs + ['flip2']
        #f2_reqs_d = { sel: True for sel in f2_reqs }
        #flip_sel2 = selection.require(**f2_reqs_d)
                                        
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(electron)[baseline], weight=weight.weight()[baseline])
        output['electron_flips'].fill(dataset=dataset, multiplicity=n_flips[baseline], weight=weight.weight()[baseline])

        output["electron"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[baseline].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[baseline].eta)),
            phi = ak.to_numpy(ak.flatten(leading_electron[baseline].phi)),
            weight = weight.weight()[baseline]
        )
        
        output["flipped_electron"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(flipped_electron[flip_sel].pt)),
            eta = ak.to_numpy(ak.flatten(flipped_electron[flip_sel].eta)),
            phi = ak.to_numpy(ak.flatten(flipped_electron[flip_sel].phi)),
            weight = weight.weight()[flip_sel]
        )
        
        return output

    def postprocess(self, accumulator):
        return accumulator




if __name__ == '__main__':

    from klepto.archives import dir_archive
    from processor.std_acumulators import desired_output, add_processes_to_output

    from Tools.helpers import get_samples
    from Tools.config_helpers import redirector_ucsd, redirector_fnal
    from Tools.nano_mapping import make_fileset

    overwrite = True
    
    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'nano_analysis'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    histograms = sorted(list(desired_output.keys()))
    
    year = 2018
    
    samples = get_samples()

    fileset = make_fileset(['DY', 'TTZ'], samples, redirector=redirector_ucsd, small=True)

    add_processes_to_output(fileset, desired_output)

    exe_args = {
        'workers': 16,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
    }
    exe = processor.futures_executor
    
    if not overwrite:
        cache.load()
    
    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
        output = cache.get('simple_output')
    
    else:
        print ("I'm running now")
        
        output = processor.run_uproot_job(
            fileset,
            "Events",
            nano_analysis(year=year, variations=[], accumulator=desired_output),
            exe,
            exe_args,
            chunksize=250000,
        )
        
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()
