import awkward as ak
import numpy as np

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiData, LumiMask, LumiList


# this is all slightly better practice
from Tools.objects import Collections, choose, cross, match
from Tools.basic_objects import getJets
from Tools.config_helpers import loadConfig
from Tools.helpers import build_weight_like
from Tools.triggers import getFilters
from Tools.charge_flip import charge_flip

class dielectron_mass(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        self._accumulator = processor.dict_accumulator( accumulator )
        if self.year == 2016:
            self.charge_flip_ratio = charge_flip('../histos/chargeflipfull2016June.pkl.gz')
        if self.year == 2017:
            self.charge_flip_ratio = charge_flip('histos/chargeflipfull2017June.pkl.gz')
        if self.year == 2018:
            self.charge_flip_ratio = charge_flip('../histos/chargeflipfullpt152018.pkl.gz')

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>0
        
        if self.year == 2016:
            lumimask = LumiMask('../data/lumi/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt')
        if self.year == 2017:
            lumimask = LumiMask('../data/lumi/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt')
        if self.year == 2018:
            lumimask = LumiMask('../data/lumi/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt')
        
        ev = events[presel]
        dataset = ev.metadata['dataset']
        
        # load the config - probably not needed anymore
        cfg = loadConfig()
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
        if self.year == 2018:
            triggers = ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL
        elif self.year == 2017:
            triggers = ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL
        elif self.year == 2016:
            triggers = ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ
        
      
        ## Electrons
        electron = Collections(ev, "Electron", "tightFCNC", 0, self.year).get()
        electron = electron[(electron.pt > 25) & (np.abs(electron.eta) < 2.4)]
        
        loose_electron = Collections(ev, "Electron", "looseFCNC", 0, self.year).get()
        loose_electron = loose_electron[(loose_electron.pt > 25) & (np.abs(loose_electron.eta) < 2.4)]
        
        SSelectron = (ak.sum(electron.charge, axis=1) != 0) & (ak.num(electron)==2)
        OSelectron = (ak.sum(electron.charge, axis=1) == 0) & (ak.num(electron)==2)
        
        dielectron = choose(electron, 2)
        dielectron_mass = (dielectron['0']+dielectron['1']).mass
        dielectron_pt = (dielectron['0']+dielectron['1']).pt
        
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[(leading_electron_idx)]
        leading_electron = leading_electron[(leading_electron.pt > 30)]
        
        trailing_electron_idx = ak.singletons(ak.argmin(electron.pt, axis=1))
        trailing_electron = electron[trailing_electron_idx]
        
        ##Muons
        
        loose_muon = Collections(ev, "Muon", "looseFCNC", 0, self.year).get()
        loose_muon = loose_muon[(loose_muon.pt > 20) & (np.abs(loose_muon.eta) < 2.4)]
        
        #jets
        jet       = getJets(ev, minPt=40, maxEta=2.4, pt_var='pt')
        jet       = jet[~match(jet, loose_muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons
 
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi                                   
                      
        #weights
        weight = Weights( len(ev) )
        weight2 = Weights( len(ev) )
        weight2.add("charge flip", self.charge_flip_ratio.flip_weight(electron))
        
        #selections    
        filters   = getFilters(ev, year=self.year, dataset=dataset, UL = False)
        mask = lumimask(ev.run, ev.luminosityBlock)
        ss = (SSelectron)
        os = (OSelectron)
        mass = (ak.min(np.abs(dielectron_mass-91.2), axis = 1) < 15)
        lead_electron = (ak.min(leading_electron.pt, axis = 1) > 30)
        jet1 = (ak.num(jet) >= 1)
        jet2 = (ak.num(jet) >= 2)
        num_loose = ( (ak.num(loose_electron) == 2) & (ak.num(loose_muon) == 0) )

        
        
        selection = PackedSelection()
        selection.add('filter',      (filters) )
        selection.add('mask',        (mask) )
        selection.add('ss',          ss)
        selection.add('os',          os)
        selection.add('mass',        mass)
        selection.add('leading',     lead_electron)
        selection.add('triggers',    triggers)
        selection.add('one jet',     jet1)
        selection.add('two jets',    jet2)
        selection.add('num_loose',   num_loose)

        
        bl_reqs = ['filter'] + ['triggers'] + ['mask']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)
        
        s_reqs = bl_reqs + ['ss'] + ['mass']+ ['num_loose'] + ['leading']
        s_reqs_d = { sel: True for sel in s_reqs }
        ss_sel = selection.require(**s_reqs_d)
        
        o_reqs = bl_reqs + ['os'] + ['mass']+ ['num_loose'] + ['leading']
        o_reqs_d = { sel: True for sel in o_reqs }
        os_sel = selection.require(**o_reqs_d)
        
        j1s_reqs = s_reqs + ['one jet']
        j1s_reqs_d = { sel: True for sel in j1s_reqs }
        j1ss_sel = selection.require(**j1s_reqs_d)
        
        j1o_reqs = o_reqs + ['one jet']
        j1o_reqs_d = {sel: True for sel in j1o_reqs }
        j1os_sel = selection.require(**j1o_reqs_d)
        
        j2s_reqs = s_reqs + ['two jets']
        j2s_reqs_d = { sel: True for sel in j2s_reqs }
        j2ss_sel = selection.require(**j2s_reqs_d)
        
        j2o_reqs = o_reqs + ['two jets']
        j2o_reqs_d = {sel: True for sel in j2o_reqs }
        j2os_sel = selection.require(**j2o_reqs_d)
   
        
        #outputs
        
        output["electron_data1"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(leading_electron[os_sel].phi)),
            weight=weight2.weight()[os_sel]
        )
        
        output["electron_data2"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_electron[os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_electron[os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_electron[os_sel].phi)),
            weight=weight2.weight()[os_sel]
        )
        
        output["electron_data3"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[j1os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[j1os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(leading_electron[j1os_sel].phi)),
            weight=weight2.weight()[j1os_sel]
        )
        
        output["electron_data4"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_electron[j1os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_electron[j1os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_electron[j1os_sel].phi)),
            weight=weight2.weight()[j1os_sel]
        )
        
        output["electron_data5"].fill(
           dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[j2os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[j2os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(leading_electron[j2os_sel].phi)),
            weight=weight2.weight()[j2os_sel]
        )
        
        output["electron_data6"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_electron[j2os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_electron[j2os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_electron[j2os_sel].phi)),
            weight=weight2.weight()[j2os_sel]
        )
        
        output["electron_data7"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[ss_sel].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[ss_sel].eta)),
            phi = ak.to_numpy(ak.flatten(leading_electron[ss_sel].phi)),
            weight = weight.weight()[ss_sel]
        )
        
        output["electron_data8"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_electron[ss_sel].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_electron[ss_sel].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_electron[ss_sel].phi)),
            weight = weight.weight()[ss_sel]
        )
        
        output["electron_data9"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[j1ss_sel].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[j1ss_sel].eta)),
            phi = ak.to_numpy(ak.flatten(leading_electron[j1ss_sel].phi)),
            weight = weight.weight()[j1ss_sel]
        )
        
        output["electron_data10"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_electron[j1ss_sel].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_electron[j1ss_sel].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_electron[j1ss_sel].phi)),
            weight = weight.weight()[j1ss_sel]
        )
        
        output["electron_data11"].fill(
           dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[j2ss_sel].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[j2ss_sel].eta)),
            phi = ak.to_numpy(ak.flatten(leading_electron[j2ss_sel].phi)),
            weight = weight.weight()[j2ss_sel]
        )
        
        output["electron_data12"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_electron[j2ss_sel].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_electron[j2ss_sel].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_electron[j2ss_sel].phi)),
            weight = weight.weight()[j2ss_sel]
        )
        
        output["dilep_mass1"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dielectron_mass[os_sel])),
            pt = ak.to_numpy(ak.flatten(dielectron_pt[os_sel])),
            weight=weight2.weight()[os_sel]
        )
        
        output["dilep_mass2"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dielectron_mass[j1os_sel])),
            pt = ak.to_numpy(ak.flatten(dielectron_pt[j1os_sel])),
            weight=weight2.weight()[j1os_sel]
        )
        
        output["dilep_mass3"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dielectron_mass[j2os_sel])),
            pt = ak.to_numpy(ak.flatten(dielectron_pt[j2os_sel])),
            weight=weight2.weight()[j2os_sel]
        )
        
        output["dilep_mass4"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dielectron_mass[ss_sel])),
            pt = ak.to_numpy(ak.flatten(dielectron_pt[ss_sel])),
            weight = weight.weight()[ss_sel]
        )
        
        output["dilep_mass5"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dielectron_mass[j1ss_sel])),
            pt = ak.to_numpy(ak.flatten(dielectron_pt[j1ss_sel])),
            weight = weight.weight()[j1ss_sel]
        )
        
        output["dilep_mass6"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dielectron_mass[j2ss_sel])),
            pt = ak.to_numpy(ak.flatten(dielectron_pt[j2ss_sel])),
            weight = weight.weight()[j2ss_sel]
        )
        
        output["MET"].fill(
            dataset = dataset,
            pt = met_pt[os_sel],
            weight=weight2.weight()[os_sel]
        )
        
        output["MET2"].fill(
            dataset = dataset,
            pt = met_pt[j1os_sel],
            weight=weight2.weight()[j1os_sel]
        )
        
        output["MET3"].fill(
            dataset = dataset,
            pt = met_pt[j2os_sel],
            weight=weight2.weight()[j2os_sel]
        )
        
        output["MET4"].fill(
            dataset = dataset,
            pt = met_pt[ss_sel],
            weight = weight.weight()[ss_sel]
        )
        
        output["MET5"].fill(
            dataset = dataset,
            pt = met_pt[j1ss_sel],
            weight = weight.weight()[j1ss_sel]
        )
        
        output["MET6"].fill(
            dataset = dataset,
            pt = met_pt[j2ss_sel],
            weight = weight.weight()[j2ss_sel]
        )
        
        output["N_jet"].fill(
            dataset = dataset,
            multiplicity = ak.num(jet)[os_sel],
            weight=weight2.weight()[os_sel]
        )
        
        output["N_jet2"].fill(
            dataset = dataset,
            multiplicity = ak.num(jet)[j1os_sel],
            weight=weight2.weight()[j1os_sel]
        )
            
        output["N_jet3"].fill(
            dataset = dataset,
            multiplicity = ak.num(jet)[j2os_sel],
            weight=weight2.weight()[j2os_sel]
        )
        
        output["N_jet4"].fill(
            dataset = dataset,
            multiplicity = ak.num(jet)[ss_sel],
            weight = weight.weight()[ss_sel]
        )
        
        output["N_jet5"].fill(
            dataset = dataset,
            multiplicity = ak.num(jet)[j1ss_sel],
            weight = weight.weight()[j1ss_sel]
        )
            
        output["N_jet6"].fill(
            dataset = dataset,
            multiplicity = ak.num(jet)[j2ss_sel],
            weight = weight.weight()[j2ss_sel]
        )
        
        output["PV_npvsGood"].fill(
            dataset = dataset,
            multiplicity = ev.PV[os_sel].npvsGood,
            weight=weight2.weight()[os_sel]
        )
       
        output["PV_npvsGood2"].fill(
            dataset = dataset,
            multiplicity = ev.PV[j1os_sel].npvsGood,
            weight=weight2.weight()[j1os_sel]
        )
        
        output["PV_npvsGood3"].fill(
            dataset = dataset,
            multiplicity = ev.PV[j2os_sel].npvsGood,
            weight=weight2.weight()[j2os_sel]
        )
        
        output["PV_npvsGood4"].fill(
            dataset = dataset,
            multiplicity = ev.PV[ss_sel].npvsGood,
            weight = weight.weight()[ss_sel]
        )
       
        output["PV_npvsGood5"].fill(
            dataset = dataset,
            multiplicity = ev.PV[j1ss_sel].npvsGood,
            weight = weight.weight()[j1ss_sel]
        )
        
        output["PV_npvsGood6"].fill(
            dataset = dataset,
            multiplicity = ev.PV[j2ss_sel].npvsGood,
            weight = weight.weight()[j2ss_sel]
        )

        return output

    def postprocess(self, accumulator):
        return accumulator
    

if __name__ == '__main__':
    from Tools.helpers import get_samples
    from Tools.config_helpers import redirector_ucsd
    from Tools.nano_mapping import make_fileset
    from processor.default_accumulators import desired_output

    year = 2018

    fileset = make_fileset(['Data'], year, redirector=redirector_ucsd, small=False)

    exe_args = {
        'workers': 8,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
        "skipbadfiles": True,
    }
    exe = processor.futures_executor

    output = processor.run_uproot_job(
            fileset,
            "Events",
            dielectron_mass(year=year, variations=[], accumulator=desired_output),
            exe,
            exe_args,
            chunksize=250000,
    )