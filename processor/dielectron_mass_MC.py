try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection


# this is all very bad practice
from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import *
from Tools.helpers import build_weight_like
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.lepton_scalefactors2 import LeptonSF2
from Tools.charge_flip import *
from Tools.pileup import pileup

class dielectron_mass(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        self.leptonSF = LeptonSF2(year=year)
        self.PU = pileup(year=year, UL = False)
        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>0
        
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

        electron = electron[(electron.genPartIdx >= 0)]
        electron = electron[(np.abs(electron.matched_gen.pdgId)==11)]  #from here on all leptons are gen-matched
        electron = electron[( (electron.genPartFlav==1) | (electron.genPartFlav==15) )] #and now they are all prompt
        
        loose_electron = Collections(ev, "Electron", "looseFCNC", 0, self.year).get()
        loose_electron = loose_electron[(loose_electron.pt > 25) & (np.abs(loose_electron.eta) < 2.4)]
    
        SSelectron = (ak.sum(electron.charge, axis=1) != 0) & (ak.num(electron)==2)
        OSelectron = (ak.sum(electron.charge, axis=1) == 0) & (ak.num(electron)==2)
        
        dielectron = choose(electron, 2)
        dielectron_mass = (dielectron['0']+dielectron['1']).mass
        dielectron_pt = (dielectron['0']+dielectron['1']).pt
        
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        leading_electron = leading_electron[(leading_electron.pt > 30)]

        trailing_electron_idx = ak.singletons(ak.argmin(electron.pt, axis=1))
        trailing_electron = electron[trailing_electron_idx]
        
        ##Muons
        muon = Collections(ev, "Muon", "tightFCNC").get()
        muon = muon[(muon.pt > 15) & (np.abs(muon.eta) < 2.4)]
        
        muon = muon[(muon.genPartIdx >= 0)]
        muon = muon[(np.abs(muon.matched_gen.pdgId)==13)] #from here, all muons are gen-matched
        muon = muon[( (muon.genPartFlav==1) | (muon.genPartFlav==15) )] #and now they are all prompt
        
        loose_muon = Collections(ev, "Muon", "looseFCNC").get()
        loose_muon = loose_muon[(loose_muon.pt > 20) & (np.abs(loose_muon.eta) < 2.4)]
        
        #jets
        jet       = getJets(ev, minPt=40, maxEta=2.4, pt_var='pt')
        jet       = jet[ak.argsort(jet.pt, ascending=False)] # need to sort wrt smeared and recorrected jet pt
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons
        
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        # setting up the various weights
        weight = Weights( len(ev) )
        
        if not dataset=='MuonEG':
            # generator weight
            weight.add("weight", ev.genWeight)  
            weight.add("lepton", self.leptonSF.get(electron, muon))
            weight.add("pileup", self.PU.reweight(ak.to_numpy(ev.Pileup.nTrueInt), to='central'), weightUp = self.PU.reweight(ak.to_numpy(ev.Pileup.nTrueInt), to='up'), weightDown = self.PU.reweight(ak.to_numpy(ev.Pileup.nTrueInt), to='down'), shift=False)
                      
        #selections    
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        ss = (SSelectron)
        os = (OSelectron)
        mass = (ak.min(np.abs(dielectron_mass-91.2), axis = 1) < 15)
        lead_electron = (ak.min(leading_electron.pt, axis = 1) > 30)
        jet1 = (ak.num(jet) >= 1)
        jet2 = (ak.num(jet) >= 2)
        num_loose = ( (ak.num(loose_electron) == 2) & (ak.num(loose_muon) == 0) )

        
        selection = PackedSelection()
        selection.add('filter',      (filters) )
        selection.add('ss',          ss )
        selection.add('os',          os )
        selection.add('mass',        mass)
        selection.add('triggers',    triggers)
        selection.add('leading',     lead_electron)
        selection.add('one jet',     jet1)
        selection.add('two jets',    jet2)
        selection.add('num_loose',   num_loose)
        
        bl_reqs = ['filter'] + ['triggers'] + ['num_loose']
        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)
        
        s_reqs = bl_reqs + ['ss'] + ['leading'] + ['mass']
        s_reqs_d = { sel: True for sel in s_reqs }
        ss_sel = selection.require(**s_reqs_d)
        
        o_reqs = bl_reqs + ['os'] + ['leading'] + ['mass']
        o_reqs_d = {sel: True for sel in o_reqs }
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
            weight = weight.weight()[os_sel]
        )
        
        output["electron_data2"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_electron[os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_electron[os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_electron[os_sel].phi)),
            weight = weight.weight()[os_sel]
        )
        
        output["electron_data3"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[j1os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[j1os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(leading_electron[j1os_sel].phi)),
            weight = weight.weight()[j1os_sel]
        )
        
        output["electron_data4"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_electron[j1os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_electron[j1os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_electron[j1os_sel].phi)),
            weight = weight.weight()[j1os_sel]
        )
        
        output["electron_data5"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[j2os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[j2os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(leading_electron[j2os_sel].phi)),
            weight = weight.weight()[j2os_sel]
        )
        
        output["electron_data6"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_electron[j2os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_electron[j2os_sel].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_electron[j2os_sel].phi)),
            weight = weight.weight()[j2os_sel]
        )
        
        output["dilep_mass1"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dielectron_mass[os_sel])),
            pt = ak.to_numpy(ak.flatten(dielectron_pt[os_sel])),
            weight = weight.weight()[os_sel]
        )
        
        output["dilep_mass2"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dielectron_mass[j1os_sel])),
            pt = ak.to_numpy(ak.flatten(dielectron_pt[j1os_sel])),
            weight = weight.weight()[j1os_sel]
        )
        
        output["dilep_mass3"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dielectron_mass[j2os_sel])),
            pt = ak.to_numpy(ak.flatten(dielectron_pt[j2os_sel])),
            weight = weight.weight()[j2os_sel]
        )
        
        output["MET"].fill(
            dataset = dataset,
            pt = met_pt[os_sel],
            phi = met_phi[os_sel],
            weight = weight.weight()[os_sel]
        )
        
        output["N_jet"].fill(
            dataset = dataset,
            multiplicity = ak.num(jet)[os_sel],
            weight = weight.weight()[os_sel]
        )
        
        output["PV_npvsGood"].fill(
            dataset = dataset,
            multiplicity = ev.PV[os_sel].npvsGood,
            weight = weight.weight()[os_sel]
        )
        
        output["MET2"].fill(
            dataset = dataset,
            pt = met_pt[j1os_sel],
            phi = met_phi[j1os_sel],
            weight = weight.weight()[j1os_sel]
        )
        
        output["N_jet2"].fill(
            dataset = dataset,
            multiplicity = ak.num(jet)[j1os_sel],
            weight = weight.weight()[j1os_sel]
        )
        
        output["PV_npvsGood2"].fill(
            dataset = dataset,
            multiplicity = ev.PV[j1os_sel].npvsGood,
            weight = weight.weight()[j1os_sel]
        )
        
        output["MET3"].fill(
            dataset = dataset,
            pt = met_pt[j2os_sel],
            phi = met_phi[j2os_sel],
            weight = weight.weight()[j2os_sel]
        )
        
        output["N_jet3"].fill(
            dataset = dataset,
            multiplicity = ak.num(jet)[j2os_sel],
            weight = weight.weight()[j2os_sel]
        )
        
        output["PV_npvsGood3"].fill(
            dataset = dataset,
            multiplicity = ev.PV[j2os_sel].npvsGood,
            weight = weight.weight()[j2os_sel]
        )

        return output

    def postprocess(self, accumulator):
        return accumulator