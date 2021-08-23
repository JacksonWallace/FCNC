import awkward as ak
import numpy as np

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection


# this is all slightly better practice
from Tools.basic_objects import getJets, getBTagsDeepFlavB
from Tools.btag_scalefactors import btag_scalefactor
from Tools.config_helpers import loadConfig
from Tools.cutflow import Cutflow
from Tools.helpers import build_weight_like
from Tools.lepton_scalefactors2 import LeptonSF2
from Tools.objects import Collections, choose, cross, match
from Tools.pileup import pileup
from Tools.trigger_scalefactors import triggerSF
from Tools.triggers import getFilters

class FCNC_cutflow(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        self.leptonSF = LeptonSF2(year=year)
        self.PU = pileup(year=year, UL = False)
        self.btagSF = btag_scalefactor(year, UL = False)
        self.triggerSF = triggerSF(year=year)
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
        
        #Begin defining objects
        
        ## Electrons
        electron = Collections(ev, "Electron", "tightFCNC", self.year, 0).get()
        #electron = Collections(ev, "Electron", "tightSSTTH", self.year, 0).get()
        electron = electron[( (electron.pt > 20) & (np.abs(electron.eta) < 2.4) )] #matches skim 
        
        gen_electron = electron[(electron.genPartIdx >= 0)]
        gen_electron = gen_electron[(np.abs(gen_electron.matched_gen.pdgId)==11)]  #from here on all leptons are gen-matched
        non_prompt_electron = gen_electron[( (gen_electron.genPartFlav!=1) & (gen_electron.genPartFlav!=15) )]
        
        prompt_electron = gen_electron[( (gen_electron.genPartFlav==1) | (gen_electron.genPartFlav==15) )] #and now they are all prompt
        
        flip_electron = prompt_electron[( (prompt_electron.matched_gen.pdgId*(-1) == prompt_electron.pdgId) & (np.abs(prompt_electron.pdgId) == 11) )]        
        
        electron       = electron[ak.argsort(electron.pt, ascending=False)]
        leading_electron = electron[:,0:1]
        trailing_electron = electron[:,1:2]
        
        dielectron = choose(electron, 2)
        dielectron_mass = dielectron.mass
        dielectron_pt = dielectron.pt
        
        SSelectron = (ak.sum(electron.charge, axis=1) != 0) & (ak.num(electron)==2)
        OSelectron = (ak.sum(electron.charge, axis=1) == 0) & (ak.num(electron)==2)
        
        loose_electron = Collections(ev, "Electron", "fakeFCNC", self.year, 0).get()
        #loose_electron = Collections(ev, "Electron", "vetoTTH", self.year, 0).get()
        loose_electron = loose_electron[( ((loose_electron.pt > 20) | (loose_electron.conePt > 20) ) & (np.abs(loose_electron.eta) < 2.4) )] #matches skim 
        
        loose_electron       = loose_electron[ak.argsort(loose_electron.pt, ascending=False)]
        leading_loose_electron = loose_electron[:,0:1]
        trailing_loose_electron = loose_electron[:,1:2]
        
        diloose_electron = choose(loose_electron, 2)
        diloose_electron_OS = diloose_electron[(diloose_electron.charge == 0)]
        
        
        ##Muons
        muon = Collections(ev, "Muon", "tightFCNC", self.year, 0).get()
        #muon = Collections(ev, "Muon", "tightSSTTH", self.year, 0).get()
        muon = muon[( (muon.pt > 20) & (np.abs(muon.eta) < 2.4) )]
        gen_muon = muon[(muon.genPartIdx >= 0)]
        gen_muon = gen_muon[(np.abs(gen_muon.matched_gen.pdgId)==13)]
        non_prompt_muon = gen_muon[( (gen_muon.genPartFlav!=1) & (gen_muon.genPartFlav!=15) )]
        
        muon       = muon[ak.argsort(muon.pt, ascending=False)]
        leading_muon = muon[:,0:1]
        trailing_muon = muon[:,1:2]
        
        loose_muon = Collections(ev, "Muon", "fakeFCNC", self.year, 0).get()
        #loose_muon = Collections(ev, "Muon", "vetoTTH", self.year, 0).get()
        loose_muon = loose_muon[( ((loose_muon.pt > 20) | (loose_muon.conePt > 20)) & (np.abs(loose_muon.eta) < 2.4) )] #matches skim 
   
        loose_muon       = loose_muon[ak.argsort(loose_muon.pt, ascending=False)]
        leading_loose_muon = loose_muon[:,0:1]
        trailing_loose_muon = loose_muon[:,1:2]
        
        diloose_muon = choose(loose_muon, 2)
        diloose_muon_OS = diloose_muon[(diloose_muon.charge == 0)]

        
        ##Leptons
        lepton   = ak.concatenate([muon, electron], axis=1) #tight leptons, matches skim
        
        non_prompt_lepton = ak.concatenate([non_prompt_muon, non_prompt_electron], axis=1)
        
        lepton = lepton[ak.argsort(lepton.pt, ascending = False)]
        leading_lepton = lepton[:,0:1]
        subleading_lepton = lepton[:,1:2]
        subsubleading_lepton = lepton[:,2:3]
        
        dilepton = choose(lepton, 2)
        dilepton_mass = dilepton.mass
        dilepton_pt = dilepton.pt
        
        SSlepton = ( (ak.sum(lepton.charge, axis=1) != 0) & (ak.num(lepton)==2) )
        OSlepton = ( (ak.sum(lepton.charge, axis=1) == 0) & (ak.num(lepton)==2) )
        
        loose_lepton = ak.concatenate([loose_muon, loose_electron], axis=1) #matches skim
        
                
        #jets and btags
        jet       = getJets(ev, minPt=25, maxEta=2.4, pt_var='pt', UL = False)
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons
        
        btag      = getBTagsDeepFlavB(jet, year=self.year, UL=False)
        light     = getBTagsDeepFlavB(jet, year=self.year, UL=False, invert=True)
        btag      = btag[ak.argsort(btag.pt, ascending=False)]
        
        leading_btag = btag[:, 0:1]
        subleading_btag = btag[:, 1:2]
        
        jet = jet[jet.pt>30]
        
        leading_jet = jet[:,0:1]
        subleading_jet = jet[:,1:2]
        subsubleading_jet = jet[:,2:3]
                
        
        ## MET -> can switch to puppi MET
        if self.year == 2016 or self.year == 2018:
            met_pt = ev.MET.pt
        elif self.year == 2017:
            met_pt = ev.METFixEE2017.pt
        met_phi = ev.MET.phi      


        # setting up the various weights
        weight = Weights( len(ev) )
        weight2 = Weights( len(ev) )
        
        if not dataset=='EGamma':
            #generator weight
            weight.add("weight", ev.genWeight)  
            weight.add("lepton", self.leptonSF.get(electron, muon))
            weight.add("pileup", self.PU.reweight(ak.to_numpy(ev.Pileup.nTrueInt), to='central'), weightUp = self.PU.reweight(ak.to_numpy(ev.Pileup.nTrueInt), to='up'), weightDown = self.PU.reweight(ak.to_numpy(ev.Pileup.nTrueInt), to='down'), shift=False)
            weight.add("btag", self.btagSF.Method1a(btag, light))     #are these ready?
            weight.add("trigger", self.triggerSF.get(electron, muon)) #are these ready?
                       
                      
        #selections
        skim = (((ak.num(loose_lepton) == 2) & (ak.sum(loose_lepton.charge, axis=1) != 0)) | (ak.num(loose_lepton) > 2))
        tight = (((ak.num(lepton) == 2) & (ak.sum(lepton.charge, axis=1) != 0)) | (ak.num(lepton) > 2))
        leading = (ak.min(leading_lepton.pt, axis = 1) > 25)
        filters = getFilters(ev, year=self.year, dataset=dataset, UL = False)
        met =  ( met_pt > 50 )
        
        #triggers
        if self.year == 2018:
            triggers = ev.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ | ev.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ | ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL | ev.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8
        if self.year == 2017:
            triggers = ev.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 | ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL | ev.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ | ev.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ
        if self.year == 2016:
            triggers = ev.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ | ev.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL | ev.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ | ev.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL | ev.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ | ev.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL | ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ

        
        #SS SRs
        gmass = (((ak.num(lepton) >= 2) & (ak.all(dilepton_mass > 12, axis=1))) | (ak.num(lepton) < 2) )
        Z_mr_veto = ( ( (ak.num(loose_lepton) == 3) & ( (ak.all(np.abs(diloose_electron_OS.mass-91) > 15, axis=1)) & (ak.all(np.abs(diloose_muon_OS.mass-91) > 15, axis=1)) )  )  | (ak.num(loose_lepton) != 3) )
        g_mr_veto = ( ( (ak.num(loose_lepton) == 3) & ( (ak.all(np.abs(diloose_electron_OS.mass) > 12, axis=1)) & (ak.all(np.abs(diloose_muon_OS.mass) > 12, axis=1)) )  )  | (ak.num(loose_lepton) != 3) )
        ss = (SSlepton & gmass & Z_mr_veto & g_mr_veto)
        Zmass = (((ak.num(electron) >= 2) & (ak.all(np.abs(dielectron_mass-91) > 15, axis=1))) | (ak.num(electron) < 2) )
        #Zmass = ak.fill_none(((ak.num(electron) == 2) & ((ak.min(np.abs(dielectron_mass-90), axis = 1) > 15))) | (ak.num(electron) == 1) | (ak.num(electron) == 0), True)
        two_jets = (ak.num(jet) >= 2)
        
        #ML SRs
        Z_mr_veto2 = ( ( (ak.num(loose_lepton) >= 3) & ( (ak.all(np.abs(diloose_electron_OS.mass-91) > 15, axis=1)) & (ak.all(np.abs(diloose_muon_OS.mass-91) > 15, axis=1)) )  )  | (ak.num(loose_lepton) < 3) )
        ml =  ( (ak.num(lepton) > 2) & gmass & Z_mr_veto2)
        one_jet = (ak.num(jet) >= 1)
        
        #flips and fakes
        no_flip = (ak.num(flip_electron) == 0)
        flip = (ak.num(flip_electron) >= 1)  
        fake = (ak.num(non_prompt_lepton) >= 1)
        no_fake = (ak.num(non_prompt_lepton) == 0)



        selection = PackedSelection()
        selection.add('"skim"',            skim)
        selection.add('lepton selection',  tight)
        selection.add('leading lepton',    leading)
        selection.add('triggers',          (triggers))
        selection.add('filter',            (filters))
        selection.add('ss',                ss)
        selection.add('SS onZ veto',       Zmass)
        selection.add('two jets',          two_jets)
        selection.add('MET > 50',          met)
        selection.add('multilep',          ml)
        selection.add('one jet',           one_jet)
        selection.add('flips',             flip)
        selection.add('fakes',             fake)
        selection.add('no flips',          no_flip)
        selection.add('no fakes',          no_fake)

        sk_reqs = ['"skim"']
        sk_reqs_d = { sel: True for sel in sk_reqs }
        skim = selection.require(**sk_reqs_d)
        
        bl_reqs = sk_reqs + ['lepton selection'] + ['leading lepton'] + ['triggers'] + ['filter']
        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)
        
        ss_reqs = bl_reqs + ['ss'] + ['SS onZ veto'] + ['two jets'] + ['MET > 50']
        ss_reqs_d = { sel: True for sel in ss_reqs }
        ss_sel = selection.require(**ss_reqs_d)
        
        ml_reqs = bl_reqs + ['multilep'] + ['one jet'] + ['MET > 50']
        ml_reqs_d = { sel: True for sel in ml_reqs }
        ml_sel = selection.require(**ml_reqs_d)
        
        flip_reqs = ss_reqs + ['flips'] + ['no fakes']
        flip_reqs_d = { sel: True for sel in flip_reqs }
        flip_sel = selection.require(**flip_reqs_d)
        
        fake_ss_reqs = ss_reqs + ['fakes']
        fake_ss_reqs_d = { sel: True for sel in fake_ss_reqs }
        fake_ss_sel = selection.require(**fake_ss_reqs_d)
        
        fake_ml_reqs = ml_reqs + ['fakes']
        fake_ml_reqs_d = { sel: True for sel in fake_ml_reqs }
        fake_ml_sel = selection.require(**fake_ml_reqs_d)
        
        #cutflow
        cutflow1 = Cutflow(output, ev, weight=weight2)
        #cutflow2 = Cutflow(output, ev, weight=weight)
        
        cutflow1_reqs_d = {}
        for req in ss_reqs:
            cutflow1_reqs_d.update({req: True})
            cutflow1.addRow(req, selection.require(**cutflow1_reqs_d) )
        
        #cutflow2_reqs_d = {}
        #for req in ml_reqs:
        #    cutflow2_reqs_d.update({req: True})
        #    cutflow2.addRow(req, selection.require(**cutflow2_reqs_d) )
     
        
        #outputs
        
        output["j_vs_b_ss"].fill(
            dataset = dataset,
            n1 = ak.num(jet[ss_sel]),
            n2 = ak.num(btag[ss_sel]),
            weight = weight.weight()[ss_sel]
        )
        
        output["j_vs_b_ss_flips"].fill(
            dataset = dataset,
            n1 = ak.num(jet[flip_sel]),
            n2 = ak.num(btag[flip_sel]),
            weight = weight.weight()[flip_sel]
        )
        
        output["j_vs_b_ss_fakes"].fill(
            dataset = dataset,
            n1 = ak.num(jet[fake_ss_sel]),
            n2 = ak.num(btag[fake_ss_sel]),
            weight = weight.weight()[fake_ss_sel]
        )
        
        output["j_vs_b_ss_non_fakes_flips"].fill(
            dataset = dataset,
            n1 = ak.num(jet[ss_sel & no_fake & no_flip]),
            n2 = ak.num(btag[ss_sel & no_fake & no_flip]),
            weight = weight.weight()[ss_sel & no_fake & no_flip]
        )
        
        output["j_vs_b_ml"].fill(
            dataset = dataset,
            n1 = ak.num(jet[ml_sel]),
            n2 = ak.num(btag[ml_sel]),
            weight = weight.weight()[ml_sel]
        )
        
        output["j_vs_b_ml_fakes"].fill(
            dataset = dataset,
            n1 = ak.num(jet[fake_ml_sel]),
            n2 = ak.num(btag[fake_ml_sel]),
            weight = weight.weight()[fake_ml_sel]
        )
        
        output["j_vs_b_ml_non_fakes"].fill(
            dataset = dataset,
            n1 = ak.num(jet[ml_sel & no_fake]),
            n2 = ak.num(btag[ml_sel & no_fake]),
            weight = weight.weight()[ml_sel & no_fake]
        )
        
        return output    

    def postprocess(self, accumulator):
        return accumulator