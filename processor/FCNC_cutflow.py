import awkward as ak
import numpy as np

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection


# this is all slightly better practice
from Tools.objects import Collections, choose, cross, match
from Tools.basic_objects import getJets, getBTagsDeepFlavB
from Tools.config_helpers import loadConfig
from Tools.helpers import build_weight_like
from Tools.cutflow import Cutflow
from Tools.triggers import getFilters
from Tools.lepton_scalefactors2 import LeptonSF2
from Tools.pileup import pileup

class FCNC_cutflow(processor.ProcessorABC):
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
        
        #triggers
        if self.year == 2018:
            triggers = ev.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ | ev.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ | ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL | ev.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8
        if self.year == 2017:
            triggers = ev.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 | ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL | ev.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ | ev.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ
        if self.year == 2016:
            triggers = ev.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ | ev.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ | ev.HLT.TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ | ev.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ
        
        
        ## Electrons
        electron = Collections(ev, "Electron", "tightFCNC", self.year, 0).get()
        electron = electron[((electron.pt > 20) | (electron.conePt > 20)) & (np.abs(electron.eta) < 2.4)] #matches skim 
        
        loose_electron = Collections(ev, "Electron", "fakeFCNC", self.year, 0).get()
        loose_electron = loose_electron[((loose_electron.pt > 20) | (loose_electron.conePt > 20) ) & (np.abs(loose_electron.eta) < 2.4)] #matches skim 
        
        SSelectron = (ak.sum(electron.charge, axis=1) != 0) & (ak.num(electron)==2)
        OSelectron = (ak.sum(electron.charge, axis=1) == 0) & (ak.num(electron)==2)
        
        dielectron = choose(electron, 2)
        dielectron_mass = dielectron.mass
        dielectron_pt = dielectron.pt
        
        electron       = electron[ak.argsort(electron.pt, ascending=False)]
        leading_electron = electron[:,0:1]
        trailing_electron = electron[:,1:2]
        
        loose_electron       = loose_electron[ak.argsort(loose_electron.pt, ascending=False)]
        leading_loose_electron = loose_electron[:,0:1]
        trailing_loose_electron = loose_electron[:,1:2]
        
        diloose_electron = choose(loose_electron, 2)
        diloose_electron_OS = diloose_electron[(diloose_electron.charge == 0)]
        
        ##Muons
        muon = Collections(ev, "Muon", "tightFCNC", self.year, 0).get()
        muon = muon[((muon.pt > 20) | (muon.conePt > 20)) & (np.abs(muon.eta) < 2.4)] #matches skim
        
        loose_muon = Collections(ev, "Muon", "fakeFCNC", self.year, 0).get()
        loose_muon = loose_muon[((loose_muon.pt > 20) | (loose_muon.conePt > 20)) & (np.abs(loose_muon.eta) < 2.4)] #matches skim 
        
        muon       = muon[ak.argsort(muon.pt, ascending=False)]
        leading_muon = muon[:,0:1]
        trailing_muon = muon[:,1:2]
        
        loose_muon       = loose_muon[ak.argsort(loose_muon.pt, ascending=False)]
        leading_loose_muon = loose_muon[:,0:1]
        trailing_loose_muon = loose_muon[:,1:2]
        
        diloose_muon = choose(loose_muon, 2)
        diloose_muon_OS = diloose_muon[(diloose_muon.charge == 0)]

        
        ##Leptons
        lepton   = ak.concatenate([muon, electron], axis=1) #tight leptons, matches skim
        
        SSlepton = ( (ak.sum(lepton.charge, axis=1) != 0) & (ak.num(lepton)==2) )
        OSlepton = ( (ak.sum(lepton.charge, axis=1) == 0) & (ak.num(lepton)==2) )
        
        lepton = lepton[ak.argsort(lepton.pt, ascending = False)]
        leading_lepton = lepton[:,0:1]
        subleading_lepton = lepton[:,1:2]
        subsubleading_lepton = lepton[:,2:3]
        
        dilepton = choose(lepton, 2)
        dilepton_mass = dilepton.mass
        dilepton_pt = dilepton.pt
        
        loose_lepton = ak.concatenate([loose_muon, loose_electron], axis=1) #matches skim
                
        #jets
        jet       = getJets(ev, minPt=25, maxEta=2.4, pt_var='pt', UL = False)
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons
        
        leading_jet = jet[:,0:1]
        subleading_jet = jet[:,1:2]
        subsubleading_jet = jet[:,2:3]
        
        #btags
        
        btag      = getBTagsDeepFlavB(jet, year=self.year, UL=False)
        btag      = btag[ak.argsort(btag.pt, ascending=False)]
        
        jet = jet[jet.pt>30]
        
        leading_btag = btag[:, 0:1]
        subleading_btag = btag[:, 1:2]
                
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi
        
        #gen information
        gen_particles = ev.GenPart
        
        gen_Higgs = gen_particles[(gen_particles.pdgId == 25)][:,-1]
        
        #Higgs momentum
        gHp = gen_Higgs.pvec.absolute()
        
        gen_t = gen_particles[(np.abs(gen_particles.pdgId) == 6)][:,-1]
        
        gen_lepton = ev.GenDressedLepton
        gen_lepton = gen_lepton[ak.argsort(gen_lepton.pt, ascending=False)]
        
        leading_gen_lepton = gen_lepton[:,0:1]
        trailing_gen_lepton = gen_lepton[:,1:2]
        
        gen = ev.Generator
        X = ak.concatenate([gen[np.abs(gen.id2) != 21].x2, gen[np.abs(gen.id1) != 21].x1])        


        # setting up the various weights
        weight = Weights( len(ev) )
        weight2 = Weights( len(ev) )
        
        if not dataset=='EGamma':
            #generator weight
            weight.add("weight", ev.genWeight)  
            weight.add("lepton", self.leptonSF.get(electron, muon))
            weight.add("pileup", self.PU.reweight(ak.to_numpy(ev.Pileup.nTrueInt), to='central'), weightUp = self.PU.reweight(ak.to_numpy(ev.Pileup.nTrueInt), to='up'), weightDown = self.PU.reweight(ak.to_numpy(ev.Pileup.nTrueInt), to='down'), shift=False)
            #add trigger scale factors
            
                      
        #selections
        skim = (((ak.num(loose_lepton) == 2) & (ak.sum(loose_lepton.charge, axis=1) != 0)) | (ak.num(loose_lepton) > 2))
        tight = (((ak.num(lepton) == 2) & (ak.sum(lepton.charge, axis=1) != 0)) | (ak.num(lepton) > 2))
        leading = (ak.min(leading_lepton.pt, axis = 1) > 25)
        filters = getFilters(ev, year=self.year, dataset=dataset, UL = False)
        met =  ( met_pt > 50 )

        
        #SS SRs
        gmass = (((ak.num(electron) == 2) & (ak.all(dielectron_mass > 12, axis=1))) | (ak.num(electron) == 1) | (ak.num(electron) == 0))
        mr_veto = ( ( (ak.num(loose_lepton) == 3) & ( (ak.all(np.abs(diloose_electron_OS.mass-91) > 15, axis=1)) & (ak.all(np.abs(diloose_muon_OS.mass-91) > 15, axis=1)) )  )  | (ak.num(loose_lepton) != 3) )
        mr_veto2 = ( ( (ak.num(loose_lepton) == 3) & ( (ak.all(np.abs(diloose_electron_OS.mass) > 12, axis=1)) & (ak.all(np.abs(diloose_muon_OS.mass) > 12, axis=1)) )  )  | (ak.num(loose_lepton) != 3) )
        ss = (SSlepton & gmass & mr_veto & mr_veto2)
        Zmass = (((ak.num(electron) == 2) & (ak.all(np.abs(dielectron_mass-91) > 15, axis=1))) | (ak.num(electron) == 1) | (ak.num(electron) == 0))
        #Zmass = ak.fill_none(((ak.num(electron) == 2) & ((ak.min(np.abs(dielectron_mass-90), axis = 1) > 15))) | (ak.num(electron) == 1) | (ak.num(electron) == 0), True)
        two_jets = (ak.num(jet) >= 2)
        
        #ML SRs
        ml =  ( (ak.num(lepton) > 2) )
        one_jet = (ak.num(jet) >= 1)


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

        sk_reqs = ['"skim"']
        sk_reqs_d = { sel: True for sel in sk_reqs }
        skim = selection.require(**sk_reqs_d)
        
        bl_reqs = sk_reqs + ['lepton selection'] + ['leading lepton'] + ['triggers'] + ['filter']
        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)
        
        ss_reqs = bl_reqs + ['ss']
        ss_reqs_d = { sel: True for sel in ss_reqs }
        ss_sel = selection.require(**ss_reqs_d)
        
        mass_reqs = ss_reqs + ['SS onZ veto']  
        mass_reqs_d = { sel: True for sel in mass_reqs }
        mass_veto_sel = selection.require(**mass_reqs_d)
        
        j2s_reqs = mass_reqs + ['two jets']
        j2s_reqs_d = { sel: True for sel in j2s_reqs }
        j2ss_sel = selection.require(**j2s_reqs_d)
        
        met_reqs = j2s_reqs + ['MET > 50']
        met_reqs_d = { sel: True for sel in met_reqs }
        met_sel_ss = selection.require(**met_reqs_d)
        
        ml_reqs = bl_reqs + ['multilep']
        ml_reqs_d = { sel: True for sel in ml_reqs }
        ml_sel = selection.require(**ml_reqs_d)
        
        j1ml_reqs = ml_reqs + ['one jet']
        j1ml_reqs_d = { sel: True for sel in j1ml_reqs }
        j1ml_sel = selection.require(**j1ml_reqs_d)
        
        met_reqs2 = j1ml_reqs + ['MET > 50']
        met_reqs_d2 = { sel: True for sel in met_reqs2 }
        met_sel_ml = selection.require(**met_reqs_d2)
        
        #cutflow
        #cutflow1 = Cutflow(output, ev, weight=weight)
        cutflow2 = Cutflow(output, ev, weight=weight2)
        
        #cutflow1_reqs_d = {}
        #for req in met_reqs:
        #    cutflow1_reqs_d.update({req: True})
        #    cutflow1.addRow(req, selection.require(**cutflow1_reqs_d) )
        
        cutflow2_reqs_d = {}
        for req in met_reqs2:
            cutflow2_reqs_d.update({req: True})
            cutflow2.addRow(req, selection.require(**cutflow2_reqs_d) )
     
        
        #outputs
        
        output["j_vs_b_ss"].fill(
            dataset = dataset,
            n1 = ak.num(jet[met_sel_ss]),
            n2 = ak.num(btag[met_sel_ss]),
            weight = weight.weight()[met_sel_ss]
        )
        
        output["j_vs_b_ml"].fill(
            dataset = dataset,
            n1 = ak.num(jet[met_sel_ml]),
            n2 = ak.num(btag[met_sel_ml]),
            weight = weight.weight()[met_sel_ml]
        )
        
        #output["lead_gen_lep"].fill(
        #    dataset = dataset,
        #    pt = ak.to_numpy(ak.flatten(leading_gen_lepton[(ak.num(leading_gen_lepton) == 1)].pt)),
        #    eta = ak.to_numpy(ak.flatten(leading_gen_lepton[(ak.num(leading_gen_lepton) == 1)].eta)),
        #    phi = ak.to_numpy(ak.flatten(leading_gen_lepton[(ak.num(leading_gen_lepton) == 1)].phi)),
        #    weight = weight.weight()[(ak.num(leading_gen_lepton) == 1)]
        #)
        
        #output["trail_gen_lep"].fill(
        #    dataset = dataset,
        #    pt = ak.to_numpy(ak.flatten(trailing_gen_lepton[(ak.num(trailing_gen_lepton) == 1)].pt)),
        #    eta = ak.to_numpy(ak.flatten(trailing_gen_lepton[(ak.num(trailing_gen_lepton) == 1)].eta)),
        #    phi = ak.to_numpy(ak.flatten(trailing_gen_lepton[(ak.num(trailing_gen_lepton) == 1)].phi)),
        #    weight = weight.weight()[(ak.num(trailing_gen_lepton) == 1)]
        #)
        
        #output["lead_gen_lep2"].fill(
        #    dataset = dataset,
        #    pt = ak.to_numpy(ak.flatten(leading_gen_lepton[skim & (ak.num(leading_gen_lepton) == 1)].pt)),
        #    eta = ak.to_numpy(ak.flatten(leading_gen_lepton[skim & (ak.num(leading_gen_lepton) == 1)].eta)),
        #    phi = ak.to_numpy(ak.flatten(leading_gen_lepton[skim & (ak.num(leading_gen_lepton) == 1)].phi)),
        #    weight = weight.weight()[skim & (ak.num(leading_gen_lepton) == 1)]
        #)
        
        #output["trail_gen_lep2"].fill(
        #    dataset = dataset,
        #    pt = ak.to_numpy(ak.flatten(trailing_gen_lepton[skim & (ak.num(trailing_gen_lepton) == 1)].pt)),
        #    eta = ak.to_numpy(ak.flatten(trailing_gen_lepton[skim & (ak.num(trailing_gen_lepton) == 1)].eta)),
        #    phi = ak.to_numpy(ak.flatten(trailing_gen_lepton[skim & (ak.num(trailing_gen_lepton) == 1)].phi)),
        #    weight = weight.weight()[skim & (ak.num(trailing_gen_lepton) == 1)]
        #)
        
        #output["N_ele"].fill(
        #    dataset = dataset,
        #    multiplicity = ak.num(loose_lepton),
        #    weight = weight.weight()
        #)
        
        #output["N_ele2"].fill(
        #    dataset = dataset,
        #    multiplicity = ak.num(loose_lepton[skim]),
        #    weight = weight.weight()[skim]
        #)
        
        #output["flipped_electron"].fill(       #don't worry, this is actually a Higgs
        #    dataset = dataset,
        #    pt = gHp,             #full momentum, not just transverse
        #    eta = gen_Higgs.eta,
        #    weight = weight.weight()
        #)
        
        #output["X"].fill(
        #    dataset = dataset,
        #    score = X,
        #    weight = weight.weight()
        #)
             
        #output["flipped_electron2"].fill(       #don't worry, this is actually a top
        #    dataset = dataset,
        #    pt = ak.to_numpy(ak.flatten(gen_t.pt)),
        #    eta = ak.to_numpy(ak.flatten(gen_t.eta)),
        #    weight = weight.weight()
        #)
        
        #output["j1"].fill(
        #    dataset = dataset,
        #    pt = ak.to_numpy(ak.flatten(leading_jet[mass_veto_sel & (ak.num(jet2) >= 1)].pt)),
        #    eta = ak.to_numpy(ak.flatten(leading_jet[mass_veto_sel & (ak.num(jet2) >= 1)].eta)),
        #    phi = ak.to_numpy(ak.flatten(leading_jet[mass_veto_sel & (ak.num(jet2) >= 1)].phi)),
        #    weight=weight.weight()[mass_veto_sel & (ak.num(jet2) >= 1)]

        #)
            
        #output["j2"].fill(
        #    dataset = dataset,
        #    pt = ak.to_numpy(ak.flatten(subleading_jet[mass_veto_sel & (ak.num(jet2) >= 2)].pt)),
        #    eta = ak.to_numpy(ak.flatten(subleading_jet[mass_veto_sel & (ak.num(jet2) >= 2)].eta)),
        #    phi = ak.to_numpy(ak.flatten(subleading_jet[mass_veto_sel & (ak.num(jet2) >= 2)].phi)),
        #    weight=weight.weight()[mass_veto_sel & (ak.num(jet2) >= 2)]
        #)
        
        #output["j3"].fill(
        #    dataset = dataset,
        #    pt = ak.to_numpy(ak.flatten(subsubleading_jet[mass_veto_sel & (ak.num(jet2) >= 3)].pt)),
        #    eta = ak.to_numpy(ak.flatten(subsubleading_jet[mass_veto_sel & (ak.num(jet2) >= 3)].eta)),
        #    phi = ak.to_numpy(ak.flatten(subsubleading_jet[mass_veto_sel & (ak.num(jet2) >= 3)].phi)),
        #    weight=weight.weight()[mass_veto_sel & (ak.num(jet2) >= 3)]
        #)
        
        #output["b1"].fill(
        #    dataset = dataset,
        #    pt = ak.to_numpy(ak.flatten(leading_btag[mass_veto_sel & (ak.num(btag) >= 1)].pt)),
        #    eta = ak.to_numpy(ak.flatten(leading_btag[mass_veto_sel & (ak.num(btag) >= 1)].eta)),
        #    phi = ak.to_numpy(ak.flatten(leading_btag[mass_veto_sel & (ak.num(btag) >= 1)].phi)),
        #    weight=weight.weight()[mass_veto_sel & (ak.num(btag) >= 1)]
        #)
            
        #output["b2"].fill(
        #    dataset = dataset,
        #    pt = ak.to_numpy(ak.flatten(subleading_btag[mass_veto_sel & (ak.num(btag) >= 2)].pt)),
        #    eta = ak.to_numpy(ak.flatten(subleading_btag[mass_veto_sel & (ak.num(btag) >= 2)].eta)),
        #    phi = ak.to_numpy(ak.flatten(subleading_btag[mass_veto_sel & (ak.num(btag) >= 2)].phi)),
        #    weight=weight.weight()[mass_veto_sel & (ak.num(btag) >= 2)]
        #)
        
        #output["N_jet"].fill(
        #    dataset = dataset,
        #    multiplicity = ak.num(jet3)[mass_veto_sel],
        #    weight=weight.weight()[mass_veto_sel]
        #)
        
        #output["N_b"].fill(
        #    dataset = dataset,
        #    multiplicity = ak.num(btag3)[mass_veto_sel],
        #    weight=weight.weight()[mass_veto_sel]
        #)

        
        return output    

    def postprocess(self, accumulator):
        return accumulator