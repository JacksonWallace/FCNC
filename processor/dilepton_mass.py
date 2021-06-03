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
from Tools.lepton_scalefactors import *
from Tools.charge_flip import *
from Tools.gen import find_first_parent, get_charge_parent


class dilepton_mass(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        
        self.btagSF = btag_scalefactor(year)
        
        #self.leptonSF = LeptonSF(year=year)
        
        self.charge_flip_ratio = charge_flip('../histos/chargeflipfull2018.pkl.gz')
        
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
        dataset_weight = ev.metadata['dataset']

        # load the config - probably not needed anymore
        cfg = loadConfig()
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
      
        ## Electrons
        electron = Collections(ev, "Electron", "tightFCNC", 0, self.year).get()
        electron = electron[(electron.pt > 20) & (np.abs(electron.eta) < 2.4)]

        electron = electron[(electron.genPartIdx >= 0)]
        electron = electron[(np.abs(electron.matched_gen.pdgId)==11)]  #from here on all leptons are gen-matched
        electron = electron[( (electron.genPartFlav==1) | (electron.genPartFlav==15) )] #and now they are all prompt
        
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        
        trailing_electron_idx = ak.singletons(ak.argmin(electron.pt, axis=1))
        trailing_electron = electron[trailing_electron_idx]
       
        is_flipped = ( ( (electron.matched_gen.pdgId*(-1) == electron.pdgId) | (find_first_parent(electron.matched_gen)*(-1) == electron.pdgId) ) & (np.abs(electron.pdgId) == 11) )
        
        
        flipped_electron = electron[is_flipped]
        flipped_electron = flipped_electron[(ak.fill_none(flipped_electron.pt, 0)>0)]
        flipped_electron = flipped_electron[~(ak.is_none(flipped_electron))]
        n_flips = ak.num(flipped_electron)
                
        ##Muons
        muon     = Collections(ev, "Muon", "tightFCNC").get()
        muon = muon[(muon.pt > 20) & (np.abs(muon.eta) < 2.4)]
        
        muon = muon[(muon.genPartIdx >= 0)]
        muon = muon[(np.abs(muon.matched_gen.pdgId)==13)] #from here, all muons are gen-matched
        muon = muon[( (muon.genPartFlav==1) | (muon.genPartFlav==15) )] #and now they are all prompt
       
        
        ##Leptons

        lepton   = ak.concatenate([muon, electron], axis=1)
        SSlepton = (ak.sum(lepton.charge, axis=1) != 0) & (ak.num(lepton)==2)
        OSlepton = (ak.sum(lepton.charge, axis=1) == 0) & (ak.num(lepton)==2)
        
        no_mumu = (ak.num(muon) <= 1)
        
        two_lepton = lepton[(ak.num(lepton)>=2)]
        dilepton = choose(two_lepton, 2)
        dilepton_mass = (dilepton['0']+dilepton['1']).mass
        
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]
        
        
        
        #jets
        jet       = getJets(ev, minPt=40, maxEta=2.4, pt_var='pt')
        jet       = jet[ak.argsort(jet.pt, ascending=False)] # need to sort wrt smeared and recorrected jet pt
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] 
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        # setting up the various weights
        weight = Weights( len(ev) )
        weight2 = Weights( len(ev))
        
        if not dataset=='MuonEG':
            # generator weight
            weight.add("weight", ev.genWeight)
            weight2.add("weight", ev.genWeight)
            
        weight2.add("charge flip", self.charge_flip_ratio.flip_weight(electron))
                                   
                      
        #selections    
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        ss = (SSlepton)
        os = (OSlepton)
        jet_all = (ak.num(jet) >= 2)
        flip = (n_flips == 1)
        no_flips = (n_flips == 0)
        nmm = no_mumu
        
        
        selection = PackedSelection()
        selection.add('filter',      (filters) )
        selection.add('ss',          ss)
        selection.add('os',          os)
        selection.add('flip',        flip)
        selection.add('nflip',       no_flips)
        selection.add('jet_all',     jet_all)
        selection.add('no_mumu',     nmm) 
        
        bl_reqs = ['filter', 'no_mumu']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)
        
        s_reqs = bl_reqs + ['ss']
        s_reqs_d = { sel: True for sel in s_reqs }
        ss_sel = selection.require(**s_reqs_d)
        
        os_reqs = bl_reqs + ['os']
        os_reqs_d = { sel: True for sel in os_reqs }
        os_sel = selection.require(**os_reqs_d)
        
        f_reqs = bl_reqs + ['flip']
        f_reqs_d = { sel: True for sel in f_reqs }
        flip_sel = selection.require(**f_reqs_d)
        
        nf_reqs = bl_reqs + ['nflip']
        nf_reqs_d = { sel: True for sel in nf_reqs }
        nflip_sel = selection.require(**nf_reqs_d)
        
   
        
        #outputs
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(lepton)[ss_sel], weight=weight.weight()[ss_sel])
        output['electron_flips'].fill(dataset=dataset, multiplicity=n_flips[ss_sel], weight=weight.weight()[ss_sel])
                      
        output['N_ele2'].fill(dataset=dataset_weight, multiplicity=ak.num(lepton)[os_sel], weight=weight2.weight()[os_sel])
        output['electron_flips2'].fill(dataset=dataset_weight, multiplicity=n_flips[os_sel], weight=weight2.weight()[os_sel])
        
        output["electron"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[ss_sel].pt)),
            eta = ak.to_numpy(ak.flatten(abs(leading_electron[ss_sel].eta))),
            #phi = ak.to_numpy(ak.flatten(leading_electron[baseline].phi)),
            weight = weight.weight()[ss_sel]
        )
        
        output["electron2"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_electron[os_sel].pt)),
            eta = ak.to_numpy(ak.flatten(leading_electron[os_sel].eta)),
            #phi = ak.to_numpy(ak.flatten(leading_electron[baseline].phi)),
            weight = weight2.weight()[os_sel]
        )
        
        output["dilep_mass"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dilepton_mass[ss_sel])),
            weight = weight.weight()[ss_sel],
        )
        
        output["dilep_mass2"].fill(
            dataset = dataset,
            mass = ak.to_numpy(ak.flatten(dilepton_mass[os_sel])),
            weight = weight2.weight()[os_sel],
        )

        return output

    def postprocess(self, accumulator):
        return accumulator